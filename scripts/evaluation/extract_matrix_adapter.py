#!/usr/bin/env python3
"""Extract the trained MatrixAdapter from a DrivoR checkpoint.

SVD is performed on the feature map F = W_A @ x  (shape: dim × H*W).
Each rank-k mode is a spatial pattern (right singular vector) encoded by
a hidden-space direction (left singular vector). W_B maps it back to RGB.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsim.agents.drivoR.matrix_adapter import MatrixAdapter

CHECKPOINT = '/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/5-18_5-24/adapt_matrix/lightning_logs/version_6861420/checkpoints/last.ckpt'
INPUT = '/fs/nexus-projects/sim2real/aliu/navsim/scripts/evaluation/images/test_images'
OUTPUT = '/fs/nexus-projects/sim2real/aliu/DrivoR/scripts/evaluation/adapted_images/adapt_matrix'
SAVE_ADAPTER = None  # set to a path like 'adapter_weights.pt' to save adapter weights separately
LIMIT = 1
SAVE_SIDE_BY_SIDE = True  # save original and adapted image side by side

# "real", "virtual", or None (all images under INPUT)
DOMAIN = "virtual"

# None  → run top TOP_RANKS modes, each saved in its own subfolder rank1/, rank2/, ...
# int   → run only that rank (1-indexed)
RANK = None
TOP_RANKS = 5

# output_size=(H, W) used during training: image_size=[1148, 672] → (672, 1148)
OUTPUT_SIZE = (672, 1148)


def feature_map_svd_rank_k(x, W_A, W_B, bias, k):
    """
    Compute the rank-k spatial mode of the feature map F = W_A @ x_flat.

    F shape: (dim, H*W). SVD decomposes it into spatial patterns (rows of Vh)
    weighted by hidden-space directions (cols of U). W_B maps mode k back to RGB.
    Returns the reconstructed image tensor (C, H, W) and the full singular value vector.
    """
    C, H, W = x.shape
    x_flat = x.view(C, -1)                          # (3, H*W)

    F_feat = W_A @ x_flat                            # (dim, H*W)
    U, S, Vh = torch.linalg.svd(F_feat, full_matrices=False)
    # U: (dim, r),  S: (r,),  Vh: (r, H*W),  r = min(dim, H*W)

    # Rank-1 reconstruction for mode k
    F_k = S[k] * torch.outer(U[:, k], Vh[k])        # (dim, H*W)

    out = W_B @ F_k + bias.view(3, 1)                # (3, H*W)  adapter contribution only
    out = out.view(C, H, W)
    out = F.interpolate(out.unsqueeze(0), size=OUTPUT_SIZE,
                        mode="bilinear", align_corners=False).squeeze(0)
    return out, S


def save_image(out_tensor, img_pil, orig_size, out_path, side_by_side):
    to_pil = transforms.ToPILImage()
    out = (out_tensor - out_tensor.min()) / (out_tensor.max() - out_tensor.min() + 1e-8)
    adapted_img = to_pil(out.clamp(0, 1)).resize(orig_size, Image.LANCZOS)

    if side_by_side:
        W, H = orig_size
        canvas = Image.new("RGB", (W * 2, H))
        canvas.paste(img_pil, (0, 0))
        canvas.paste(adapted_img, (W, 0))
        canvas.save(out_path, format="JPEG")
    else:
        adapted_img.save(out_path, format="JPEG")


def main(checkpoint):
    print(f"Loading checkpoint: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]

    prefix = "agent._drivor_model.adapter."
    adapter_sd = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
    if not adapter_sd:
        available = [k for k in state_dict if "adapter" in k]
        print(f"ERROR: No keys found with prefix '{prefix}'.")
        print(f"Keys containing 'adapter': {available}")
        sys.exit(1)
    print(f"Found {len(adapter_sd)} adapter parameter tensors.")

    dim = adapter_sd["A.weight"].shape[0]
    print(f"MatrixAdapter dim={dim}")
    adapter = MatrixAdapter(input_dim=3, dim=dim, output_size=OUTPUT_SIZE)
    adapter.load_state_dict(adapter_sd)
    adapter.eval()

    if SAVE_ADAPTER:
        torch.save(adapter_sd, SAVE_ADAPTER)
        print(f"Saved adapter weights to: {SAVE_ADAPTER}")

    W_A = adapter.A.weight.squeeze(-1).squeeze(-1)   # (dim, 3)
    W_B = adapter.B.weight.squeeze(-1).squeeze(-1)   # (3, dim)
    bias = adapter.B.bias.data                        # (3,)

    input_root = Path(INPUT)
    output_root = Path(OUTPUT)
    search_root = input_root / DOMAIN if DOMAIN else input_root
    image_paths = sorted(search_root.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images under {search_root}")

    to_tensor = transforms.ToTensor()

    counter = 0
    for img_path in image_paths:
        counter += 1
        if LIMIT > 0 and counter > LIMIT:
            break

        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        x = to_tensor(img)   # (3, H, W), no batch dim

        with torch.no_grad():
            # Compute singular values on this image (printed once per image)
            _, S = feature_map_svd_rank_k(x, W_A, W_B, bias, k=0)

        max_rank = min(TOP_RANKS if RANK is None else RANK, S.shape[0])
        print(f"\n{img_path.name}  — singular values of F=W_A@x (dim={dim}, {x.shape[1]}×{x.shape[2]} pixels):")
        for i in range(max_rank):
            print(f"  σ_{i+1} = {S[i].item():.6f}")

        ranks = list(range(max_rank)) if RANK is None else [RANK - 1]

        for k in ranks:
            label = f"rank{k + 1}"
            rel     = img_path.relative_to(input_root)
            out_dir = output_root / label / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{label}_{img_path.stem}.jpg"

            with torch.no_grad():
                out, _ = feature_map_svd_rank_k(x, W_A, W_B, bias, k)

            save_image(out, img, orig_size, out_path, SAVE_SIDE_BY_SIDE)
            print(f"  [{label}] saved → {out_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main(CHECKPOINT)
