#!/usr/bin/env python3
"""Extract the trained adapter from a DrivoR Lightning checkpoint and run images through it."""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsim.agents.drivoR.v2r_adapter import V2RAdapter
from navsim.agents.drivoR.matrix_adapter import MatrixAdapter

CHECKPOINT = '/fs/nexus-projects/sim2real/aliu/DrivoR/exp/ke/5-18_5-24/adapt_percept/lightning_logs/version_6861133/checkpoints/last.ckpt'
INPUT = '/fs/nexus-projects/sim2real/aliu/navsim/scripts/evaluation/images/test_images'
# INPUT  = '/fs/nexus-projects/sim2real/aliu/navsim/dataset/sensor_blobs/test/2021.05.25.14.16.10_veh-35_01100_01664/CAM_F0'
# INPUT = '/fs/nexus-projects/sim2real/aliu/carla_garage_data_navsim_converted/sensor_blobs/Town01_Rep0_routes_training_route47_02_21_01_17_14/CAM_F0'
OUTPUT = '/fs/nexus-projects/sim2real/aliu/DrivoR/scripts/evaluation/adapted_images'
SAVE_ADAPTER = None  # set to a path like 'adapter_weights.pt' to save adapter weights separately
LIMIT = 100
SAVE_SIDE_BY_SIDE = True  # save original and adapted image side by side

# output_size=(H, W) used during training: image_size=[1148, 672] → (672, 1148)
OUTPUT_SIZE = (672, 1148)

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

    # Auto-detect adapter type from checkpoint keys.
    if "encoder_conv1.weight" in adapter_sd:
        hidden_dim = adapter_sd["encoder_conv1.weight"].shape[0]
        print(f"Detected V2RAdapter (hidden_dim={hidden_dim}).")
        adapter = V2RAdapter(input_dim=3, hidden_dim=hidden_dim, dropout_rate=0.1, output_size=OUTPUT_SIZE)
    elif "A.weight" in adapter_sd:
        dim = adapter_sd["A.weight"].shape[0]
        print(f"Detected MatrixAdapter (dim={dim}).")
        adapter = MatrixAdapter(input_dim=3, dim=dim, dropout_rate=0.1, output_size=OUTPUT_SIZE)
    else:
        print(f"ERROR: Cannot identify adapter type from keys: {list(adapter_sd.keys())}")
        sys.exit(1)

    adapter.load_state_dict(adapter_sd)
    adapter.eval()

    if SAVE_ADAPTER:
        torch.save(adapter_sd, SAVE_ADAPTER)
        print(f"Saved adapter weights to: {SAVE_ADAPTER}")

    input_root = Path(INPUT)
    output_root = Path(OUTPUT)
    image_paths = sorted(input_root.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images under {input_root}")

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    counter = 0

    for img_path in image_paths:
        counter += 1
        if LIMIT > 0 and counter > LIMIT:
            break
        rel     = img_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"adapted_{img_path.stem}.jpg"

        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        x = to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            out = adapter(x).squeeze(0)
            out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        adapted_img = to_pil(out.clamp(0, 1)).resize(orig_size, Image.LANCZOS)

        if SAVE_SIDE_BY_SIDE:
            W, H = orig_size
            side_by_side = Image.new("RGB", (W * 2, H))
            side_by_side.paste(img, (0, 0))
            side_by_side.paste(adapted_img, (W, 0))
            side_by_side.save(out_path, format="JPEG")
        else:
            adapted_img.save(out_path, format="JPEG")

    print(f"Done. Saved {len(image_paths)} adapted images to {output_root}")


if __name__ == "__main__":
    main(CHECKPOINT)
