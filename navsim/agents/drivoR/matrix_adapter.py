import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixAdapter(nn.Module):
    """
    Sim-to-real adapter parameterized as a low-rank channel-mixing matrix.

    The effective transform is M = W_B @ W_A (rank ≤ dim), applied per-pixel
    via two 1×1 convs with a residual connection so the adapter starts at
    identity and learns only the domain-shift delta.
    """

    def __init__(self, input_dim=3, dim=16, output_size=(378, 378)):
        super(MatrixAdapter, self).__init__()

        self.A = nn.Conv2d(input_dim, dim, kernel_size=1, bias=False)
        self.B = nn.Conv2d(dim, input_dim, kernel_size=1, bias=True)

        self.output_size = output_size
        self._debug_saved = False

        nn.init.normal_(self.A.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, H, W)
        Returns:
            Adapted tensor of shape (B, input_dim, *output_size)
        """
        out = self.B(self.A(x)) + x        # residual: starts at identity (B.weight=0), learns delta

        out = F.interpolate(out, size=self.output_size, mode="bilinear", align_corners=False)

        if torch.isnan(out).any():
            print(f"[NaN DEBUG - MatrixAdapter] Output contains NaN! count={torch.isnan(out).sum().item()}")

        # # Save one input/output image pair on rank 0, first forward call only
        # rank = int(os.environ.get("LOCAL_RANK", 0))
        # if not self._debug_saved and rank == 0:
        #     self._debug_saved = True
        #     try:
        #         from torchvision.utils import save_image
        #         save_dir = "/fs/nexus-projects/sim2real/aliu/DrivoR/scripts/evaluation/adapted_images"
        #         os.makedirs(save_dir, exist_ok=True)
        #         x_in_resized = F.interpolate(x[:1].detach().float(), size=self.output_size, mode="bilinear", align_corners=False)
        #         save_image(x_in_resized[0].clamp(0, 1), os.path.join(save_dir, "input.png"))
        #         save_image(out[:1].detach().float()[0].clamp(0, 1), os.path.join(save_dir, "adapted.png"))
        #         i, o = x_in_resized[0], out[0].detach().float()
        #         print(
        #             f"[MatrixAdapter debug] saved to {save_dir} | "
        #             f"in  mean={i.mean():.3f} std={i.std():.3f} min={i.min():.3f} max={i.max():.3f} | "
        #             f"out mean={o.mean():.3f} std={o.std():.3f} min={o.min():.3f} max={o.max():.3f}"
        #         )
        #     except Exception as e:
        #         print(f"[MatrixAdapter debug] image save failed: {e}")

        return out