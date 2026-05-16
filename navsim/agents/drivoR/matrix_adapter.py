import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixAdapter(nn.Module):
    """
    Sim-to-real adapter parameterized as a low-rank channel-mixing matrix.

    The effective transform is M = W_B @ W_A (rank ≤ dim), applied per-pixel
    via two 1×1 convs with a residual connection. W_B is zero-initialized so
    training starts from identity (stable early gradient flow, same as LoRA).
    """

    def __init__(self, input_dim=3, dim=16, dropout_rate=0.1, output_size=(378, 378)):
        super(MatrixAdapter, self).__init__()

        # A: input_dim → dim  (up-projection)
        self.A = nn.Conv2d(input_dim, dim, kernel_size=1, bias=False)
        # B: dim → input_dim  (down-projection, zero-init for identity start)
        self.B = nn.Conv2d(dim, input_dim, kernel_size=1, bias=True)

        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU()
        self.output_size = output_size

        nn.init.kaiming_uniform_(self.A.weight, nonlinearity="relu")
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, H, W)
        Returns:
            Adapted tensor of shape (B, input_dim, *output_size)
        """
        out = self.activation(self.A(x))   # (B, dim, H, W)
        out = self.dropout(out)
        out = self.B(out)                  # (B, input_dim, H, W)
        out = out + x                      # residual

        out = F.interpolate(out, size=self.output_size, mode="bilinear", align_corners=False)

        if torch.isnan(out).any():
            print(f"[NaN DEBUG - MatrixAdapter] Output contains NaN! count={torch.isnan(out).sum().item()}")

        return out