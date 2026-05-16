import torch
import torch.nn as nn
import torch.nn.functional as F


class V2RAdapter(nn.Module):
    """
    Virtual-to-Real adapter for sim-to-real domain adaptation.
    Transforms simulated driving images (3×384×1024) to align with real-world
    visual characteristics through a spatial bottleneck autoencoder.
    
    Inspired by: Shen et al. "Task-Driven Domain-Agnostic Learning with 
    Information Bottleneck for Autonomous Steering" (ICRA 2024)
    """

    def __init__(self, input_dim=3, hidden_dim=64, dropout_rate=0.1, output_size=(378, 378)):
        super(V2RAdapter, self).__init__()

        # Encoder: compress spatial dimensions (down-sampling)
        self.encoder_conv1 = nn.Conv2d(input_dim, hidden_dim,
                                       kernel_size=3, stride=2, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(hidden_dim)

        self.encoder_conv2 = nn.Conv2d(hidden_dim, hidden_dim,
                                       kernel_size=3, stride=2, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(hidden_dim)

        # Bottleneck (smallest spatial dimensions)

        # Decoder: expand spatial dimensions (up-sampling)
        self.decoder_conv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                                kernel_size=4, stride=2, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(hidden_dim)

        self.decoder_conv2 = nn.ConvTranspose2d(hidden_dim, input_dim,
                                                kernel_size=4, stride=2, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(input_dim)

        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU()

        # Output size must be multiple of 14 for DINOv2 (patch_size=14)
        # 378 = 14 × 27, 1022 = 14 × 73
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: Input simulation image tensor of shape (B, 3, 384, 1024)

        Returns:
            Adapted image tensor of shape (B, 3, 378, 1022)
        """
        identity = x

        # Encoder: compress spatial dimensions
        # (B, 3, 384, 1024) → (B, 64, 192, 512)
        out = self.activation(self.encoder_bn1(self.encoder_conv1(x)))
        out = self.dropout(out)

        # (B, 64, 192, 512) → (B, 64, 96, 256) BOTTLENECK
        out = self.activation(self.encoder_bn2(self.encoder_conv2(out)))
        out = self.dropout(out)

        # Decoder: expand spatial dimensions
        # (B, 64, 96, 256) → (B, 64, 192, 512)
        out = self.activation(self.decoder_bn1(self.decoder_conv1(out)))
        out = self.dropout(out)

        # (B, 64, 192, 512) → (B, 3, 384, 1024)
        out = self.decoder_bn2(self.decoder_conv2(out))

        # Residual connection (if spatial dimensions match)
        if identity.shape[2:] == out.shape[2:]:
            out = out + identity

        # Resize to DINOv2-compatible size (378×1022)
        # This handles the 384×1024 → 378×1022 transformation
        out = F.interpolate(out, size=self.output_size,
                           mode='bilinear', align_corners=False)

        # NaN check before returning
        if torch.isnan(out).any():
            print(f"[NaN DEBUG - V2RAdapter] Output contains NaN! NaN count: {torch.isnan(out).sum().item()}")

        return out