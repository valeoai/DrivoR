import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_LOAD_LOCK = threading.Lock()


class DINOv2Encoder(nn.Module):
    """Frozen HuggingFace DINOv2 image encoder with a learned projection neck.

    Loads a pretrained DINOv2 backbone via `transformers.Dinov2Backbone` and
    freezes all weights. A single linear layer (`neck`) projects backbone
    features from `hidden_size` down to `output_dim`.

    The forward pass expects a batch of multi-camera images with shape
    `[B, N, C, H, W]` and returns patch token features with shape
    `[B, N*num_tokens, output_dim]`. If your codebase uses single images,
    set `num_cameras=1` and reshape accordingly.

    Args:
        model_name: HuggingFace model ID, e.g. "facebook/dinov2-base".
            Hidden sizes by variant: small=384, base=768, large=1024, giant=1536.
        output_dim: Output feature dimension after the neck projection.
        num_tokens: Number of spatial patch tokens to keep per image (taken
            from the top-left of the flattened spatial grid). Must be <=
            (backbone_size[0] // 14) * (backbone_size[1] // 14).
        backbone_size: (H, W) to resize images to before the backbone.
            Both dims must be multiples of 14 (DINOv2 patch size). Smaller
            values reduce memory and compute; default (224, 336) yields 384 patches.
        use_grid_mask: If True, apply GridMask augmentation during training.
            Requires `grid_mask.py` (GridMask class) to be importable from the
            same package. Set False if you do not have this dependency.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        output_dim: int = 256,
        num_tokens: int = 64,
        backbone_size: Tuple[int, int] = (224, 336),
        use_grid_mask: bool = True,
    ):
        super().__init__()
        from transformers import Dinov2Backbone, Dinov2Config

        with _LOAD_LOCK:
            dinov2_cfg = Dinov2Config.from_pretrained(model_name)
            dinov2_cfg.out_indices = [dinov2_cfg.num_hidden_layers - 1]
            dinov2_cfg.reshape_hidden_states = True  # output shape: [B, C, h, w]
            self.backbone = Dinov2Backbone.from_pretrained(
                model_name, config=dinov2_cfg, low_cpu_mem_usage=False,
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Recompute each transformer layer's activations during backward rather
        # than storing them — bounds peak memory to one layer at a time.
        # use_reentrant=False is required for correctness under DDP + AMP.
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.backbone_size = backbone_size
        self.num_tokens = num_tokens
        self.hidden_size = dinov2_cfg.hidden_size  # 384 / 768 / 1024 / 1536

        self.neck = nn.Linear(self.hidden_size, output_dim)

        self.use_grid_mask = use_grid_mask
        if use_grid_mask:
            from .grid_mask import GridMask
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a batch of multi-camera images.

        Args:
            img: [B, N, C, H, W] — batch of N camera images per scene.

        Returns:
            tokens: [B, N*num_tokens, output_dim]
        """
        B, N, C, H, W = img.shape
        img = rearrange(img, 'b n c h w -> (b n) c h w')

        if self.training and self.use_grid_mask:
            img = self.grid_mask(img)

        if (H, W) != self.backbone_size:
            img = F.interpolate(img, size=self.backbone_size, mode='bilinear', align_corners=False)

        # feature_maps[-1]: [B*N, hidden_size, h, w]
        feat = self.backbone(img).feature_maps[-1]

        # Flatten spatial dims to a token sequence: [B*N, h*w, hidden_size]
        feat = feat.flatten(2).transpose(1, 2)

        # Keep the first num_tokens patch tokens
        tokens = feat[:, :self.num_tokens]          # [B*N, num_tokens, hidden_size]
        tokens = self.neck(tokens)                  # [B*N, num_tokens, output_dim]
        tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)

        return tokens
