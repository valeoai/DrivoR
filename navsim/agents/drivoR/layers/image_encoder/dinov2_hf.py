import torch
import torch.nn as nn
from einops import rearrange
from .dinov2 import DINOv2Encoder

from .grid_mask import GridMask


class HFDINOv2Encoder(nn.Module):
    """Thin DrivoR wrapper around the standalone DINOv2Encoder.

    Adapts the config-object constructor and (img, scene_tokens) forward
    signature expected by DrivoR's model to DINOv2Encoder's explicit-arg API.
    scene_tokens is accepted for interface compatibility but not used.
    GridMask augmentation is applied here (before the encoder) using DrivoR's
    local GridMask, so DINOv2Encoder is constructed with use_grid_mask=False.

    Exposes num_features (= backbone hidden_size) for drivor_model.py callers.
    """

    def __init__(self, config):
        super().__init__()
        model_name = config.get("hf_model_name", "facebook/dinov2-base")
        _backbone_size = config.get("hf_backbone_size", None)
        backbone_size = tuple(_backbone_size) if _backbone_size is not None else (224, 336)

        self.encoder = DINOv2Encoder(
            model_name=model_name,
            output_dim=config.tf_d_model,
            num_tokens=config.num_scene_tokens,
            backbone_size=backbone_size,
            use_grid_mask=False,
        )
        self.num_features = self.encoder.hidden_size

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = img.shape

        if self.use_grid_mask:
            img = rearrange(img, 'b n c h w -> (b n) c h w')
            img = self.grid_mask(img)
            img = rearrange(img, '(b n) c h w -> b n c h w', b=B, n=N)

        return self.encoder(img)
