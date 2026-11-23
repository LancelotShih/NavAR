# Using a one line u-net from segmentation models from pytorch

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255

def build_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    return smp.Unet(
        encoder_name = encoder,
        encoder_weights = "imagenet" if pretrained else None,
        classes = 1
    )

class FloorLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1).float()
        valid = (target != IGNORE).float()
        y = (target == 1).float()

        # BCE: average only over valid pixels
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1)

        # Dice: soft overlap over the whole batch, valid pixels only
        p = torch.sigmoid(logits) * valid
        inter = (p * y).sum()
        dice = 1 - (2 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)

        return bce + self.dice_weight * dice
