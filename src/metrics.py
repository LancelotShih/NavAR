# We need to give train.py the ability to score each epoch

import torch

IGNORE = 255

class FloorMetrics():
    def __init__(self):
        self.reset()

    def reset(self):
        # reset the outcome
        self.tp = 0 # true positive
        self.fp = 0 # false positive
        self.fn = 0 # false negative
        # we don't need to know true negative since we only care about marking the floor

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # check if the prediction matches the target, and then update the tp, fp, and fn metrics accordingly
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

        valid = target != IGNORE
        pred = pred.bool() & valid
        true = (target == 1) & valid

        self.tp += (pred & true).sum().item()
        self.fp += (pred & ~true).sum().item()
        self.fn += (~pred & true).sum().item()

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0
        }

    