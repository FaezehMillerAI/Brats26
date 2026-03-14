import torch
from monai.metrics import HausdorffDistanceMetric


def _compute_basic(pred, target, eps=1e-6):
    dims = tuple(range(2, pred.ndim))
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    tn = ((1 - pred) * (1 - target)).sum(dims)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return dice, iou, precision, recall, specificity


class MetricTracker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
        self.hd95 = HausdorffDistanceMetric(percentile=95, include_background=True)

    def reset(self):
        self.values = {"dice": [], "iou": [], "precision": [], "recall": [], "specificity": [], "hd95": []}

    def update(self, preds, targets):
        preds = (preds > self.threshold).float()
        dice, iou, precision, recall, specificity = _compute_basic(preds, targets)
        self.values["dice"].append(dice.detach().cpu())
        self.values["iou"].append(iou.detach().cpu())
        self.values["precision"].append(precision.detach().cpu())
        self.values["recall"].append(recall.detach().cpu())
        self.values["specificity"].append(specificity.detach().cpu())
        hd = self.hd95(preds, targets).detach().cpu()
        self.values["hd95"].append(hd)

    def compute(self):
        result = {}
        for k, vals in self.values.items():
            if not vals:
                result[k] = None
                continue
            stack = torch.cat(vals, dim=0)
            result[k] = stack.mean(dim=0).tolist()
        return result
