import argparse
import os
from tqdm import tqdm
import torch

from .config import load_config
from .data.brats_peds import build_dataloaders
from .metrics import MetricTracker
from .models.model import EFFUNet, brats_label_map
from .utils import ensure_dir, get_device, save_json
from .viz.plots import plot_dice_boxplot, save_overlay


def _prep_batch(batch, device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    if labels.ndim == 5:
        labels = labels[:, 0]
    labels = brats_label_map(labels)
    meta = batch.get("meta")
    if meta is not None:
        meta = torch.tensor(meta, device=device)
    return images, labels, meta


def _dice_per_case(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    dims = tuple(range(2, preds.ndim))
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.detach().cpu().tolist()


def evaluate(cfg, split="test", weights=None):
    device = get_device()
    out_dir = os.path.join(cfg["run"]["output_dir"], cfg["run"]["name"])
    ensure_dir(out_dir)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    model = EFFUNet(
        in_modalities=cfg["model"]["in_modalities"],
        base_channels=cfg["model"]["base_channels"],
        edge_enabled=cfg["model"]["edge_enabled"],
        freq_enabled=cfg["model"]["freq_enabled"],
        modality_attention=cfg["model"]["modality_attention"],
        film_metadata=cfg["model"]["film_metadata"],
        meta_dim=2,
        out_channels=3,
    ).to(device)
    weights = weights or os.path.join(out_dir, "best.pt")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))

    model.eval()
    tracker = MetricTracker(threshold=cfg["eval"]["threshold"])
    per_case = []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Eval {split}")
        for i, batch in enumerate(pbar):
            images, labels, meta = _prep_batch(batch, device)
            logits = model(images, meta)
            probs = torch.sigmoid(logits)
            tracker.update(probs, labels)
            per_case += _dice_per_case(probs, labels)

            if i < 5:
                pred = (probs[0, 0] > 0.5).float()
                lbl = labels[0, 0]
                save_overlay(
                    images[0],
                    pred,
                    lbl,
                    os.path.join(out_dir, "sample_overlays", f"case_{i}.png"),
                )

    metrics = tracker.compute()
    save_json(metrics, os.path.join(out_dir, f"metrics_{split}.json"))

    dice_df = [{"WT": d[0], "TC": d[1], "ET": d[2]} for d in per_case]
    plot_dice_boxplot(dice_df, os.path.join(out_dir, f"dice_boxplot_{split}.png"))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, split=args.split, weights=args.weights)


if __name__ == "__main__":
    main()
