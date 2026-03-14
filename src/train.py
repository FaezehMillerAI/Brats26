import os
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast

from .config import parse_args, load_config
from .data.brats_peds import build_dataloaders
from .losses import build_loss
from .metrics import MetricTracker
from .models.model import EFFUNet, brats_label_map
from .utils import ensure_dir, get_device, save_json, set_seed
from .viz.plots import plot_training_curves


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


def train(cfg):
    set_seed(cfg["run"]["seed"])
    device = get_device()
    out_dir = os.path.join(cfg["run"]["output_dir"], cfg["run"]["name"])
    ensure_dir(out_dir)

    train_loader, val_loader, _ = build_dataloaders(cfg)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = build_loss(cfg["loss"]["dice_weight"], cfg["loss"]["ce_weight"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    best_dice = -1.0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            images, labels, meta = _prep_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                logits = model(images, meta)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        val_metrics = {}
        if epoch % cfg["train"]["val_interval"] == 0:
            model.eval()
            tracker = MetricTracker(threshold=cfg["eval"]["threshold"])
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False)
                for batch in pbar:
                    images, labels, meta = _prep_batch(batch, device)
                    logits = model(images, meta)
                    probs = torch.sigmoid(logits)
                    tracker.update(probs, labels)
            val_metrics = tracker.compute()

            dice_vals = val_metrics.get("dice") or [0, 0, 0]
            mean_dice = float(sum(dice_vals) / len(dice_vals))
            if cfg["train"]["save_best"] and mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, len(train_loader)),
                "val_dice": float(sum(val_metrics.get("dice") or [0, 0, 0]) / 3)
                if val_metrics
                else 0.0,
            }
        )

    save_json(history, os.path.join(out_dir, "history.json"))
    plot_training_curves(history, os.path.join(out_dir, "curves.html"))
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
