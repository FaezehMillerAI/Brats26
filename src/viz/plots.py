import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, out_html):
    df = pd.DataFrame(history)
    fig = px.line(df, x="epoch", y=["train_loss", "val_dice"], markers=True)
    fig.update_layout(title="Training Curves", xaxis_title="Epoch")
    fig.write_html(out_html)


def plot_dice_boxplot(dice_per_case, out_png):
    df = pd.DataFrame(dice_per_case)
    df_m = df.melt(var_name="region", value_name="dice")
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_m, x="region", y="dice")
    plt.title("Per-Region Dice Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_overlay(image, pred, label, out_png):
    # image: (C,D,H,W) torch or numpy
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(label, "detach"):
        label = label.detach().cpu().numpy()

    img = image[0]
    mid = img.shape[0] // 2
    img_slice = img[mid]
    pred_slice = pred[mid]
    label_slice = label[mid]

    plt.figure(figsize=(6, 4))
    plt.imshow(img_slice, cmap="gray")
    plt.contour(label_slice, colors="g", linewidths=1)
    plt.contour(pred_slice, colors="r", linewidths=1)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
