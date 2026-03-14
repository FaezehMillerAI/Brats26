# BraTS-PEDs: Edge-Frequency Fusion U-Net (EFFU-Net)

This repository provides a publication-oriented, modular pipeline for pediatric brain tumor segmentation on BraTS-PEDs. It extends a strong 3D U-Net baseline with two explicit novelties inspired by recent edge- and frequency-aware segmentation work:

1. Edge-Enhanced Input (EEI): multi-scale 3D gradient magnitude channels to emphasize tumor boundaries.
2. Frequency-Aware Branch (FAB): log-magnitude FFT features fused into the main encoder.

Optional extensions include modality attention and clinical metadata FiLM conditioning. The codebase includes comprehensive metrics, ablation runners, and high-quality visualizations (Plotly and Seaborn).

## Quick start

1. Install dependencies
```
pip install -r requirements.txt
```

2. Prepare data (expected structure)

```
Brats/
  BraTS-PEDs_metadata.tsv
  BraTS-PEDs_Imaging_Info.tsv
  BraTS-PED-00001-000/
    BraTS-PED-00001-000_t1n.nii.gz
    BraTS-PED-00001-000_t1c.nii.gz
    BraTS-PED-00001-000_t2w.nii.gz
    BraTS-PED-00001-000_t2f.nii.gz
    BraTS-PED-00001-000_seg.nii.gz
  BraTS-PED-00002-000/
  ...
```

3. Train
```
python -m src.train --config configs/default.yaml
```

4. Evaluate
```
python -m src.eval --config configs/default.yaml --split test
```

5. Run ablations
```
python -m scripts.run_ablation --config configs/ablation.yaml
```

## Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

Then set `data_root` in the config to:
```
/content/drive/MyDrive/Brats
```

## Outputs

All artifacts are saved to `outputs/`:
- training curves (Plotly HTML)
- per-class metric plots (Seaborn)
- prediction overlays
```
outputs/
  runs/<run_name>/
    metrics.json
    curves.html
    dice_boxplot.png
    sample_overlays/
```

## Notes

If your dataset is zipped, use `scripts/unpack_brats.py` to unpack it before training.
