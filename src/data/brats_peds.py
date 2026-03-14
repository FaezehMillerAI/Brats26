import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
)


MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def _find_files_for_subject(subject_dir: str, subject_id: str) -> Dict[str, str]:
    nii_files = glob.glob(os.path.join(subject_dir, "**", "*.nii*"), recursive=True)
    if not nii_files:
        nii_files = glob.glob(os.path.join(os.path.dirname(subject_dir), f"{subject_id}*.nii*"))

    mapping = {}
    for path in nii_files:
        lower = os.path.basename(path).lower()
        for mod in MODALITIES:
            if re.search(rf"_{mod}(\\.|_)", lower):
                mapping[mod] = path
        if "seg" in lower or "label" in lower:
            mapping["seg"] = path

    return mapping


def _meta_vector(row: pd.Series) -> np.ndarray:
    age = row.get("Age at imaging (days)", np.nan)
    sex = row.get("Sex_at_birth", "Unknown")
    age = float(age) if age == age else 0.0
    age = age / 7000.0
    sex_val = 0.0
    if isinstance(sex, str):
        sex_val = 1.0 if sex.lower().startswith("m") else 0.0
    return np.array([age, sex_val], dtype=np.float32)


def _resolve_roots(root: str) -> Tuple[str, str]:
    # meta_root may differ from data_root in some releases
    pkg_root = os.path.join(root, "PKG - BraTS-PEDs-v1", "BraTS-PEDs-v1")
    alt_root = os.path.join(root, "BraTS-PEDs-v1")

    meta_root = root
    data_root = root

    if os.path.exists(os.path.join(root, "BraTS-PEDs_metadata.tsv")):
        if os.path.isdir(pkg_root):
            data_root = pkg_root
        elif os.path.isdir(alt_root):
            data_root = alt_root
        return meta_root, data_root

    for c in [pkg_root, alt_root, root]:
        if os.path.exists(os.path.join(c, "BraTS-PEDs_metadata.tsv")):
            return c, c

    return root, root


def _subject_dirs(root: str) -> Dict[str, str]:
    dirs = {}
    for split in ["Training", "Validation", "Test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            path = os.path.join(split_dir, name)
            if os.path.isdir(path) and name.startswith("BraTS-PED"):
                dirs[name] = path
    return dirs


def build_subjects(root: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    meta_root, data_root = _resolve_roots(root)
    meta_path = os.path.join(meta_root, "BraTS-PEDs_metadata.tsv")
    df = pd.read_csv(meta_path, sep="\\t", engine="python")
    subject_dirs = _subject_dirs(data_root)

    subjects = []
    for _, row in df.iterrows():
        sid = row["BraTS-SubjectID"]
        subject_dir = subject_dirs.get(sid)
        if subject_dir is None:
            continue
        files = _find_files_for_subject(subject_dir, sid)
        if not all(m in files for m in MODALITIES):
            continue
        if files.get("seg") is None:
            continue
        item = {
            "subject_id": sid,
            "image": [files[m] for m in MODALITIES],
            "label": files.get("seg"),
            "meta": _meta_vector(row),
            "cohort": str(row.get("BraTS2025_cohort", "Training")),
        }
        subjects.append(item)

    split = {"train": [], "val": [], "test": []}
    for item in subjects:
        cohort = item["cohort"].lower()
        if "train" in cohort:
            split["train"].append(item)
        elif "val" in cohort:
            split["val"].append(item)
        else:
            split["test"].append(item)
    return subjects, split


def build_transforms(patch_size, spacing, is_train):
    keys = ["image", "label"]
    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        ConcatItemsd(keys=["image"], name="image", dim=0),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    if is_train:
        transforms += [
            RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    transforms += [EnsureTyped(keys=keys)]
    return transforms


def build_dataloaders(cfg):
    _, split = build_subjects(cfg["data"]["root"])
    if not split["train"] and not split["val"] and not split["test"]:
        raise ValueError(
            "No subjects found. Ensure NIfTI files are extracted from zip archives "
            "and that the root points to the folder containing BraTS-PEDs_metadata.tsv."
        )
    patch_size = cfg["data"]["patch_size"]
    spacing = cfg["data"]["spacing"]

    train_ds = CacheDataset(
        data=split["train"],
        transform=build_transforms(patch_size, spacing, True),
        cache_rate=0.1,
        num_workers=cfg["data"]["num_workers"],
    )
    val_ds = CacheDataset(
        data=split["val"],
        transform=build_transforms(patch_size, spacing, False),
        cache_rate=0.1,
        num_workers=cfg["data"]["num_workers"],
    )
    test_ds = CacheDataset(
        data=split["test"],
        transform=build_transforms(patch_size, spacing, False),
        cache_rate=0.1,
        num_workers=cfg["data"]["num_workers"],
    )

    loader_args = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **loader_args),
        DataLoader(val_ds, shuffle=False, **loader_args),
        DataLoader(test_ds, shuffle=False, **loader_args),
    )
