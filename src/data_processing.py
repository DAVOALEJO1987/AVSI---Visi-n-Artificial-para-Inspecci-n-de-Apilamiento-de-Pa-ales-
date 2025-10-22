
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import shutil

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from .utils import ensure_dir, empty_dir, list_images_by_class, imread_rgb, set_seed

def clean_to_dir(src_root: Path, dst_root: Path, size: Tuple[int,int]=(224,224)) -> pd.DataFrame:
    set_seed(42)
    src_root, dst_root = Path(src_root), Path(dst_root)
    ensure_dir(dst_root)
    rows = []
    items = list_images_by_class(src_root)
    for fp, label in items:
        img = imread_rgb(fp)
        if img is None: 
            continue
        img_res = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        out_folder = dst_root / label
        ensure_dir(out_folder)
        out_path = out_folder / Path(fp).name
        cv2.imwrite(str(out_path), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))
        rows.append((str(out_path), label))
    return pd.DataFrame(rows, columns=["filepath","label"])

def split_to_dirs(df: pd.DataFrame, split_dir: Path, val_size=0.15, test_size=0.15) -> Dict[str, Path]:
    split_dir = Path(split_dir)
    for d in ["train", "val", "test"]:
        ensure_dir(split_dir/d)
        empty_dir(split_dir/d)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1.0-test_size), stratify=train_val["label"], random_state=42)

    def _copy(rows: pd.DataFrame, name: str):
        base = split_dir/name
        for cls in sorted(rows["label"].unique()):
            ensure_dir(base/cls)
        for _, r in rows.iterrows():
            src = Path(r["filepath"]); dst = base/r["label"]/src.name
            shutil.copy2(src, dst)

    _copy(train, "train"); _copy(val, "val"); _copy(test, "test")
    return {"train": split_dir/"train", "val": split_dir/"val", "test": split_dir/"test"}

def build_dataloaders(split_dir: Path, img_size=(224,224), batch_size=32, num_workers=2, balance_with_sampler=True):
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import numpy as np

    imagenet_norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    train_tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        imagenet_norm
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        imagenet_norm
    ])

    train_ds = datasets.ImageFolder(str(Path(split_dir)/"train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(str(Path(split_dir)/"val"),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(str(Path(split_dir)/"test"),  transform=eval_tfms)

    if balance_with_sampler:
        class_count = np.bincount([y for _,y in train_ds.samples])
        class_weights = 1.0 / np.maximum(class_count, 1)
        sample_weights = [class_weights[y] for _,y in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    classes = train_ds.classes
    return train_loader, val_loader, test_loader, classes
