
from __future__ import annotations
import os, shutil, random, time, hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def set_seed(seed: int = 42):
    import numpy as _np, random as _rnd
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    _np.random.seed(seed); _rnd.seed(seed)

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def empty_dir(p: Path):
    p = Path(p)
    if not p.exists():
        return
    for item in p.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

def list_images_by_class(root: Path):
    rows = []
    root = Path(root)
    if not root.exists():
        return rows
    for cdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = cdir.name
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rows.append((str(p), label))
    return rows

def imread_rgb(path_str: str):
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            with Image.open(path_str) as im:
                return np.array(im.convert("RGB"))
        except Exception:
            return None

def brightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())

def contrast(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.std())

def blur_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def perceptual_hash(path: Path, size: int = 8) -> Optional[str]:
    try:
        im = Image.open(path).convert("L").resize((size+1, size), Image.LANCZOS)
        diff = []
        for y in range(size):
            for x in range(size):
                diff.append(im.getpixel((x, y)) > im.getpixel((x+1, y)))
        v = 0
        for i, bit in enumerate(diff):
            v |= (1 if bit else 0) << i
        return hex(v)
    except Exception:
        return None
