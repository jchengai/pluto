import math
import os
import pprint
from pathlib import Path
import numpy
import numpy as np
import pandas as pd
import torch
import cv2


def to_tensor(data):
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, numpy.ndarray):
        if data.dtype == numpy.float64:
            return torch.from_numpy(data).float()
        else:
            return torch.from_numpy(data)
    elif isinstance(data, numpy.number):
        return torch.tensor(data).float()
    elif isinstance(data, list):
        return data
    elif isinstance(data, int):
        return torch.tensor(data)
    elif isinstance(data, tuple):
        return to_tensor(data[0])
    else:
        print(type(data), data)
        raise NotImplementedError


def to_numpy(data):
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            return data.detach().cpu().numpy()
        else:
            return data.cpu().numpy()
    else:
        print(type(data), data)
        raise NotImplementedError


def enable_grad(data):
    if isinstance(data, dict):
        return {k: enable_grad(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.float32:
            data.requires_grad = True
    else:
        raise NotImplementedError


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise NotImplementedError


def print_dict_tensor(data, prefix=""):
    for k, v in data.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"{prefix}{k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict_tensor(v, "    ")


def print_simulation_results(file=None):
    if file is not None:
        df = pd.read_parquet(file)
    else:
        root = Path(os.getcwd()) / "aggregator_metric"
        result = list(root.glob("*.parquet"))
        result = max(result, key=lambda item: item.stat().st_ctime)
        df = pd.read_parquet(result)
    final_score = df[df["scenario"] == "final_score"]
    final_score = final_score.to_dict(orient="records")[0]
    pprint.PrettyPrinter(indent=4).pprint(final_score)


def load_checkpoint(checkpoint):
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    return state_dict


def safe_index(ls, value):
    try:
        return ls.index(value)
    except ValueError:
        return None


def shift_and_rotate_img(img, shift, angle, resolution, cval=-200):
    """
    img: (H, W, C)
    shift: (H_shift, W_shift, 0)
    resolution: float
    angle: float
    """
    rows, cols = img.shape[:2]
    shift = shift / resolution
    translation_matrix = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    translated_img = cv2.warpAffine(
        img, translation_matrix, (cols, rows), borderValue=cval
    )
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
    rotated_img = cv2.warpAffine(translated_img, M, (cols, rows), borderValue=cval)
    if len(img.shape) == 3 and len(rotated_img.shape) == 2:
        rotated_img = rotated_img[..., np.newaxis]
    return rotated_img.astype(np.float32)


def crop_img_from_center(img, crop_size):
    h, w = img.shape[:2]
    h_crop, w_crop = crop_size
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    return img[h_start : h_start + h_crop, w_start : w_start + w_crop].astype(
        np.float32
    )
