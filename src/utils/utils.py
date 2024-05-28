import os
import pprint
from pathlib import Path
import numpy
import numpy as np
import pandas as pd
import torch


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
