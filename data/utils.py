import numpy as np
import SimpleITK as sitk


def load_image(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    return arr.astype(np.float32)

def zscore_norm(img):
    """Standard per-image z-score normalization."""
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-8)


def percentile_norm(img, p_min=1, p_max=99):
    """Percentile-based normalization (robust to outliers)."""
    lower = np.percentile(img, p_min) if p_min > 0 else 0
    upper = np.percentile(img, p_max)
    denom = (upper - lower) if upper > lower else 1.0
    return np.clip((img - lower) / (denom + 1e-8), 0, 1)


def minmax_norm(img):
    """Normalize to [0, 1] based on min/max."""
    vmin, vmax = np.min(img), np.max(img)
    denom = (vmax - vmin) if vmax > vmin else 1.0
    return (img - vmin) / (denom + 1e-8)

def linear_norm(img, vmin, vmax):
    if vmax <= vmin:
        raise ValueError(f"Invalid vmin/vmax: {vmin}, {vmax}")
    return (img - vmin) / (vmax - vmin)

def rescale(img):
    """Rescale from [0, 1] to [-1, 1]."""
    return (img - 0.5) * 2.0

def revert_rescale(img):
    """Rescale from [-1, 1] to [0, 1]."""
    return (img / 2.0) + 0.5

def linear_denorm(img, vmin, vmax):
    if vmax <= vmin:
        raise ValueError(f"Invalid vmin/vmax: {vmin}, {vmax}")
    return img * (vmax - vmin) + vmin

def pop_denorm(img, mean, std):
    """Revert population normalization."""
    return img * std + mean

def normalize_fn(img, method="pop_zscore", modality=None, region=None, **kwargs):
    method = method.lower()
    if method in ["zscore", "standard"]:
        return zscore_norm(img)
    elif method in ["percentile", "pctl"]:
        p_min = kwargs.get("p_min", 0)
        p_max = kwargs.get("p_max", 99)
        return percentile_norm(img, p_min, p_max)
    elif method in ["minmax"]:
        return minmax_norm(img)
    elif method in ["linear"]:
        vmin = kwargs.get("vmin")
        vmax = kwargs.get("vmax")
        return linear_norm(img, vmin, vmax)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")



