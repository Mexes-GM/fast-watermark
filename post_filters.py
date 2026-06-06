"""Post-processing filters: resize, Kuwahara blur, median filter, Gaussian noise."""
from __future__ import annotations

import numpy as np
from PIL import Image
import cv2


# ---------------------------------------------------------------------------
# Resize relative
# ---------------------------------------------------------------------------
_PIL_SAMPLERS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "hamming": Image.HAMMING,
    "bilinear": Image.BILINEAR,
    "box": Image.BOX,
    "nearest": Image.NEAREST,
}


def resize_relative(img: Image.Image, scale_w: float, scale_h: float,
                    method: str = "lanczos") -> Image.Image:
    if scale_w == 1.0 and scale_h == 1.0:
        return img
    sampler = _PIL_SAMPLERS.get(method.lower(), Image.LANCZOS)
    # int() truncation for pixel dimensions.
    new_w = max(1, int(img.width * scale_w))
    new_h = max(1, int(img.height * scale_h))
    return img.resize((new_w, new_h), sampler)


# ---------------------------------------------------------------------------
# Kuwahara blur
# ---------------------------------------------------------------------------
def _kuwahara_rgb_uint8(orig_img: np.ndarray, radius: int,
                        method: str = "mean") -> np.ndarray:
    """
    `orig_img` must be RGB uint8 with shape (H, W, 3).
    Returns RGB uint8 of the same shape.

    MEMORY-OPTIMISED: processes quadrants sequentially instead of
    storing all 4 at once.  For 8K images this saves ~2 GB per call.
    """
    image = orig_img.astype(np.float32, copy=False)
    H, W = image.shape[:2]
    # Convert to grayscale for variance computation.
    image_2d = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY).astype(np.float32, copy=False)
    squared_img = image_2d ** 2

    if method == "mean":
        kxy = np.ones(radius + 1, dtype=np.float32) / (radius + 1)
    elif method == "gaussian":
        kxy = cv2.getGaussianKernel(2 * radius + 1, -1, ktype=cv2.CV_32F)
        kxy /= kxy[radius:].sum()
        klr = np.array([kxy[:radius + 1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]
    else:
        raise ValueError(f"unknown kuwahara method: {method}")

    shift = [(0, 0), (0, radius), (radius, 0), (radius, radius)]

    # ── Sequential processing: one quadrant at a time ──
    best_filtered = np.empty_like(image)
    best_stddev = np.full(image.shape[:2], np.inf, dtype=np.float32)
    # Reusable temporary buffers
    tmp_avg_3ch = np.empty_like(image)
    tmp_avg_2d = np.empty(image.shape[:2], dtype=np.float32)
    tmp_stddev = np.empty(image.shape[:2], dtype=np.float32)

    for k in range(4):
        if method == "mean":
            kx, ky = kxy, kxy
        else:
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image,        -1, kx, ky, tmp_avg_3ch, shift[k])
        cv2.sepFilter2D(image_2d,     -1, kx, ky, tmp_avg_2d,  shift[k])
        cv2.sepFilter2D(squared_img,  -1, kx, ky, tmp_stddev,  shift[k])
        tmp_stddev -= tmp_avg_2d ** 2

        # Update best where this quadrant has lower variance
        mask = tmp_stddev < best_stddev
        best_stddev[mask] = tmp_stddev[mask]
        best_filtered[mask] = tmp_avg_3ch[mask]

    return best_filtered.astype(orig_img.dtype)


def kuwahara_blur(img: Image.Image, radius: int = 3,
                  method: str = "mean") -> Image.Image:
    if radius <= 0:
        return img
    arr = np.asarray(img)
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
        out = _kuwahara_rgb_uint8(rgb, radius, method)
        return Image.fromarray(out[..., 0], mode="L")
    has_alpha = arr.shape[2] == 4
    rgb = arr[..., :3].copy()
    out = _kuwahara_rgb_uint8(rgb, radius, method)
    if has_alpha:
        return Image.fromarray(np.dstack([out, arr[..., 3]]), mode="RGBA")
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------
def median_filter_image(img: Image.Image, size: int = 1) -> Image.Image:
    """Kernel size = 2*size + 1."""
    if size < 1:
        return img
    d = size * 2 + 1
    arr = np.asarray(img)

    if arr.ndim == 2:
        return Image.fromarray(cv2.medianBlur(arr, d), mode="L")

    has_alpha = arr.shape[2] == 4
    rgb = arr[..., :3]

    if d <= 5:
        f = rgb.astype(np.float32) / 255.0
        f = cv2.medianBlur(f, d)
        out = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        out = cv2.medianBlur(rgb.copy(), d)

    if has_alpha:
        return Image.fromarray(np.dstack([out, arr[..., 3]]), mode="RGBA")
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------
def _channel_indices(channels: str, has_alpha: bool) -> list:
    table = {
        "rgb":  [0, 1, 2], "rgba": [0, 1, 2, 3],
        "rg":   [0, 1], "rb": [0, 2], "ra": [0, 3],
        "gb":   [1, 2], "ga": [1, 3], "ba": [2, 3],
        "r":    [0], "g": [1], "b": [2], "a": [3],
    }
    idx = table.get(channels.lower(), [0, 1, 2])
    if not has_alpha:
        idx = [i for i in idx if i != 3]
    return idx


def gaussian_noise(img: Image.Image, strength: float = 0.5,
                   monochromatic: bool = False, invert: bool = False,
                   channels: str = "rgb",
                   seed=None) -> Image.Image:
    """
    Add Gaussian half-normal noise: n = |N(0,1)|;  n /= n.max();
    out = img +/- n * strength;  clip [0,1].
    """
    if strength <= 0:
        return img
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[..., None]
    has_alpha = arr.shape[2] == 4

    idx = _channel_indices(channels, has_alpha)
    if not idx:
        return img

    sub = arr[..., idx]  # (H, W, C')
    if monochromatic and sub.shape[2] > 1:
        noise = rng.standard_normal(sub.shape[:2]).astype(np.float32)
    else:
        noise = rng.standard_normal(sub.shape).astype(np.float32)

    noise = np.abs(noise)
    m = noise.max()
    if m > 0:
        noise = noise / m

    if monochromatic and sub.shape[2] > 1:
        noise = noise[..., None].repeat(sub.shape[2], axis=-1)

    if invert:
        sub = sub - noise * strength
    else:
        sub = sub + noise * strength

    sub = np.clip(sub, 0.0, 1.0)
    arr[..., idx] = sub

    out = (arr * 255.0).astype(np.uint8)
    if out.shape[2] == 1:
        return Image.fromarray(out[..., 0], mode="L")
    if out.shape[2] == 4:
        return Image.fromarray(out, mode="RGBA")
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
DEFAULT_PIPELINE = {
    "enabled": False,
    "upscale": 2.0,
    "upscale_method": "lanczos",
    "kuwahara_radius": 2,
    "kuwahara_method": "mean",
    "median_size": 1,
    "downscale": 0.5,
    "downscale_method": "lanczos",
    "noise_strength": 0.08,
    "noise_monochromatic": True,
    "noise_invert": False,
    "noise_channels": "rgb",
}


def apply_pipeline(img: Image.Image, cfg: dict) -> Image.Image:
    if not cfg.get("enabled", True):
        return img
    out = img
    s_up = float(cfg.get("upscale", 2.0))
    if s_up != 1.0:
        out = resize_relative(out, s_up, s_up,
                              cfg.get("upscale_method", "lanczos"))
    r = int(cfg.get("kuwahara_radius", 2))
    if r > 0:
        out = kuwahara_blur(out, r, cfg.get("kuwahara_method", "mean"))
    m = int(cfg.get("median_size", 1))
    if m > 0:
        out = median_filter_image(out, m)
    s_dn = float(cfg.get("downscale", 0.5))
    if s_dn != 1.0:
        out = resize_relative(out, s_dn, s_dn,
                              cfg.get("downscale_method", "lanczos"))
    ns = float(cfg.get("noise_strength", 0.08))
    if ns > 0:
        out = gaussian_noise(
            out,
            strength=ns,
            monochromatic=bool(cfg.get("noise_monochromatic", True)),
            invert=bool(cfg.get("noise_invert", False)),
            channels=str(cfg.get("noise_channels", "rgb")),
        )
    return out
