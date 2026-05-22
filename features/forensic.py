import torch
import torch.nn.functional as F


# ---------------------------
# Kernel cache (Issue #4)
# Avoids re-creating small tensors every forward pass
# ---------------------------
_kernel_cache = {}


def _get_srm_kernels(device):
    key = ('srm', device)
    if key not in _kernel_cache:
        kernels = torch.tensor([
            [[0, 0, 0],
             [0, 1, -1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, -1, 1],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, -1, 0],
             [0, 0, 0]],
        ], dtype=torch.float32, device=device).unsqueeze(1)  # (3,1,3,3)
        _kernel_cache[key] = kernels
    return _kernel_cache[key]


def _get_laplacian_kernel(device):
    key = ('lap', device)
    if key not in _kernel_cache:
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        _kernel_cache[key] = kernel
    return _kernel_cache[key]


# ---------------------------
# Helper: ensure BCHW
# ---------------------------
def _to_bchw(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x


# ---------------------------
# SRM (High-pass residual)
# ---------------------------
def compute_srm(image):
    """
    Simple SRM-like high-pass filters.
    Input:  (B,3,H,W) or (3,H,W)
    Output: (B,3,H,W)
    """
    x = _to_bchw(image)

    # Issue #4: use cached kernels instead of creating new tensors each call
    kernels = _get_srm_kernels(x.device)

    # Apply per channel
    x_gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
    out = F.conv2d(x_gray, kernels, padding=1)  # (B,3,H,W)

    return out.squeeze(0) if image.dim() == 3 else out


# ---------------------------
# FFT magnitude
# ---------------------------
def compute_fft(image):
    """
    FFT magnitude spectrum — computed per-channel (R, G, B independently)
    to capture colour-channel-specific frequency artifacts.
    """
    x = _to_bchw(image)

    # Per-channel FFT (B, 3, H, W) — each colour channel has distinct
    # frequency characteristics under deepfake manipulation
    fft = torch.fft.fft2(x)
    fft_shift = torch.fft.fftshift(fft)

    magnitude = torch.log(1 + torch.abs(fft_shift))  # log-scale stabilise

    # Normalise to zero mean / unit std per sample so FFT values (~[0,8])
    # match the scale of SRM / Laplacian / Wavelet channels (~[-1, 1]).
    # Without this the FFT channels dominate the 12-ch forensic stack.
    mean = magnitude.mean(dim=(-2, -1), keepdim=True)
    std  = magnitude.std(dim=(-2, -1), keepdim=True)
    magnitude = (magnitude - mean) / (std + 1e-8)

    return magnitude.squeeze(0) if image.dim() == 3 else magnitude


# ---------------------------
# Wavelet (Haar approx)
# ---------------------------
def compute_wavelet(image):
    """
    Simple Haar-like decomposition using pooling differences.
    """
    x = _to_bchw(image)

    # Low frequency
    low = F.avg_pool2d(x, kernel_size=2, stride=2)

    # Issue #2: use size= instead of scale_factor=2 to handle odd dimensions
    # scale_factor=2 would produce 127→254 instead of 255, causing shape mismatch
    low_up = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)

    # High frequency residual
    high = x - low_up

    return high.squeeze(0) if image.dim() == 3 else high


# ---------------------------
# Laplacian edge
# ---------------------------
def compute_laplacian(image):
    """
    Laplacian edge detector — computed per-channel (R, G, B independently)
    to capture colour-channel-specific edge artifacts.
    """
    x = _to_bchw(image)

    # Issue #4: use cached kernel
    kernel = _get_laplacian_kernel(x.device)

    # Depthwise convolution: apply the same Laplacian kernel to each
    # of the 3 RGB channels independently (groups=3)
    kernel_3ch = kernel.repeat(3, 1, 1, 1)  # (3, 1, 3, 3)
    edges = F.conv2d(x, kernel_3ch, padding=1, groups=3)  # (B, 3, H, W)

    return edges.squeeze(0) if image.dim() == 3 else edges


# ---------------------------
# Final stack
# ---------------------------
def build_forensic_stack(image):
    """
    Input:  (3,H,W) or (B,3,H,W)
    Output: (C,H,W) or (B,C,H,W)

    C = 3 (srm) + 3 (fft) + 3 (wavelet) + 3 (laplacian) = 12
    """
    srm = compute_srm(image)
    fft = compute_fft(image)
    wavelet = compute_wavelet(image)
    laplacian = compute_laplacian(image)

    return torch.cat([srm, fft, wavelet, laplacian], dim=-3)