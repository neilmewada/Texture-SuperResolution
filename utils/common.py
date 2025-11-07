import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import os
import math
import torch, torch.nn.functional as F

def rgb_to_y(t: torch.Tensor) -> torch.Tensor:
    # [B,3,H,W] -> [B,1,H,W] in [0,1]
    r, g, b = t[:,0:1], t[:,1:2], t[:,2:3]
    return 0.299*r + 0.587*g + 0.114*b

def _gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g /= g.sum()
    window_1d = g.view(1, 1, -1)                                # 1x1xK
    window_2d = (window_1d.transpose(1,2) @ window_1d).unsqueeze(0)  # 1x1xKxK
    window = window_2d.repeat(channels, 1, 1, 1)                 # Cx1xKxK (depthwise)
    return window

@torch.no_grad()
def ssim_y(img1: torch.Tensor, img2: torch.Tensor, data_range=1.0, window_size=11, sigma=1.5, shave=0, reduction="mean"):
    """
    SSIM on Y channel. img1,img2: [B,3,H,W] in [0,1].
    """
    print(img1.shape, " ", img2.shape, " ", img1.dim(), " ", img1.size(1))
    assert img1.shape == img2.shape and img1.dim() == 3 and img1.size(1) == 3
    y1 = rgb_to_y(img1)
    y2 = rgb_to_y(img2)

    if shave > 0:
        y1 = y1[..., shave:-shave, shave:-shave]
        y2 = y2[..., shave:-shave, shave:-shave]

    B, C, H, W = y1.shape
    window = _gaussian_window(window_size, sigma, channels=C, device=y1.device, dtype=y1.dtype)

    # means
    mu1 = F.conv2d(y1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(y2, window, padding=window_size//2, groups=C)

    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2

    # variances & covariance
    sigma1_sq = F.conv2d(y1*y1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(y2*y2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12   = F.conv2d(y1*y2, window, padding=window_size//2, groups=C) - mu12

    # constants (default K1=0.01, K2=0.03)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # reduce over C,H,W
    if reduction == "mean":
        return ssim_map.mean(dim=(1,2,3)).mean().item()
    elif reduction == "none":
        return ssim_map.mean(dim=(1,2,3))   # per-image
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

def read_image(filepath):
    image = io.read_image(filepath, io.ImageReadMode.RGB)
    return image

def write_image(filepath, src):
    io.write_png(src, filepath)

# https://www.researchgate.net/publication/284923134
def rgb2ycbcr(src):
    R = src[0]
    G = src[1]
    B = src[2]
    
    ycbcr = torch.zeros(size=src.shape)
    # *Intel IPP
    # ycbcr[0] = 0.257 * R + 0.504 * G + 0.098 * B + 16
    # ycbcr[1] = -0.148 * R - 0.291 * G + 0.439 * B + 128
    # ycbcr[2] = 0.439 * R - 0.368 * G - 0.071 * B + 128
    # *Intel IPP specific for the JPEG codec
    ycbcr[0] =  0.299 * R + 0.587 * G + 0.114 * B
    ycbcr[1] =  -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    ycbcr[2] =  0.5 * R - 0.41869 * G - 0.08131 * B + 128
    
    # Y in range [16, 235]
    ycbcr[0] = torch.clip(ycbcr[0], 16, 235)
    # Cb, Cr in range [16, 240]
    ycbcr[[1, 2]] = torch.clip(ycbcr[[1, 2]], 16, 240)
    ycbcr = ycbcr.type(torch.uint8)
    return ycbcr

# https://www.researchgate.net/publication/284923134
def ycbcr2rgb(src):
    Y = src[0]
    Cb = src[1]
    Cr = src[2]

    rgb = torch.zeros(size=src.shape)
    # *Intel IPP
    # rgb[0] = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    # rgb[1] = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    # rgb[2] = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    # *Intel IPP specific for the JPEG codec
    rgb[0] = Y + 1.402 * Cr - 179.456
    rgb[1] = Y - 0.34414 * Cb - 0.71414 * Cr + 135.45984
    rgb[2] = Y + 1.772 * Cb - 226.816

    rgb = torch.clip(rgb, 0, 255)
    rgb = rgb.type(torch.uint8)
    return rgb

# list all file in dir and sort
def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def resize_bicubic(src, h, w):
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image

def gaussian_blur(src, ksize=3, sigma=0.5):
    blur_image = transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)(src)
    return blur_image
    
def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = resize_bicubic(src, h, w)
    return image

def downscale(src, scale):
    h = int(src.shape[1] / scale)
    w = int(src.shape[2] / scale)
    image = resize_bicubic(src, h, w)
    return image

def make_lr(src, scale=3):
    h = src.shape[1]
    w = src.shape[2]
    lr_image = downscale(src, scale)
    lr_image = resize_bicubic(lr_image, h, w)
    return lr_image

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def exists(path):
    return os.path.exists(path)

def PSNR(y_true, y_pred, max_val=1.0):
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)
    MSE = torch.mean(torch.square(y_true - y_pred))
    return 10 * torch.log10(max_val * max_val / MSE)

def random_crop(src, h, w):
    crop = transforms.RandomCrop([h, w])(src)
    return crop

def random_transform(src):
    _90_left, _90_right, _180 = 1, 3, 2
    operations = {
        0 : (lambda x : x                                       ),
        1 : (lambda x : torch.rot90(x, k=_90_left,  dims=(1, 2))),
        2 : (lambda x : torch.rot90(x, k=_90_right, dims=(1, 2))),
        3 : (lambda x : torch.rot90(x, k=_180,      dims=(1, 2))),
        4 : (lambda x : torch.fliplr(x)                         ),
        5 : (lambda x : torch.flipud(x)                         ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src)
    return image_transform

def shuffle(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of elements")
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)
    X = torch.index_select(X, dim=0, index=torch.as_tensor(indices))
    Y = torch.index_select(Y, dim=0, index=torch.as_tensor(indices))
    return X, Y

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_cpu(tensor):
    return tensor.detach().cpu()