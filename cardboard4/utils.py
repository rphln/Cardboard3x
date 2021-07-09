from kornia.losses import psnr, ssim
from torch import Tensor


def mean_psnr(u: Tensor, v: Tensor) -> Tensor:
    return psnr(u, v, 1).mean()


def mean_ssim(u: Tensor, v: Tensor) -> Tensor:
    return ssim(u, v, 9).mean()
