from typing import Tuple

import torch
from piqa.ssim import ssim
from piqa.utils.functional import gaussian_kernel
from torch import Tensor
from torch.nn import Module


class MultiScaleSSIM(Module):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: Tuple[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
        n_channels: int = 3,
    ):
        super().__init__()

        kernels: Tensor = torch.stack(
            [gaussian_kernel(kernel_size, s).repeat(n_channels, 1, 1) for s in sigma]
        )

        self.register_buffer("kernels", kernels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        css = []

        for index, kernel in enumerate(reversed(self.kernels)):
            ss, cs = ssim(x, y, kernel, channel_avg=False)

            if index == 0:
                css.append(torch.relu(ss))
            else:
                css.append(torch.relu(cs))

        return torch.stack(css, dim=-1).prod(dim=-1).mean(dim=-1)
