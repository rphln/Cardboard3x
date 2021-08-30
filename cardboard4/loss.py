from typing import Tuple

import torch
from kornia.filters import get_gaussian_kernel2d
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter


class MultiScaleSSIM(Module):
    def __init__(
        self,
        kernel_size: int,
        sigma: Tuple[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
        upper_bound: float = 1.0,
    ) -> None:
        super().__init__()

        self.c1: float = (0.01 * upper_bound) ** 2
        self.c2: float = (0.03 * upper_bound) ** 2

        filters = [
            get_gaussian_kernel2d((kernel_size, kernel_size), (s, s)) for s in sigma
        ]

        self.filters = Parameter(torch.stack(filters).unsqueeze(1), requires_grad=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        µx: Tensor = conv2d(x, self.filters)
        µy: Tensor = conv2d(y, self.filters)

        mµx = µx[:, 0:1, :, :]
        mµy = µy[:, 0:1, :, :]

        σx2: Tensor = conv2d(x ** 2, self.filters) - µx ** 2
        σy2: Tensor = conv2d(y ** 2, self.filters) - µy ** 2
        σxy: Tensor = conv2d(x * y, self.filters) - µx * µy

        l = (2.0 * mµx * mµy + self.c1) / (mµx ** 2 + mµy ** 2 + self.c1)
        cs = (2.0 * σxy + self.c2) / (σx2 + σy2 + self.c2)

        return l * torch.prod(cs, dim=1, keepdim=True)
