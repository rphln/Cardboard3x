from torch import nn
from torch.nn.functional import interpolate, relu


def conv2d(in_: int, out: int, kernel_size: int):
    return nn.Conv2d(
        in_,
        out,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )


class SRCNN(nn.Module):
    def __init__(self, channels=1, n1=64, n2=32, f1=9, f2=5, f3=5):
        super().__init__()

        self.conv_1 = conv2d(channels, n1, f1)
        self.conv_2 = conv2d(n1, n2, f2)
        self.conv_3 = conv2d(n2, channels, f3)

        self.apply(SRCNN.init_weights)

    def forward(self, x):
        x = interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)
        x = relu(self.conv_1(x), inplace=True)
        x = relu(self.conv_2(x), inplace=True)
        x = self.conv_3(x)

        return x

    @classmethod
    def init_weights(cls, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, val=0.0)
