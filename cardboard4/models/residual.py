from torch import Tensor
from torch.nn import Conv2d, InstanceNorm2d, Module, PixelShuffle, Sequential, SiLU
from torch.nn.init import calculate_gain, xavier_uniform_, zeros_
from torchvision.transforms import Normalize


def init_weights(module):
    if isinstance(module, Conv2d):
        zeros_(module.bias)
        xavier_uniform_(module.weight, gain=calculate_gain("relu"))


class ResidualBlock(Module):
    def __init__(self, outer_features: int, inner_features: int):
        super().__init__()

        self.inner = Sequential(
            Conv2d(outer_features, inner_features, kernel_size=3, padding=1),
            InstanceNorm2d(inner_features),
            SiLU(),
            Conv2d(inner_features, outer_features, kernel_size=3, padding=1),
            InstanceNorm2d(outer_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.inner(x)


class ResidualNetwork(Module):
    def __init__(
        self,
        outer_features: int = 128,
        inner_features: int = 256,
        upscale_factor: int = 3,
        channels: int = 3,
    ):
        super().__init__()

        self.normalize = Normalize(
            std=[0.2931, 0.2985, 0.2946],
            mean=[0.7026, 0.6407, 0.6265],
        )
        self.denormalize = Normalize(
            std=[1.0 / 0.2931, 1.0 / 0.2985, 1.0 / 0.2946],
            mean=[-0.7026 / 0.2931, -0.6407 / 0.2985, -0.6265 / 0.2946],
        )

        self.root = Sequential(
            Conv2d(channels, outer_features, kernel_size=3, padding=1),
            SiLU(),
        )

        self.stem = Sequential(
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
            ResidualBlock(outer_features, inner_features),
        )

        self.leaf = Sequential(
            Conv2d(
                outer_features,
                channels * upscale_factor ** 2,
                kernel_size=3,
                padding=1,
            ),
            PixelShuffle(upscale_factor),
        )

        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.normalize(x)
        x = self.root(x)
        x = self.stem(x) + x
        x = self.leaf(x)
        x = self.denormalize(x)

        return x
