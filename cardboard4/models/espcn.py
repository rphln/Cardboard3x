from torch.nn import Conv2d, PixelShuffle, Sequential, Tanh
from torchvision.transforms import Normalize


class ESPCN(Sequential):
    SCALE = 3
    N0 = 3

    F1 = 5
    N1 = 64

    F2 = 3
    N2 = 32

    F3 = 3
    N3 = N0 * (SCALE ** 2)

    def __init__(self):
        super().__init__()

        self.normalize = Normalize(
            std=[0.2931, 0.2985, 0.2946],
            mean=[0.7026, 0.6407, 0.6265],
        )

        self.stem = Sequential(
            Conv2d(self.N0, self.N1, self.F1, padding="same"),
            Tanh(),
            Conv2d(self.N1, self.N2, self.F2, padding="same"),
            Tanh(),
        )
        self.head = Sequential(
            Conv2d(self.N2, self.N3, self.F3, padding="same"),
            PixelShuffle(self.SCALE),
        )
