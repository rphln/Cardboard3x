# -*- coding: utf-8 -*-
# %% [markdown]
# # SRCNN
#
# O modelo descrito é o "Super-Resolution Convolutional Neural Network", ou SRCNN, criado por Chao Dong *et al.* em 2015.
#
# A rede é composta por três camadas convolucionais, intercaladas pela função de ativação ReLU. As imagens passam por uma interpolação bicúbica antes de serem alimentadas à rede.
#
# Aqui, usamos a configuração 9-5-5 para tamanho de filtros. A primeira camada produz 64 canais, enquanto a segunda camada produz 32 canais.
#
# ```
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ├─Conv2d: 1-1                            [-1, 64, 96, 96]          15,616
# ├─Conv2d: 1-2                            [-1, 32, 96, 96]          51,232
# ├─Conv2d: 1-3                            [-1, 3, 96, 96]           2,403
# ==========================================================================================
# Total params: 69,251
# Trainable params: 69,251
# Non-trainable params: 0
# Total mult-adds (M): 637.30
# ==========================================================================================
# Input size (MB): 0.01
# Forward/backward pass size (MB): 6.96
# Params size (MB): 0.26
# Estimated Total Size (MB): 7.24
# ==========================================================================================
# ```

# %%

from torch.nn import Conv2d, Module, init
from torch.nn.functional import interpolate, mse_loss, relu
from torchsummary import summary
from torchvision.transforms.functional import normalize


def conv2d(in_channels, out_channels, kernel_size) -> Module:
    return Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def init_parameters(module):
    if isinstance(module, Conv2d):
        init.normal_(module.weight, std=1e-3)
        init.constant_(module.bias, val=0e-0)


class SRCNN(Module):
    N0 = 3
    N1 = 64
    N2 = 32

    F1 = 9
    F2 = 5
    F3 = 5

    def __init__(self):
        super().__init__()

        self.conv1 = conv2d(
            in_channels=self.N0, out_channels=self.N1, kernel_size=self.F1
        )
        self.conv2 = conv2d(
            in_channels=self.N1, out_channels=self.N2, kernel_size=self.F2
        )
        self.conv3 = conv2d(
            in_channels=self.N2, out_channels=self.N0, kernel_size=self.F3
        )

    def forward(self, x):
        x = interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)
        x = relu(self.conv1(x), inplace=True)
        x = relu(self.conv2(x), inplace=True)
        x = self.conv3(x)

        return x


# %% [markdown]
# ## Dependências comuns
#
# Definimos aqui as dependências comuns a todos os modelos treinados e o conjunto de treinamento; também exibimos uma pequena parcela, selecionada aleatoriamente, do conjunto.

# %%

from pathlib import Path

from IPython.display import display
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from srnn import training
from srnn.dataset import TensorPairsDataset

dataset = TensorPairsDataset("var/rphln-safebooru2020-medium.train.h5")

loader = DataLoader(dataset, batch_size=64, shuffle=True)
lr, hr = next(iter(loader))

display(to_pil_image(make_grid(hr)))

# %% [markdown]
# ## Treinamento
#
# Por fim, realizamos o treinamento da rede. O artigo especifica a taxa de aprendizado das duas primeiras camadas como 10<sup>-4</sup> e da última camada como 10<sup>-5</sup>.

# %%

LEARNING_RATE = 1e-4

model = SRCNN()
summary(model, (3, 32, 32))

parameters = [
    {"params": model.conv1.parameters()},
    {"params": model.conv2.parameters()},
    {"params": model.conv3.parameters(), "lr": LEARNING_RATE * 0.1},
]

training(
    model=model,
    parameters=parameters,
    init_parameters=init_parameters,
    criterion=mse_loss,
    learning_rate=LEARNING_RATE,
    epochs=300,
    batch_size=256,
    save_interval=10,
    device="cuda:0",
    dataset=dataset,
    checkpoints=Path("var/checkpoints/"),
    # resume=Path("var/checkpoints/SRCNN-L2-90-4.pth"),
    basename="{model}-Unnormalized-L2-{epoch}-{fold}.pth",
)
