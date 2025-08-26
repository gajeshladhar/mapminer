import torch
from .nafnet import NAFNet
from .convlstm import ConvLSTM
from .dinov3 import DiNOV3


# Aliases for convenience
DINOv3 = DiNOV3
DINOV3 = DiNOV3
DiNOv3 = DiNOV3

if __name__=="__main__":
    model = NAFNet(in_channels=12)
    x = torch.randn(size=(1,12,60,60))
    with torch.no_grad():
        y = model(x)