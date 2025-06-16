import torch
from .nafnet import NAFNet

if __name__=="__main__":
    model = NAFNet(in_channels=12)
    x = torch.randn(size=(1,12,60,60))
    with torch.no_grad():
        y = model(x)