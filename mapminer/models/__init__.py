import torch
from .nafnet import NAFNet
from .convlstm import ConvLSTM
from .dinov3 import DiNOV3


# Aliases for convenience
DINOv3 = DiNOV3
DINOV3 = DiNOV3
DiNOv3 = DiNOV3


if __name__=="__main__":
    try : 
        from IPython.display import clear_output
        clear_output()
    except : 
        pass