import torch
from torch import nn
from IPython.display import clear_output


class DiNOV3(nn.Module):
    """
    Generalized DINOv3 models from torch.hub with optional pretrained weights.

    Args:
        architecture (str): Which backbone to use. Supported: ["vit-s","vit-s-plus","vit-l","vit-h-plus","vit-l-sat"].
                           Default: "vit-l"
        pretrained (bool): If True, loads pretrained weights from HuggingFace URL.
                           Default: True.

    Example:
        >>> from mapminer.models import DiNOV3
        >>> model = DiNOV3(architecture="vit-l", pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224).cuda()
        >>> out = model(x)
    """

    # URLs to checkpoints (can be expanded later)
    pretrained_urls = {
        "vit-s" : "https://huggingface.co/datasets/gajeshladharai/artifacts/resolve/main/dinov3/dinov3_vits16_pretrain_lvd1689m.pth",
        "vit-s-plus" : "https://huggingface.co/datasets/gajeshladharai/artifacts/resolve/main/dinov3/dinov3_vits16plus_pretrain_lvd1689m.pth",
        "vit-l" : "https://huggingface.co/datasets/gajeshladharai/artifacts/resolve/main/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth",
        "vit-h-plus" : "https://huggingface.co/datasets/gajeshladharai/artifacts/resolve/main/dinov3/dinov3_vith16plus_pretrain_lvd1689m.pth",
        "vit-l-sat": "https://huggingface.co/datasets/gajeshladharai/artifacts/resolve/main/dinov3/dinov3_vitl16_pretrain_sat493m.pth",
    }

    hub_entrypoints = {
        "vit-s" : "dinov3_vits16",
        "vit-s-plus" : "dinov3_vits16plus",
        "vit-l" : "dinov3_vitl16",
        "vit-h-plus" : "dinov3_vith16plus",
        "vit-l-sat": "dinov3_vitl16",
    }

    def __init__(self, architecture: str = "vit-l-sat", pretrained: bool = True):
        super().__init__()
        if architecture not in self.hub_entrypoints:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                             f"Choose from {list(self.hub_entrypoints.keys())}")
        self.architecture = architecture

        # Load architecture from GitHub (no weights)
        self.model = torch.hub.load(
            "facebookresearch/dinov3",
            self.hub_entrypoints[architecture],
            source="github",
            pretrained=False)

        # Optionally load pretrained weights
        if pretrained:
            url = self.pretrained_urls[architecture]
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        # Clear output if running in Jupyter (keeps notebook clean)
        try:
            clear_output()
        except Exception:
            pass

    def forward(self, x, tokens=False):
        Hp_t, Wp_t = x.shape[-2:]
        y = self.model.forward_features(x)['x_norm_patchtokens']
        if tokens : 
            return y
        B, N, Cemb = y.shape
        y = y.reshape(B,Cemb,Hp_t//16, Wp_t//16)
        return y

    def normalize(self,x):
        """
        Apply ImageNet normalization (mean/std).
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W) in range [0, 1] or [0, 255].
        Returns:
            Tensor: Normalized tensor.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # If input is in [0, 255], scale to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0

        return (x - mean) / std