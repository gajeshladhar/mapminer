import os, sys, subprocess
import torch
from torch import nn
import s3fs


class DINOCHM(nn.Module):
    """
    🌲 DINOCHM canopy height predictor using Meta's HighResCanopyHeight repo.

    Args:
        version (str): Backbone version. Supports 'v2' (ViT-H, DINOv2-based).
        pretrained (bool): If True, downloads & loads pretrained weights.

    Features:
        - Auto-clones the HighResCanopyHeight repo (~/.dinochm/HighResCanopyHeight).
        - Loads ViT-H backbone + DPT decoder (SSLAE).
        - Forward: tensor (B,3,H,W) → canopy height in meters.
    """
    def __init__(self,version='v2',pretrained=True):
        """
        Initialize the DINOCHM model.

        Args:
            version (str): Backbone version. Supports 'v2' (ViT-H, DINOv2-based).
            pretrained (bool): If True, downloads & loads pretrained checkpoint.
        """
        super().__init__()

        if version=='v2':
            checkpoint="SSLhuge_satellite.pth"
            repo_url="https://github.com/facebookresearch/HighResCanopyHeight"
            repo_dir="~/.dinochm/HighResCanopyHeight"
            local_dir="~/.dinochm"
        else : 
            raise ValueError(f"DINO {version} backbone is not supported yet")

        self.version = version
        self.repo_dir = os.path.expanduser(repo_dir)
        self.local_dir = os.path.expanduser(local_dir)
        os.makedirs(self.local_dir, exist_ok=True)

        # clone repo if needed
        if not os.path.exists(self.repo_dir):
            subprocess.check_call(["git", "clone", "--depth", "1", repo_url, self.repo_dir])
        sys.path.append(self.repo_dir)
        
        if pretrained : 
            # download checkpoint via s3fs
            self.ckpt_path = os.path.join(self.local_dir, checkpoint)
            if not os.path.exists(self.ckpt_path):
                print("Downloading weights from Meta s3://dataforgood-fb-data ...")
                fs = s3fs.S3FileSystem(anon=True)
                s3_path = f"dataforgood-fb-data/forests/v1/models/saved_checkpoints/{checkpoint}"
                with fs.open(s3_path, "rb") as src, open(self.ckpt_path, "wb") as dst:
                    dst.write(src.read())
                try : 
                    from IPython.display import clear_output
                    clear_output() 
                except : 
                    pass

        # import repo modules and build model
        from models.backbone import SSLVisionTransformer
        from models.dpt_head import DPTHead

        class SSLAE(nn.Module):
            def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
                super().__init__()
                if huge:
                    self.backbone = SSLVisionTransformer(
                        embed_dim=1280,
                        num_heads=20,
                        out_indices=(9, 16, 22, 29),
                        depth=32,
                        pretrained=pretrained,
                    )
                    self.decode_head = DPTHead(
                        classify=classify,
                        in_channels=(1280, 1280, 1280, 1280),
                        embed_dims=1280,
                        post_process_channels=[160, 320, 640, 1280],
                    )
                else:
                    self.backbone = SSLVisionTransformer(pretrained=pretrained)
                    self.decode_head = DPTHead(classify=classify, n_bins=256)

            def forward(self, x):
                x = self.backbone(x)
                x = self.decode_head(x)
                return x

        # wrap into main module
        self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        if pretrained :
            ckpt = torch.load(self.ckpt_path, map_location="cpu",weights_only=False)['state_dict']
            self.chm_module_.load_state_dict(ckpt, strict=False)


    def forward(self, x):
        """
        Run forward inference.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W), range [0,1] Normalized with given values.
        Returns:
            Tensor: Predicted canopy height map (B,1,H,W), scaled in meters.
        """
        outputs = self.chm_module_(x)
        pred = 10 * outputs + 0.001
        return pred.relu()
    
    
    def normalize(self,x):
        """
        Apply ImageNet normalization (mean/std).
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W) in range [0, 1] or [0, 255].
        Returns:
            Tensor: Normalized tensor.
        """
        mean = torch.tensor([0.430, 0.411, 0.296], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.213, 0.156, 0.143], device=x.device).view(1, 3, 1, 1)

        # If input is in [0, 255], scale to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0

        return (x - mean) / std