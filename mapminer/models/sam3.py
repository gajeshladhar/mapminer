import numpy as np
import xarray as xr
import torch 
from torch import nn
import torch.nn.functional as F
import dill

from PIL import Image
import geopandas as gpd
from shapely.geometry import shape, box
import rasterio.features
from affine import Affine

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


class SAM3(nn.Module):
    """
    Meta's SAM3 integration in mapminer for concept-based instance segmentation.

    This class provides a high-level inference interface for running SAM3 on
    geospatial imagery represented as an xarray DataArray. It supports
    text-based prompts and optional exemplar geometries, and returns results
    as a GeoDataFrame in the original CRS.

    The model operates in inference-only mode (no gradients).

    Parameters
    ----------
    model : torch.nn.Module, optional
        Preloaded SAM3 model. If None, the model is loaded from Hugging Face.
    processor : transformers.Processor, optional
        Corresponding SAM3 processor. Required if model is provided.
    device : str, default="cuda"
        Device to run inference on ("cuda" or "cpu").
    """
    def __init__(self, model=None, processor=None,device='cuda'):
        """
        Initialize the Meta's SAM3 model.

        If model and processor are not provided, they are automatically
        downloaded and loaded from Hugging Face artifacts.

        Parameters
        ----------
        model : torch.nn.Module, optional
            Preloaded SAM3 model.
        processor : transformers.Processor, optional
            SAM3 processor corresponding to the model.
        device : str, default="cuda"
            Device for inference.
        """
        super().__init__()
        self.device = device
        if model is not None : 
            self.model = model.to(self.device)
            self.processor = processor
        else : 
            self.model, self.processor = self._load_model()

    def forward(self,**kwargs):
        """
        Disabled forward pass.

        SAM3 right now supports inference-only usage in MapMiner.
        Use `inference()` instead.

        Raises
        ------
        NotImplementedError
            Always raised to prevent gradient-based usage.
        """
        raise NotImplementedError("Gradient Enabled Forward pass Not implemented yet, please use inference()")

    def inference(self,ds,text=None,exemplars=None,conf=0.5,pixel_conf=0.4,full_graph=False):
        """
        Run SAM3 inference using text and/or exemplar prompts.

        When `full_graph=True`, returns the complete SAM3 output dictionary
        containing intermediate results from all stages of inference
        (encoder, decoder, masks, logits, boxes, etc.). Vision embeddings
        resampled to input image space are exposed under the `ds_embedding` key.

        Parameters
        ----------
        ds : xarray.DataArray
            Input image with dimensions (y, x, band) and spatial coordinates.
        text : str, optional
            Concept prompt (e.g., "building", "road", "vehicle").
        exemplars : geopandas.GeoDataFrame, optional
            Exemplar geometries with optional binary labels (1 = positive, 0 = negative).
        conf : float, default=0.5
            Instance-level confidence threshold.
        pixel_conf : float, default=0.4
            Pixel-level mask threshold.
        full_graph : bool, default=False
            If True, return the full SAM3 output graph including all intermediate
            tensors and post-processed artifacts.

        Returns
        -------
        geopandas.GeoDataFrame or dict
            GeoDataFrame of predicted instances when `full_graph=False`;
            otherwise, the full SAM3 output dictionary with vision embeddings
            available via `outputs["ds_embedding"]`.
        """

        if exemplars is None:
            exemplars, labels = None, None
        else : 
            exemplars, labels = self._exemplars_to_boxes(ds,exemplars)
        inputs = self.processor(
            images=Image.fromarray(ds.transpose('y','x','band').data),
            input_boxes=exemplars,
            input_boxes_labels=labels,
            text=text,
            return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
            **inputs,
            output_hidden_states=full_graph,
            return_dict=True)
        
        results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=conf,
                mask_threshold=pixel_conf,
                target_sizes=inputs.get("original_sizes").tolist())[0]
        df = self._to_gdf(ds,results)

        if full_graph : 
            return df, self._process_graph(ds,outputs)
        return df 

    def _process_graph(self,ds,outputs):
        """
        Normalize SAM3 encoder embeddings, resample to input image resolution,
        and attach them as a dense xarray DataArray to the outputs dict.
        """
        _ = self._normalize_image_embedding(outputs['vision_hidden_states'][-1]).data.cpu()
        H, W = ds.sizes["y"], ds.sizes["x"]
        _ = F.interpolate(
            _,
            size=(H, W),
            mode="bilinear",
            align_corners=False)[0]
        outputs['ds_embedding'] = xr.DataArray(_.numpy(),dims=['band','y','x'],coords={'band':[f"A{str(i).zfill(2)}" for i in range(1, 257)],'y':ds.y.values,'x':ds.x.values})

        return outputs

    def _exemplars_to_boxes(self,ds,exemplars):
        """
        Convert exemplar geometries into pixel-space bounding boxes.

        Exemplars are clipped to the spatial extent of the input image and
        transformed from CRS coordinates into pixel coordinates.

        Parameters
        ----------
        ds : xarray.DataArray
            Input image with spatial coordinates.
        exemplars : geopandas.GeoDataFrame
            GeoDataFrame containing exemplar geometries and optional labels.

        Returns
        -------
        tuple
            - list of bounding boxes [[xmin, ymin, xmax, ymax]]
            - list of integer labels
        """
        if 'label' not in exemplars.columns:
            exemplars['label'] = 1

        extent = box(ds.x.min(), ds.y.min(), ds.x.max(), ds.y.max())
        exemplars = exemplars.assign(geometry=exemplars.geometry.intersection(extent))
        exemplars = exemplars[~exemplars.geometry.is_empty]

        if len(exemplars) == 0:
            return None, None

        # --- affine from xarray ---
        x = ds.x.values
        y = ds.y.values

        transform = Affine(
            x[1] - x[0], 0, x[0],
            0, y[1] - y[0], y[0]
        )

        inv_transform = ~transform  # inverse affine

        # --- convert geometry to pixel space ---
        gdf_pixel = exemplars.copy()
        gdf_pixel["geometry"] = gdf_pixel.geometry.apply(
            lambda g: gpd.GeoSeries([g]).affine_transform(
                [inv_transform.a, inv_transform.b,
                inv_transform.d, inv_transform.e,
                inv_transform.c, inv_transform.f]
            ).iloc[0]
        )
        labels = gdf_pixel["label"].astype(int).tolist()

        exemplars = [
            [int(xmin), int(ymin), int(xmax), int(ymax)]
            for xmin, ymin, xmax, ymax in gdf_pixel.geometry.bounds.values
        ]
        return [exemplars], [labels]


    def _load_model(self, device="cuda"):
        """
        Load SAM3 model and processor from Hugging Face artifacts.

        Requires a recent Transformers version with SAM3 support.

        Parameters
        ----------
        device : str, default="cuda"
            Device to load the model onto.

        Returns
        -------
        tuple
            (model, processor)

        Raises
        ------
        Exception
            If an incompatible Transformers version is installed.
        """
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError:
            raise RuntimeError(
                "Install SAM3-compatible transformers: "
                "pip install transformers==5.0.0rc0"
            )

        local_dir = snapshot_download(
            repo_id="gajeshladharai/artifacts",
            repo_type="dataset",
            allow_patterns=[
                "sam3/config.json",
                "sam3/model.safetensors",
                "sam3/processor_config.json",
                "sam3/tokenizer.json",
                "sam3/tokenizer_config.json",
            ],
            token=False
        )

        sam3_dir = f"{local_dir}/sam3"

        processor = Sam3Processor.from_pretrained(
            sam3_dir,
            trust_remote_code=True
        )

        model = Sam3Model.from_pretrained(
            sam3_dir,
            torch_dtype="auto",
            trust_remote_code=True
        )

        model = model.to(device).eval()

        try : 
            from IPython.display import clear_output 
            clear_output()
        except : 
            pass
        return model, processor
    
    def _to_gdf(self,ds,results):
        """
        Convert SAM3 segmentation outputs into a GeoDataFrame.

        Pixel-space masks are vectorized into polygons and transformed back
        into the original CRS of the input image.

        Parameters
        ----------
        ds : xarray.DataArray
            Input image with spatial coordinates.
        results : dict
            Output dictionary from SAM3 post-processing containing masks and scores.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with columns:
            - geometry : shapely geometry
            - score : confidence score
        """
        if len(results['masks']) == 0:
            return gpd.GeoDataFrame()
        
        x = ds.x.values
        y = ds.y.values

        transform = Affine(
            x[1] - x[0], 0, x[0],
            0, y[1] - y[0], y[0]
        )

        records = []
        for mask, score in zip(results["masks"].data.cpu().numpy(), results["scores"].data.cpu()):
            mask = mask.astype(np.uint8)

            for geom, val in rasterio.features.shapes(mask, transform=transform):
                if val == 1:
                    records.append({
                        "score": float(score),
                        "geometry": shape(geom)
                    })
        gdf = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=ds.rio.crs if hasattr(ds, "rio") else ds.attrs.get("crs")
        )
        gdf["geometry"] = gdf.geometry.buffer(0)
        return gdf
    
    def _normalize_image_embedding(self,emb):
        """
        Normalize SAM3 encoder embedding to (B, C, H, W)
        """
        if emb.dim() == 3:
            # (B, N, C) → (B, C, H, W)
            B, N, C = emb.shape
            H = W = int(N ** 0.5)
            emb = emb.view(B, H, W, C).permute(0, 3, 1, 2)
        elif emb.dim() == 4:
            # (B, H, W, C) → (B, C, H, W)
            emb = emb.permute(0, 3, 1, 2)
        return emb


if __name__=="__main__":
    sam = SAM3()