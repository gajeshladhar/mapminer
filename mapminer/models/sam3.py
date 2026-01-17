import numpy as np
import xarray as xr
import torch 
from torch import nn
import torch.nn.functional as F

from PIL import Image
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import shape, box, Point, Polygon, MultiPolygon
import rasterio.features
from affine import Affine

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

from joblib import Parallel, delayed


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
            self.model, self.processor, self.sam3_dir = self._load_model(device=device)
        
        self.pvs = None

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

    def inference(self,ds,text=None,exemplars=None,df=None,conf=0.5,pixel_conf=0.4,full_graph=False):
        """
        Run SAM3 inference using textual and/or visual prompts.

        This method supports three types of prompts:
        1. Text prompts (semantic concepts)
        2. Exemplar geometries (training-style positive/negative regions)
        3. Visual prompts via spatial geometries (points or polygons)

        Visual prompts (`df`) are converted into SAM3-compatible point or
        bounding-box prompts in pixel space. The input image (`ds`) and the
        visual prompt GeoDataFrame (`df`) **must be in the same CRS**.

        When `full_graph=True`, the complete SAM3 inference graph is returned,
        including encoder features, decoder outputs, masks, logits, and IoU
        scores. Vision embeddings resampled to the input image resolution are
        exposed under the `ds_embedding` key.

        Parameters
        ----------
        ds : xarray.DataArray
            Input image with dimensions (y, x, band) and spatial coordinates.
            Must define valid `x` and `y` coordinates.
        text : str, optional
            Concept prompt (e.g., "building", "road", "vehicle").
        exemplars : geopandas.GeoDataFrame, optional
            Exemplar regions used as positive / negative references.
            Expected columns:
            - geometry : Polygon / MultiPolygon
            - label : int (1 = positive, 0 = negative)
        df : geopandas.GeoDataFrame, optional
            Visual prompt geometries in the **same CRS as `ds`**.
            Supported geometry types:
            - Point           → interpreted as point-click prompts
            - Polygon / MultiPolygon → interpreted as bounding-box prompts
        conf : float, default=0.5
            Instance-level confidence threshold applied to final predictions.
        pixel_conf : float, default=0.4
            Pixel-level mask threshold used during mask post-processing.
        full_graph : bool, default=False
            If True, return the full SAM3 output dictionary containing all
            intermediate tensors and artifacts.

        Returns
        -------
        geopandas.GeoDataFrame or dict
            GeoDataFrame of predicted instance geometries and confidence scores
            when `full_graph=False`; otherwise, the full SAM3 output dictionary
            with dense vision embeddings available via `outputs["ds_embedding"]`.
        """
        if df is not None : 
            if self.pvs is None : 
                self.pvs = SAM3PVS(self.sam3_dir,device=self.device)
            
            df, outputs = self.pvs.inference(ds,df,conf=conf,pixel_conf=pixel_conf,full_graph=full_graph)
        
        else : 
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
        latent_dims = _.shape[0]
        outputs['ds_embedding'] = xr.DataArray(_.numpy(),dims=['band','y','x'],coords={'band':[f"A{str(i).zfill(2)}" for i in range(1, latent_dims+1)],'y':ds.y.values,'x':ds.x.values})

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
        return model, processor, sam3_dir
    
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
    



class SAM3PVS(nn.Module):
    """
    SAM3-based Prompted Visual Segmentation (PVS) engine.

    Supports point and polygon visual prompts provided as GeoDataFrames
    in the same CRS as the input image, and returns geospatial instance
    masks with confidence scores.
    """
    def __init__(self, sam3_dir,device='cuda'):
        """
        Initialize SAM3 PVS model and processor.

        Parameters
        ----------
        sam3_dir : str
            Path or HF repo containing SAM3 artifacts.
        device : str, default="cuda"
            Device for inference.
        """
        super().__init__()
        self.model, self.processor = self._load_model(sam3_dir,device=device)
        self.device = device

    def inference(self,ds,df,conf=0.0,pixel_conf=0.0,full_graph=False):
        """
        Run SAM3 inference using visual prompts.

        Parameters
        ----------
        ds : xarray.DataArray
            Input image (y, x, band).
        df : geopandas.GeoDataFrame
            Visual prompts (Point → clicks, Polygon → boxes), same CRS as ds.
        conf : float
            Confidence threshold on IoU scores.
        pixel_conf : float
            Pixel-level mask threshold.
        full_graph : bool
            If True, return full SAM3 outputs.

        Returns
        -------
        geopandas.GeoDataFrame, dict
            Predicted geometries with scores and optional full graph.
        """
        payload = SAM3PVS._process_geoms(ds,df)
        points = payload['points']['points']
        boxes = payload['boxes']['boxes']

        if len(points[0])>0 :
            masks_points, scores_points, outputs = self.process_points(ds,points,pixel_conf=pixel_conf,full_graph=full_graph)
        if len(boxes[0])>0 :
            masks_boxes, scores_boxes, outputs = self.process_polygons(ds,boxes,pixel_conf=pixel_conf,full_graph=full_graph)
        masks_list, scores_list = [], []
        if len(points[0])>0 :
            masks_list.append(masks_points)
            scores_list.append(scores_points)

        if len(boxes[0])>0 :
            masks_list.append(masks_boxes)
            scores_list.append(scores_boxes)

        masks = torch.cat(masks_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        df = self._masks_to_gdf(masks,ds)
        df['score'] = scores 
        df = df[(df.score > conf) & (df.is_valid)]
        return df, outputs


    def process_polygons(self,ds,polygons,labels=None,pixel_conf=0.0,full_graph=False):
        inputs = self.processor(
            images=Image.fromarray(ds.transpose('y','x','band').data), 
            input_boxes=[[p[0] for p in polygons[0]]],return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs,output_hidden_states=full_graph)

        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"],mask_threshold=pixel_conf)[0]
        iou = outputs.iou_scores.cpu()
        if iou.dim() == 3:
            iou = iou.squeeze(0)
        best_ids = iou.argmax(dim=1)
        best_scores = iou.max(dim=1).values

        masks = torch.stack(
            [masks[i, best_ids[i]] for i in range(masks.shape[0])],
            dim=0
        ).unsqueeze(1) 

        return masks, best_scores, outputs


    def process_points(self,ds,points,labels=None,pixel_conf=0.4,full_graph=False):
        if labels is None : 
            labels = [[[1]]*len(points)]
        inputs = self.processor(
            images=Image.fromarray(ds.transpose('y','x','band').data), 
            input_points=points, 
            return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs,output_hidden_states=full_graph)

        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"],mask_threshold=pixel_conf)[0]
        iou = outputs.iou_scores.cpu()
        if iou.dim() == 3:
            iou = iou.squeeze(0)

        best_ids = iou.argmax(dim=1)
        best_scores = iou.max(dim=1).values

        masks = torch.stack(
            [masks[i, best_ids[i]] for i in range(masks.shape[0])],
            dim=0
        ).unsqueeze(1) 

        return masks, best_scores, outputs

    @staticmethod
    def _process_geoms(ds,df):
        points, boxes = SAM3PVS._gdf_to_pixels(ds,df)
        payload = {
            'points': {
                'points': points,
                'labels': [[1]*len(points)] if points else None
            },
            'boxes': {
                'boxes': boxes,
                'labels': [[1]*len(boxes)] if boxes else None
            }
        }
        return payload


    @staticmethod
    def _process_single_geom(geom, inv):
        if geom is None or geom.is_empty:
            return None, None

        if isinstance(geom, Point):
            px, py = inv * (geom.x, geom.y)
            return [[int(px), int(py)]], None

        if isinstance(geom, (Polygon, MultiPolygon)):
            xmin, ymin, xmax, ymax = geom.bounds
            pxmin, pymin = inv * (xmin, ymin)
            pxmax, pymax = inv * (xmax, ymax)
            return None, [[
                int(pxmin), int(pymin),
                int(pxmax), int(pymax)
            ]]

        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    @staticmethod
    def _process_single_mask(mask, transform):
        geoms = [
            shape(geom)
            for geom, val in rasterio.features.shapes(mask, transform=transform)
            if val == 1
        ]
        if not geoms:
            return None
        return unary_union(geoms)


    @staticmethod
    def _gdf_to_pixels(ds, gdf, n_jobs=-1):
        """
        Parallel conversion of GeoDataFrame geometries to pixel-space prompts.
        """
        x = ds.x.values
        y = ds.y.values

        transform = Affine(
            x[1] - x[0], 0, x[0],
            0, y[1] - y[0], y[0]
        )
        inv = ~transform

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(SAM3PVS._process_single_geom)(geom, inv)
            for geom in gdf.geometry
        )

        points, boxes = [], []

        for p, b in results:
            if p is not None:
                points.append(p)
            if b is not None:
                boxes.append(b)

        return [points] or None, [boxes] or None

    @staticmethod
    def _masks_to_gdf(masks, ds, n_jobs=-1):
        """
        Parallel conversion of SAM-style masks to GeoDataFrame
        (1 geometry per mask).
        """
        if masks is None or masks.numel() == 0:
            return gpd.GeoDataFrame()

        # --- affine from xarray ---
        x = ds.x.values
        y = ds.y.values
        transform = Affine(
            x[1] - x[0], 0, x[0],
            0, y[1] - y[0], y[0]
        )

        best_masks = masks[:, 0].cpu().numpy().astype(np.uint8)
        geometries = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(SAM3PVS._process_single_mask)(mask, transform)
            for mask in best_masks
        )
        records = [{"geometry": g} for g in geometries]

        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=ds.rio.crs if hasattr(ds, "rio") else ds.attrs.get("crs")
        ).assign(
            geometry=lambda g: g.geometry.buffer(0)
        )

    @staticmethod
    def _load_model(sam3_dir,device='cuda'):
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        model = Sam3TrackerModel.from_pretrained(sam3_dir,
            torch_dtype="auto",
            trust_remote_code=True)
        model = model.to(device).eval()

        processor = Sam3TrackerProcessor.from_pretrained(sam3_dir,
                    torch_dtype="auto",
                    trust_remote_code=True)
        return model, processor
    

if __name__=="__main__":
    sam = SAM3()