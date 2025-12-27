import numpy as np
import xarray as xr
import torch 
from torch import nn
import dill

from PIL import Image
import geopandas as gpd
from shapely.geometry import shape, box
import rasterio.features
from affine import Affine

from huggingface_hub import hf_hub_download


class SAM3(nn.Module):
    def __init__(self, model=None, processor=None,device='cuda'):
        super().__init__()
        self.device = device
        if model is not None : 
            self.model = model.to(self.device)
            self.processor = processor
        else : 
            self.model, self.processor = self._load_model()

    def forward(self,**kwargs):
        raise NotImplementedError("Gradient Enabled Forward pass Not implemented yet, please use inference()")

    def inference(self,ds,text=None,exemplars=None,conf=0.5,pixel_conf=0.4):
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
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=conf,
                mask_threshold=pixel_conf,
                target_sizes=inputs.get("original_sizes").tolist())[0]
        df = self._to_gdf(ds,results)
        return df

    def _exemplars_to_boxes(self,ds,exemplars):
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


    def _load_model(self, device='cuda'):
        try : 
            from transformers import Sam3Processor, Sam3Model
        except : 
            raise Exception("please install latest version of transformers or pip install transformers==5.0.0rc0")
        path = hf_hub_download(
            repo_id="gajeshladharai/artifacts",
            filename="sam3/model.pt",
            repo_type="dataset",
            token=False
        )

        with open(path, "rb") as f:
            data = dill.load(f)

        model = data["model"].to(device)
        processor = data["processor"]
        return model, processor
    
    def _to_gdf(self,ds,results):
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

if __name__=="__main__":
    sam = SAM3()