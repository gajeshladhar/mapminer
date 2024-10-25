import mercantile
import requests
import re
from PIL import Image
import io
from io import BytesIO
import numpy as np
import pandas as pd
import xarray as xr

from paddleocr import PaddleOCR
from shapely import Polygon, Point, box
from rasterio.transform import from_bounds
import dask
from IPython.display import clear_output

import locale
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

import shutil
import platform
import os
import subprocess


class GoogleTrafficMiner():
    """
    GoogleTrafficMiner is a tool for fetching and processing Real Time Google Traffic imagery and metadata.

    This class provides functionality to:
    - Download Google Traffic tiles and stitch them into an xarray.DataArray.
    """
    
    def __init__(self,ocr='paddle'):
        """
        Initializes the GoogleMiner with headless Chrome for web scraping and an OCR reader.
        """
        self.ocr = ocr
                    
    
    def fetch(self,lat=None,lon=None,radius=None,bbox=None,polygon=None,resolution=1):
        """
        Fetches basemap imagery and metadata for a given location or bounding box or Polygon.

        Parameters:
        - lat (float): Latitude of the center point (if bbox is None).
        - lon (float): Longitude of the center point (if bbox is None).
        - radius (float): Radius around the center point (if bbox is None).
        - bbox (tuple): Bounding box as (west, south, east, north) EPSG:4326.
        - Polygon (shapely.geometry.Polygon): Polygon in EPSG:4326.
        - resolution (float): Desired resolution in meters per pixel.

        Returns:
        - xarray.Dataset: Basemap imagery with metadata.
        """
        if polygon is not None : 
            bbox = polygon.bounds
        elif radius is not None : 
            bbox = Point(lon,lat).buffer(radius/111/1000).bounds
        else : 
            bbox = bbox
            
        ds = self.fetch_imagery(bbox,resolution).compute()
       
        return ds
    
    @dask.delayed()
    def fetch_imagery(self,bbox,resolution):
        """
        Lazily downloads and stitches basemap tiles for the given bbox and resolution.

        Parameters:
        - bbox (tuple): Bounding box as (west, south, east, north).
        - resolution (float): Desired resolution in meters per pixel.

        Returns:
        - xarray.DataArray: Stitched basemap imagery.
        """
        ds = self.download_google_basemap(bbox, resolution)
        return ds
    
    
    @staticmethod
    def download_google_basemap(bbox, resolution):
        """
        Downloads and stitches Google basemap tiles into an xarray.DataArray.

        Parameters:
        bbox (tuple): (west, south, east, north) bounding box in WGS 84 coordinates.
        resolution (float): Desired resolution in meters per pixel.

        Returns:
        xarray.DataArray: Stitched basemap as an xarray.DataArray with georeferencing.
        """

        def resolution_to_zoom(resolution):
            zoom = np.log2(156543.03 / resolution)
            return int(np.ceil(zoom))

        # Determine the zoom level
        zoom = resolution_to_zoom(resolution)

        # Calculate tile bounds using mercantile
        tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom))

        # Download the tiles
        tile_images = []
        for tile in tiles:
            url = f"https://mt1.google.com/vt?hl=es&lyrs=s,traffic|seconds_into_week:-1&x={tile.x}&y={tile.y}&z={zoom}"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            tile_images.append((img, tile))

        # Determine the size of the output image
        tile_width, tile_height = tile_images[0][0].size
        total_width = tile_width * len(set([tile.x for _, tile in tile_images]))
        total_height = tile_height * len(set([tile.y for _, tile in tile_images]))

        # Create an empty image to paste the tiles into
        mosaic = Image.new('RGB', (total_width, total_height))

        # Determine the overall bounding box
        west, south, east, north = mercantile.xy_bounds(tiles[0])
        for _, tile in tile_images[1:]:
            tile_west, tile_south, tile_east, tile_north = mercantile.xy_bounds(tile)
            west = min(west, tile_west)
            south = min(south, tile_south)
            east = max(east, tile_east)
            north = max(north, tile_north)

        # Paste tiles into the mosaic
        for img, tile in tile_images:
            x_offset = (tile.x - tiles[0].x) * tile_width
            y_offset = (tile.y - tiles[0].y) * tile_height
            mosaic.paste(img, (x_offset, y_offset))

        # Calculate the geotransform
        transform = from_bounds(west, south, east, north, total_width, total_height)

        # Convert the image to a NumPy array
        data = np.array(mosaic)

        # Create an xarray.DataArray
        da = xr.DataArray(
            data.transpose(2, 0, 1),  # Transpose to (bands, y, x) format
            dims=["band", "y", "x"],
            coords={
                "band": [1, 2, 3],
                "y": np.linspace(north, south, total_height),
                "x": np.linspace(west, east, total_width)
            },
            attrs={
                "transform": transform,
                "crs": "EPSG:3857"  # Corrected to Web Mercator CRS
            }
        )
        return da
    
if __name__ == '__main__':
    miner = GoogleTrafficMiner()
    ds = miner.fetch(lon=-95.665, lat=39.8283, radius=100)
    print(ds)
