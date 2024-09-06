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
import easyocr
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


class GoogleMiner():
    """
    GoogleMiner is a tool for fetching and processing Google basemap imagery and metadata.

    This class provides functionality to:
    - Download Google basemap tiles and stitch them into an xarray.DataArray.
    - Extract metadata (e.g., date of the imagery) from the downloaded tiles using OCR.
    - Generate Google Earth URLs based on coordinates.
    """
    
    def __init__(self,ocr='paddle'):
        """
        Initializes the GoogleMiner with headless Chrome for web scraping and an OCR reader.
        """
        self.ocr = ocr
        self.driver = self.get_driver()
        self.reader = self.get_ocr_reader()
        clear_output()
    
    def get_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = uc.Chrome(options=chrome_options)
        return driver
    
    def get_ocr_reader(self):
        if self.ocr=='easy':
            return easyocr.Reader(['en'])
        else : 
            return PaddleOCR(use_angle_cls=True, lang='en')
    
    def read_text_easy(self,data):
        text = self.reader.readtext(image=data)
        date = None
        confidence = None
        for _ in text:
            try : 
                if len(re.sub(r'[^0-9/]', '',_[1]))>5:
                    t = _[1]
                    t = re.sub(r'[^0-9/]', '',t)
                    if len(re.findall('\/',t))<=1:
                        t = re.sub(r'\/', '',t)
                        t = re.sub(r'(\d{1,2})(\d{1,2})(\d{4})', r'\1/\2/\3',t)
                    date = str(pd.to_datetime(t, format='%m/%d/%Y' if locale.getlocale()[0] == 'en_US' else '%d/%m/%Y').date())
                    confidence = _[2]
                    if len(date)>5:
                        break
                    else : 
                        raise
            except : 
                date = None
                confidence = None
                continue
        return date,confidence
    
    def read_text_paddle(self,data):
        text = self.reader.ocr(data)[0]
        date = None
        confidence = None
        for _ in text:
            try : 
                if len(re.sub(r'[^0-9/]', '',_[1][0]))>5:
                    t = _[1][0]
                    t = re.sub(r'[^0-9/]', '',t)
                    if len(re.findall('\/',t))<=1:
                        t = re.sub(r'\/', '',t)
                        t = re.sub(r'(\d{1,2})(\d{1,2})(\d{4})', r'\1/\2/\3',t)
                    date = str(pd.to_datetime(t, format='%m/%d/%Y' if locale.getlocale()[0] == 'en_US' else '%d/%m/%Y').date())
                    confidence = _[1][1]
                    if len(date)>5:
                        break
                    else : 
                        raise
            except : 
                date = None
                confidence = None
                continue
        return date,confidence
                
    
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
            
        ds,metadata = dask.compute(self.fetch_imagery(bbox,resolution),self.fetch_metadata(bbox,resolution))
        ds.attrs['metadata'] = metadata
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
    
    @dask.delayed()
    def fetch_metadata(self,bbox,resolution):
        """
        Lazily extracts metadata (e.g., capture date) from the basemap imagery using OCR.

        Parameters:
        - bbox (tuple): Bounding box as (west, south, east, north).
        - resolution (float): Desired resolution in meters per pixel.

        Returns:
        - dict: Extracted metadata including the capture date.
        """
        lon,lat = list(box(*bbox).centroid.coords)[0]
        
        self.driver.get(self.generate_google_earth_url(lat,lon,11))
        time.sleep(2)
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ESCAPE)
        max_tries=10
        date = None
        confidence = None
        
        while max_tries>0:
            max_tries-=1
            time.sleep(1)
            png = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(png))
            self.image = image
            data = np.array(image)
            ds = xr.DataArray(data=data,dims=['y','x','band'],coords={'band':[0,1,2],'y':range(data.shape[0]),'x':range(data.shape[1])})
            data = data[int(data.shape[0]*0.940):,int(data.shape[1]*0.05):int(data.shape[1]*0.48)]
            try : 
                date,confidence = self.read_text_paddle(data) if self.ocr=='paddle' else self.read_text_easy(data)
            except : 
                pass
            clear_output()
            if date is not None:
                break
        metadata = {
            'date':{
                'value':date,
                'confidence':confidence
            }
        }
        return metadata
    
    @staticmethod
    def generate_google_earth_url(latitude, longitude, zoom_level):
        """
        Generates a Google Earth URL for a given location and zoom level.

        Parameters:
        - latitude (float): Latitude of the location.
        - longitude (float): Longitude of the location.
        - zoom_level (float): Zoom level (approx. altitude).

        Returns:
        - str: Google Earth URL.
        """
        # Approximate the altitude based on zoom level (this is a simplified approximation)
        # Note: The exact relationship between zoom level and altitude in Google Earth is complex.
        # Here, altitude is just a rough estimate.
        altitude = 40000000 / (2 ** zoom_level)
        distance = altitude * 0.3  # Adjust distance based on altitude (simplified assumption)

        # Fixed values for tilt, heading, pitch, and roll for simplicity
        tilt = 0
        heading = 0
        pitch = 0
        roll = 0

        url = f"https://earth.google.com/web/@{latitude},{longitude},{altitude:.2f}a,{distance:.2f}d,{tilt}y,{heading}h,{pitch}t,{roll}r"
        return url
    
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
            url = f"https://mt1.google.com/vt/lyrs=s&x={tile.x}&y={tile.y}&z={zoom}"
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