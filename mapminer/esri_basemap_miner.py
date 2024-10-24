import requests
import numpy as np
import xarray as xr
import rioxarray
from PIL import Image
from io import BytesIO
from shapely.geometry import Point
import geopandas as gpd


class ESRIBaseMapMiner:
    """
    A class to fetch and process ESRI basemap imagery, reproject it to UTM, and return it as an xarray.DataArray,
    with capture date added to the dataset's attributes (metadata).
    """

    def __init__(self):
        """
        Initializes the ESRIBaseMapMiner class.
        """
        self.service_url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer"

    def _get_utm_crs(self, lon, lat):
        """
        Determines the UTM CRS (EPSG code) based on the provided longitude and latitude.
        """
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = f"EPSG:{326 if lat >= 0 else 327}{utm_zone:02d}"  # Use EPSG:326XX for Northern Hemisphere, 327XX for Southern
        return utm_crs

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, resolution=1.5):
        """
        Fetches the ESRI basemap imagery, reprojects it to UTM, and returns it as an xarray.DataArray with metadata.

        Parameters:
        - lat (float): Latitude of the point in WGS84.
        - lon (float): Longitude of the point in WGS84.
        - radius (float): Radius in meters.
        - polygon (shapely.geometry.Polygon): Optional polygon bounding box.
        - resolution (float): Resolution in meters for the image.

        Returns:
        - xarray.DataArray: The basemap imagery with capture date stored in attributes.
        """
        if polygon is None:
            # Create a bounding box around the point (lat, lon) using radius
            polygon = Point(lon, lat).buffer(radius / 111320)  # Approx conversion to degrees
        xmin, ymin, xmax, ymax = polygon.bounds

        # Reproject the bounding box coordinates to EPSG:3857 (Web Mercator)
        xmin_3857, ymin_3857 = self._transform_wgs_to_mercator(xmin, ymin)
        xmax_3857, ymax_3857 = self._transform_wgs_to_mercator(xmax, ymax)

        # Fetch basemap data and capture metadata
        data_array, capture_date = self._fetch_and_process_basemap(xmin_3857, ymin_3857, xmax_3857, ymax_3857, resolution)

        # Add the capture date to the DataArray's attributes
        data_array.attrs['metadata'] = {'date':{'value': capture_date}}
        
        return data_array.rio.set_crs('epsg:3857')

    def _transform_wgs_to_mercator(self, lon, lat):
        """
        Transforms coordinates from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857).
        """
        point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857")
        return point.x[0], point.y[0]

    def _fetch_and_process_basemap(self, xmin_merc, ymin_merc, xmax_merc, ymax_merc, resolution):
        """
        Fetches basemap data from the ESRI server using EPSG:3857 (Web Mercator) and returns it as an xarray.DataArray.
        It also returns the capture date as metadata.
        """
        service_url = f"{self.service_url}/export"

        # Set export parameters for image request in Web Mercator (EPSG:3857)
        export_params = {
            'bbox': f'{xmin_merc},{ymin_merc},{xmax_merc},{ymax_merc}',
            'bboxSR': '3857',  # Spatial reference of the bounding box (EPSG:3857)
            'size': f'{int((xmax_merc - xmin_merc) / resolution)},{int((ymax_merc - ymin_merc) / resolution)}',
            'imageSR': '3857',  # Spatial reference of the output image (EPSG:3857)
            'format': 'png',
            'transparent': 'true',
            'f': 'image'
        }

        # Make request to ESRI service
        response = requests.get(service_url, params=export_params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP {response.status_code}")

        # Fetch the capture date from metadata
        capture_date = self._fetch_capture_date(xmin_merc, ymin_merc, xmax_merc, ymax_merc)

        # Load the image
        image = Image.open(BytesIO(response.content))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to NumPy array and ensure it's 3D
        image_array = np.array(image)

        if len(image_array.shape) == 2:  # If single-band (grayscale), convert to RGB
            image_array = np.stack([image_array] * 3, axis=-1)

        # Create xarray DataArray with Web Mercator coordinates
        x_coords = np.linspace(xmin_merc, xmax_merc, image_array.shape[1])
        y_coords = np.linspace(ymin_merc, ymax_merc, image_array.shape[0])
        data_array = xr.DataArray(image_array, coords=[y_coords, x_coords, ['R', 'G', 'B']], dims=["y", "x", "band"])

        return data_array, capture_date

    def _fetch_capture_date(self, xmin_merc, ymin_merc, xmax_merc, ymax_merc):
        """
        Fetches the capture date from the ESRI service for the specified bounding box.
        """
        service_url = f"{self.service_url}/identify"

        # Query for available data within the UTM bounding box
        params = {
            'f': 'json',
            'geometry': f'{(xmin_merc + xmax_merc) / 2},{(ymin_merc + ymax_merc) / 2}',
            'geometryType': 'esriGeometryPoint',
            'sr': '3857',  # Spatial reference of the request
            'mapExtent': f'{xmin_merc},{ymin_merc},{xmax_merc},{ymax_merc}',
            'imageDisplay': '1000,1000,96',
            'tolerance': 1,
            'returnGeometry': 'false',
            'returnCatalogItems': 'true',
        }

        response = requests.get(service_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch capture date: HTTP {response.status_code}")

        response_data = response.json()

        # Extract the capture date from the result
        capture_date = None
        for result in response_data.get('results', []):
            if 'attributes' in result and 'DATE (YYYYMMDD)' in result['attributes']:
                capture_date = result['attributes']['DATE (YYYYMMDD)']
                break

        if not capture_date:
            raise Exception("No capture date found in the response")

        return capture_date
