import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

from pystac_client import Client
from odc.stac import load
import planetary_computer

class DEMMiner:
    """
    A class to authenticate with Planetary Computer and fetch DEM data. If unavailable, it generates a dummy DEM.
    """
    
    def __init__(self):
        pass
    
    def authenticate(self):
        """
        Authenticates to Planetary Computer using the provided API key.
        """
        planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")
    
    def fetch(self, lat=None, lon=None, radius=None, polygon=None):
        """
        Fetches DEM data from the Planetary Computer or generates a dummy DEM if the fetch fails.
        
        Args:
            polygon (Polygon): Input polygon for the area of interest.
            crs (str): Coordinate reference system of the input polygon, defaults to 'epsg:4326'.
        
        Returns:
            xr.DataArray: DEM data array or dummy DEM in case of failure.
        """
        # Reproject polygon to WGS84 if not already in 'epsg:4326'
        
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)

        try:
            # Query DEM data from Planetary Computer
            planetary_catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
            query = planetary_catalog.search(collections=["cop-dem-glo-30"], limit=100, bbox=polygon.buffer(300 * (1 / 111 / 1000)).bounds)
            query = list(query.items())

            # Load DEM data using the specified bounds and reproject to UTM
            ds_dem = load(query, bbox=polygon.buffer(300 * (1 / 111 / 1000)).bounds, chunks={}, crs="epsg:4326").astype("float32")["data"].rio.reproject(
                self._get_utm_crs(lat=polygon.centroid.y, lon=polygon.centroid.x)).isel(time=0)
            
            # Replace invalid values (high values) with NaN
            ds_dem.data[:] = np.where(ds_dem.data > 1e10, np.nan, ds_dem.data)
        except:
            # Generate dummy DEM in case of failure
            ds_dem = self.get_dummy_dem(polygon)

        return ds_dem
    
    def _get_utm_crs(self, lat, lon):
        """
        Determines the appropriate UTM CRS based on the latitude and longitude.
        
        Parameters:
        - lat (float): Latitude of the location.
        - lon (float): Longitude of the location.
        
        Returns:
        - str: The EPSG code for the local UTM CRS.
        """
        # Calculate the UTM zone based on longitude
        utm_zone = int((lon + 180) // 6) + 1
        
        # Determine the EPSG code for the northern or southern hemisphere
        if lat >= 0:
            return f"EPSG:326{utm_zone:02d}"  # Northern hemisphere UTM (EPSG:326XX)
        else:
            return f"EPSG:327{utm_zone:02d}"  # Southern hemisphere UTM (EPSG:327XX)
        
    def get_dummy_dem(self, polygon: Polygon, resolution: int = 30) -> xr.DataArray:
        """
        Generates a zero-valued DEM as a fallback.
        
        Args:
            polygon (Polygon): The polygon defining the area of interest.
            resolution (int): The resolution of the grid (in meters). Defaults to 30 meters.
        
        Returns:
            xr.DataArray: Zero-valued xarray DataArray representing the DEM.
        """
        # Determine the UTM CRS from the polygon's centroid
        crs = self._get_utm_crs(lat=polygon.centroid.y, lon=polygon.centroid.x)
        
        # Get the bounds of the buffered polygon in the UTM CRS
        buffered_polygon = polygon.buffer(300 * (1 / 111 / 1000))  # Buffer the polygon slightly
        bounds = gpd.GeoDataFrame([{'geometry': buffered_polygon}], crs='epsg:4326').to_crs(crs).iloc[0, -1].bounds

        # Extract bounds (min/max x and y coordinates)
        min_x, min_y, max_x, max_y = bounds

        # Generate grid coordinates based on the resolution
        xs = np.arange(min_x, max_x, resolution)  # X coordinates (eastings)
        ys = np.arange(min_y, max_y, resolution)  # Y coordinates (northings)

        # Create a zero-valued 2D array
        dem_data = np.zeros((len(ys), len(xs)))

        # Create the DataArray with coordinates and metadata
        ds_dem = xr.DataArray(
            dem_data,
            coords={'y': ys, 'x': xs},  # Set Y (northing) and X (easting) coordinates
            dims=['y', 'x'],  # Specify dimension names
            attrs={
                'crs': crs,  # CRS attribute
                'resolution': resolution,  # Resolution attribute
                'description': 'Dummy DEM with zero elevation values'
            }
        )

        return ds_dem
    
if __name__ == '__main__':
    # Initialize DEMMiner with the service account JSON file
    miner = DEMMiner()
    
    # Fetch dem data for a small area around a given point
    ds = miner.fetch(polygon=Point(73.31452961, 28.01306571).buffer(1e-5))
    
    # Output the results
    print(ds)
