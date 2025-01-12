import planetary_computer
import pystac_client
import numpy as np
import geopandas as gpd
from shapely import unary_union
import rioxarray
from rioxarray.merge import merge_arrays
from odc.stac import load
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon, Point, box


class NAIPMiner:
    """
    A class for fetching and processing NAIP imagery from Microsoft's Planetary Computer.
    """
    
    def __init__(self):
        """
        Initializes the NAIPMiner class with a Planetary Computer API key.
        """
        planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")  # Replace with your key
        self.catalog_url = "https://stac-api.d2s.org"
        self.catalog = pystac_client.Client.open(self.catalog_url)

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2020-01-01/2021-01-01"):
        """
        Fetches NAIP imagery for a given date range and bounding box or polygon.
        
        Parameters:
        - lat (float): Latitude of the center point (if polygon is None).
        - lon (float): Longitude of the center point (if polygon is None).
        - radius (float): Radius around the center point in kilometers (if polygon is None).
        - polygon (shapely.geometry.Polygon): Polygon defining the area of interest (optional).
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format (default is 2020).
        
        Returns:
        - xarray.Dataset: NAIP imagery for the given area and date range.
        """
        if polygon is None:
            # Create a polygon around the lat/lon with a given radius in kilometers
            polygon = Point(lon, lat).buffer(radius/111/1000)  # Convert radius from km to degrees

        # Convert the polygon to a bounding box
        bbox = polygon.bounds

        # Search the Planetary Computer for NAIP imagery
        query = self.catalog.search(
            collections=["naip"],   # NAIP Collection
            datetime=daterange,     # Date range
            bbox=bbox,              # Bounding box of the AOI
            limit=100               # Limit to 100 results
        )
        query = list(query.items())
        # If no items found, raise an error
        if len(query) == 0:
            raise ValueError("No NAIP data found for the given date range and bounding box.")
        
        collections = {}
        for item in query:
            ds = rioxarray.open_rasterio(
                item.assets["image"].href, 
                chunks={"x": 1000, "y": 1000}
            )
            ds = ds.rio.clip(
                geometries=[box(*gpd.GeoDataFrame([{'geometry': box(*bbox)}], crs='epsg:4326')
                            .to_crs(ds.rio.crs).iloc[0, -1].bounds)],
                drop=True
            )
            naip_date = str(pd.to_datetime(item.datetime)).split(" ")[0]
            if naip_date not in collections : 
                collections[naip_date] = []
            collection = {
                'naip_date':naip_date,
                'crs': ds.rio.crs, 
                'polygon': box(*ds.rio.bounds()),
                'ds':ds
            }
            collections[naip_date].append(collection) 
        
        collections = sorted(
                collections.items(),
                key=lambda x: unary_union([d['polygon'] for d in x[1]]).area,
                reverse=True
            )
        collection = collections[0]
        ds =  merge_arrays([c['ds'] for c in collection[1]])
        ds.attrs['metadata'] = {'date': {'value': collection[0], 'confidence': 100}}
        return ds

# Example usage:
if __name__ == "__main__":
    naip_miner = NAIPMiner()
    lat,lon = 33.88120789,-91.48968559
    radius = 1000
    ds_naip = naip_miner.fetch(lat,lon,radius,daterange='2021-01-01/2024-01-01')#.compute()
    print(ds_naip)
