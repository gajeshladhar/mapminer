import planetary_computer
import pystac_client
import geopandas as gpd
import rioxarray
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
        
        naip_date = str(pd.to_datetime(query[0].datetime)).split(" ")[0]
        query = query[0]
        ds = rioxarray.open_rasterio(query.assets["image"].href,chunks={"x":1000,"y":1000}).sortby('y').sortby('x')
        ds = ds.rio.clip(geometries=[gpd.GeoDataFrame([{'geometry':box(*bbox)}],crs='epsg:4326').to_crs(ds.rio.crs).iloc[0,-1]],drop=True)
        attrs = {'metadata':{'date': {'value': naip_date, 'confidence': 100}}}
        ds = xr.DataArray(data=ds.data,dims=['band','y','x'],coords={
            'band':['Red','Green','Blue','NIR'],
            'y':ds.y.values,
            'x':ds.x.values},attrs=attrs).rio.write_crs(ds.rio.crs)
        
        return ds

# Example usage:
if __name__ == "__main__":
    naip_miner = NAIPMiner()
    lat,lon = 32.89370884,-97.18257253
    radius = 200
    ds_naip = naip_miner.fetch(lat,lon,radius).compute()
    print(ds_naip)
