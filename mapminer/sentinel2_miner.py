import planetary_computer
from odc.stac import load
import xarray as xr
import numpy as np
import rioxarray
from pystac_client import Client
from shapely.geometry import Polygon, Point, box

class Sentinel2Miner:
    """
    A class for fetching and processing Sentinel-2 imagery from Microsoft's Planetary Computer.
    """
    
    def __init__(self):
        """
        Initializes the Sentinel2Miner class with a Planetary Computer API key.
        """
        planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")
        self.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)

    def fetch(self,lat=None,lon=None,radius=None,polygon=None,daterange="2024-01-01/2024-01-10",merge_nodata=False):
        """
        Fetches Sentinel-2 imagery for a given date range and bounding box.
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - bbox (list): Bounding box as [west, south, east, north].
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Sentinel-2 imagery with georeferencing and nodata merged if specified.
        """
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)
        ds_sentinel = self.fetch_imagery(daterange, polygon.bounds, merge_nodata)
        return ds_sentinel

    def fetch_imagery(self, daterange, bbox, merge_nodata=False):
        """
        Returns Dask Datacube of Sentinel-2 based on the provided bounding box and date range (Lazy Loading).
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - bbox (list): Bounding box as [west, south, east, north].
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Sentinel-2 dataset.
        """
        query = self.catalog.search(
            collections=["sentinel-2-l2a"],
            datetime=daterange,
            limit=100,
            bbox=bbox
        )
        query = list(query.items())

        # Load the dataset (grouping by solar day)
        ds_sentinel = load(query, bbox=bbox, groupby="solar_day", chunks={}).astype("float32").sortby('time', ascending=True)

        if merge_nodata:
            ds_sentinel = self._merge_nodata(ds_sentinel)
        
        return ds_sentinel

    def _merge_nodata(self, ds_sentinel):
        """
        Merges nodata values from neighboring tiles in the dataset.
        
        Parameters:
        - ds_sentinel (xarray.Dataset): The Sentinel-2 dataset.
        
        Returns:
        - xarray.Dataset: The dataset with nodata values merged.
        """
        # Merge nodata for each time step and band
        for time_index in range(len(ds_sentinel.time.values[:-1])):
            curr_time = ds_sentinel.time.values[time_index]
            nearest_time = ds_sentinel.time.values[time_index-1] if (abs(ds_sentinel.time.values[time_index-1]-curr_time)-abs(ds_sentinel.time.values[time_index+1]-curr_time))<0 else ds_sentinel.time.values[time_index+1]
            
            for band in ds_sentinel.data_vars:
                if 'nodata' not in ds_sentinel[band].attrs:
                    ds_sentinel[band].loc[curr_time, :, :] = xr.where(
                        np.isnan(ds_sentinel[band].sel(time=curr_time)),
                        ds_sentinel[band].sel(time=nearest_time),
                        ds_sentinel[band].sel(time=curr_time)
                    )
                else:
                    ds_sentinel[band].loc[curr_time, :, :] = xr.where(
                        ds_sentinel[band].sel(time=curr_time) == ds_sentinel[band].attrs['nodata'],
                        ds_sentinel[band].sel(time=nearest_time),
                        ds_sentinel[band].sel(time=curr_time)
                    )
        
        return ds_sentinel


if __name__=="__main__":
    miner = Sentinel2Miner()
    daterange = "2024-01-01/2024-01-10"
    polygon = box(*[77.1025, 28.7041, 77.4125, 28.8541])  # Bounding box for New Delhi
    # Fetch the dataset with nodata merging enabled
    ds_sentinel = miner.fetch(daterange, polygon = polygon, merge_nodata=True)
    print(ds_sentinel)