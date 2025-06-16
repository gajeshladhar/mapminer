import planetary_computer
import dask
from odc.stac import load
import xarray as xr
import rioxarray
import numpy as np
from pystac_client import Client
from shapely.geometry import Polygon, Point, box

class MODISMiner:
    """
    A class for fetching and processing MODIS imagery from Microsoft's Planetary Computer.
    """
    
    def __init__(self):
        """
        Initializes the LandsatMiner class with a Planetary Computer API key.
        """
        planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")
        self.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2024-01-01/2024-01-10", merge_nodata=False):
        """
        Fetches Landsat imagery for a given date range and bounding box or polygon.
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - polygon (Polygon): Polygon defining the area of interest (optional).
        - lat (float): Latitude of the center point (if polygon is None).
        - lon (float): Longitude of the center point (if polygon is None).
        - radius (float): Radius around the center point in kilometers (if polygon is None).
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Landsat imagery with georeferencing and nodata merged if specified.
        """
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)

        # Determine the local UTM CRS based on the bounding box
        utm_crs = self._get_utm_crs(polygon.centroid.y, polygon.centroid.x)

        ds_landsat = self.fetch_imagery(daterange, polygon.bounds, utm_crs, merge_nodata)
        return ds_landsat

    def fetch_imagery(self, daterange, bbox, crs, merge_nodata=False):
        """
        Returns Dask Datacube of Landsat based on the provided bounding box and date range (Lazy Loading).
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - bbox (list): Bounding box as [west, south, east, north].
        - crs (str): CRS to use for the dataset (typically UTM).
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Landsat dataset.
        """
        query_250m = self.catalog.search(
            collections=["modis-09Q1-061"],  # Collection for Landsat Collection 2 Level 2 data
            datetime=daterange,
            limit=100,
            bbox=bbox
        )
        query_500m = self.catalog.search(
            collections=["modis-09A1-061"],  # Collection for Landsat Collection 2 Level 2 data
            datetime=daterange,
            limit=100,
            bbox=bbox
        )
        query_250m = list(query_250m.items())
        query_500m = list(query_500m.items())

        # Load the dataset with specified CRS (UTM) and resolution (30 meters for Landsat)
        ds_modis_250 = load(
            query_250m,
            bbox=bbox,
            groupby="solar_day",  # Grouping by scene ID
            crs=crs,             # Use the dynamically calculated UTM CRS
            resolution=250,       # Landsat data has a 30-meter resolution
            chunks={}
        ).astype("float32").sortby('time', ascending=True)

        if merge_nodata:
            ds_modis_250 = self._merge_nodata(ds_modis_250)

        # Load the dataset with specified CRS (UTM) and resolution (30 meters for Landsat)
        ds_modis_500 = load(
            query_500m,
            bbox=bbox,
            groupby="solar_day",  # Grouping by scene ID
            crs=crs,             # Use the dynamically calculated UTM CRS
            resolution=500,       # Landsat data has a 30-meter resolution
            chunks={}
        ).astype("float32").sortby('time', ascending=True)

        if merge_nodata:
            ds_modis_500 = self._merge_nodata(ds_modis_500)
        
        ds_modis = ds_modis_500.sel(x=ds_modis_250.x.values,y=ds_modis_250.y.values,time=ds_modis_250.time.values,method='nearest')  
        ds_modis['x'], ds_modis['y'] = ds_modis_250.x.values,  ds_modis_250.y.values
        ds_modis['sur_refl_b01'].data[:] = ds_modis_250['sur_refl_b01'].data
        ds_modis['sur_refl_b02'].data[:] = ds_modis_250['sur_refl_b02'].data
        
        ds_modis = xr.Dataset({var:ds_modis[var] for var in [
                                'sur_refl_b01',
                                'sur_refl_b02',
                                'sur_refl_b03',
                                'sur_refl_b04',
                                'sur_refl_b05',
                                'sur_refl_b06',
                                'sur_refl_b07',
                                ]})
        return ds_modis

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

    def _merge_nodata(self, ds_landsat):
        """
        Merges nodata values from neighboring tiles in the dataset.
        
        Parameters:
        - ds_landsat (xarray.Dataset): The Landsat dataset.
        
        Returns:
        - xarray.Dataset: The dataset with nodata values merged.
        """
        # Merge nodata for each time step and band
        for time_index in range(len(ds_landsat.time.values[:-1])):
            curr_time = ds_landsat.time.values[time_index]
            nearest_time = ds_landsat.time.values[time_index - 1] if (abs(ds_landsat.time.values[time_index - 1] - curr_time) - abs(ds_landsat.time.values[time_index + 1] - curr_time)) < 0 else ds_landsat.time.values[time_index + 1]
            
            for band in ds_landsat.data_vars:
                if 'nodata' not in ds_landsat[band].attrs:
                    ds_landsat[band].loc[curr_time, :, :] = xr.where(
                        np.isnan(ds_landsat[band].sel(time=curr_time)),
                        ds_landsat[band].sel(time=nearest_time),
                        ds_landsat[band].sel(time=curr_time)
                    )
                else:
                    ds_landsat[band].loc[curr_time, :, :] = xr.where(
                        ds_landsat[band].sel(time=curr_time) == ds_landsat[band].attrs['nodata'],
                        ds_landsat[band].sel(time=nearest_time),
                        ds_landsat[band].sel(time=curr_time)
                    )
        
        return ds_landsat

# Example usage
if __name__ == "__main__":
    miner = MODISMiner()
    daterange = "2024-01-01/2024-01-10"
    polygon = box(*[77.1025, 28.7041, 77.4125, 28.8541])  # Bounding box for New Delhi
    
    # Fetch the dataset with nodata merging enabled and in local UTM CRS
    ds_modis = miner.fetch(daterange=daterange, polygon=polygon, merge_nodata=True)
    print(ds_modis.compute())
