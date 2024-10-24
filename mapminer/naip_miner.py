import planetary_computer
import pystac_client
import rioxarray
from odc.stac import load
import xarray as xr
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
        self.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = pystac_client.Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)

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
            polygon = Point(lon, lat).buffer(radius* (111/1000))  # Convert radius from km to degrees

        # Convert the polygon to a bounding box
        bbox = polygon.bounds

        # Search the Planetary Computer for NAIP imagery
        query = self.catalog.search(
            collections=["naip"],   # NAIP Collection
            datetime=daterange,     # Date range
            bbox=bbox,              # Bounding box of the AOI
            limit=100               # Limit to 100 results
        )
        query_items = list(query.items())

        # If no items found, raise an error
        if len(query_items) == 0:
            raise ValueError("No NAIP data found for the given date range and bounding box.")
        
        # Load the data using odc.stac and Dask for lazy loading
        ds_naip = load(
            query_items,
            bbox=bbox,
            chunks={}
        ).astype("float32").sortby('time', ascending=True)

        return ds_naip

# Example usage:
if __name__ == "__main__":
    naip_miner = NAIPMiner()
    daterange = "2020-01-01/2021-01-01"
    polygon = box(-100.75, 35.25, -100.5, 35.5)  # Example bounding box in Texas

    # Fetch NAIP imagery for the specified polygon and date range
    ds_naip = naip_miner.fetch(polygon=polygon, daterange=daterange)
    print(ds_naip)
