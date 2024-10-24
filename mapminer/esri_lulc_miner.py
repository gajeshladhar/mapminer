import planetary_computer
import pystac_client
import dask
from odc.stac import load
import xarray as xr
from shapely.geometry import Polygon, Point, box


class ESRILULCMiner:
    """
    A class for fetching and processing the 10m Annual Land Use Land Cover (9-class) V2 from Microsoft's Planetary Computer.
    """
    
    def __init__(self):
        """
        Initializes the LULCMiner class with a Planetary Computer API key.
        """
        planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")  # Replace with your API key
        self.catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = pystac_client.Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2024-01-01/2024-12-31"):
        """
        Fetches the 10m Annual Land Use Land Cover (9-class) for a given date range and bounding box or polygon.
        
        Parameters:
        - lat (float): Latitude of the center point (if polygon is None).
        - lon (float): Longitude of the center point (if polygon is None).
        - radius (float): Radius around the center point in kilometers (if polygon is None).
        - polygon (shapely.geometry.Polygon): Polygon defining the area of interest (optional).
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format (default: 2021).

        Returns:
        - xarray.Dataset: LULC imagery for the given area and date range.
        """
        if polygon is None:
            # Create a polygon around the lat/lon with a given radius in kilometers
            polygon = Point(lon, lat).buffer(radius / 111/1000)  # Convert radius from km to degrees

        # Convert the polygon to a bounding box
        bbox = polygon.bounds

        # Search the Planetary Computer for LULC data
        query = self.catalog.search(
            collections=["io-lulc-annual-v02"],   # ESA WorldCover LULC Collection
            datetime=daterange,               # Date range for LULC data
            bbox=bbox,                        # Bounding box of the AOI
            limit=100                         # Limit to 100 results
        )
        query_items = list(query.items())

        # If no items found, raise an error
        if len(query_items) == 0:
            raise ValueError("No LULC data found for the given date range and bounding box.")

        # Load the data using odc.stac and Dask for lazy loading
        ds_lulc = load(
            query_items,
            bbox=bbox,
            chunks={}
        ).astype("float32").sortby('time', ascending=True)

        return ds_lulc

# Example usage:
if __name__ == "__main__":
    lulc_miner = ESRILULCMiner()
    daterange = "2024-01-01/2024-12-31"  # Specify the year you want to fetch data for
    polygon = box(-100.75, 35.25, -100.5, 35.5)  # Example bounding box (in degrees)

    # Fetch LULC data for the specified polygon and date range
    ds_lulc = lulc_miner.fetch(polygon=polygon, daterange=daterange)
    print(ds_lulc)