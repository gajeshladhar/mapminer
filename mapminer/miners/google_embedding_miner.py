import ee
import os
import json
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, Point
from cryptography.fernet import Fernet


class GoogleEmbeddingMiner:
    """
    GoogleEmbeddingMiner extracts annual satellite embedding data from Google Earth Engine (GEE)
    using the 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL' dataset and loads it as an xarray Dataset.
    
    It supports both point-based and polygon-based queries and returns a stack of 64-dimensional
    image embeddings per pixel for the specified year(s).
    """
    def __init__(self, json_path: str = None):
        """
        Initializes GoogleEmbeddingMiner by authenticating using the provided JSON key file.
        
        Args:
            json_path (str): Path to the service account JSON file.
        """
        self.authenticate(json_path)
    
    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2024-01-01/2025-01-01"):
        """
        Fetches data for the given polygon or point within the specified date range.
        
        Args:
            lat (float): Latitude of the center point (if no polygon provided).
            lon (float): Longitude of the center point (if no polygon provided).
            radius (float): Radius around the point (in meters) to define the area.
            polygon (shapely.geometry.Polygon): Input polygon in EPSG:4326 (if provided).
            daterange (str): Date range to filter data (e.g., "2024-01-01/2025-01-01").
        
        Returns:
            xarray.Dataset: Data as an xarray Dataset.
        """
        # Split the date range into start and end dates
        start_date, end_date = daterange.split("/")
        start_year, end_year = int(start_date[:4]), int(end_date[:4])
        if polygon is None:
            # Create a buffer polygon around the point
            polygon = Point(lon, lat).buffer(radius / 111000.0)  # Radius in degrees

        geometry = ee.Geometry.Polygon(list(polygon.exterior.coords))
        ic = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterDate(start_date, end_date).filterBounds(geometry)
        
        image = ic.first().select(0)
        # Get original projection
        proj = image.projection().getInfo()
        transform = proj['transform']
        # Patch Y-resolution to be negative
        if transform[4] > 0:
            transform[4] = -transform[4]
        # Re-attach to projection
        patched_proj = ee.Projection(proj['crs'], transform=transform)
        # Check if collection has data
        count = ic.size().getInfo()
        if count == 0:
            raise ValueError("No data found for the given area and date range.")

        ic = ic.filter(ee.Filter.calendarRange(start_year, end_year, 'year'))
        # Use xarray to open the dataset with the correct engine and projection
        ds = xr.open_dataset(
            ic,
            engine='ee',
            projection=patched_proj,
            geometry=geometry
        )
        
        return ds

    def authenticate(self, json_path: str = None):
        """
        Authenticates with Google Earth Engine using a service account.
        
        Args:
            json_path (str): Path to the service account JSON file. If not provided,
                            the method will use the encrypted key from the external 'keys' folder.
        """
        # Get the absolute path of the 'keys' folder, assuming it's at the root level of the project
        keys_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'keys'))

        if json_path is None:
            # Access secret_key.key and google_service_account.json from the separate 'keys' folder
            secret_key_path = os.path.join(keys_dir, 'secret_key.key')
            service_account_json_path = os.path.join(keys_dir, 'google_service_account.json')

            key = open(secret_key_path, 'rb').read()
            cipher = Fernet(key)

            # Decrypt the Google service account JSON file
            encrypted_json = open(service_account_json_path, 'rb').read()
            decrypted_key = cipher.decrypt(encrypted_json)
            service_account_config = json.loads(decrypted_key)
        else:
            # Load service account configuration from the provided JSON path
            with open(json_path, 'r') as f:
                service_account_config = json.load(f)

        # Authenticate with GEE using service account credentials
        credentials = ee.ServiceAccountCredentials(
            service_account_config['client_email'],
            key_data=json.dumps(service_account_config)
        )
        ee.Initialize(credentials)


if __name__ == '__main__':
    # Initialize GoogleEmbeddingMiner with the service account JSON file
    miner = GoogleEmbeddingMiner()
    # Fetch GoogleEmbedding data for a small area around a given point with a date range
    ds = miner.fetch(lon=-121.8036, lat=39.0372, radius=100, daterange="2023-01-01/2025-01-01")
    # Output the results
    print(ds)
