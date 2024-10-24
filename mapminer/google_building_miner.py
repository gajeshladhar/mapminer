import ee
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import dask
from cryptography.fernet import Fernet


class GoogleBuildingMiner:
    """
    GoogleMiner class for extracting building data from Google Open Buildings dataset using Google Earth Engine (GEE).
    """

    def __init__(self, json_path: str=None):
        """
        Initializes GoogleMiner by authenticating using the provided JSON key file.
        
        Args:
            json_path (str): Path to the service account JSON file.
        """
        self.autheticate(json_path)
    
    def fetch(self,lat=None,lon=None,radius=None,polygon=None):
        """
        Fetches building data for the given polygon after splitting it for efficient processing.
        
        Args:
            polygon (shapely.geometry.Polygon): Input polygon EPSG:4326.
            crs (str): Coordinate reference system of the input polygon, defaults to 'epsg:4326'.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with building information.
        """
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)
        
        # Split polygon into smaller parts for efficient processing
        filtered_polygons = self.split_polygon(polygon, max_counts=5000)

        # Fetch building data for all the split polygons using Dask
        df_building = pd.concat(dask.compute(*[self.fetch_buildings(polygon) for polygon in filtered_polygons]), axis=0)
        if len(df_building)==0: 
            df_building = gpd.GeoDataFrame(pd.DataFrame(columns=["layer_id","geometry"]), geometry="geometry").set_crs('epsg:4326')
            
        return df_building
    
    def autheticate(self, json_path: str):
        """
        Authenticates with Google Earth Engine using a service account.
        
        Args:
            json_path (str): Path to the service account JSON file.
        """
        if json_path is None : 
            key = open('keys/secret_key.key', 'rb').read(); cipher = Fernet(key)
            decrypted_key = cipher.decrypt(open('keys/google_service_account.json', 'rb').read())
            service_account_config = json.loads(decrypted_key)
        else : 
            # Load service account configuration
            service_account_config = json.load(open(json_path))
        
        # Authenticate with GEE using service account credentials
        credentials = ee.ServiceAccountCredentials(
            service_account_config['client_email'],
            key_data=json.dumps(service_account_config)
        )
        ee.Initialize(credentials)
    
    @dask.delayed
    def fetch_buildings(self, polygon: Polygon) -> gpd.GeoDataFrame:
        """
        Fetches building features lazily from the Google Open Buildings dataset for a given polygon.
        
        Args:
            polygon (shapely.geometry.Polygon): Input polygon.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with building features.
        """
        # Filter buildings within the polygon and return as a GeoDataFrame
        features = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(
            ee.Geometry(polygon.__geo_interface__)).getInfo()
        return gpd.GeoDataFrame.from_features(features)
    
    @dask.delayed
    def fetch_counts(self, polygon: Polygon) -> int:
        """
        Fetches the count of building features lazily for a given polygon.
        
        Args:
            polygon (shapely.geometry.Polygon): Input polygon.

        Returns:
            int: Count of buildings within the polygon.
        """
        # Return the count of buildings in the polygon
        return ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(
            ee.Geometry(polygon.__geo_interface__)).size().getInfo()
        
    def split_polygon(self, polygon: Polygon, grid_size: float = 20000*(9e-6), max_counts: int = 10000) -> list:
        """
        Splits a polygon into smaller grid-based rectangles to manage large feature counts.
        
        Args:
            polygon (shapely.geometry.Polygon): Input polygon.
            grid_size (float): Size of each grid cell.
            max_counts (int): Maximum allowed feature count per grid.

        Returns:
            list: List of split polygons that meet the max_counts requirement.
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        rectangles = []
        counts = []
        
        # Generate grid of rectangles within the polygon bounds
        for x in np.arange(min_x, max_x, grid_size):
            for y in np.arange(min_y, max_y, grid_size):
                rectangle = Polygon([
                    (x, y),
                    (x + grid_size, y),
                    (x + grid_size, y + grid_size),
                    (x, y + grid_size)
                ]).intersection(polygon).buffer(grid_size/100)
                
                # Skip if rectangle area is zero
                if rectangle.area == 0:
                    continue

                # Fetch building counts for each rectangle
                counts.append(self.fetch_counts(rectangle))
                rectangles.append(rectangle)
        
        # Compute building counts for all rectangles
        counts = dask.compute(*counts)
        
        refined_rectangles = []
        
        # Recursively split rectangles if they exceed max_counts
        for index in range(len(counts)):
            if counts[index] > max_counts:
                refined_rectangles.extend(self.split_polygon(rectangles[index], grid_size=grid_size/2, max_counts=max_counts))
            else:
                refined_rectangles.append(rectangles[index])
        
        return refined_rectangles


if __name__ == '__main__':
    # Initialize GoogleMiner with the service account JSON file
    miner = GoogleBuildingMiner()#"/root/projects/deployables/satsearch/satsearch/geodatabase/miner/google-earth-service-account.json")
    
    # Fetch building data for a small area around a given point
    df = miner.fetch(lon=73.31452961, lat=28.01306571,radius=100)
    
    # Output the results
    print(df)
