import ee
import os
import json
import mercantile
import numpy as np
import pandas as pd
import geopandas as gpd
import dask
from shapely.geometry import Polygon, Point, shape, box

class MSBuildingMiner:
    """
    Microsoft Building Miner class for extracting building data from Microsoft Open Buildings dataset using Dask & Numba.
    """

    def __init__(self):
        """
        Initializes the miner by loading global quadkey-indexed tile metadata as a GeoDataFrame.
        """
        df_ms = pd.read_csv("https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv")
        df_ms['geometry'] = df_ms.QuadKey.apply(lambda key : self._quadkey_to_geom(str(key)))
        df_ms = gpd.GeoDataFrame(df_ms,crs='epsg:4326')
        self.df_ms = df_ms
    
    def _quadkey_to_geom(self,quadkey):
        """
        Converts a quadkey to a bounding box geometry (EPSG:4326).
        
        Args:
            quadkey (str): Quadkey tile string.
        
        Returns:
            shapely.geometry.Polygon: Bounding box geometry of the tile.
        """
        tile = mercantile.quadkey_to_tile(quadkey)
        return box(*mercantile.bounds(tile))
    
    
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
        
        
        urls = self.df_ms[self.df_ms.intersects(polygon)].Url.values 
        if len(urls)==0:
            raise ValueError("No data found in Microsoft Buildings dataset for the given region.")
        
        dfs = dask.compute(*[self.fetch_buildings(url,polygon) for url in urls],scheduler='threads')
        df = pd.concat(dfs)
        df['confidence'] = df.properties.apply(lambda properties : properties['confidence'] if 'confidence' in properties else 1.0).values
        return df
    
    @dask.delayed
    def fetch_buildings(self,url,polygon):
        df = pd.read_json(url, lines=True)
        df['geometry'] = df['geometry'].apply(shape)
        df = gpd.GeoDataFrame(df, crs='epsg:4326')
        df = df[df.intersects(polygon)]
        return df
        
    
 


if __name__ == '__main__':
    miner = MSBuildingMiner()
    df = miner.fetch(25.9727714,69.29731942,radius=1000)
    print(df)
