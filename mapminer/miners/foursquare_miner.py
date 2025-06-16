import duckdb
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely import wkt
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


class FourSquareMiner:
    def __init__(self, engine='duckdb'):
        """
        Initialize the FourSquareMiner with the specified engine.
        """
        self.engine = engine
        self.parquet_path = 's3://fsq-os-places-us-east-1/release/dt=2024-11-19/places/parquet/'
        if engine == 'duckdb':
            self._setup_duckdb()

    def _setup_duckdb(self):
        """
        Set up the DuckDB connection and configure S3 and spatial extensions.
        """
        self.con = duckdb.connect()
        self.con.execute("INSTALL httpfs;")
        self.con.execute("LOAD httpfs;")
        self.con.execute("INSTALL spatial;")
        self.con.execute("LOAD spatial;")
        self.con.execute("SET s3_region='us-east-1';")

    def _fetch_duckdb(self, polygon):
        """
        Fetch data using DuckDB and filter using a spatial polygon.
        """
        query = f"""
        SELECT *,
               ST_AsText(ST_Point(longitude, latitude)) AS wkt_geometry
        FROM read_parquet('{self.parquet_path}*.parquet')
        WHERE ST_Within(ST_Point(longitude, latitude), ST_GeomFromText('{polygon.wkt}'))
        """
        result_df = self.con.execute(query).fetchdf()
        result_df['geometry'] = result_df['wkt_geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(result_df, geometry='geometry')
        gdf.set_crs(epsg=4326, inplace=True)
        return gdf

    def _fetch_dask(self, polygon):
        """
        Fetch data using Dask and filter using bounding box of the polygon.
        """
        ddf = dd.read_parquet(self.parquet_path, storage_options={"anon": True})
        minx, miny, maxx, maxy = polygon.bounds
        ddf_filtered = ddf[(ddf['latitude'] >= miny) & (ddf['latitude'] <= maxy) &
                           (ddf['longitude'] >= minx) & (ddf['longitude'] <= maxx)]
        
        # Apply point creation
        def make_point(row):
            return Point(row['longitude'], row['latitude'])
        
        meta = ddf_filtered.head(0)
        meta['geometry'] = gpd.GeoSeries([Point()])
        ddf_filtered['geometry'] = ddf_filtered.apply(make_point, axis=1, meta=('geometry', 'object'))
        dgdf = ddf_filtered.map_partitions(gpd.GeoDataFrame, geometry='geometry', crs='EPSG:4326', meta=meta)
        
        with ProgressBar():
            dgdf = dgdf.compute()
        
        return dgdf

    def _get_parquet_path(self, daterange):
        """
        Update the parquet path based on the closest available date.
        """
        fs = fsspec.filesystem('s3', anon=True)
        s3_directory = 's3://fsq-os-places-us-east-1/release/'
        files = fs.ls(s3_directory)
        file_dates = [pd.to_datetime(f.split("dt=")[1]) for f in files]
        closest_file = files[np.argmin(abs(pd.to_datetime(daterange) - pd.to_datetime(file_dates)))]
        return f's3://{closest_file}/places/parquet/'

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2024-11-19"):
        """
        Fetch data based on input location (lat, lon) and radius or a polygon.
        Supports fetching by DuckDB or Dask engine.
        """
        daterange = daterange.split("/")[-1]
        if polygon is None:
            polygon = Point(lon, lat).buffer(radius / 111 / 1000)  # Approx. 1 degree = 111 km
        
        self.parquet_path = self._get_parquet_path(daterange)

        if self.engine == 'duckdb':
            return self._fetch_duckdb(polygon)
        return self._fetch_dask(polygon)
