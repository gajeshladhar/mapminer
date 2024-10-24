import numpy as np
import pandas as pd
import geopandas as gpd
import json
import time
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import requests
import dask
from typing import List, Dict, Union


class OSMMiner:
    """
    A class to query and process OSM data using Overpass API for multiple layers.
    The data is fetched in parallel using Dask and then processed.
    """
    
    def __init__(self):
        """
        Initialize the OSMMiner class with predefined layer configuration.
        """
        # Config containing OSM queries for various geographical features
        self.config = [
            {'layer_id': 45, 'layer_name': 'wind_tower', 'queries': [{'query': '["generator:source"~"wind"]', 'type': 'node'}]},
            {'layer_id': 51, 'layer_name': 'htl_tower', 'queries': [{'query': '["power"~"tower"]', 'type': 'node'}]},
            {'layer_id': 40, 'layer_name': 'nalla', 'queries': [{'query': '["waterway"~"stream"]', 'type': 'way'},{'query': '["waterway"~"tidal_channel"]', 'type': 'way'}]},
            {'layer_id': 36, 'layer_name': 'canal', 'queries': [{'query': '["waterway"~"canal|drain|ditch"]', 'type': 'way'}]},
            {'layer_id': 5, 'layer_name': 'defence_land', 'queries': [{'query': '["landuse"~"military"]', 'type': 'way'},{'query': '["military"~"barracks"]', 'type': 'way'}]},
            {'layer_id': 10, 'layer_name': 'mine', 'queries': [{'query': '["landuse"~"quarry"]', 'type': 'way'}]},
            {'layer_id': 8, 'layer_name': 'reserve_forest', 'queries': [{'query': '["landuse"~"forest"]', 'type': 'way'},{'query': '["leisure"~"nature_reserve"]', 'type': 'way'}]},
            {'layer_id': 9, 'layer_name': 'protected_forest', 'queries': [{'query': '["boundary"~"protected_area"]', 'type': 'way'}]},
            {'layer_id': 25, 'layer_name': 'railway_station', 'queries': [{'query': '["railway"~"platform"]', 'type': 'way'},{'query': '["railway"~"halt"]', 'type': 'way'},{'query': '["landuse"~"railway"]', 'type': 'way'}]},
            {'layer_id': 12, 'layer_name': 'railway_line', 'queries': [{'query': '["railway"~"rail"]', 'type': 'way'}]},
            {'layer_id': 23, 'layer_name': 'salt_pane', 'queries': [{'query': '["landuse"~"salt_pond"]', 'type': 'way'}]},
            {'layer_id': 31, 'layer_name': 'power_line', 'queries': [{'query': '["power"~"line"]', 'type': 'way'}]},
            {'layer_id': 17, 'layer_name': 'national_highway', 'queries': [{'query': '["highway"~"trunk|motorway"]', 'type': 'way'}]},
            {'layer_id': 15, 'layer_name': 'village_road', 'queries': [{'query': '["highway"~"secondary|tertiary|unclassified"]', 'type': 'way'}]},
            {'layer_id': 24, 'layer_name': 'solar_park', 'queries': [{'query': '["generator:source"~"solar"]', 'type': 'way'},{'query': '["plant:source"~"solar"]', 'type': 'way'}]},
            {'layer_id': 16, 'layer_name': 'state_highway', 'queries': [{'query': '["highway"~"primary"]', 'type': 'way'}]},
            {'layer_id': 37, 'layer_name': 'dam', 'queries': [{'query': '["waterway"~"dam"]', 'type': 'way'}]},
            {'layer_id': 13, 'layer_name': 'metro_line', 'queries': [{'query': '["railway"~"subway"]', 'type': 'way'},{'query': '["construction"~"subway"]', 'type': 'way'}]},
            {'layer_id': 14, 'layer_name': 'bridge', 'queries': [{'query': '["bridge"]', 'type': 'way'}]},
            {'layer_id': 27, 'layer_name': 'helipad', 'queries': [{'query': '["aeroway"~"helipad|heliport"]', 'type': 'way'}]}
        ]
    
    def fetch(self,lat=None,lon=None,radius=None,polygon=None):
        """
        Fetch OSM data for all layers in parallel and process the results.
        
        Args:
            polygon (Polygon): Geographical polygon bounding box.
        
        Returns:
            pd.DataFrame: Combined DataFrame containing processed OSM data.
        """
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)
        # Create a list to store delayed tasks
        delayed_tasks = []
        
        # Loop through each layer and its queries in self.config
        for layer in self.config:
            for query in layer['queries']:
                # Append the delayed task for each query
                delayed_task = self.fetch_overpass_query(polygon, query['query'])
                delayed_tasks.append((layer, query, delayed_task))

        # Use dask.compute to run all delayed tasks in parallel
        results = dask.compute(*[task[2] for task in delayed_tasks])

        dfs = []
        # Create a list of delayed tasks for processing OSM data
        delayed_dfs = []
        for i, result in enumerate(results):
            layer, query, _ = delayed_tasks[i]
            # Use lambda and dask.delayed for processing OSM data
            delayed_task = dask.delayed(lambda res: OSMProcessor(res).process_osm_data())(result)
            delayed_dfs.append((layer, delayed_task))
        
        # Compute all delayed tasks in parallel using the 'processes' scheduler
        processed_results = dask.compute(*[task[1] for task in delayed_dfs], scheduler='processes')
        
        # Process the computed results and append to the dfs list
        for i, processed_df in enumerate(processed_results):
            layer = delayed_dfs[i][0]
            if len(processed_df) > 0:
                processed_df['layer_id'] = layer['layer_id']
                processed_df['layer_name'] = layer['layer_name']
                dfs.append(processed_df.loc[:, ['layer_id', 'layer_name','geometry']])
        
        return pd.concat(dfs) if dfs else gpd.GeoDataFrame(pd.DataFrame(columns=["layer_id","layer_name","geometry"]), geometry="geometry").set_crs('epsg:4326')
            
        
    @dask.delayed
    def fetch_overpass_query(self, polygon: Polygon, query: str) -> Union[Dict, None]:
        """
        Send a request to the Overpass API to fetch OSM data for a given query and polygon.

        Args:
            polygon (Polygon): Geographical polygon bounding box.
            query (str): OSM query to fetch data.

        Returns:
            dict: Parsed OSM data in JSON format.
            None: If the request fails after retries.
        """
        bbox = polygon.bounds
        overpass_query = f"""
            [out:json];
            (
            way({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}){query};
            node({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}){query};
            relation({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}){query};
            );
            (._;>;);
            out body;
        """
        
        max_retries = 10
        retry_delay = 2  # seconds between retries
        
        # Retry the request if it fails
        for attempt in range(max_retries):
            try:
                response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query)
                response.raise_for_status()  # Raise an exception for HTTP errors
                osm_data = response.json()  # Try to decode JSON
                return osm_data
            except requests.exceptions.RequestException:
                time.sleep(retry_delay)  # Wait before retrying
            except json.JSONDecodeError:
                time.sleep(retry_delay)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
        
        return None  # Return None if all retries fail

    
class OSMProcessor:
    """
    A class to process OSM data into usable geospatial formats (GeoDataFrames).
    """
    
    def __init__(self, osm_data: Dict):
        """
        Initialize the OSMProcessor class with OSM data.

        Args:
            osm_data (dict): OSM data to be processed.
        """
        self.osm_data = osm_data
        self.df_nodes, self.df_ways, self.df_relations = self.create_osm_dataframes()
        self.processed_way_ids = set()
        self.processed_node_ids = set()
        self.processed_features = []

    def create_osm_dataframes(self) -> tuple:
        """
        Create DataFrames for OSM nodes, ways, and relations.

        Returns:
            tuple: (df_nodes, df_ways, df_relations) as pandas DataFrames.
        """
        df_nodes = pd.DataFrame(
            [el for el in self.osm_data['elements'] if el['type'] == 'node']
        ).set_index('id') if any(el['type'] == 'node' for el in self.osm_data['elements']) else pd.DataFrame(columns=['id', 'lon', 'lat', 'tags'])

        df_ways = pd.DataFrame(
            [el for el in self.osm_data['elements'] if el['type'] == 'way']
        ).set_index('id') if any(el['type'] == 'way' for el in self.osm_data['elements']) else pd.DataFrame(columns=['id', 'nodes', 'tags'])

        df_relations = pd.DataFrame(
            [el for el in self.osm_data['elements'] if el['type'] == 'relation']
        ).set_index('id') if any(el['type'] == 'relation' for el in self.osm_data['elements']) else pd.DataFrame(columns=['id', 'members', 'tags'])

        return df_nodes, df_ways, df_relations

    def get_way_geometry(self, way_id: int) -> Union[LineString, Polygon]:
        """
        Get the geometry of a way (LineString or Polygon) using its node coordinates.

        Args:
            way_id (int): ID of the way to get geometry for.

        Returns:
            LineString or Polygon: Geometry of the way.
        """
        nodes = self.df_ways.loc[way_id, 'nodes']
        coords = [(self.df_nodes.loc[node_id, 'lon'], self.df_nodes.loc[node_id, 'lat']) for node_id in nodes]
        return Polygon(coords) if coords[0] == coords[-1] else LineString(coords)

    def process_relations(self):
        """
        Process OSM relations, combining way geometries into unified features.
        """
        if self.df_relations.empty:
            return

        for relation_id in self.df_relations.index:
            members = self.df_relations.loc[relation_id, 'members']
            member_geometries = []

            for member in members:
                if member['type'] == 'way' and member['ref'] not in self.processed_way_ids:
                    way_geometry = self.get_way_geometry(member['ref'])
                    member_geometries.append(way_geometry)

                    # Mark the way and its nodes as processed
                    self.processed_way_ids.add(member['ref'])
                    self.processed_node_ids.update(self.df_ways.loc[member['ref'], 'nodes'])

            if member_geometries:
                combined_geom = unary_union(member_geometries) if len(member_geometries) > 1 else member_geometries[0]
                self.processed_features.append({
                    'feature_id': relation_id,
                    'type': 'relation',
                    'tags': self.df_relations.loc[relation_id, 'tags'],
                    'geometry': combined_geom
                })

    def process_ways(self):
        """
        Process OSM ways that are not part of relations, creating geometries for each way.
        """
        if self.df_ways.empty:
            return

        for way_id in self.df_ways.index.difference(self.processed_way_ids):
            way_geometry = self.get_way_geometry(way_id)

            # Mark the nodes of this way as processed
            self.processed_node_ids.update(self.df_ways.loc[way_id, 'nodes'])

            self.processed_features.append({
                'feature_id': way_id,
                'type': 'way',
                'tags': self.df_ways.loc[way_id, 'tags'],
                'geometry': way_geometry
            })

    def process_nodes(self):
        """
        Process OSM nodes that are not part of ways or relations.
        """
        if self.df_nodes.empty:
            return

        for node_id in self.df_nodes.index.difference(self.processed_node_ids):
            self.processed_features.append({
                'feature_id': node_id,
                'type': 'node',
                'tags': self.df_nodes.loc[node_id, 'tags'] if 'tags' in self.df_nodes.columns else {},
                'geometry': Point(self.df_nodes.loc[node_id, ['lon', 'lat']])
            })

    def process_osm_data(self) -> gpd.GeoDataFrame:
        """
        Process OSM data into a GeoDataFrame containing all processed features.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing processed OSM data features.
        """
        # Process relations first
        self.process_relations()

        # Process remaining ways that are not part of relations
        self.process_ways()

        # Process remaining nodes that are not part of ways or relations
        self.process_nodes()

        # Convert processed features to GeoDataFrame
        if self.processed_features:
            return gpd.GeoDataFrame(self.processed_features, crs="EPSG:4326")
        else:
            return gpd.GeoDataFrame(self.processed_features)


if __name__ == '__main__':
    # Initialize DEMMiner with the service account JSON file
    miner = OSMMiner()
    # Fetch dem data for a small area around a given point
    df = miner.fetch(lon=73.31452961, lat=28.01306571,radius=100)
    # Output the results
    print(df)
