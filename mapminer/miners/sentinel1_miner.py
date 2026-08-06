import os
import requests
import s3fs
import xml.etree.ElementTree as ET
import planetary_computer
from odc.stac import load
import xarray as xr
import numpy as np
import rioxarray
from pystac_client import Client
from shapely.geometry import Polygon, Point, box


class Sentinel1Miner:
    """
    A class for fetching and processing Sentinel-1 GRD imagery from Microsoft's Planetary Computer.
    """
    available_engines = {
        "planetary_computer": {
            "catalog_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "collection": "sentinel-1-grd"
        },
        "copernicus": {
            "catalog_url": "https://stac.dataspace.copernicus.eu/v1",
            "collection": "sentinel-1-grd"
        },
        "element84": {
            "catalog_url": "https://earth-search.aws.element84.com/v1",
            "collection": "sentinel-1-grd"
        },
    }
    def __init__(self,engine="planetary_computer"):
        """
        Initializes the Sentinel1GRDMiner class using the specified STAC engine.
        """
        engine = self.available_engines.get(engine)
        self.catalog_url = engine["catalog_url"]
        self.collection = engine["collection"]

        if self.catalog_url == self.available_engines["planetary_computer"]["catalog_url"]:
            planetary_computer.settings.set_subscription_key("1d7ae9ea9d3843749757036a903ddb6c")
            self.catalog = Client.open(self.catalog_url, modifier=planetary_computer.sign_inplace)
        else:
            if self.catalog_url == self.available_engines["element84"]["catalog_url"]:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            self.catalog = Client.open(self.catalog_url)

    def fetch(self, lat=None, lon=None, radius=None, polygon=None, daterange="2024-01-01/2024-01-10", merge_nodata=False):
        """
        Fetches Sentinel-1 GRD imagery for a given date range and bounding box or polygon.
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - polygon (Polygon): Polygon defining the area of interest (optional).
        - lat (float): Latitude of the center point (if polygon is None).
        - lon (float): Longitude of the center point (if polygon is None).
        - radius (float): Radius around the center point in kilometers (if polygon is None).
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Sentinel-1 GRD imagery with georeferencing and nodata merged if specified.
        """
        if polygon is None : 
            polygon = Point(lon,lat).buffer(radius/111/1000)

        # Determine the local UTM CRS based on the bounding box
        utm_crs = self._get_utm_crs(polygon.centroid.y, polygon.centroid.x)

        ds_sentinel = self.fetch_imagery(daterange, polygon.bounds, utm_crs, merge_nodata)
        return ds_sentinel

    def fetch_imagery(self, daterange, bbox, crs, merge_nodata=False):
        """
        Returns Dask Datacube of Sentinel-1 GRD based on the provided bounding box and date range (Lazy Loading).
        
        Parameters:
        - daterange (str): Date range in 'YYYY-MM-DD/YYYY-MM-DD' format.
        - bbox (list): Bounding box as [west, south, east, north].
        - crs (str): CRS to use for the dataset (typically UTM).
        - merge_nodata (bool): Whether to merge nodata values from neighboring tiles (default: False).
        
        Returns:
        - xarray.Dataset: Sentinel-1 GRD dataset.
        """
        query = self.catalog.search(
            collections=[self.collection],
            datetime=daterange,
            limit=100,
            bbox=bbox
        )
        query = list(query.items())
        query = sorted(query, key=lambda item: item.properties.get("datetime"))

        # Load the dataset with specified CRS (UTM) and resolution (10 meters for Sentinel-1 GRD)
        ds_sentinel = load(
            query,
            bbox=bbox,
            crs=crs,             # Use the dynamically calculated UTM CRS
            resolution=10,       # Resolution for Sentinel-1 GRD (10 meters)
            chunks={}
        ).astype("float32").sortby('time', ascending=True)

        if merge_nodata:
            ds_sentinel = self._merge_nodata(ds_sentinel)

        # Attach per-scene metadata (SAR properties + radiometric calibration LUTs) to attrs
        ds_sentinel.attrs['metadata'] = [self._extract_metadata(item) for item in query]

        return ds_sentinel

    def _fetch_asset_bytes(self, href):
        """
        Downloads the raw bytes of a STAC asset, transparently handling both
        signed HTTPS hrefs (Planetary Computer) and s3:// hrefs (Copernicus / element84).

        Parameters:
        - href (str): Asset href as returned by the STAC item.

        Returns:
        - bytes: Raw contents of the asset.
        """
        if href.startswith("s3://"):
            fs = s3fs.S3FileSystem(anon=True)
            with fs.open(href.replace("s3://", ""), "rb") as f:
                return f.read()
        else:
            response = requests.get(href, timeout=30)
            response.raise_for_status()
            return response.content

    def _parse_calibration_xml(self, xml_bytes):
        """
        Parses a Sentinel-1 `calibration-iw-{pol}.xml` annotation file into the
        sigmaNought calibration LUT (used to convert digital numbers to sigma0):

            sigma0 = (raw_data.astype(float) ** 2) / (K ** 2)

        where K is obtained by interpolating this LUT to the pixel/line of interest.

        Parameters:
        - xml_bytes (bytes): Raw contents of the calibration XML file.

        Returns:
        - dict: absoluteCalibrationConstant plus per-vector line/pixel/sigmaNought arrays.
        """
        root = ET.fromstring(xml_bytes)

        absolute_calibration_constant = float(root.findtext(".//absoluteCalibrationConstant", default="1.0"))

        vectors = []
        for vector in root.findall(".//calibrationVectorList/calibrationVector"):
            vectors.append({
                "azimuthTime": vector.findtext("azimuthTime"),
                "line": int(vector.findtext("line")),
                "pixel": [int(v) for v in vector.findtext("pixel").split()],
                "sigmaNought": [float(v) for v in vector.findtext("sigmaNought").split()],
            })

        return {
            "absoluteCalibrationConstant": absolute_calibration_constant,
            "calibrationVectorList": vectors,
        }

    def _extract_metadata(self, item):
        """
        Builds a metadata dictionary for a single Sentinel-1 STAC item, combining
        its SAR/orbit properties with the per-polarization calibration LUTs
        (`calibration-iw-vv.xml` / `calibration-iw-vh.xml`) needed for radiometric
        calibration (DN -> sigma0 -> dB).

        Parameters:
        - item (pystac.Item): STAC item for a single Sentinel-1 scene.

        Returns:
        - dict: Scene properties plus a 'calibration' entry keyed by polarization.
        """
        metadata = dict(item.properties)
        metadata["id"] = item.id

        calibration = {}
        for pol in ("vv", "vh"):
            asset_key = f"schema-calibration-{pol}"
            if asset_key not in item.assets:
                continue
            try:
                xml_bytes = self._fetch_asset_bytes(item.assets[asset_key].href)
                calibration[pol] = self._parse_calibration_xml(xml_bytes)
            except Exception as e:
                calibration[pol] = {"error": str(e)}

        metadata["calibration"] = calibration
        return metadata

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

    def _merge_nodata(self, ds_sentinel):
        """
        Merges nodata values from neighboring tiles in the dataset.
        
        Parameters:
        - ds_sentinel (xarray.Dataset): The Sentinel-1 GRD dataset.
        
        Returns:
        - xarray.Dataset: The dataset with nodata values merged.
        """
        # Merge nodata for each time step and band
        for time_index in range(len(ds_sentinel.time.values[:-1])):
            curr_time = ds_sentinel.time.values[time_index]
            nearest_time = ds_sentinel.time.values[time_index - 1] if (abs(ds_sentinel.time.values[time_index - 1] - curr_time) - abs(ds_sentinel.time.values[time_index + 1] - curr_time)) < 0 else ds_sentinel.time.values[time_index + 1]
            
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

# Example usage
if __name__ == "__main__":
    miner = Sentinel1Miner()
    daterange = "2024-01-01/2024-01-10"
    polygon = box(*[77.1025, 28.7041, 77.4125, 28.8541])  # Bounding box for New Delhi
    
    # Fetch the dataset with nodata merging enabled and in local UTM CRS
    ds_sentinel = miner.fetch(daterange=daterange, polygon=polygon, merge_nodata=True)
    print(ds_sentinel)
