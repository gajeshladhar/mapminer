from .google_basemap_miner import GoogleBaseMapMiner
from .google_traffic_miner import GoogleTrafficMiner
from .google_building_miner import GoogleBuildingMiner
from .ms_building_miner import MSBuildingMiner
from .osm_miner import OSMMiner
from .landsat_miner import LandsatMiner
from .modis_miner import MODISMiner
from .sentinel1_miner import Sentinel1Miner
from .sentinel2_miner import Sentinel2Miner
from .dem_miner import DEMMiner
from .esri_lulc_miner import ESRILULCMiner
from .esri_basemap_miner import ESRIBaseMapMiner
from .naip_miner import NAIPMiner
from .cdl_miner import CDLMiner
from .foursquare_miner import FourSquareMiner

if __name__=="__main__":
    miner = GoogleBaseMapMiner()
    ds = miner.fetch(-13.25049643,35.24904667,radius=500,resolution=0.5)