from google_miner import GoogleMiner

if __name__=="__main__":
    miner = GoogleMiner()
    ds = miner.fetch(-13.25049643,35.24904667,radius=500,resolution=0.5)