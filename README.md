<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>🌍 <strong>MapMiner</strong> </h1>
    <p>
    <a href="https://colab.research.google.com/drive/1steVa5hY0SqUabvFLb0J4ypRWgSs7io9?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
</a>
        <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square&logo=python" alt="Python">
        <img src="https://img.shields.io/badge/Xarray-0.18+-orange.svg?style=flat-square&logo=xarray" alt="Xarray">
        <img src="https://img.shields.io/badge/Dask-Powered-yellow.svg?style=flat-square&logo=dask" alt="Dask">
        <img src="https://img.shields.io/badge/Numba-Accelerated-green.svg?style=flat-square&logo=numba" alt="Numba">
        <img src="https://img.shields.io/badge/Selenium-Automated-informational.svg?style=flat-square&logo=selenium" alt="Selenium">
    </p>
    <p><strong>MapMiner</strong> is a geospatial tool designed to efficiently download and process geospatial data and metadata from various sources. It leverages powerful Python libraries like <strong>Selenium</strong>, <strong>Dask</strong>, <strong>Numba</strong>, and <strong>Xarray</strong> to provide high-performance data retrieval and processing capabilities for geospatial analysis and visualization.</p><br>
    <h2>🚀 <strong>Key Features</strong></h2>
    <ul>
        <li><strong>🌐 Selenium:</strong> Automated web interactions for metadata extraction.</li>
        <li><strong>⚙️ Dask:</strong> Distributed computing to manage large datasets.</li>
        <li><strong>🚀 Numba:</strong> JIT compilation for accelerating numerical computations.</li>
        <li><strong>📊 Xarray:</strong> Multi-dimensional array data handling for seamless integration.</li>
    </ul><br><h2>📚 <strong>Supported Datasets</strong></h2>
<p>MapMiner supports a variety of geospatial datasets across multiple categories:</p>
<div>


| Category                            | Datasets                                                                 |
|-------------------------------------|--------------------------------------------------------------------------|
| 🌍 **Satellite**                    | `Sentinel-2`, `Sentinel-1`, `MODIS`, `Landsat`                            |
| 🚁 **Aerial**                       | `NAIP`                                                                   |
| 🗺️ **Basemap**                      | `Google`, `ESRI`                                                         |
| 📍 **Vectors**                      | `Google Building Footprint`, `OSM`                                        |
| 🏔️ **DEM (Digital Elevation Model)** | `Copernicus DEM 30m`, `ALOS DEM`                                          |
| 🌍 **LULC (Land Use Land Cover)**    | `ESRI LULC`                                                              |
| 🌾 **Crop Layer**                   | `CDL Crop Mask`    



<h2>🛠 <strong>Installation</strong></h2>
<p>Ensure you have the necessary dependencies installed:</p>
<pre><code class="highlight">git clone https://github.com/gajeshladhar/mapminer.git
cd mapminer
pip3 install .</code></pre>
    <h2>📝 <strong>Usage</strong></h2>
    <p>MapMiner provides multiple classes to fetch and process different types of geospatial data:</p>
    <h3><strong>1️⃣ GoogleBaseMapMiner</strong></h3>
    <pre><code>from mapminer.miner import GoogleBaseMapMiner
miner = GoogleBaseMapMiner()
ds = miner.fetch(lat=40.748817, lon=-73.985428, radius=500)</code></pre>
    <h3><strong>2️⃣ CDLMiner</strong></h3>
    <pre><code>from mapminer.miner import CDLMiner
miner = CDLMiner()
ds = miner.fetch(lon=-95.665, lat=39.8283, radius=10000, daterange="2024-01-01/2024-01-10")</code></pre>
    <h3><strong>3️⃣ GoogleBuildingMiner</strong></h3>
    <pre><code>from mapminer.miner import GoogleBuildingMiner
miner = GoogleBuildingMiner()
ds = miner.fetch(lat=34.052235, lon=-118.243683, radius=1000)</code></pre>
    <h2>🖼 <strong>Visualizing the Data</strong></h2>
    <p>You can easily visualize the data fetched using <code class="highlight">hvplot</code>:</p>
    <pre><code>import hvplot.xarray
ds.hvplot.image(title=f"Captured on {ds.attrs['metadata']['date']['value']}")</code></pre>
    <h2>📦 <strong>Dependencies</strong></h2>
    <p>MapMiner relies on several Python libraries:</p>
    <ul>
        <li><strong class="important">Selenium:</strong> For automated browser control.</li>
        <li><strong class="important">Dask:</strong> For distributed computing and handling large data.</li>
        <li><strong class="important">Numba:</strong> For accelerating numerical operations.</li>
        <li><strong class="important">Xarray:</strong> For handling multi-dimensional array data.</li>
        <li><strong class="important">EasyOCR:</strong> For extracting text from images.</li>
        <li><strong class="important">HvPlot:</strong> For visualizing xarray data.</li>
    </ul>
    <h2>🛠 <strong>Contributing</strong></h2>
    <p>Contributions are welcome! Fork the repository and submit pull requests. Include tests for any new features or bug fixes.</p>
</body>
</html>
