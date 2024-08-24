<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MapMiner README</title>
</head>
<body>
    <h1>🌍 <strong>MapMiner</strong> 🌍</h1>
    <p>
        <img src="https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square&logo=python" alt="Python">
        <img src="https://img.shields.io/badge/Xarray-0.18+-orange.svg?style=flat-square&logo=xarray" alt="Xarray">
        <img src="https://img.shields.io/badge/Dask-Powered-yellow.svg?style=flat-square&logo=dask" alt="Dask">
        <img src="https://img.shields.io/badge/Numba-Accelerated-green.svg?style=flat-square&logo=numba" alt="Numba">
        <img src="https://img.shields.io/badge/Selenium-Automated-informational.svg?style=flat-square&logo=selenium" alt="Selenium">
    </p>
    <p><strong>MapMiner</strong> is an advanced geospatial tool designed to efficiently download and process Google basemap imagery and associated metadata. Leveraging powerful Python libraries like <strong>Selenium</strong>, <strong>Dask</strong>, <strong>Numba</strong>, and <strong>Xarray</strong>, MapMiner offers high-performance data retrieval and processing capabilities for geospatial analysis and visualization.</p>

    <h2>🚀 <strong>Key Features</strong></h2>
    <ul>
        <li><strong>🌐 Selenium:</strong> Automated web interactions to extract metadata from Google Earth.</li>
        <li><strong>⚙️ Dask:</strong> Distributed computing to manage large datasets and parallelize tasks.</li>
        <li><strong>🚀 Numba:</strong> JIT compilation for accelerating numerical computations.</li>
        <li><strong>📊 Xarray:</strong> Advanced data handling for multi-dimensional arrays, enabling seamless integration with geospatial data.</li>
    </ul>

    <h2>🛠 <strong>Installation</strong></h2>
    <p>Ensure you have the necessary dependencies installed:</p>
    <pre><code>pip install selenium dask numba xarray requests easyocr</code></pre>

    <h2>📝 <strong>Usage</strong></h2>
    <p>MapMiner provides a streamlined API to fetch and process imagery and metadata:</p>

    <h3><strong>1️⃣ Initialize the Miner</strong></h3>
    <p>Start by creating an instance of the <code>GoogleMiner</code> class:</p>
    <pre><code>from mapminer import GoogleMiner

miner = GoogleMiner()</code></pre>

    <h3><strong>2️⃣ Fetch Imagery and Metadata</strong></h3>
    <p>Specify a location using latitude, longitude, and radius, or define a bounding box (<code>bbox</code>), and fetch the corresponding data:</p>
    <pre><code># Fetch imagery and metadata by location
ds = miner.fetch(lat=40.748817, lon=-73.985428, radius=500)

# Alternatively, fetch by bounding box
bbox = (-74.0, 40.7, -73.9, 40.8)
ds = miner.fetch(bbox=bbox, resolution=1)</code></pre>

    <h3><strong>3️⃣ Access the Data</strong></h3>
    <p>The returned <code>xarray.Dataset</code> contains the imagery and associated metadata:</p>
    <pre><code># Access the image data
image_data = ds.data

# Access the metadata
metadata = ds.attrs['metadata']
print(metadata['date']['value'])  # Print the date of the imagery</code></pre>

    <h2>⚙️ <strong>How It Works</strong></h2>
    <p>MapMiner combines several powerful libraries to provide a seamless and efficient workflow:</p>
    <ul>
        <li><strong>Selenium</strong> automates the process of metadata extraction from Google Earth, taking screenshots and using OCR to extract relevant information.</li>
        <li><strong>Dask</strong> manages large datasets, enabling parallelized downloads and computations, which is crucial for handling extensive basemap data.</li>
        <li><strong>Numba</strong> is utilized to speed up computationally expensive tasks, ensuring that the processing is both fast and efficient.</li>
        <li><strong>Xarray</strong> is used to handle the multi-dimensional array data, providing an intuitive interface for working with geospatial imagery and metadata.</li>
    </ul>

    <h2>🔧 <strong>Customization</strong></h2>
    <p>MapMiner is highly customizable to suit various needs. You can modify the resolution, bounding box, and even the zoom level for Google Earth:</p>
    <pre><code># Custom fetch with different zoom level and resolution
ds = miner.fetch(lat=34.052235, lon=-118.243683, radius=1000, resolution=0.5)</code></pre>

    <h2>🧪 <strong>Example</strong></h2>
    <p>Below is an example of how to use MapMiner to download basemap tiles and retrieve metadata:</p>
    <pre><code>from mapminer import GoogleMiner

# Initialize the miner
miner = GoogleMiner()

# Fetch the data
ds = miner.fetch(lat=51.5074, lon=-0.1278, radius=1000)

# Access image data and metadata
image = ds.data
metadata = ds.attrs['metadata']

# Output the date the image was captured
print("Captured Date:", metadata['date']['value'])</code></pre>

    <h2>🖼 <strong>Visualizing the Data</strong></h2>
    <p>MapMiner outputs can be easily visualized using popular Python libraries like <code>matplotlib</code> or <code>hvplot</code>:</p>
    <pre><code>import matplotlib.pyplot as plt

# Visualize the fetched imagery
plt.imshow(ds.data)
plt.title(f"Captured on {metadata['date']['value']}")
plt.show()</code></pre>

    <h2>📦 <strong>Dependencies</strong></h2>
    <p>MapMiner relies on the following Python libraries:</p>
    <ul>
        <li><strong>Selenium</strong>: For automated browser control.</li>
        <li><strong>Dask</strong>: For distributed computing and handling large data.</li>
        <li><strong>Numba</strong>: For speeding up numerical operations.</li>
        <li><strong>Xarray</strong>: For handling multi-dimensional array data.</li>
        <li><strong>EasyOCR</strong>: For extracting text from images.</li>
    </ul>

    <h2>🛠 <strong>Contributing</strong></h2>
    <p>Contributions are welcome! Please fork this repository and submit pull requests. Make sure to include tests for any new features or bug fixes.</p>

    <h2>📝 <strong>License</strong></h2>
    <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
</html>
