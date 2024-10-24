from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Read in the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mapminer',
    version='0.1.1',
    description='An advanced geospatial data extraction and processing toolkit for Earth observation datasets.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/gajeshladhar/mapminer',
    author='Gajesh Ladhar',
    author_email='gajeshladhar@gmail.com',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        'geospatial', 'GIS', 'Earth observation', 'satellite imagery',
        'data processing', 'remote sensing', 'machine learning', 
        'map tiles', 'metadata extraction', 'planetary datasets', 
        'xarray', 'spatial analysis'
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'mapminer': ['keys/*'],
    },
    install_requires=requirements,
    python_requires='>=3.6',
    project_urls={
        'Documentation': 'https://github.com/gajeshladhar/mapminer#readme',
        'Source': 'https://github.com/gajeshladhar/mapminer',
        'Tracker': 'https://github.com/gajeshladhar/mapminer/issues',
    },
)
