from setuptools import setup, find_packages

# Read in the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mapminer',
    version='0.1',
    description='A tool to download and process Geospatial data and metadata.',
    author='Gajesh Ladhar',
    author_email='gajeshladhar@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
