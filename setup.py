from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# ---- ✅ Minimal Fix: Clean reader for requirements ----
def read_requirements(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r"):
                # recursively include referenced file
                ref_file = line.split(" ", 1)[1]
                ref_path = pathlib.Path(path).parent / ref_file
                lines.extend(read_requirements(ref_path))
            else:
                lines.append(line)
    return lines

# Read base and all requirements
base_requirements = read_requirements('requirements/base.txt')
all_requirements = read_requirements('requirements/all.txt')

# ---- Setup configuration ----
setup(
    name='mapminer',
    version='0.1.78',
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
        'mapminer': ['miners/keys/*'],
    },
    install_requires=base_requirements,
    extras_require={
        "all": all_requirements,
    },
    python_requires='>=3.9',  # ✅ updated since many deps need >=3.9
    project_urls={
        'Documentation': 'https://github.com/gajeshladhar/mapminer#readme',
        'Source': 'https://github.com/gajeshladhar/mapminer',
        'Tracker': 'https://github.com/gajeshladhar/mapminer/issues',
    },
)
