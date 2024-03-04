#!/usr/bin/env python

import setuptools
import distutils.command.build
import distutils.command.sdist

setuptools.setup(
    name='emerald-shapeutils',
    version='0.0.3',
    description='Utils for handling shapes and rasters',
    long_description='Utils for handling shapes and rasters',
    long_description_content_type="text/markdown",
    author='Craig William Christensen, Egil Moeller and others ',
    author_email='cch@emrld.no, em@emrld.no',
    url='https://github.com/emerald-geomodelling/emerald-shapeutils',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "shapely",
        "geopandas",
        "scipy",
        "rasterio",
        "pandas",
        "pyproj"
    ]
)
