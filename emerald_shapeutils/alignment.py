import geopandas as gpd
import rasterio

from scipy.interpolate import interp1d
import numpy as np

import pandas as pd
from pyproj import Transformer

from shapely import wkt
from shapely.geometry import LineString, Point

def resample_shape(geom, distance):
    """Resamples shapely shape `geom` at positions `distance` apart
    (measured in coordinate units). Currently only supports LineString
    and MultiLineString.
    """
    # adapted from
    # https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely
    # todo : this function assumes that the coordinate system is a cartesian system using metres. CCh, 2021-01-12

    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [resample_shape(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

def sample_raster(raster, x, y, xy_crs):
    """Sample data from a rasterio raster taking care to transform
    coordinates. Returns numpy array (Npos, Mchannels)."""
    x_trans, y_trans = Transformer.from_crs(xy_crs, raster.crs, always_xy=True).transform(x,y)
    return np.array(list(raster.sample(np.column_stack((x_trans, y_trans)), 1)))

def sample_single_channel_raster_file(path_raster, x, y, xy_crs):
    """Sample data from a geotiff file taking care to transform
    coordinates. Returns numpy array (Npos, Mchannels)."""
    with rasterio.open(path_raster) as raster:
        return sample_raster(raster, x, y, xy_crs).T[0]
    
def sample_shape_to_points(shape, sampling_distance, crs):
    """Sample a Shapely shape at regular intervals and generate a
    GeoPandas GeoDataFrame with point geometries, x, y and z columns,
    as well as xdist, the distance along the shape."""

    shape = resample_shape(shape, sampling_distance)
    coords = np.array(shape.coords)
    xdists = np.arange(len(coords)) * sampling_distance
    coords = np.column_stack((coords, xdists))

    x, y = coords[:,0], coords[:,1]
    z = coords[:,2] if shape.has_z else np.full(len(coords), np.nan)
        
    return gpd.GeoDataFrame({'xdist':xdists,
                             'geometry':gpd.points_from_xy(x, y, z),
                             'x':x,
                             'y':y,
                             'z':z},
                            crs=crs)

def generate_interpolation_points_geodataframe_from_gdf(shape_gdf, sampling_distance, dtm_tif, xdist_shift = 0):
    """Sample a GeoPandas GeoDataFrame (with a single row) at even
    intervals and sample a dtm at the same positions. Returns
    GeoPandas GeoDataFrame with point geometries, x, y, z, xdist and
    topo columns.
    """
    
    points = sample_shape_to_points(shape_gdf.geometry.iloc[0], sampling_distance, shape_gdf.crs)

    if xdist_shift is not None and xdist_shift !=0.0:
        points.xdist = points.xdist+xdist_shift

    # if DTM specified, sample raster values at interpolation points along line
    if dtm_tif is not None:
        points.loc[:,'topo'] = sample_single_channel_raster_file(dtm_tif,
                                                              points.x.to_numpy(),
                                                              points.y.to_numpy(),
                                                              points.crs)
    else:
        points.loc[:, 'topo'] = np.nan

    return points

def generate_interpolation_points_geodataframe(shape_gdf_shp,sampling_distance, dtm_tif, xdist_shift=0):
    #read the tunnel alignment shapefile as a GeoDataFrame
    shape_gdf = gpd.read_file(shape_gdf_shp)
    return generate_interpolation_points_geodataframe_from_gdf(
        shape_gdf,sampling_distance,
        dtm_tif, xdist_shift)
