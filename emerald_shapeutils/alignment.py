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
    with rasterio.open(path_raster) as raster:
        return sample_raster(raster, x, y, xy_crs).T[0]
    
def generate_interpolation_points_geodataframe_from_gdf2(coords, crs, dtm_tif, plot=False, xdist_shift = 0):
    gs_x = coords[:,0]
    gs_y = coords[:,1]
    xdist = coords[:,-1]
    
    if coords.shape[1] > 3:
        gs_z = coords[:,2]
    else:
        gs_z = np.full(len(coords), np.nan)
        
    # create new dataframe as points
    d = {'xdist':xdist,
        'geometry':gpd.points_from_xy(gs_x, gs_y, gs_z),
        'x':gs_x,
        'y':gs_y,
        'z':gs_z}
    gdf = gpd.GeoDataFrame(d,crs=crs)

    if xdist_shift is not None and xdist_shift !=0.0:
        gdf.xdist = gdf.xdist+xdist_shift

    # if DTM specified, sample raster values at interpolation points along line
    if dtm_tif is not None:
        gdf.loc[:,'topo'] = sample_single_channel_raster_file(dtm_tif,
                                          gdf.x.to_numpy(),
                                          gdf.y.to_numpy(),
                                          gdf.crs)
    else:
        gdf.loc[:, 'topo'] = np.nan

    return gdf

def generate_interpolation_points_geodataframe_from_gdf(tunnel_alignment, sampling_distance, *arg, **kw):
    shape = resample_shape(tunnel_alignment.geometry.iloc[0], sampling_distance)
    coords = np.array(shape.coords)
    xdists = np.arange(len(coords)) * sampling_distance
    coords = np.column_stack((coords, xdists))
        
    return generate_interpolation_points_geodataframe_from_gdf2(
        coords,
        tunnel_alignment.crs,
        *arg, **kw)

def generate_interpolation_points_geodataframe(tunnel_alignment_shp,sampling_distance,
                                               dtm_tif, plot=False, xdist_shift=0):
    #read the tunnel alignment shapefile as a GeoDataFrame
    tunnel_alignment = gpd.read_file(tunnel_alignment_shp)
    return generate_interpolation_points_geodataframe_from_gdf(
        tunnel_alignment,sampling_distance,
        dtm_tif, plot, xdist_shift)
