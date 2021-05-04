import geopandas as gpd
import rasterio

from scipy.interpolate import interp1d
import numpy as np

import pandas as pd
from pyproj import Transformer

from shapely import wkt
from shapely.geometry import LineString, Point

def redistribute_vertices(geom, distance):
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
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

def sample_raster(path_raster, x, y, crs_points):
    with rasterio.open(path_raster) as tif:
        x_trans, y_trans = Transformer.from_crs(crs_points, tif.crs, always_xy=True).transform(x,y)

        samples = tif.sample(np.column_stack((x_trans, y_trans)), 1)

        terrain_elevations = []
        for location in samples:
            for band in location:
                terrain_elevations.append(band)

    return np.array(terrain_elevations)

def generate_interpolation_points_geodataframe(tunnel_alignment_shp,sampling_distance,
                                               dtm_tif, plot=False, xdist_shift=0):
    #read the tunnel alignment shapefile as a GeoDataFrame
    tunnel_alignment = gpd.read_file(tunnel_alignment_shp)
    return generate_interpolation_points_geodataframe_from_gdf(
        tunnel_alignment,sampling_distance,
        dtm_tif, plot, xdist_shift)
    
def generate_interpolation_points_geodataframe_from_gdf(tunnel_alignment,sampling_distance,
                                                        dtm_tif, plot=False, xdist_shift = 0):
    gs = tunnel_alignment

    # Interpolate points with regular spacing along the alignment
    multiline_r = redistribute_vertices(gs.geometry.iloc[0], sampling_distance)
    coords = np.array(multiline_r.coords)
    gs_x = coords[:,0]
    gs_y = coords[:,1]
    if coords.shape[1] > 2:
        gs_z = coords[:,2]
    else:
        gs_z = np.full(len(coords), np.nan)
        
    # create new dataframe as points
    point_geoms = []
    xdist = []
    for i in range(len(gs_x)):
        point_geoms.append(Point(gs_x[i], gs_y[i], gs_z[i]))
        xdist.append(i*sampling_distance) # todo: this computation of xdist assumes that we have only one line object in the shapefile. CCh, 2021-01-12
    d = {'xdist':xdist,
        'geometry':point_geoms,
        'x':gs_x,
        'y':gs_y,
        'z':gs_z}
    gdf = gpd.GeoDataFrame(d,crs=gs.crs)

    if xdist_shift is not None and xdist_shift !=0.0:
        gdf.xdist = gdf.xdist+xdist_shift

    # if DTM specified, sample raster values at interpolation points along line
    if dtm_tif is not None:
        gdf.loc[:,'topo'] = sample_raster(dtm_tif,
                                          gdf.x.to_numpy(),
                                          gdf.y.to_numpy(),
                                          gdf.crs)
    else:
        gdf.loc[:, 'topo'] = np.nan

    return gdf

