# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:05:48 2015

@author: heistermann
"""

from osgeo import osr
import wradlib
import pylab as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
import datetime as dt

   
def testplot(cats, catsavg, xy, data, levels = np.arange(0,320,20), title=""):
    """Quick test plot layout for this example file
    """
    colors = plt.cm.spectral(np.linspace(0,1,len(levels)) )    
    mycmap, mynorm = from_levels_and_colors(levels, colors, extend="max")

    radolevels = levels.copy()#[0,1,2,3,4,5,10,15,20,25,30,40,50,100]
    radocolors = plt.cm.spectral(np.linspace(0,1,len(radolevels)) )    
    radocmap, radonorm = from_levels_and_colors(radolevels, radocolors, extend="max")

    fig = plt.figure(figsize=(12,12))
    # Average rainfall sum
    ax = fig.add_subplot(111, aspect="equal")
    wradlib.vis.add_lines(ax, cats, color='black', lw=0.5)
    catsavg[np.isnan(catsavg)]=-9999
    coll = PolyCollection(cats, array=catsavg, cmap=mycmap, norm=mynorm, edgecolors='none')
    ax.add_collection(coll)
    ax.autoscale()
    cb = plt.colorbar(coll, ax=ax, shrink=0.75)
    plt.xlabel("UTM 51N Easting")
    plt.ylabel("UTM 51N Northing")
#    plt.title("Subcatchment rainfall depth")
    plt.grid()
    plt.draw()
    plt.savefig(r"E:\docs\_projektantraege\SUSTAIN_EU_ASIA\workshop\maik\pampanga_cat.png")
#    plt.close()
    # Original RADOLAN data
    fig = plt.figure(figsize=(12,12))
    # Average rainfall sum
    ax1 = fig.add_subplot(111, aspect="equal")
    pm = plt.pcolormesh(xy[:, :, 0], xy[:, :, 1], np.ma.masked_invalid(data), cmap=radocmap, norm=radonorm)
    wradlib.vis.add_lines(ax1, cats, color='black', lw=0.5)
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    cb = plt.colorbar(pm, ax=ax1, shrink=0.75)
    cb.set_label("(mm/h)")
    plt.xlabel("UTM 51N Easting")
    plt.ylabel("UTM 51N Northing")
#    plt.title("Composite rainfall depths")
    plt.grid()
    plt.draw()
    plt.savefig(r"E:\docs\_projektantraege\SUSTAIN_EU_ASIA\workshop\maik\pampanga_comp.png")
    plt.close()
#    plt.tight_layout()


if __name__ == '__main__':

    # Get RADOLAN grid coordinates
#    grid_xy_radolan = wradlib.georef.get_radolan_grid(900, 900)
#    x_radolan = grid_xy_radolan[:, :, 0]
#    y_radolan = grid_xy_radolan[:, :, 1]
#    
#    # create radolan projection osr object
#    proj_stereo = wradlib.georef.create_osr("dwd-radolan")

    # create Gauss Krueger zone 4 projection osr object
    proj_gk = osr.SpatialReference()
    proj_gk.ImportFromEPSG(32651)

    # transform radolan polar stereographic projection to GK4
#    xy = wradlib.georef.reproject(grid_xy_radolan,
#                                  projection_source=proj_stereo,
#                                  projection_target=proj_gk)

    # Open shapefile (already in GK4)
    shpfile = r"P:\progress\ECHSE_projects\echse\echse_proj\pampanga\data\topocatch\out\shpfiles\proj_shp.shp"
    dataset, inLayer = wradlib.io.open_shape(shpfile)
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='ID')

    # Read and prepare the actual data (RADOLAN)
    f = r"P:\progress\test\pampanga_comp.hdf5"
    data, attrs = wradlib.io.from_hdf5(f)
    xy = wradlib.util.gridaspoints(attrs["y"], attrs["x"])
    X = xy[:,0].reshape((1000,1000))
    Y = xy[:,1].reshape((1000,1000))
    
    # Reduce grid size using a bounding box (to enhancing performance)
    bbox = inLayer.GetExtent()
    buffer = 5000.
    bbox = dict(left=bbox[0]-buffer, right=bbox[1]+buffer, bottom=bbox[2]-buffer, top=bbox[3]+buffer)
    mask, shape = wradlib.zonalstats.mask_from_bbox(X,Y, bbox)
    xy_ = np.vstack((X[mask].ravel(),Y[mask].ravel())).T
    data_ = data[mask]
    
    ###########################################################################
    # Approach #1: Assign grid points to each polygon and compute the average.
    # 
    # - Uses matplotlib.path.Path
    # - Each point is weighted equally (assumption: polygon >> grid cell)
    # - this is quick, but theoretically dirty     
    ###########################################################################

    t1 = dt.datetime.now()

    # Create instances of type GridPointsToPoly (one instance for each target polygon)
    obj1 = wradlib.zonalstats.GridPointsToPoly(xy_, cats, buffer=500.)

    # Compute stats for target polygons
    avg1 =  obj1.mean( data_.ravel() )
    
    # Plot average rainfall and original data
#    testplot(cats, avg1, xy.reshape((1000,1000,2)), data, title="Catchment rainfall mean (GridPointsToPoly)")
   

    ###########################################################################
    # Approach #2: Compute weighted mean based on fraction of source polygons in target polygons
    # 
    # - This is more accurate (no assumptions), but probably slower...
    ###########################################################################

    # Create vertices for each grid cell (MUST BE DONE IN NATIVE RADOLAN COORDINATES)
    grdverts = wradlib.zonalstats.grid_centers_to_vertices(X[mask],Y[mask],1.,1.)
    # And reproject to Cartesian reference system (here: GK4)
#    grdverts = wradlib.georef.reproject(grdverts,
#                                  projection_source=proj_stereo,
#                                  projection_target=proj_gk)

    # Create instances of type GridCellsToPoly (one instance for each target polygon)
    obj3 = wradlib.zonalstats.GridCellsToPoly(grdverts, cats)

    # Compute stats for target polygons
    avg3 =  obj3.mean( data_.ravel() )

    # Plot average rainfall and original data
    testplot(cats, avg3, xy.reshape((1000,1000,2)), data, title="Catchment rainfall mean (GridCellsToPoly)")
    

