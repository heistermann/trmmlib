# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         vis
# Purpose:
#
# Authors:      Maik Heistermann
#
# Created:      2015-11-6
# Copyright:    (c) Maik Heistermann
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python


import wradlib
import numpy as np
import pylab as plt
import os
import datetime as dt
from matplotlib.collections import PolyCollection
from matplotlib.colors import from_levels_and_colors
from scipy.stats.stats import pearsonr
from scipy.signal import medfilt
import trmmlib as tl 

    
def plot_grid(X, Y, R, dx=0.25, dy=0.25, ax=None, overlays=[], bbox=None, **kwargs):
    """Plots grid using pcolormesh.
    
    This function assumes that the grid coordinates X and Y represent center
    points of the grid cells. Since pcolormesh requires corner points, you also
    need to specify the pixel size using dx and dy.
    
    X: array of X coordinates having the same shape as R
    Y: array of Y coordinates having the same shape as R
    R: 2-D grid to be plotted
    dx: pixel size in x direction
    dy: pixel size in y direction
    ax: axes object (if None, a new axes will be created)
    shape: list of strings
        paths to shapefiles for overlay
    kwargs: keyword arguments for matplotlib.pcolormesh
    
    """
    # Pre-process X and Y coords to represent corner points instead of center points
    X = X.copy() - dx/2.
    X = np.hstack((X,X[:,-1].reshape((-1,1)) + dx) )
    X = np.vstack((X,X[-1,:].reshape((1,-1))) )
    Y = Y.copy() - dy/2.
    Y = np.vstack((Y,Y[-1,:].reshape((1,-1)) + dy) )
    Y = np.hstack((Y,Y[:,-1].reshape((-1,1))) )

    # Create axes object of not passed
    if ax==None:
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111, aspect="equal")
    pm = ax.pcolormesh(X, Y, R, **kwargs)
    plt.colorbar(pm, shrink=0.5)
    plt.grid(color="white")
    if bbox:
        plt.xlim(bbox["left"],bbox["right"])
        plt.ylim(bbox["bottom"],bbox["top"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Shapefile overlay
#    for overlay in overlays:
#        dataset, inLayer = wradlib.io.open_shape(shp)
#        borders, keys = wradlib.georef.get_shape_coordinates(inLayer)
#        wradlib.vis.add_lines(ax, overlay, color='white', lw=0.5)


def plot_cats(cats, R, ax=None, bbox=None,  **kwargs):
    """Plots grid using pcolormesh.
    
    This function assumes that the grid coordinates X and Y represent center
    points of the grid cells. Since pcolormesh requires corner points, you also
    need to specify the pixel size using dx and dy.
    
    """
    if ax==None:
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111, aspect="equal")
    wradlib.vis.add_lines(ax, cats, color='black', lw=0.5)
    coll = PolyCollection(cats, array=R, **kwargs)
    ax.add_collection(coll)
    ax.autoscale()
    if bbox:
        plt.xlim(bbox["left"],bbox["right"])
        plt.ylim(bbox["bottom"],bbox["top"])
    plt.draw()
    return ax, coll
    

def plot_trmm_grid_lines(ax):
    """
    """
    y = np.arange(60, 60-480*0.25, -0.25)
    x = np.arange(0, 360, 0.25) - 180.
    ax.hlines(y, xmin=x.min()-0, xmax=x.max()+0, color="grey")
    ax.vlines(x, ymin=y.min()-0, ymax=y.max()+0, color="grey")


    
def maps_from_echse(conf):
    """Produces time series of rainfall maps from ECHSE input data and catchment shapefiles.
    """
    # Read sub-catchment rainfall from file
    fromfile = np.loadtxt(conf["f_data"], dtype="string", delimiter="\t")
    if len(fromfile)==2:    
        rowix = 1
    elif len(fromfile)>2:
        rowix = slice(1,len(fromfile))
    else:
        raise Exception("Data file is empty: %s" % conf["f_data"])
        
    var = fromfile[rowix,1:].astype("f4")
    dtimes = fromfile[rowix,0]
    dtimes = np.array([wradlib.util.iso2datetime(dtime) for dtime in dtimes])
    dtimesfromconf = wradlib.util.from_to(conf["tstart"], conf["tend"], conf["interval"])
    dtimes = np.intersect1d(dtimes, dtimesfromconf)
    if len(dtimes)==0:
        print "No datetimes for mapping based on intersection of data file and config info."
        return(0)
    
#    objects = fromfile[0,1:]

    cats = plt.genfromtxt(conf["f_coords"], delimiter="\t", names=True,
                          dtype=[('id', '|S20'), ('lat', 'f4'), ('lon', 'f4'), 
                                 ('x', 'f4'), ('y', 'f4')])
    mapx, mapy = wradlib.georef.reproject(cats["x"],cats["y"], 
                                          projection_source=conf["trg_proj"], 
                                          projection_target=conf["map_proj"])

    # Read shapefile
    dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
    polys, keys = wradlib.georef.get_shape_coordinates(inLayer, key='DN')
    keys = np.array(keys)
    # Preprocess polygons (remove minors, sort in same order as in coords file)
    polys2 = []
    for i, id in enumerate(cats["id"]):
        keyix = np.where( keys==eval(id.strip("cats_")) )[0]
        if len(keyix) > 1:
            # More than one key matching? Find largest matching polygon
            keyix = keyix[np.argmax([len(polys[key]) for key in keyix])]
        else:
            keyix = keyix[0]            
        poly = polys[keyix].copy()
        if poly.ndim==1:
            # Multi-Polygons - keep only the largest polygon 
            # (just for plotting - no harm done)
            poly2 = poly[np.argmax([len(subpoly) for subpoly in poly])].copy()
        else:
            poly2 = poly.copy()
        polys2.append ( wradlib.georef.reproject(poly2, 
                                           projection_source=conf["trg_proj"], 
                                           projection_target=conf["map_proj"]) )
    
    colors = plt.cm.spectral(np.linspace(0,1,len(conf["levels"])))    
    mycmap, mynorm = from_levels_and_colors(conf["levels"], colors, extend="max")
    
    plt.interactive(False)
    for i, dtime in enumerate(dtimes):
        datestr = (dtime-dt.timedelta(seconds=conf["interval"])).strftime("%Y%m%d.png")
        print datestr
        figpath = os.path.join(conf["savefigs"], datestr)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, aspect="equal")
        ax, coll = plot_cats(polys2, var[i], ax=ax, bbox=conf["bbox"], cmap=mycmap, 
                  norm=mynorm, edgecolors='none')
        cb = plt.colorbar(coll, ax=ax, ticks=conf["levels"], shrink=0.6)
        cb.ax.tick_params(labelsize="small")
        cb.set_label("(mm)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plot_trmm_grid_lines(ax)
        plt.text(conf["bbox"]["left"]+0.25, conf["bbox"]["top"]-0.25, 
                 "%s\n%s to\n%s" % (conf["figtxtbody"], 
                                    (dtime-dt.timedelta(seconds=conf["interval"])).isoformat(" "),
                                     dtime.isoformat(" ") ),
                 color="red", fontsize="small", verticalalignment="top")
        plt.tight_layout()
        plt.savefig(figpath)
        plt.close()
    plt.interactive(True)


def simple_scatter(x, y, xlab, ylab, lim, txt="", **kwargs):
    """Simple scatter plot
    """
    plt.scatter(x, y, **kwargs)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(lim[0],lim[1])
    plt.ylim(lim[0],lim[1])
    plt.plot([lim[0]-100,lim[1]+100], [lim[0]-100,lim[1]+100], color="black", linestyle="--")
    plt.text(lim[0]+10, lim[1]-10, txt, verticalalignment="top", color="red")

def time_series_intercomparison(conf_imd4, conf_gages, conf_trmm, conf_trmm_rt):
    """
    """
    
    tstart = dt.datetime.strptime("2001-04-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    tend = dt.datetime.strptime("2010-12-30 00:00:00", "%Y-%m-%d %H:%M:%S")

    
    dtimes_imd4, _, imd4 = tl.echse.read_echse_data_file(conf_imd4["f_data"])
    dtimes_trmm, _, trmm = tl.echse.read_echse_data_file(conf_trmm["f_data"])
    dtimes_trmmrt, _, trmmrt = tl.echse.read_echse_data_file(conf_trmm_rt["f_data"])
    dtimes_gage, _, gage = tl.echse.read_echse_data_file(conf_gages["f_data"])
    
    ix_imd4   = (dtimes_imd4   >= tstart) & (dtimes_imd4   <tend)
    ix_trmm   = (dtimes_trmm   >= tstart) & (dtimes_trmm   <tend)
    ix_trmmrt = (dtimes_trmmrt >= tstart) & (dtimes_trmmrt <tend)
    ix_gage   = (dtimes_gage   >= tstart) & (dtimes_gage   <tend)
    
    lim = (0,300)
    txt = "Daily rainfall\nSubcatchment average\n%s to %s" % (tstart.strftime("%Y-%m-%d"),tend.strftime("%Y-%m-%d"))
    kwargs = {"edgecolor":"None", "alpha":0.05}
    
    plt.interactive(False)
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(221, aspect="equal")
    corr = "\nR=%.2f" % pearsonr(gage[ix_gage,:].ravel(), trmm[ix_trmm,:].ravel())[0]
    tl.vis.simple_scatter(gage[ix_gage,:], trmm[ix_trmm,:], "GAGE (mm)", "TRMM (mm)", lim, txt="GAGE vs. TRMM\n"+txt+corr, **kwargs  )

    ax = fig.add_subplot(222, aspect="equal")
    corr = "\nR=%.2f" % pearsonr(imd4[ix_imd4,:].ravel(), trmm[ix_trmm,:].ravel())[0]
    tl.vis.simple_scatter(imd4[ix_imd4,:], trmm[ix_trmm,:],     "IMD4 (mm)", "TRMM (mm)", lim, txt="IMD4 vs. TRMM\n"+txt+corr, **kwargs)

    ax = fig.add_subplot(223, aspect="equal")
    corr = "\nR=%.2f" % pearsonr(gage[ix_gage,:].ravel(), imd4[ix_imd4,:].ravel())[0]
    tl.vis.simple_scatter(gage[ix_gage,:], imd4[ix_imd4,:],     "GAGE (mm)", "IMD4 (mm)", lim, txt="GAGE vs. IMD4\n"+txt+corr, **kwargs)

    ax = fig.add_subplot(224, aspect="equal")
    corr = "\nR=%.2f" % pearsonr(trmm[ix_trmm,:].ravel(), trmmrt[ix_trmmrt,:].ravel())[0]
    tl.vis.simple_scatter(trmm[ix_trmm,:], trmmrt[ix_trmmrt,:], "TRMM (mm)", "TRMM RT (mm)", lim, txt="TRMM vs. TRMM RT\n"+txt+corr, **kwargs)
    
    plt.tight_layout()

    plt.savefig("P:/progress/mahanadi/_qpe/inter_product_scatter.png")
    plt.interactive(True)
    
    plt.figure(figsize=(12,12))
    plt.subplot(311)
    plt.plot(dtimes_imd4[ix_imd4], medfilt( np.mean(imd4[ix_imd4,:],axis=1), 1 ), color="black", label="IMD4" )
    plt.plot(dtimes_gage[ix_gage], medfilt( np.mean(gage[ix_gage,:],axis=1), 1 ), color="green",  label="GAGE", alpha=0.7  )
    plt.plot(dtimes_trmm[ix_trmm], medfilt( np.mean(trmm[ix_trmm,:],axis=1), 1 ), color="red",  label="TRMM", alpha=0.5  )
    plt.plot(dtimes_trmmrt[ix_trmmrt], medfilt( np.mean(trmmrt[ix_trmmrt,:],axis=1), 1 ), color="blue",  label="TRMM RT", alpha=0.5  )
    plt.xlabel("Year")
    plt.ylabel("Daily rainfall (mm)")
    plt.title("Unsmoothed")
    plt.legend()
    plt.subplot(312)
    plt.plot(dtimes_imd4[ix_imd4], medfilt( np.mean(imd4[ix_imd4,:],axis=1), 31 ), color="black", label="IMD4" )
    plt.plot(dtimes_gage[ix_gage], medfilt( np.mean(gage[ix_gage,:],axis=1), 31 ), color="green",  label="GAGE", alpha=0.7  )
    plt.plot(dtimes_trmm[ix_trmm], medfilt( np.mean(trmm[ix_trmm,:],axis=1), 31 ), color="red",  label="TRMM", alpha=0.5  )
    plt.plot(dtimes_trmmrt[ix_trmmrt], medfilt( np.mean(trmmrt[ix_trmmrt,:],axis=1), 31 ), color="blue",  label="TRMM RT", alpha=0.5  )
    plt.xlabel("Year")
    plt.ylabel("Daily rainfall (mm)")
    plt.title("Smoothed with 31 day median filter")
    plt.subplot(313)
    plt.plot(dtimes_imd4[ix_imd4], medfilt( np.mean(imd4[ix_imd4,:],axis=1), 91 ), color="black", label="IMD4" )
    plt.plot(dtimes_gage[ix_gage], medfilt( np.mean(gage[ix_gage,:],axis=1), 91 ), color="green",  label="GAGE", alpha=0.7  )
    plt.plot(dtimes_trmm[ix_trmm], medfilt( np.mean(trmm[ix_trmm,:],axis=1), 91 ), color="red",  label="TRMM", alpha=0.5  )
    plt.plot(dtimes_trmmrt[ix_trmmrt], medfilt( np.mean(trmmrt[ix_trmmrt,:],axis=1), 91 ), color="blue",  label="TRMM RT", alpha=0.5  )
    plt.xlabel("Year")
    plt.title("Smoothed with 91 day median filter")
    plt.tight_layout()
    
    plt.savefig("P:/progress/mahanadi/_qpe/inter_product_timeseries.png")



if __name__ == '__main__':

    pass