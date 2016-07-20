# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
e:\home\.spyder2\.temp.py
"""

import wradlib
import numpy as np
import pylab as plt
from scipy.spatial import cKDTree
import os
import gc
import sys
import datetime as dt
from matplotlib.collections import PolyCollection
from matplotlib.colors import from_levels_and_colors
from scipy.signal import medfilt


def read_trmm(f):
    """Read TRMM data that comes on NetCDF.
    
    Parameters
    ----------
    f : string (TRMM file path)
    
    Returns
    -------
    out : X, Y, R
        Two dimensional arrays of longitudes, latitudes, and rainfall
        
    """
    data = wradlib.io.read_generic_netcdf(f)
    
    x = data["variables"]["longitude"]["data"]
    y = data["variables"]["latitude"]["data"]
    X, Y = np.meshgrid(x,y)
    X = X - 180.
    R = data["variables"]["r"]["data"][0]
    R = np.roll(R,720,axis=1)
    
    return X, Y, R
    

def read_trmm_bin(f):
    """Read TRMM data that comes as binary data (bin).
    
    Parameters
    ----------
    f : string (TRMM file path)
    
    Returns
    -------
    out : X, Y, R
        Two dimensional arrays of longitudes, latitudes, and rainfall
        
    """
    nlat = 480
##    nlon = 1440
    
    # Read data
    R = np.fromfile(f, dtype="f4")
    if sys.byteorder=="little":
        R = R.byteswap()
    R = np.reshape(R, (1440,480), order="F")
    R = np.rot90(R)
    R = np.roll(R, 720, axis=1)
    
    # Make grid
    y = np.arange(59.875, 59.875-nlat*0.25, -0.25)
    x = np.arange(0, 360, 0.25) - 180.
    X, Y = np.meshgrid(x,y)
    
#    R = np.roll(R,720,axis=1)    

    return X, Y, R
    

def read_imerg(f, var = "Grid/precipitationCal"):
    """Read IMERG data that comes on HDF5.
    
    Parameters
    ----------
    f : string (IMERG file path)
    var : string
        The variable to be extracted from the HDF5 file
    
    Returns
    -------
    out : X, Y, R
        Two dimensional arrays of longitudes, latitudes, and rainfall
        
    """ 
    data = wradlib.io.read_generic_hdf5(f)

    y = data["Grid/lat"]["data"]
    x = data["Grid/lon"]["data"]
    X, Y = np.meshgrid(x,y)
#    X = X - 180.    
    var = data[var]["data"].T
#    var = np.roll(var,len(x)/2,axis=1)
    del data
    gc.collect()
    
    return X, Y, var



def mask_from_bbox(x, y, bbox):
    """Return index array based on spatial selection from a bounding box.
    """
    ny, nx = x.shape
    
    ix = np.arange(x.size).reshape(x.shape)

    # Find bbox corners
    #    Plant a tree
    tree = cKDTree(np.vstack((x.ravel(),y.ravel())).transpose())
    # find lower left corner index
    dists, ixll = tree.query([bbox["left"], bbox["bottom"]], k=1)
    ill, jll = np.array(np.where(ix==ixll))[:,0]
    ill = (ixll / nx)-1
    jll = (ixll % nx)-1
    # find lower left corner index
    dists, ixur = tree.query([bbox["right"],bbox["top"]], k=1)
    iur, jur = np.array(np.where(ix==ixur))[:,0]
    iur = (ixur / nx)+1
    jur = (ixur % nx)+1
    
    mask = np.repeat(False, ix.size).reshape(ix.shape)
    if iur>ill:
        mask[ill:iur,jll:jur] = True
        shape = (iur-ill, jur-jll)
    else:
        mask[iur:ill,jll:jur] = True
        shape = (ill-iur, jur-jll)
    
##    print ill, iur, jll, jur
    
    return mask, shape
        
#    return ix[ill:iur,jll:jur].ravel() 
    
    
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
    

def write_echse_coords_file(f, x, y, lat, lon, id=None, header="# Dummy header\n"):
    """Writes grid point coordinates in ECHSE ready format.
    """
    if id==None:
        id = np.arange(1,len(x)+1)
    with open(f,"w") as fcoords:
#        fcoords.write(header)
#        fcoords.write("#\n")
        fcoords.write("id\tlat\tlon\tx\ty\n")
        for i in xrange(x.size):
            fcoords.write("%s\t%.6f\t%.6f\t%.2f\t%.2f\n" % (id[i], lon.ravel()[i], lat.ravel()[i], x.ravel()[i], y.ravel()[i]))


def write_echse_data(fpath, dtime, data, id=None, header=True):
    """
    """
    if header:
        writemode = "w"
    else:
        writemode = "a"
    # open data file
    with open(fpath,writemode) as f:
        if header:
            # write header line
            f.write("datetime\t")
            if id==None:
                id = np.arange(data.size)+1
            np.savetxt(f, id.reshape((1,-1)), delimiter="\t", fmt="%s")
        # write data line
        f.write( dtime.strftime("%Y-%m-%d %H:%M:%S\t") )
        np.savetxt(f, data.reshape(1,-1), delimiter="\t", fmt="%.1f")
        

def trmm_to_echse(conf):
    """
    """
    trgproj = wradlib.georef.epsg_to_osr(conf["epsg"])
    dtimes = wradlib.util.from_to(conf["tstart"], conf["tend"], conf["interval"])
    newcoords = conf["newcoords"]
    header = True
    
    for dtime in dtimes:
        fname = "3B42_daily.%s.7.nc" % dtime.strftime("%Y.%m.%d")
        fpath = os.path.join(conf["srcdir"], fname)
        lon, lat, R = read_trmm(fpath)
        print fname
        if newcoords:
            # Only once
            mask, shape = mask_from_bbox(lon, lat, conf["bbox"])
            id = np.where(mask.ravel())[0]
            x, y = wradlib.georef.reproject(lon, lat, projection_target=trgproj)
            write_echse_coords_file(conf["f_coords"], x[mask], y[mask], lat[mask], lon[mask], id=None)
            newcoords = False
        write_echse_data(conf["f_data"], dtime, R[mask], id=id, header=header)
        if header:
            header = False


def trmm_to_echse2(conf):
    """Read and and interpolate TRMM data to ECHSE-ready format on catchment level.
    
    """
    dtimes = wradlib.util.from_to(conf["tstart"], conf["tend"], conf["interval"])
    newcoords = conf["newcoords"]
    header = True
    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
    # Just to write also lat/long to the ECHSE coords file
    catlon, catlat = wradlib.georef.reproject(cats["x"], cats["y"], 
                                              projection_source=conf["trg_proj"],
                                              projection_target=wradlib.georef.get_default_projection())
    
    for dtime in dtimes:
        if conf["trmm_product"]=="daily3b42late":
            fname = "3B42_daily.%s.7.nc" % dtime.strftime("%Y.%m.%d")
            reader = read_trmm
        elif conf["trmm_product"]=="daily3b42rt":
            fname = "%s/3B42RT_daily.%s.bin" % (dtime.strftime("%Y"), dtime.strftime("%Y.%m.%d") )
            reader = read_trmm_bin
        else:
            raise Exception("Unknown TRMM product: %s." % conf["trmm_format"])
            
        fpath = os.path.join(conf["srcdir"], fname)
        try:
            lon, lat, R = reader(fpath)
        except IOError:
            print "File not found: %s." % fname
            continue
        print fname
        if newcoords:
            # Only once
            mask, shape = mask_from_bbox(lon, lat, conf["bbox"])
            id = np.array(["cat_%d" % cat for cat in cats["DN"].astype("i4")])
            x, y = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])
            src = np.vstack((x[mask].ravel(), y[mask].ravel())).transpose()
            trg = np.vstack((cats["x"], cats["y"])).transpose()
            ip = wradlib.ipol.Idw(src, trg, nnearest=4)
            write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], catlat, catlon, id=id)
            newcoords = False
        write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(R[mask]), id=id, header=header)
        if header:
            header = False

def read_echse_data_file(f):
    """
    """
    # Read file as string
    fromfile = np.loadtxt(f, dtype="string", delimiter="\t")
    # Check for file size
    if len(fromfile)==2:    
        rowix = 1
    elif len(fromfile)>2:
        rowix = slice(1,len(fromfile))
    else:
        raise Exception("Data file is empty: %s" % f)

    # Convert strings to data, date times, and IDs
    var = fromfile[rowix,1:].astype("f4")
    dtimes = fromfile[rowix,0]
    dtimes = np.array([wradlib.util.iso2datetime(dtime) for dtime in dtimes])
    ids = fromfile[0,1:]
    
    return dtimes, ids, var


def gages_to_echse(conf):
    """Interpolate rain gage observations to catchment centroids.
    """
    newcoords = conf["newcoords"]
    header = True
    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
    locs = np.genfromtxt(conf["f_gage_coords"], delimiter="\t", names=True)
    dtimes, ids, var = read_echse_data_file(conf["f_gage_data"])
    # Just to write also lat/long to the ECHSE coords file
    catlon, catlat = wradlib.georef.reproject(cats["x"], cats["y"], 
                                              projection_source=conf["trg_proj"],
                                              projection_target=wradlib.georef.get_default_projection())
    
    tstart = dt.datetime.strptime(conf["tstart"], "%Y-%m-%d %H:%M:%S")
    tend   = dt.datetime.strptime(conf["tend"], "%Y-%m-%d %H:%M:%S")
    
    for i, dtime in enumerate(dtimes):
        if (dtime < tstart) or (dtime > tend):
            continue        
        print dtime.isoformat(" ")
        
        if newcoords:
            # Only once
            id = np.array(["cat_%d" % cat for cat in cats["DN"].astype("i4")])
            src = np.vstack( (locs["x"], locs["y"]) ).transpose()
            trg = np.vstack( (cats["x"], cats["y"]) ).transpose()
            ip = wradlib.ipol.Idw(src, trg, nnearest=4)
            write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], 
                                    catlat, catlon, id=id)
            newcoords = False
        write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(var[i]), id=id, header=header)
        if header:
            header = False


def imd4_to_echse(conf):
    """Read IMD4 gridded precipiation dataset and interpolate to subcatchments.
    
    """
    # Read data from file
    fromfile = np.loadtxt(conf["f_imd4"], dtype="string", delimiter=",")
    var = fromfile[1:,3:].astype("f4")
    years = fromfile[1:,0].astype("i4")
    months = fromfile[1:,1].astype("i4")
    days = fromfile[1:,2].astype("i4")
    dtimes = np.array( [dt.datetime(years[i], months[i], days[i]) for i in range(len(years))] )
    lon = np.array([])
    lat = np.array([])
    for item in fromfile[0,3:]:
        lon_, lat_ = item.split("_")
        lon = np.append(lon, float(lon_) )
        lat = np.append(lat, float(lat_) )
    x, y = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])

    # Read target subcatchment centroids
    newcoords = conf["newcoords"]
    header = True
    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
    # Just to write also lat/long to the ECHSE coords file
    catlon, catlat = wradlib.georef.reproject(cats["x"], cats["y"], 
                                              projection_source=conf["trg_proj"],
                                              projection_target=wradlib.georef.get_default_projection())
    
    tstart = dt.datetime.strptime(conf["tstart"], "%Y-%m-%d %H:%M:%S")
    tend   = dt.datetime.strptime(conf["tend"], "%Y-%m-%d %H:%M:%S")
    
    for i, dtime in enumerate(dtimes):
        if (dtime < tstart) or (dtime > tend):
            continue        
        print dtime.isoformat(" ")
        
        if newcoords:
            # Only once
            #mask, shape = mask_from_bbox(lon, lat, conf["bbox"])
            id = np.array(["cat_%d" % cat for cat in cats["DN"].astype("i4")])
            #x, y = wradlib.georef.reproject(lon, lat, projection_target=trgproj)
            src = np.vstack( (x, y) ).transpose()
            trg = np.vstack( (cats["x"], cats["y"]) ).transpose()
            ip = wradlib.ipol.Idw(src, trg, nnearest=4)
            write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], 
                                    catlat, catlon, id=id)
            newcoords = False
        write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(var[i]), id=id, header=header)
        if header:
            header = False




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


if __name__ == '__main__':
    
    # Interpolate rain gauges
    conf_gages = dict(
        tstart = "1998-01-01 03:30:00",
        tend = "2015-04-30 03:30:00",
        interval = 86400,
        timeshift = -dt.timedelta(seconds=4.5*3600),

        f_coords      = "P:/progress/mahanadi/qpe/gages_idw/locs.txt",
        f_data        = "P:/progress/mahanadi/qpe/gages_idw/data.txt",
        savefigs      = "P:/progress/mahanadi/qpe/gages_idw/maps",
        
        f_gage_data   = "P:/progress/mahanadi/Precipitation/rain_gages/precip_data_noGaps.txt",
        f_gage_coords = "P:/progress/mahanadi/Precipitation/rain_gages/precip_locs_noGaps.txt",
        
        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",

        figtxtbody    = "Rain Gauge Interpolation IDW\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )
    
    # TRMM 3B42 Research (Late)
    conf_trmm = dict(
        trmm_product = "daily3b42late",
        tstart = "1998-01-01 00:00:00",
        tend = "2015-04-30 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=24*3600),
        srcdir        = "X:/trmm/3B42_daily_V07",
        
        f_coords      = "P:/progress/mahanadi/qpe/3B42_daily_research/coords.txt",
        f_data        = "P:/progress/mahanadi/qpe/3B42_daily_research/data.txt",
        savefigs      = "P:/progress/mahanadi/qpe/3B42_daily_research/maps",
        
        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",

        figtxtbody    = "TRMM 3B42 Research\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )

    # TRMM 3B42 Real-Time (Early)
    conf_trmm_rt = dict(
        trmm_product = "daily3b42rt",
        tstart = "1998-01-01 00:00:00",
        tend = "2015-04-30 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=24*3600),

        srcdir        = "X:/trmm/3B42RT_daily_V07",
        
        f_coords      = "P:/progress/mahanadi/qpe/3B42_daily_realtime/coords.txt",
        f_data        = "P:/progress/mahanadi/qpe/3B42_daily_realtime/data.txt",
        savefigs      = "P:/progress/mahanadi/qpe/3B42_daily_realtime/maps",
        
        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",
        figtxtbody    = "TRMM 3B42 Real-Time\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )

    # IMD4 Gridded rainfall Dataset
    conf_imd4 = dict(
        tstart = "1998-01-01 00:00:00",
        tend = "2015-04-30 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=24*3600),
        f_imd4        = "P:/progress/mahanadi/Precipitation/gridded_IMD/P.csv",
        
        f_coords      = "P:/progress/mahanadi/qpe/imd4/locs.txt",
        f_data        = "P:/progress/mahanadi/qpe/imd4/data.txt",
        savefigs      = "P:/progress/mahanadi/qpe/imd4/maps",

        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",
        figtxtbody    = "IMD4 Subsample\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )
    
    # Compute subcatchment rainfall
#    imd4_to_echse(conf_imd4)
#    gages_to_echse(conf_gages)
#    trmm_to_echse2(conf_trmm_rt)
    
    # Make maps for all time steps based on config
#    maps_from_echse(conf_imd4)
#    maps_from_echse(conf_gages)
    
    tstart = dt.datetime.strptime("2000-04-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    tend = dt.datetime.strptime("2010-12-31 00:00:00", "%Y-%m-%d %H:%M:%S")
    
    dtimes_imd4, _, imd4 = read_echse_data_file(conf_imd4["f_data"])
    dtimes_trmm, _, trmm = read_echse_data_file(conf_trmm["f_data"])
    dtimes_trmmrt, _, trmmrt = read_echse_data_file(conf_trmm_rt["f_data"])
    dtimes_gage, _, gage = read_echse_data_file(conf_gages["f_data"])
    
    ix_imd4   = (dtimes_imd4   >= tstart) & (dtimes_imd4   <tend)
    ix_trmm   = (dtimes_trmm   >= tstart) & (dtimes_trmm   <tend)
    ix_trmmrt = (dtimes_trmmrt >= tstart) & (dtimes_trmmrt <tend)
    ix_gage   = (dtimes_gage   >= tstart) & (dtimes_gage   <tend)
    
    lim = (0,300)
    txt = "Daily rainfall\nSubcatchment average\n%s to %s" % (tstart.strftime("%Y-%m-%d"),tend.strftime("%Y-%m-%d"))
    kwargs = {"edgecolor":"None", "alpha":0.05}
    
#    fig = plt.figure(figsize=(8,8))
#
#    ax = fig.add_subplot(221, aspect="equal")
#    simple_scatter(gage[ix_gage,:], trmm[ix_trmm,:],     "GAGE (mm)", "TRMM (mm)", lim, txt="GAGE vs. TRMM\n"+txt, **kwargs  )
#
#    ax = fig.add_subplot(222, aspect="equal")
#    simple_scatter(imd4[ix_imd4,:], trmm[ix_trmm,:],     "IMD4 (mm)", "TRMM (mm)", lim, txt="IMD4 vs. TRMM\n"+txt, **kwargs)
#
#    ax = fig.add_subplot(223, aspect="equal")
#    simple_scatter(gage[ix_gage,:], imd4[ix_imd4,:],     "GAGE (mm)", "IMD4 (mm)", lim, txt="GAGE vs. IMD4\n"+txt, **kwargs)
#
#    ax = fig.add_subplot(224, aspect="equal")
#    simple_scatter(trmm[ix_trmm,:], trmmrt[ix_trmmrt,:], "TRMM (mm)", "TRMM RT (mm)", lim, txt="TRMM vs. TRMM RT\n"+txt, **kwargs)
#    
#    plt.tight_layout()
    
    plt.figure(figsize=(15,6))
    plt.plot(dtimes_imd4[ix_imd4], medfilt( np.mean(imd4[ix_imd4,:],axis=1), 91 ), color="black", label="IMD4" )
    plt.plot(dtimes_gage[ix_gage], medfilt( np.mean(gage[ix_gage,:],axis=1), 91 ), color="blue",  label="GAGE"  )
    plt.plot(dtimes_trmm[ix_trmm], medfilt( np.mean(trmm[ix_trmm,:],axis=1), 91 ), color="red",  label="TRMM"  )
    plt.xlabel("Year")
    plt.ylabel("Smoothed daily rainfall (mm)")
    plt.legend()

#    plt.plot(dtimes_trmmrt[ix_trmmrt], medfilt( np.mean(trmmrt[ix_trmmrt,:],axis=1), 31 ), color="green",  label="TRMM RT"  )    
    
#    dtimes, lon, lat, var = read_imd4()
#    nlon = len(np.unique(lon))
#    nlat = len(np.unique(lat))
#    plt.scatter(x=lon, y=lat, c=var[100])
#    plt.pcolormesh(lon.reshape(nlon, nlat), lat.reshape(nlon, nlat), var[100].reshape(nlon, nlat))

    
    
#    X2, Y2, R2 = read_trmm_bin(r"X:\trmm\3B42RT_daily_V07\2013\3B42RT_daily.2013.08.06.bin")

#    fpath = r"X:\gpm\imerg\2014\3B-HHR.MS.MRG.3IMERG.20141010-S000000-E002959.0000.V03D.HDF5"
#    X2, Y2, R2 = read_imerg(fpath)
    
#    shapes = ["E:/data/shape_collection/TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp"]
    
    # Get original TRMM data
#    X1, Y1, R1 = read_trmm(r"X:\trmm\3B42_daily_V07\3B42_daily.2013.08.06.7.nc")
    
#    fig = plt.figure(figsize=(12,12))
#    ax = fig.add_subplot(211, aspect="equal")
#    plot_grid(X1, Y1, np.ma.masked_less(R1,0), ax=ax, overlays=[cats], 
#              bbox=conf["bbox"], cmap="spectral", vmax=52)
#    wradlib.vis.add_lines(ax, polys2, color='white', lw=0.5)    
#    plt.title("TRMM 3B42 Native Resolution")
#    
#    ax = fig.add_subplot(212, aspect="equal")
#    plot_cats(polys2, Ravg, ax=ax, bbox=conf["bbox"], cmap=plt.cm.spectral, edgecolors='none')
#    ax.hlines(Y1[:,0]-0.25/2, xmin=X1.min()-0.25/2, xmax=X1.max()+0.25/2, color="grey")
#    ax.vlines(X1[0,:]-0.25/2, ymin=Y1.min()-0.25/2, ymax=Y1.max()+0.25/2, color="grey")

    
#    plot_map(X2, Y2, np.ma.masked_less(R2, 0), ax=ax, shapes=shapes, cmap="spectral", vmax=260)
#    plt.title("TRMM 3B42RT")
    
#    import wradlib
#    data = wradlib.io.read_generic_netcdf(r"X:\trmm\3B42_daily_V07\3B42_daily.2004.01.31.7.nc")
#    print data["CoreMetadata.0"]
    

    



            
            

        


    


    



