# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         wflows
# Purpose:
#
# Authors:      Maik Heistermann
#
# Created:      2015-11-6
# Copyright:    (c) Maik Heistermann
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python


import trmmlib as tl
import wradlib

import numpy as np
import pylab as plt
import os
import datetime as dt
import fnmatch
import errno

from matplotlib.colors import from_levels_and_colors
 


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
        lon, lat, R = tl.products.read_trmm(fpath)
        print fname
        if newcoords:
            # Only once
            mask, shape = tl.util.mask_from_bbox(lon, lat, conf["bbox"])
            id = np.where(mask.ravel())[0]
            x, y = wradlib.georef.reproject(lon, lat, projection_target=trgproj)
            tl.echse.write_echse_coords_file(conf["f_coords"], x[mask], y[mask], lat[mask], lon[mask], id=None)
            newcoords = False
        tl.echse.write_echse_data(conf["f_data"], dtime, R[mask], id=id, header=header)
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
            reader = tl.products.read_trmm
        elif conf["trmm_product"]=="daily3b42rt":
            fname = "%s/3B42RT_daily.%s.bin" % (dtime.strftime("%Y"), dtime.strftime("%Y.%m.%d") )
            reader = tl.products.read_trmm_bin
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
            mask, shape = tl.util.mask_from_bbox(lon, lat, conf["bbox"])
            id = np.array(["cat_%d" % cat for cat in cats["DN"].astype("i4")])
            x, y = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])
            src = np.vstack((x[mask].ravel(), y[mask].ravel())).transpose()
            trg = np.vstack((cats["x"], cats["y"])).transpose()
            ip = wradlib.ipol.Idw(src, trg, nnearest=4)
            tl.echse.write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], catlat, catlon, id=id)
            newcoords = False
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(R[mask]), id=id, header=header)
        if header:
            header = False


def trmm_to_echse3(conf):
    """Read and and interpolate TRMM data to ECHSE-ready format on catchment level.
    
    Uses Zonal Stats
    
    """
    dtimes = wradlib.util.from_to(conf["tstart"], conf["tend"], conf["interval"])
    newcoords = conf["newcoords"]
    header = True
    
    for dtime in dtimes:
        if conf["trmm_product"]=="daily3b42late":
            fname = "3B42_daily.%s.7.nc" % dtime.strftime("%Y.%m.%d")
            reader = tl.products.read_trmm
        elif conf["trmm_product"]=="daily3b42rt":
            fname = "%s/3B42RT_daily.%s.bin" % (dtime.strftime("%Y"), dtime.strftime("%Y.%m.%d") )
            reader = tl.products.read_trmm_bin
        elif conf["trmm_product"]=="imerg/final/daily":
            fname = "imerg_final_daily_%s.h5" % dtime.strftime("%Y%m%d")
            reader = tl.products.read_imerg_custom_h5
        else:
            raise Exception("Unknown TRMM product: %s." % conf["trmm_product"])
            
        fpath = os.path.join(conf["srcdir"], fname)
        try:
            lon, lat, R = reader(fpath)
        except IOError:
            print "File not found: %s." % fname
            continue
        print fname
        if newcoords:
            
            # Read target polygons (catchments) --> already projected
            dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
            cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
            cats, keys = tl.util.make_ids_unique(cats, keys)
            cats = tl.util.reduce_multipolygons(cats)
            xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
            lonlat = wradlib.georef.reproject(xy, 
                                              projection_source=conf["trg_proj"],
                                              projection_target=wradlib.georef.get_default_projection())

            # Only once
            mask, shape = tl.util.mask_from_bbox(lon, lat, conf["bbox"])
            grdverts = wradlib.zonalstats.grid_centers_to_vertices(lon[mask],lat[mask],0.25,0.25)
            xygrdverts = wradlib.georef.reproject(grdverts, projection_target=conf["trg_proj"])

            # Create instances of type GridCellsToPoly (one instance for each target polygon)
            obj = wradlib.zonalstats.GridCellsToPoly(xygrdverts, cats)

            tl.echse.write_echse_coords_file(conf["f_coords"], xy[:,0], xy[:,1], lonlat[:,0], lonlat[:,1], id=np.array(keys))
            newcoords = False
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], obj.mean(R[mask].ravel()), id=np.array(keys), header=header)
        if header:
            header = False


def gages_to_echse(conf):
    """Interpolate rain gage observations to catchment centroids.
    """
    newcoords = conf["newcoords"]
    header = True
    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
    locs = np.genfromtxt(conf["f_gage_coords"], delimiter="\t", names=True)
    dtimes, ids, var = tl.echse.read_echse_data_file(conf["f_gage_data"])
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
            tl.echse.write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], 
                                    catlat, catlon, id=id)
            newcoords = False
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(var[i]), id=id, header=header)
        if header:
            header = False


def gages_to_echse2(conf):
    """Interpolate rain gage observations to catchment centroids.
    """
    header = True
    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
    locs = np.genfromtxt(conf["f_gage_coords"], delimiter="\t", names=True)
    dtimes, ids, var = tl.echse.read_echse_data_file(conf["f_gage_data"])
    # Just to write also lat/long to the ECHSE coords file
    catlon, catlat = wradlib.georef.reproject(cats["x"], cats["y"], 
                                              projection_source=conf["trg_proj"],
                                              projection_target=wradlib.georef.get_default_projection())

    # Read target polygons (catchments) --> already projected
    dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
    cats, keys = tl.util.make_ids_unique(cats, keys)
    cats = tl.util.reduce_multipolygons(cats)
    xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
    lonlat = wradlib.georef.reproject(xy, 
                                      projection_source=conf["trg_proj"],
                                      projection_target=wradlib.georef.get_default_projection())

    src = np.vstack( (locs["x"], locs["y"]) ).transpose()
    trg = np.vstack( (xy[:,0], xy[:,1]) ).transpose()
    ip = wradlib.ipol.Idw(src, trg, nnearest=4)
    tl.echse.write_echse_coords_file(conf["f_coords"], xy[...,0], xy[...,0], 
                            lonlat[...,1], lonlat[...,0], id=keys)
   
    tstart = dt.datetime.strptime(conf["tstart"], "%Y-%m-%d %H:%M:%S")
    tend   = dt.datetime.strptime(conf["tend"], "%Y-%m-%d %H:%M:%S")
    
    for i, dtime in enumerate(dtimes):
        if (dtime < tstart) or (dtime > tend):
            continue        
        print dtime.isoformat(" ")
        
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(var[i]), id=keys, header=header)
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
            tl.echse.write_echse_coords_file(conf["f_coords"], cats["x"], cats["y"], 
                                    catlat, catlon, id=id)
            newcoords = False
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], ip(var[i]), id=id, header=header)
        if header:
            header = False


def imd4_to_echse2(conf):
    """Read IMD4 gridded precipiation dataset and interpolate to subcatchments.
    
    Uses ZonalStats.
    """
    # Read IMD4 data and coordinates from file
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
    imdx, imdy = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])

    header = True
#    cats = np.genfromtxt(conf["f_cats"], delimiter="\t", names=True)
#    # Just to write also lat/long to the ECHSE coords file
#    catlon, catlat = wradlib.georef.reproject(cats["x"], cats["y"], 
#                                              projection_source=conf["trg_proj"],
#                                              projection_target=wradlib.georef.get_default_projection())
    
    tstart = dt.datetime.strptime(conf["tstart"], "%Y-%m-%d %H:%M:%S")
    tend   = dt.datetime.strptime(conf["tend"], "%Y-%m-%d %H:%M:%S")

    # Read target polygons (catchments) --> already projected
    dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
    cats, keys = tl.util.make_ids_unique(cats, keys)
    cats = tl.util.reduce_multipolygons(cats)
    xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
    lonlat = wradlib.georef.reproject(xy, 
                                      projection_source=conf["trg_proj"],
                                      projection_target=wradlib.georef.get_default_projection())

    # Only once
    grdverts = wradlib.zonalstats.grid_centers_to_vertices(lon,lat,0.25,0.25)
    xygrdverts = wradlib.georef.reproject(grdverts, projection_target=conf["trg_proj"])
    
    try:
        # Create instance of type GridCellsToPoly from zonal data file
        obj = wradlib.zonalstats.GridCellsToPoly('imd4_zonal_poly_cart')
    except Exception, e:
        print(e)
        # Create instance of type ZonalDataPoly from source grid and
        # catchment array
        zd = wradlib.zonalstats.ZonalDataPoly(xygrdverts, cats, srs=conf["trg_proj"])
        # dump to file
        zd.dump_vector('imd4_zonal_poly_cart')
        # Create instance of type GridPointsToPoly from zonal data object
        obj = wradlib.zonalstats.GridCellsToPoly(zd)

    
#    # Create instances of type GridCellsToPoly (one instance for each target polygon)
#    obj = wradlib.zonalstats.GridCellsToPoly(xygrdverts, cats)
    
    # Deal with those target catchments that do not intersect with IMD4 cells
    src = np.vstack( (imdx, imdy) ).transpose()
    trg = xy[obj.check_empty()]
    ip = wradlib.ipol.Nearest(src, trg)


    tl.echse.write_echse_coords_file(conf["f_coords"], xy[:,0], xy[:,1], lonlat[:,0], lonlat[:,1], id=np.array(keys))

    
    for i, dtime in enumerate(dtimes):
        if (dtime < tstart) or (dtime > tend):
            continue        
        print dtime.isoformat(" ")
        # Compute areal mean precipitation
        trgvals = obj.mean(var[i])
        # Fill gaps for catchments that do not intersect
        trgvals[obj.check_empty()] = ip(var[i])        
        tl.echse.write_echse_data(conf["f_data"], dtime+conf["timeshift"], trgvals, id=np.array(keys), header=header)
        if header:
            header = False


def imd4file_to_echse2(conf, srcfile=None, tofile=None):
    """Read IMD4 gridded precipitation dataset and interpolate to subcatchments.
    
    Uses ZonalStats.
    """
    # Read IMD4 data and coordinates from file
    if srcfile is not None:
        fromfile = np.loadtxt(srcfile, dtype="string", delimiter=",")
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
    imdx, imdy = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])

    header = True
    
    tstart = dt.datetime.strptime(conf["tstart"], "%Y-%m-%d %H:%M:%S")
    tend   = dt.datetime.strptime(conf["tend"], "%Y-%m-%d %H:%M:%S")

    # Read target polygons (catchments) --> already projected
    dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
    cats, keys = tl.util.make_ids_unique(cats, keys)
    cats = tl.util.reduce_multipolygons(cats)
    xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
#    lonlat = wradlib.georef.reproject(xy, 
#                                      projection_source=conf["trg_proj"],
#                                      projection_target=wradlib.georef.get_default_projection())

    # Only once
    grdverts = wradlib.zonalstats.grid_centers_to_vertices(lon,lat,conf["gridresolution"],conf["gridresolution"])
    xygrdverts = wradlib.georef.reproject(grdverts, projection_target=conf["trg_proj"])
    
    try:
        # Create instance of type GridCellsToPoly from zonal data file
        obj = wradlib.zonalstats.GridCellsToPoly(conf["zonaldatacart"])
    except Exception, e:
        print(e)
        # Create instance of type ZonalDataPoly from source grid and
        # catchment array
        zd = wradlib.zonalstats.ZonalDataPoly(xygrdverts, cats, srs=conf["trg_proj"])
        # dump to file
        zd.dump_vector(conf["zonaldatacart"])
        # Create instance of type GridPointsToPoly from zonal data object
        obj = wradlib.zonalstats.GridCellsToPoly(zd)

    
#    # Create instances of type GridCellsToPoly (one instance for each target polygon)
#    obj = wradlib.zonalstats.GridCellsToPoly(xygrdverts, cats)
    
    # Deal with those target catchments that do not intersect with IMD4 cells
    src = np.vstack( (imdx, imdy) ).transpose()
    trg = xy[obj.check_empty()]
    ip = wradlib.ipol.Nearest(src, trg)

#    tl.echse.write_echse_coords_file(conf["f_coords"], xy[:,0], xy[:,1], lonlat[:,0], lonlat[:,1], id=np.array(keys))
    
    for i, dtime in enumerate(dtimes):
        if (dtime < tstart) or (dtime > tend):
            continue        
        #print dtime.isoformat(" ")
        # Compute areal mean precipitation
        trgvals = obj.mean(var[i])
        # Fill gaps for catchments that do not intersect
        trgvals[obj.check_empty()] = ip(var[i])        
        tl.echse.write_echse_data(tofile, dtime+conf["timeshift"], trgvals, id=np.array(keys), header=header)
        if header:
            header = False


def xds_to_echse2(conf):
    """Read XDS precipiation dataset and interpolate to subcatchments.
    
    Iterates over all matching files and writes into the same directory tree.
    """
    print "Processing..."
    for root, dirnames, filenames in os.walk(conf["xdsroot"]):
        for filename in fnmatch.filter(filenames, conf["fpattern"]):
            # Source file
            fmatch = os.path.join(root, filename)
            # Target file
            fechse, ext = os.path.splitext(fmatch)
            fechse += conf["echseext"]
            print "%s: %s --> %s" % (root, os.path.basename(fmatch), os.path.basename(fechse))
            imd4file_to_echse2(conf, srcfile=fmatch, tofile=fechse)
    

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
        ax, coll = tl.vis.plot_cats(polys2, var[i], ax=ax, bbox=conf["bbox"], cmap=mycmap, 
                  norm=mynorm, edgecolors='none')
        cb = plt.colorbar(coll, ax=ax, ticks=conf["levels"], shrink=0.6)
        cb.ax.tick_params(labelsize="small")
        cb.set_label("(mm)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        tl.vis.plot_trmm_grid_lines(ax)
        plt.text(conf["bbox"]["left"]+0.25, conf["bbox"]["top"]-0.25, 
                 "%s\n%s to\n%s" % (conf["figtxtbody"], 
                                    (dtime-dt.timedelta(seconds=conf["interval"])).isoformat(" "),
                                     dtime.isoformat(" ") ),
                 color="red", fontsize="small", verticalalignment="top")
        plt.tight_layout()
        plt.savefig(figpath)
        plt.close()
    plt.interactive(True)


if __name__ == '__main__':
    
    pass