# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         to_echse
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

import datetime as dt

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result




if __name__ == '__main__':

    
    # Interpolate rain gauges
    conf_gages = dict(
        tstart = "2001-01-01 03:30:00",
        tend = "2010-12-31 03:30:00",
        interval = 86400,
        timeshift = -dt.timedelta(seconds=4.5*3600),

        f_coords      = "P:/progress/mahanadi/_qpe/gages_idw/locs.txt",
        f_data        = "P:/progress/mahanadi/_qpe/gages_idw/data.txt",
        savefigs      = "P:/progress/mahanadi/_qpe/gages_idw/maps",
        
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
        
        f_coords      = "P:/progress/mahanadi/_qpe/3B42_daily_research/coords.txt",
        f_data        = "P:/progress/mahanadi/_qpe/3B42_daily_research/data.txt",
        savefigs      = "P:/progress/mahanadi/_qpe/3B42_daily_research/maps",
        
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
        tend = "2014-04-30 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=24*3600),

        srcdir        = "X:/trmm/3B42RT_daily_V07",
        
        f_coords      = "P:/progress/mahanadi/_qpe/3B42_daily_realtime/coords.txt",
        f_data        = "P:/progress/mahanadi/_qpe/3B42_daily_realtime/data.txt",
        savefigs      = "P:/progress/mahanadi/_qpe/3B42_daily_realtime/maps",
        
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
        tstart = "1901-01-01 00:00:00",
        tend   = "2013-12-31 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=0*3600),
        f_imd4        = "P:/progress/mahanadi/Precipitation/gridded_IMD/P.csv",
        
        f_coords      = "P:/progress/mahanadi/_qpe/imd4/locs.txt",
        f_data        = "P:/progress/mahanadi/_qpe/imd4/data.txt",
        savefigs      = "P:/progress/mahanadi/_qpe/imd4/maps",

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

    conf_xds = dict(
        gridresolution=0.5,
		#xdsroot= "P:/progress/mahanadi/xds_20160510/ei",
		xdsroot= "P:/progress/mahanadi/EC.csv",
        echseext = ".echse",
        zonaldatacart = "imd4_zonal_poly_cart",
		fpattern = "pf.*.pl.t.csv",
        #fpattern = "T.*-*-*.csv",
		#fpattern = "T.csv",
		#fpattern = "P.csv",
        tstart = "2000-01-01 00:00:00",
        tend   = "2021-01-01 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=8*3600),

#        f_coords      = "P:/progress/mahanadi/_qpe/imd4/locs.txt",
#        f_data        = "P:/progress/mahanadi/_qpe/imd4/data.txt",
#        savefigs      = "P:/progress/mahanadi/_qpe/imd4/maps",

#        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",
        figtxtbody    = "IMD4 Subsample\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )


    # IMD4 Gridded rainfall Dataset
    conf_imerg = dict(
        trmm_product = "imerg/final/daily",
        tstart = "2014-04-01 00:00:00",
        tend = "2015-04-30 00:00:00",
        interval = 86400,
        timeshift = dt.timedelta(seconds=24*3600),

        srcdir        = "X:/gpm/imerg/final/daily",
        
        f_coords      = "P:/progress/mahanadi/_qpe/imerg/final/daily/coords.txt",
        f_data        = "P:/progress/mahanadi/_qpe/imerg/final/daily/data.txt",
        savefigs      = "P:/progress/mahanadi/_qpe/imerg/final/daily/maps",
        
        f_cats        = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp    = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",
        figtxtbody    = "IMERG Final\nCatchment Rainfall",
        bbox = {"left":80., "bottom":19., "top":24., "right":86.},
        trg_proj = wradlib.georef.epsg_to_osr(32645),
        map_proj = wradlib.georef.get_default_projection(), 
        newcoords = True,
        levels = [0,1,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300], 
        verbose = False
    )    
    
    
    
    # Compute subcatchment rainfall
#    tl.wflows.gages_to_echse2(conf_gages)
#    tl.wflows.imd4_to_echse2(conf_imd4)
#    tl.wflows.trmm_to_echse3(conf_trmm)
#    tl.wflows.trmm_to_echse3(conf_trmm_rt)
#    tl.wflows.trmm_to_echse3(conf_imerg)
    tl.wflows.xds_to_echse2(conf_xds)
    
    # Make maps for all time steps based on config
#    tl.wflows.maps_from_echse(conf_gages)
#    tl.wflows.maps_from_echse(conf_imd4)
#    tl.wflows.maps_from_echse(conf_trmm)
#    tl.wflows.maps_from_echse(conf_trmm_rt)
#    tl.wflows.maps_from_echse(conf_imerg)
        
#    tstart = dt.datetime.strptime("2001-04-01 00:00:00", "%Y-%m-%d %H:%M:%S")
#    tend = dt.datetime.strptime("2010-12-30 00:00:00", "%Y-%m-%d %H:%M:%S")
#    
#    dtimes_imd4, _, imd4 = tl.echse.read_echse_data_file(conf_imd4["f_data"])
#    dtimes_trmm, _, trmm = tl.echse.read_echse_data_file(conf_trmm["f_data"])
#    dtimes_trmmrt, _, trmmrt = tl.echse.read_echse_data_file(conf_trmm_rt["f_data"])
#    dtimes_gage, _, gage = tl.echse.read_echse_data_file(conf_gages["f_data"])
#    
#    ix_imd4   = (dtimes_imd4   >= tstart) & (dtimes_imd4   <tend)
#    ix_trmm   = (dtimes_trmm   >= tstart) & (dtimes_trmm   <tend)
#    ix_trmmrt = (dtimes_trmmrt >= tstart) & (dtimes_trmmrt <tend)
#    ix_gage   = (dtimes_gage   >= tstart) & (dtimes_gage   <tend)
    




#    plt.plot(dtimes_trmmrt[ix_trmmrt], medfilt( np.mean(trmmrt[ix_trmmrt,:],axis=1), 31 ), color="green",  label="TRMM RT"  )    
    
#    dtimes, lon, lat, var = read_imd4()
#    nlon = len(np.unique(lon))
#    nlat = len(np.unique(lat))
#    plt.scatter(x=lon, y=lat, c=var[100])
#    plt.pcolormesh(lon.reshape(nlon, nlat), lat.reshape(nlon, nlat), var[100].reshape(nlon, nlat))

    
    
#    X2, Y2, R2 = read_trmm_bin(r"X:\trmm\3B42RT_daily_V07\2013\3B42RT_daily.2013.08.06.bin")
#
#    fpath = r"X:\gpm\imerg\2014\3B-HHR.MS.MRG.3IMERG.20141010-S000000-E002959.0000.V03D.HDF5"
#    X2, Y2, R2 = read_imerg(fpath)
#    
#    shapes = ["E:/data/shape_collection/TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp"]
#    
#    # Get original TRMM data
#    X1, Y1, R1 = read_trmm(r"X:\trmm\3B42_daily_V07\3B42_daily.2013.08.06.7.nc")
#    
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
#
#    
#    plot_map(X2, Y2, np.ma.masked_less(R2, 0), ax=ax, shapes=shapes, cmap="spectral", vmax=260)
#    plt.title("TRMM 3B42RT")
#    
#    import wradlib
#    data = wradlib.io.read_generic_netcdf(r"X:\trmm\3B42_daily_V07\3B42_daily.2004.01.31.7.nc")
#    print data["CoreMetadata.0"]
    
    
#    conf = conf_imd4
#    
#    fromfile = np.loadtxt(conf["f_imd4"], dtype="string", delimiter=",")
#    var = fromfile[1:,3:].astype("f4")
#    years = fromfile[1:,0].astype("i4")
#    months = fromfile[1:,1].astype("i4")
#    days = fromfile[1:,2].astype("i4")
#    dtimes = np.array( [dt.datetime(years[i], months[i], days[i]) for i in range(len(years))] )
#    lon = np.array([])
#    lat = np.array([])
#    for item in fromfile[0,3:]:
#        lon_, lat_ = item.split("_")
#        lon = np.append(lon, float(lon_) )
#        lat = np.append(lat, float(lat_) )
#    imdx, imdy = wradlib.georef.reproject(lon, lat, projection_target=conf["trg_proj"])
#    # Only once
#    grdverts = wradlib.zonalstats.grid_centers_to_vertices(lon,lat,0.25,0.25)
#    xygrdverts = wradlib.georef.reproject(grdverts, projection_target=conf["trg_proj"])


    # Get TRMM data
#    X, Y, R = tl.products.read_imerg_daily_h5(r"X:\gpm\imerg\final\2014\05\01\imerg_final_daily_20140501.h5")
#    plt.pcolormesh(X, Y, np.ma.masked_invalid(R), cmap=plt.cm.spectral)
#    mask, shape = tl.util.mask_from_bbox(X, Y, conf["bbox"])
#
#    # Only once
#    grdverts = wradlib.zonalstats.grid_centers_to_vertices(X[mask],Y[mask],0.25,0.25)
#    xygrdverts = wradlib.georef.reproject(grdverts, projection_target=conf["trg_proj"])


    # Read target polygons (catchments) --> already projected
#    dataset, inLayer = wradlib.io.open_shape(conf["f_cats_shp"])
#    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
#    cats, keys = tl.util.make_ids_unique(cats, keys)
#    cats = tl.util.reduce_multipolygons(cats)
#    xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
#    lonlat = wradlib.georef.reproject(xy, 
#                                      projection_source=conf["trg_proj"],
#                                      projection_target=wradlib.georef.get_default_projection())
#                                      
#
#    bbox = wradlib.zonalstats.get_bbox(xygrdverts[...,0], xygrdverts[...,1])
#    buffer=20000
#    from matplotlib.collections import PatchCollection
#    import matplotlib.patches as patches
#    fig = plt.figure(figsize=(12,10))
#    ax = fig.add_subplot(111, aspect="equal")
#    grd_patches = [patches.Polygon(item, True) for item in xygrdverts ]
#    p = PatchCollection(grd_patches, facecolor="None", edgecolor="black")
#    ax.add_collection(p)
#    # Target polygon patches
#    trg_patches = [patches.Polygon(item, True) for item in cats ]
#    p = PatchCollection(trg_patches, facecolor="None", edgecolor="green", linewidth=1)
#    ax.add_collection(p)
##    error_patches = [patches.Polygon(item, True) for item in cats[np.array([737,738])] ]
##    d = PatchCollection(error_patches, facecolor="None", edgecolor="red", linewidth=1)
##    ax.add_collection(d)
#
#    plt.xlim(bbox["left"]-buffer, bbox["right"]+buffer)
#    plt.ylim(bbox["bottom"]-buffer, bbox["top"]+buffer)
#    plt.draw()
#    plt.xlabel("UTM Easting (m)")
#    plt.ylabel("UTM Northing (m)")
#    plt.tight_layout()
#    plt.savefig("P:/progress/mahanadi/_qpe/imd4/imd4_coverage1.png")

    

    



            
            

        


    


    



