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
	
    # GPM IMERG product
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

	# Products from XDS (same format as IMD4)
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

    # To create a map for a specific ECHSE data file
    conf_mapping = dict(
        tstart="2000-07-15 08:00:00",
        tend="2000-07-17 08:00:00",
        interval=86400,
        f_coords = "P:/progress/mahanadi/_qpe/imd4/locs.txt",
        f_data = "P:/progress/mahanadi/xds_20160510/ei/pdd/P.echse",
        savefigs = "P:/progress/mahanadi/xds_20160510/ei/pdd/maps",
        f_cats = "P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/catAttr_DN.txt",
        f_cats_shp="P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp",
        figtxtbody="EI Subsample\nCatchment Rainfall",
        bbox={"left": 80., "bottom": 19., "top": 24., "right": 86.},
        trg_proj=wradlib.georef.epsg_to_osr(32645),
        map_proj=wradlib.georef.get_default_projection(),
        levels=[0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300],
        verbose=False
    )

    # Compute subcatchment rainfall
#    tl.wflows.gages_to_echse2(conf_gages)
#    tl.wflows.imd4_to_echse2(conf_imd4)
#    tl.wflows.trmm_to_echse3(conf_trmm)
#    tl.wflows.trmm_to_echse3(conf_trmm_rt)
#    tl.wflows.trmm_to_echse3(conf_imerg)
    tl.wflows.xds_to_echse2(conf_xds)
    
    # Make maps for all time steps based on config
#    tl.wflows.maps_from_echse(conf_mapping)
