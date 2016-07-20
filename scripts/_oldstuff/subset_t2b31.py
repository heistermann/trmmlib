# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 09:26:47 2015

@author: heistermann
"""

import pandas as pd
import pylab as plt
import numpy as np
import wradlib
import trmmlib as tl
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

if __name__ == '__main__':
    
#    # SET CORRECT WORKING DIRECTORY BEFORE
#    T2B31V7_fn = 'T2B31V7_NN_short_Long0080_0086_Lat019_024.h5'
#    
#    # open HDF store with pandas
#    df = pd.read_hdf(T2B31V7_fn, 'T2B31', mode='r')
#    df = df.set_index("NN_DateTime")
#    
#    # Select a time period for subsetting (here: July 2002)
#    dfsub = df["2002-07-01":"2002-08-01"]
#    
#    # store as hdf5
#    dfsub.to_hdf('t2b31_200207.h5','dfsub')

#    dataset, inLayer = wradlib.io.open_shape("P:/progress/mahanadi/gis/shapefiles/mahanadi_shapes/sub_catchments_DN.shp")
#    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='catAttr_DN')
#    cats, keys = tl.util.make_ids_unique(cats, keys)
#    cats = tl.util.reduce_multipolygons(cats)
#    xy = np.array([np.array(wradlib.zonalstats.get_centroid(cat)) for cat in cats])
#    cats = [wradlib.georef.reproject(cat, projection_source=wradlib.georef.epsg_to_osr(32645), projection_target=wradlib.georef.get_default_projection()) for cat in cats]

    dataset, inLayer = wradlib.io.open_shape("P:/progress/ECHSE_projects/general_shpfiles/country_borders/PHL_Dissolve.shp")
    cats, keys = wradlib.georef.get_shape_coordinates(inLayer, key='Id')


#    dfall = pd.read_hdf('X:/trmm/mahanadi_trmm_from_bodo/t2b31_200207.h5', 'dfsub', mode='r')
    dfall = pd.read_hdf('X:/trmm/philippines_trmm_from_bodo/t2b31_2012_060708.h5', 'dfsub', mode='r')
    
    df = dfall["2012-08-06":"2012-08-11"]
    orbits = np.unique(df.NN_orbit)
    orbits.sort()
    fig = plt.figure(figsize=(16,12))
    for i, orbit in enumerate(orbits[0:8]):
        ax = fig.add_subplot(2,4,i+1, aspect="equal")
        plt.scatter(df[df.NN_orbit==orbit].NN_Longitude, df[df.NN_orbit==orbit].NN_Latitude, 
                    s=15, c=df[df.NN_orbit==orbit].NN_rrSurf, cmap=plt.cm.spectral, edgecolor="None", 
                    marker="s", vmax=30.)
        plt.xlim(df.NN_Longitude.min(), df.NN_Longitude.max())
        plt.ylim(df.NN_Latitude.min(), df.NN_Latitude.max())
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
#        plt.title("%s\n%s" % (df[df.NN_orbit==orbit].index.min().isoformat(" "),df[df.NN_orbit==orbit].index.max().isoformat(" ")))
        plt.grid()
        trg_patches = [patches.Polygon(item, True) for item in cats ]
        p = PatchCollection(trg_patches, facecolor="None", edgecolor="grey", linewidth=1)
        ax.add_collection(p)
        txt = "Orbit: %s\nIn   : %s\nOut: %s" % (orbit, df[df.NN_orbit==orbit].index.min().isoformat(" "),df[df.NN_orbit==orbit].index.max().isoformat(" "))
        plt.text(df.NN_Longitude.min()+0.25, df.NN_Latitude.min()+0.25, txt, 
                 color="black", horizontalalignment='left', verticalalignment="bottom", 
                 fontsize=13, bbox=dict(facecolor='white', edgecolor="None", alpha=0.5))

    plt.tight_layout()
    plt.savefig(r"P:\progress\philippines\trmm_3b31.png", dpi=300)
    
    
    
    

