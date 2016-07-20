# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         products
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
import gc
import sys


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
    x = np.arange(0, 360, 0.25) - (180.-0.25/2)
    X, Y = np.meshgrid(x,y)
    
#    R = np.roll(R,720,axis=1)    

    return X, Y, R
    

def read_imerg(f, var = "Grid/precipitationCal", meshgrid=True):
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
    if meshgrid:
        x, y = np.meshgrid(x,y)
#    X = X - 180.    
    var = data[var]["data"].T
#    var = np.roll(var,len(x)/2,axis=1)
    
    return x, y, var


def read_imerg_custom_h5(f, meshgrid=True):
    """Read our own custom daily product.
    """
    data, meta = wradlib.io.from_hdf5(f)
    y = meta["y"]
    x = meta["x"]
    if meshgrid:
        x, y = np.meshgrid(x,y)
    
    return x, y, data

    



if __name__ == '__main__':
    
    X, Y, R = read_imerg(r"X:\gpm\imerg\2014\06\09\3B-HHR.MS.MRG.3IMERG.20140609-S000000-E002959.0000.V03D.HDF5")

    
