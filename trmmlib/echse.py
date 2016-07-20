# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         echse
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
    """Writes ECHSE data table (datetimes and variables).
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
        


def read_echse_data_file(f):
    """Read data from standard ECHSE data file.
    
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


    


if __name__ == '__main__':
    
    pass