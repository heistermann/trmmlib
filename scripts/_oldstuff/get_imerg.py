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

from __future__ import print_function
import trmmlib as tl
import wradlib

import numpy as np
import pylab as plt
from matplotlib.colors import from_levels_and_colors
import datetime as dt

import os
from ftplib import FTP
import glob
import gc



def get_file(fpath, localdir, ftp):
    """
    """
    fname = os.path.basename(fpath)
    localfile = open(os.path.join(localdir, fname), 'wb')
    try:
        ftp.retrbinary('RETR ' + fpath, localfile.write)
        localfile.close()
    except:
        localfile.close()
        raise


def ignore(text, to_ignore):
    """
    """
    for item in to_ignore:
        if item in text:
            return True
    return False

def match_day(text, dtime):
    """
    """
    if dtime.strftime(".%Y%m%d-S") in text:
        return True
    else:
        return False

        

if __name__ == '__main__':

    # FINAL parameters
    host = "arthurhou.pps.eosdis.nasa.gov"
    user = "heisterm@uni-potsdam.de"
    passw = "heisterm@uni-potsdam.de"
    localdir = "x:/gpm/imerg/final/daily/%s"
    remotedir = "gpmdata/%s/imerg"
    #start = "2014-04-01 00:00:00"
    start = "2015-05-25 00:00:00"
    end = "2015-08-31 00:00:00" 
    to_ignore = ["3B-MO.MS.MRG.3IMERG"]
    pattern = "3B-HHR.MS.MRG.3IMERG*%s*"
    trgname = "imerg_final_daily_%s.h5"
    remove_HHR = True
    min_files = 48
    remotedatestr="%Y/%m/%d"
    localdatestr1="%Y/%m/%d"
    localdatestr2=""
    replacetarget=False
    
    # LATE parameters
#    host = "jsimpson.pps.eosdis.nasa.gov"
#    user = "heisterm@uni-potsdam.de"
#    passw = "heisterm@uni-potsdam.de"
#    localdir = "x:/gpm/imerg/late/daily/%s"
#    remotedir = "NRTPUB/imerg/late/%s"
#    start = "2015-04-30 00:00:00"
#    end = "2014-04-30 00:00:00" 
#    to_ignore = ["3B-MO.MS.MRG.3IMERG"]
#    pattern = "3B-HHR-L.MS.MRG.3IMERG*%s*"
#    trgname = "imerg_late_daily_%s.h5"
#    remove_HHR = True
#    min_files = 48
#    remotedatestr="%Y%m"
#    localdatestr1="%Y/%m/%d"
#    localdatestr2=""

        
    # iterate over datetimes
    dtimes = wradlib.util.from_to(start, end, 24*3600)
    for dtime in dtimes:
        remotesubdir = dtime.strftime(remotedatestr)
        localsubdir = dtime.strftime(localdatestr1)
        # Check if product already exists
        print("Processing %s..." % remotesubdir, end="")
        trgpath = os.path.join(localdir % dtime.strftime(localdatestr2), trgname % dtime.strftime("%Y%m%d"))
        if os.path.exists(trgpath) and (not replacetarget):
            print("already exists - continue.")
            continue
        # Renew ftp connection for every day of data        
        ftp = FTP(host)
        ftp.login(user=user, passwd = passw)
        
        try:
            remotefiles = ftp.nlst(remotedir % remotesubdir)
        except:
            continue
        try:
            os.makedirs(localdir % localsubdir)
        except:
            print("...could not create directory...continue anyway...", end="")            
        
        # Downloading files
        print("downloading if required", end="")   
        for remotefile in remotefiles:
            if (not ignore(remotefile, to_ignore)) and match_day(remotefile, dtime):
                # Only download if file does not eist locally, yet
                checklocal = os.path.join(localdir % localsubdir, os.path.basename(remotefile))
                if not os.path.exists(checklocal):
                    try:
                        get_file(remotefile, localdir % localsubdir, ftp)
                    except:
                        # in case of failure: set up new connection and try once more
                        ftp.quit()
                        ftp = FTP(host)
                        ftp.login(user=user, passwd = passw)
                        get_file(remotefile, localdir % localsubdir, ftp)                                        
                print(".", end="")
        
        # Process files
        os.chdir(localdir % localsubdir)
        accum = None
        files = glob.glob(pattern % dtime.strftime("%Y%m%d"))
        if len(files) >= min_files:
            print("accumulating", end="")
            ok = True
            for f in files:
                print(".", end="")
                try:
                    X, Y, R = tl.products.read_imerg(f, meshgrid=False)
                except IOError:
                    ok = False
                    break
                R[R<0] = np.nan
                if accum==None:
                    accum = R * 0.5 # conversion from rate (mm/h) to mm 
                else:
                    accum = accum + R * 0.5
            # Only after the accumulation is complete (or has finally failed)
            if remove_HHR:
                for f in files:
                    os.remove(f)
            if ok:
                wradlib.io.to_hdf5(trgpath, accum, metadata={"x":X, "y":Y, "meshgrid":False})
                print("done.")
            else:
                print("at least one corrupt file - accumulation terminated.")
        else:
            print("found only %d files - accumulation terminated." % len(files))
        
        # Close ftp connection                         
        ftp.quit()
    
    # Test plot
#    trgpath = os.path.join(localdir % dtime.strftime(localdatestr2), trgname % dtime.strftime("%Y%m%d"))
#    X, Y, R = tl.products.read_imerg_daily_h5(trgpath)
#    colors = plt.cm.spectral(np.linspace(0,1,len(range(0,200,10))))    
#    mycmap, mynorm = from_levels_and_colors(range(0,200,10), colors, extend="max")
#
#    fig = plt.figure(figsize=(14,8))
#    ax=fig.add_subplot(111, aspect="equal")
#    cp=plt.pcolormesh(X, Y, np.ma.masked_invalid(R), cmap=mycmap, norm=mynorm)
#    plt.colorbar(cp)
#    plt.draw()
#    gc.collect()
#    del fig, ax, cp
#    gc.collect()
    
    # read geotiff
#    from osgeo import gdal
#    f = r"E:\src\python\trmmlib\scripts\3B-HHR-L.MS.MRG.3IMERG.20150430-S233000-E235959.1410.V03E.1day.tif"
#    ds = gdal.Open(f)
#    fromtiff = np.array(ds.GetRasterBand(1).ReadAsArray())
#    fig = plt.figure(figsize=(14,8))
#    ax=fig.add_subplot(111, aspect="equal")
#    cp=plt.pcolormesh(X, Y, fromtiff/10., cmap=mycmap, norm=mynorm)
#    plt.colorbar(cp)
#    plt.draw()



