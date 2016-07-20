# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:35:05 2015

@author: irene
"""

import wradlib as wrl
import numpy as np
import os
import pylab as pl
import netCDF4 as nc
import datetime as dt

def read_JBirds(data_path,fname):
    f = open(os.path.join(data_path,fname), 'rb')
    r = np.fromfile(f, dtype='B')
    f.close()

    # Raw Data Header
    #   ignore first 96 members
    raw_data_header = r[96:352]
    # Sweep Data
    data = np.reshape(r[352:],(360,352))

    # Sweep Header
    sweep_header = data[:,:32]
    sweep_data = data[:,32:] * 0.3125 # 0.3125dBZ per member value

    # collect attributes from header
    attrs = {}

    # location
    loc = raw_data_header[1]
    if loc == int('40',16):
        attrs['Location'] = 'Virac'
        attrs['sitecoords'] = (13.62, 124.33, 0)
        attrs['Latitude'] = 13.62
        attrs['Longitude'] = 124.33
    elif loc == int('41',16):
        attrs['Location'] = 'Aparri'
        attrs['sitecoords'] = (18.35, 121.64, 0)
        attrs['Latitude'] = 18.35
        attrs['Longitude'] = 121.64
    elif loc == int('42',16):
        attrs['Location'] = 'Guiuan'
    else:
        attrs['Location'] = 'other'
    # Data type
    dattype = raw_data_header[3]
    if dattype == int('75',16):
        attrs['Data Type'] = 'Doppler Velocity'
    elif dattype == int('76',16):
        attrs['Data Type'] = 'Spectrum Width'
    elif dattype == int('F1',16):
        attrs['Data Type'] = 'Reflectivity'
    else:
        attrs['Data Type'] = 'other'
    # Date Time
    datestr = ''.join([chr(f) for f in raw_data_header[8:24]])
    attrs['time'] = dt.datetime.strptime(datestr, '%Y.%m.%d.%H.%M')
    # Response Status
    if raw_data_header[29] == 0:
        attrs['Response Status'] = 'Normal'
    else:
        attrs['Response Status'] = 'Error'
    # Step no.
    attrs['Step number'] = raw_data_header[42]
    ## Elevation
    #attrs['Elevation'] = raw_data_header[44]*360./(2**16)
    # Z-Relation-B
    attrs['Z-Relation B'] = raw_data_header[52]/100.
    # Z-Relation-Beta
    attrs['Z-Relation Beta'] = raw_data_header[54]/100.

    # get attributes from filename
    elev = int(fname[::-1][18:22][::-1])
    if fname[::-1][16:18][::-1] == 'P1':
        attrs['Elevation'] = elev / 10.
    elif fname[::-1][16:18][::-1] == 'P2':
        attrs['Elevation'] = elev / 100.
    attrs['Sweep'] = int(fname[::-1][13:15][::-1])

    attrs['MissingData'] = 255 * 0.3125
    if int(attrs['Sweep']) > 4:
        attrs['r'] = np.linspace(625,200000,320)
    else:
        attrs['r'] = np.linspace(1375,440000,320)
    attrs['az'] = np.linspace(1,360,360)

    data = np.where(sweep_data==attrs['MissingData'], np.nan, sweep_data)
    
    return data, attrs

def read_IRIS_netcdf(filename, variable='Z', enforce_equidist=False):
    """Data reader for netCDF files exported by the IRIS radar software

    The netcdf files produced by the IRIS software usually contains two
    variables: reflectivity (Z) and total power (T). The default variable read
    is reflectivity.

    Parameters
    ----------
    filename : path of the netCDF file
    enforce_equidist : boolean
        Set True if the values of the azimuth angles should be forced to be equidistant
        default value is False

    Returns
    -------
    output : numpy array of image data (dBZ), dictionary of attributes

    """
    # read the data from file
    dset = nc.Dataset(filename)
    data = dset.variables[variable][:]
    # Check azimuth angles and rotate image
    az = dset.variables['radialAzim'][:]
    # These are the indices of the minimum and maximum azimuth angle
    ix_minaz = np.argmin(az)
    ix_maxaz = np.argmax(az)
##    if enforce_equidist:
##        az = np.linspace(np.round(az[ix_minaz],2), np.round(az[ix_maxaz],2), len(az))
##    else:
##        az = np.roll(az, -ix_minaz)
##    # rotate accordingly
##    data = np.roll(data, -ix_minaz, axis=0)
    data = np.where(data==dset.variables[variable].getncattr('_FillValue'), np.nan, data)
    # Ranges
    binwidth = dset.variables['gateSize'][:]
    r = np.arange(binwidth, (dset.variables['Z'].shape[-1]*binwidth) + binwidth, binwidth)
    # collect attributes
    attrs =  {}
    for attrname in dset.ncattrs():
        attrs[attrname] = dset.getncattr(attrname)

    # Set additional metadata attributes
    attrs['az'] = az
    attrs['r']  = r
    attrs['ElevationAngle'] = dset.variables['elevationAngle'][:]
    attrs['firstGateRange'] = dset.variables['firstGateRange'][:]
    attrs['gateSize'] = dset.variables['gateSize'][:]
    attrs['nyquist'] = dset.variables['nyquist'][:]
    attrs['unambigRange'] = dset.variables['unambigRange'][:]
    attrs['calibConst'] = dset.variables['calibConst'][:]
    attrs['Longitude'] = dset.variables['siteLat'][:]
    attrs['Latitude'] = dset.variables['siteLon'][:]
    attrs['sitecoords'] = (dset.variables['siteLat'][:], dset.variables['siteLon'][:], dset.variables['siteAlt'][:])
    attrs['Time'] = dt.datetime.utcfromtimestamp(dset.variables['esStartTime'][:])
    attrs['max_range'] = data.shape[1] * binwidth
    dset.close()

    return data, attrs

def read_data(radarsystem, data_path, fname, t_scan_res):
    """
    Reads raw radar data.
    Parameters
    ----------
    fname : string
        filename of data
    """
    fn = os.path.join(data_path, fname)
    # read polar data
    if radarsystem == 'EDGE':
        dset, attrs = wrl.io.read_EDGE_netcdf(fn)
    elif radarsystem == 'JBirds':
        dset, attrs = read_JBirds(data_path,fname)
    elif radarsystem == 'IRIS':
        dset, attrs = read_IRIS_netcdf(fn)

    # replace missing values with 0
    try:
        data = np.ma.masked_values(dset, attrs[u'MissingData'])
        data_dBZ = np.where(np.isnan(data),0,data)
    except:
        data_dBZ = dset

    # compute rainfall height
    data_Z = wrl.trafo.idecibel(data_dBZ)
    data_R_rate = wrl.zr.z2r(data_Z)
    data_R_height = wrl.trafo.r2depth(data_R_rate, t_scan_res)

    attrs['r'] = attrs['r']#[:400]

    #return data_R_height[:,:range_lim], attrs
    #return data_dBZ[:,:range_lim], attrs
    return data_dBZ, attrs

def bbox(*args):
    """Get bounding box from a set of radar bin coordinates
    """
    x = np.array([])
    y = np.array([])
    for arg in args:
        x = np.append(x, arg[:,0])
        y = np.append(y, arg[:,1])
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    return xmin, xmax, ymin, ymax
    
def grid_radar_data(path_data, file_name, t_scan_res, radartype):
    data, attrs = read_data(radartype, path_data, file_name, t_scan_res)
    az = np.arange(0,360)
    proj_ph = wrl.georef.epsg_to_osr(32651)
    radar_sitecoords = (attrs['Longitude'], attrs['Latitude'])
    radar_cent_lon, radar_cent_lat = wrl.georef.polar2centroids(attrs['r'], az, radar_sitecoords)
    radar_x, radar_y = wrl.georef.reproject(radar_cent_lon, radar_cent_lat, projection_target=proj_ph)
    radar_coord = np.array([radar_x.ravel(),radar_y.ravel()]).transpose()
    return radar_coord, data, attrs

def grid_radar_accum(path_file, attrs):
    data, a = wrl.io.from_hdf5(path_file)
#    az = np.arange(0,360)
#    proj_ph = wrl.georef.epsg_to_osr(32651)
#    radar_sitecoords = (attrs['Longitude'], attrs['Latitude'])
#    radar_cent_lon, radar_cent_lat = wrl.georef.polar2centroids(attrs['r'], az, radar_sitecoords)
#    radar_x, radar_y = wrl.georef.reproject(radar_cent_lon, radar_cent_lat, projection_target=proj_ph)
#    #radar_coord = np.array([radar_x.ravel(),radar_y.ravel()]).transpose()
    return data, attrs
    
    

# set path of all data
path_data = r"P:\progress\philippines"
file_SUB = "SUB-20151017-000139-04-ZH.nc"
file_SUB_ext = "SUB-20151017-000037-02-ZH.nc"
file_TAG = "TAG-20151017-000515-04-ZH.nc"
file_TAG_ext = "TAG-20151017-000444-03-ZH.nc"
file_BAL = "Output07BAL151017164516.RAW4G2M_sweep1.nc"
tscanSUB = 300
tscanTAG = 300
tscanBAL = 900


az = np.arange(0,360)


SUB_coord, data_SUB, attrs_SUB = grid_radar_data(path_data, file_name=file_SUB, t_scan_res=tscanSUB, radartype='EDGE')
SUB_ext_coord, data_SUB_ext, attrs_SUB_ext = grid_radar_data(path_data, file_name=file_SUB_ext, t_scan_res=tscanSUB, radartype='EDGE')
TAG_coord, data_TAG, attrs_TAG = grid_radar_data(path_data, file_name=file_TAG, t_scan_res=tscanTAG, radartype='EDGE')
TAG_ext_coord, data_TAG_ext, attrs_TAG_ext = grid_radar_data(path_data, file_name=file_TAG_ext, t_scan_res=tscanTAG, radartype='EDGE')
BAL_coord, data_BAL, attrs_BAL = grid_radar_data(path_data, file_name=file_BAL, t_scan_res=tscanBAL, radartype='IRIS')

# read the accumulation files
data_SUB,a = grid_radar_accum("P:/progress/philippines/composites/SUB_24h.hdf5", attrs_SUB)
data_SUB_ext,a = grid_radar_accum("P:/progress/philippines/composites/SUB_ext_3h.hdf5", attrs_SUB_ext)
data_TAG,a = grid_radar_accum("P:/progress/philippines/composites/TAG_24h.hdf5", attrs_TAG)
data_TAG_ext,a = grid_radar_accum("P:/progress/philippines/composites/TAG_ext_24h.hdf5", attrs_TAG_ext)
data_BAL,a = grid_radar_accum("P:/progress/philippines/composites/BAL_24h.hdf5", attrs_BAL)

# limit the extension to 200km
data_SUB_ext = data_SUB_ext[:,:800]
SUB_xcoord = SUB_ext_coord[:,0].reshape(360,SUB_ext_coord[:,0].shape[0]/360)
SUB_ycoord = SUB_ext_coord[:,1].reshape(360,SUB_ext_coord[:,1].shape[0]/360)
SUB_ext_coord = np.array([SUB_xcoord[:,:800].ravel(),SUB_ycoord[:,:800].ravel()]).transpose()

data_TAG_ext = data_TAG_ext[:,:480]
TAG_xcoord = TAG_ext_coord[:,0].reshape(360,TAG_ext_coord[:,0].shape[0]/360)
TAG_ycoord = TAG_ext_coord[:,1].reshape(360,TAG_ext_coord[:,1].shape[0]/360)
TAG_ext_coord = np.array([TAG_xcoord[:,:480].ravel(),TAG_ycoord[:,:480].ravel()]).transpose()

#----------------> LIMIT SUB_ext_coords to 200km!!!!!!

# adjust radar values
data_TAG = data_TAG*1.4
data_TAG_ext = data_TAG_ext*2.0
data_BAL = data_BAL*1.8

#SUBaccum = SUBaccum.ravel()
#TAGaccum = TAGaccum.ravel()
#BALaccum = BALaccum.ravel()
#APAaccum = APAaccum.ravel()
#VIRaccum = VIRaccum.ravel()

# define target grid for composition
#xmin, xmax, ymin, ymax = bbox(SUB_coord, TAG_coord, BAL_coord, APA_coord, VIR_coord)
xmin, xmax, ymin, ymax = bbox(SUB_coord, TAG_coord, BAL_coord)
x = np.linspace(xmin,xmax+1000.,1000.)
y = np.linspace(ymin,ymax+1000.,1000.)
grid_coords = wrl.util.gridaspoints(y,x)

# create the quality maps
#SUB_mask = np.ones(data_SUB.shape)
#SUB_mask[140:200,:] = np.inf

def qual_pulse(r, binlen, beamwidth, numbeams, cutoffmin, cutoffmax ):
    """
    """
    pulse_volumes = np.tile(wrl.qual.pulse_volume(r, binlen, beamwidth,),numbeams)
    
    qual = 1. - pulse_volumes/pulse_volumes.max()  
    qual = qual  / qual.max()
    return qual

def qual_altitude(r, elev, numbeams, cutoffleft=500., cutoffright=3000., minqual=0.05, maxqual=1.0 ):
    """
    """
    alt = np.tile(wrl.georef.beam_height_n(r, elev),numbeams)
    qual = np.ones(alt.shape)
    qual[alt<=cutoffleft] = maxqual
    qual[alt>cutoffright] = minqual
    ix = (alt>=cutoffleft) & (alt<=cutoffright)
    qual[ix] = maxqual - (alt[ix] - cutoffleft)*(maxqual-minqual)/(cutoffright - cutoffleft)
    return qual


SUB_qual = np.ones(data_SUB.shape)
SUB_qual[140:200,:] = 0
SUB_qual = qual_altitude(attrs_SUB['r'], 1.5, 360) * SUB_qual.ravel()

SUB_ext_qual = np.ones(data_SUB_ext.shape)
SUB_ext_qual[140:200,:] = 0
SUB_ext_qual[:,0:480] = 0
SUB_ext_qual = qual_altitude(attrs_SUB_ext['r'][:800], 1.5, 360) * SUB_ext_qual.ravel()

TAG_qual = np.ones(data_TAG.shape)
TAG_qual *= 0.5
TAG_qual = qual_altitude(attrs_TAG['r'], 1.5, 360) * TAG_qual.ravel()

TAG_ext_qual = np.ones(data_TAG_ext.shape)
#TAG_ext_qual *= 0.8
#TAG_ext_qual[:,0:240] = 0
TAG_ext_qual = qual_altitude(attrs_TAG_ext['r'][:480], 1.5, 360) * TAG_ext_qual.ravel()

BAL_qual = np.ones(data_BAL.shape)
BAL_qual[0:30,:] = 0
BAL_qual[165:360,:] = 0
#BAL_qual[165:280,:] = 0
BAL_qual = qual_altitude(attrs_BAL['r'], 1.5, 360) * BAL_qual.ravel()
           
#SUB_qual_add = qual_pulse(attrs_SUB['r'], 1000., 1., 360)    
#pulse_volumes_SUB = qual_pulse(attrs_SUB['r'], 1000., 1., 360)
#SUB_qual_pulse = 1./pulse_volumes_SUB 
#SUB_qual_pulse = SUB_qual_pulse  / SUB_qual_pulse.max()

#pulse_volumes_TAG = np.tile(wrl.qual.pulse_volume(attrs_TAG['r'], 1000., 1.),360)
#pulse_volumes_TAG = qual_pulse(attrs_TAG['r'], 1000., 1., 360)

#BAL_mask = np.ones(data_BAL.shape)
#BAL_mask[165:280,:] = np.nan
#pulse_volumes_BAL = np.tile(wrl.qual.pulse_volume(attrs_BAL['r'], 1000., 1.),360) * BAL_mask.ravel()

# grid the data
SUB_quality_gridded = wrl.comp.togrid(SUB_coord, grid_coords, attrs_SUB['r'].max()+500., SUB_coord.mean(axis=0), SUB_qual, wrl.ipol.Nearest)
SUB_gridded = wrl.comp.togrid(SUB_coord, grid_coords, attrs_SUB['r'].max()+500., SUB_coord.mean(axis=0), data_SUB.ravel(), wrl.ipol.Nearest)

SUB_ext_quality_gridded = wrl.comp.togrid(SUB_ext_coord, grid_coords, 200000+500., SUB_ext_coord.mean(axis=0), SUB_ext_qual, wrl.ipol.Nearest)
SUB_ext_gridded = wrl.comp.togrid(SUB_ext_coord, grid_coords, 200000+500., SUB_ext_coord.mean(axis=0), data_SUB_ext.ravel(), wrl.ipol.Nearest)

TAG_quality_gridded = wrl.comp.togrid(TAG_coord, grid_coords, attrs_TAG['r'].max()+500., TAG_coord.mean(axis=0), TAG_qual, wrl.ipol.Nearest)
TAG_gridded = wrl.comp.togrid(TAG_coord, grid_coords, attrs_TAG['r'].max()+500., TAG_coord.mean(axis=0), data_TAG.ravel(), wrl.ipol.Nearest)

TAG_ext_quality_gridded = wrl.comp.togrid(TAG_ext_coord, grid_coords, 240000+500., TAG_ext_coord.mean(axis=0), TAG_ext_qual, wrl.ipol.Nearest)
TAG_ext_gridded = wrl.comp.togrid(TAG_ext_coord, grid_coords, 240000+500., TAG_ext_coord.mean(axis=0), data_TAG_ext.ravel(), wrl.ipol.Nearest)

BAL_quality_gridded = wrl.comp.togrid(BAL_coord, grid_coords, 120000+250., BAL_coord.mean(axis=0), BAL_qual, wrl.ipol.Nearest)
BAL_gridded = wrl.comp.togrid(BAL_coord, grid_coords, 120000+250., BAL_coord.mean(axis=0), data_BAL.ravel(), wrl.ipol.Nearest)

## compose_weighted
#radarinfo = np.array([SUB_gridded, TAG_gridded, TAG_ext_gridded])
#qualityinfo = np.array([SUB_quality_gridded, TAG_quality_gridded, TAG_ext_quality_gridded])
#qualityinfo /= np.nansum(qualityinfo, axis=0)
#
##qualityinfo[mask] = np.nan
#composite = np.nansum(radarinfo*qualityinfo, axis=0)
#tmp = np.vstack([SUB_quality_gridded, TAG_quality_gridded, TAG_ext_quality_gridded])
#mask = np.all( np.isnan(tmp), axis=0)
#composite[mask] = np.nan

# compose_weighted
#radarinfo = np.array([SUB_gridded, TAG_gridded, BAL_gridded, SUB_ext_gridded, TAG_ext_gridded])
#qualityinfo = np.array([SUB_quality_gridded, TAG_quality_gridded, BAL_quality_gridded, SUB_ext_quality_gridded, TAG_ext_quality_gridded])
radarinfo = np.array([SUB_gridded, TAG_ext_gridded])
qualityinfo = np.array([SUB_quality_gridded, TAG_ext_quality_gridded])
qualityinfo /= np.nansum(qualityinfo, axis=0)

#qualityinfo[mask] = np.nan
composite = np.nansum(radarinfo*qualityinfo, axis=0)
#tmp = np.vstack([SUB_quality_gridded, TAG_quality_gridded, BAL_quality_gridded, SUB_ext_quality_gridded, TAG_ext_quality_gridded])
tmp = np.vstack([SUB_quality_gridded, TAG_ext_quality_gridded])
mask = np.all( np.isnan(tmp), axis=0)
composite[mask] = np.nan
#compositeSUB = wrl.comp.compose_weighted([SUB_gridded],[1./(SUB_quality_gridded+0.001)])



# plot the gridded data individually
xmin_, xmax_, ymin_, ymax_= 50000, 450000, 1400000, 1900000 
#fig = pl.figure(figsize=(25,10))
#ax = fig.add_subplot(241, aspect="equal")
#pl.pcolormesh(x, y, np.ma.masked_invalid(SUB_gridded.reshape((len(x),len(y)))), cmap="spectral", vmax=200)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#ax = fig.add_subplot(242, aspect="equal")
#pl.pcolormesh(x, y, np.ma.masked_invalid(TAG_gridded.reshape((len(x),len(y)))), cmap="spectral", vmax=200)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#ax = fig.add_subplot(243, aspect="equal")
#pl.pcolormesh(x, y, np.ma.masked_invalid(BAL_gridded.reshape((len(x),len(y)))), cmap="spectral", vmax=200)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#ax = fig.add_subplot(244, aspect="equal")
#pl.pcolormesh(x, y, np.ma.masked_invalid(TAG_ext_gridded.reshape((len(x),len(y)))), cmap="spectral", vmax=200)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#ax = fig.add_subplot(245, aspect="equal")
#pl.pcolormesh(x, y, np.ma.masked_invalid(SUB_quality_gridded.reshape((len(x),len(y)))), cmap="spectral", vmin=0, vmax=1.)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#pl.subplot(246)
#pl.pcolormesh(x, y, np.ma.masked_invalid(TAG_quality_gridded.reshape((len(x),len(y)))), cmap="spectral", vmin=0, vmax=1.)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#pl.subplot(247)
#pl.pcolormesh(x, y, np.ma.masked_invalid(TAG_quality_gridded.reshape((len(x),len(y)))), cmap="spectral", vmin=0, vmax=1.)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()
#
#pl.subplot(248)
#pl.pcolormesh(x, y, np.ma.masked_invalid(TAG_ext_quality_gridded.reshape((len(x),len(y)))), cmap="spectral", vmin=0, vmax=1.)
#pl.colorbar()
#pl.xlim(xmin_, xmax_)
#pl.ylim(ymin_, ymax_)
#pl.grid()


## create the composite
#tmp = np.vstack([SUB_gridded, TAG_gridded, BAL_gridded])
#mask = np.all( np.isnan(tmp), axis=0)
#composite = wrl.comp.compose_weighted([SUB_gridded, TAG_gridded, BAL_gridded],[1./(SUB_quality_gridded+0.001),1./(TAG_quality_gridded+0.001),1./(BAL_quality_gridded+0.001)])
#composite[mask] = np.nan
#composite = np.ma.masked_invalid(composite)

# compose_weighted
#radarinfo = np.array([SUB_gridded, TAG_gridded])
#qualityinfo = np.array([1./(SUB_quality_gridded+0.001),1./(TAG_quality_gridded+0.001)])
#qualityinfo /= np.nansum(qualityinfo, axis=0)
#composite = np.nansum(radarinfo*qualityinfo, axis=0)

## plot the composite data
#pl.figure(figsize=(20,10))
#
#ax = pl.subplot(121, aspect='equal')
#pm = pl.pcolormesh(x, y, composite.reshape((len(x),len(y))), cmap="spectral")
#pl.grid()
#pl.colorbar(pm)
#
#ax = pl.subplot(122, aspect='equal')
#pm = pl.pcolormesh(x, y, np.ma.masked_invalid(composite_BAL).reshape((len(x),len(y))), cmap="spectral")
#pl.grid()
#pl.colorbar(pm)

# Read shapefile for overlay
PATH_municipalbounds_shapefile = r"E:\data\philippines\shapes\other\Country_boundary\PHL_Dissolve.shp"#r'U:\DATA\gis\shoreline.shp'
dataset, inLayer = wrl.io.open_shape(PATH_municipalbounds_shapefile)
borders, keys = wrl.georef.get_shape_coordinates(inLayer)
for i, border in enumerate(borders):
    borders[i] = wrl.georef.reproject(border, projection_target=wrl.georef.epsg_to_osr(32651))

PATH_pampanga_shapefile = r'P:\progress\ECHSE_projects\echse\echse_proj\pampanga\data\topocatch\out\shpfiles\proj_shp.shp'
dataset_pamp, inLayer_pamp = wrl.io.open_shape(PATH_pampanga_shapefile)
borders_pamp, keys_pamp = wrl.georef.get_shape_coordinates(inLayer_pamp)

# Create rainfall maps
proj_rad = wrl.georef.get_default_projection()
zoombox = [118.5, 124, 12.75, 19.75]
# 19.75, 124 UR
# 12.75, 118.5 LL
fig = pl.figure(figsize=(10,10))
# composite
ax = fig.add_subplot(111, aspect="equal")
comp = np.ma.masked_invalid(composite.reshape((len(x),len(y))))
pm = pl.pcolormesh(x, y, np.ma.masked_less(comp,0), cmap="jet", vmax=300)
wrl.vis.add_lines(ax, borders, color='black', lw=1.)
wrl.vis.add_lines(ax, borders_pamp, color='white', lw=1.)
pl.xlabel("Latitude")
pl.ylabel("Longitude")
pl.title("Composite")
#pl.xlim(zoombox[0:2])
#pl.ylim(zoombox[2:4])

#comp_attrs = {}
#comp_attrs['x'] = x
#comp_attrs['y'] = y
#
wrl.io.to_hdf5("P:/progress/test/pampanga_comp.hdf5", comp, metadata={"x":x, "y":y})
