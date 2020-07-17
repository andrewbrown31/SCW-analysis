#Calculate a grid of ERA5 Average Reccurence Intervals, based on a given interval.
#Options for which years base the calculation on, as well as whether to use EV theory or an empirical CDF

import datetime as dt
from lmoments3 import distr
import argparse
import geopandas
import pandas as pd
from rasterio import features
from affine import Affine
import glob
from dask.diagnostics import ProgressBar
import xarray as xr
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF 
from scipy.stats import genextreme
from scipy.stats import weibull_min as wb
from tqdm import tqdm
import netCDF4 as nc

def date_seq(times,delta_type,delta):
	start_time = times[0]
	end_time = times[1]
	current_time = times[0]
	date_list = [current_time]
	while (current_time < end_time):
		if delta_type == "hours":
			current_time = current_time + dt.timedelta(hours = delta)	
		date_list.append(current_time)
	return date_list

def file_dates(files, query):

	is_in = []
	for i in np.arange(len(files)):
		t = dt.datetime.strptime(files[i].split("/")[11][:-1], "%Y%m%dT%H%M")
		t_list = date_seq([t + dt.timedelta(hours=6), t + dt.timedelta(days=10)], "hours", 6) 
		if any(np.in1d(query, t_list)):
			is_in.append(True)
		else:
			is_in.append(False)
	return is_in


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def get_lsm():
	#Load the ERA-Interim land-sea mask (land = 1)
	lsm_file = nc.Dataset("/g/data/ub4/era5/netcdf/static_era5.nc")
	lsm = np.squeeze(lsm_file.variables["lsm"][:])
	lsm_lon = np.squeeze(lsm_file.variables["longitude"][:])
	lsm_lat = np.squeeze(lsm_file.variables["latitude"][:])
	lsm_file.close()
	return [lsm,lsm_lon,lsm_lat]

def get_mask(lon,lat,thresh=0.5):
	#Return lsm for a given domain (with lats=lat and lons=lon)
	lsm,nat_lon,nat_lat = get_lsm()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[-1]) & (nat_lat <= lat[0]))[0]
	lsm_domain = lsm[(lat_ind[0]):(lat_ind[-1]+1),(lon_ind[0]):(lon_ind[-1]+1)]
	lsm_domain = np.where(lsm_domain > thresh, 1, 0)

	return lsm_domain

def get_era5_lsm(regrid=False):

	#load the era5 lsm, subset to Australia, and coarsen as in regrid_era5(). Consider the coarse
	# lsm to be land if greater than 50% of the points are land

    
	f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_19790101_19790131.nc")
	lsm = get_mask(f.lon.values,f.lat.values)
	if regrid:
		lsm_coarse = xr.DataArray(lsm, dims=("lat","lon")).coarsen({"lat":6,"lon":6},\
			    boundary="trim", side="left").sum()
		return xr.where(lsm_coarse>=18, 1, 0).values
	else:
		return lsm


def drop_duplicates(da):

	#Drop time duplicates

	a, ind = np.unique(da.time.values, return_index=True)
	return(da[ind])

def get_barra_mask(lon,lat):

	#Take 1d lat lon data from an already-loaded BARRA-R domain (e.g. sa_small or aus) and return a land-sea mask
	nat_lon,nat_lat = get_barra_lat_lon()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[0]) & (nat_lat <= lat[-1]))[0]
	lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
	lsm_domain = lsm[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
	
	return lsm_domain

def get_barra_topog(lon,lat):

	#Take 1d lat lon data from an already-loaded BARRA-R domain (e.g. sa_small or aus) and return a land-sea mask
	nat_lon,nat_lat = get_barra_lat_lon()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[0]) & (nat_lat <= lat[-1]))[0]
	topog = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/topog-an-slv-PT0H-BARRA_R-v1.nc").variables["topog"][:]
	topog_domain = topog[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
	
	return topog_domain

def get_barra_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2012"+"/"+"12"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2012"+"12"+"01"+"T"+"00"+"*.nc")[0])
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def get_shapes():
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]
	return [f, f2, shapes]

def load_era5(y1, y2, shapes, nrm=[0,1,2,3], djf=False):

	'''Load ERA5 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Resample to daily maximum, and return a numpy array, loaded into memory.
	   Returns "da" (daily maximum gridded wind gusts), "da_dmax" (domain-maximum daily maximum gusts), 
		"da_annmax" (annual maximum gridded wind gusts), "da_dmax_annmax" (domain and annual maximum gust)
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ub4/era5/netcdf/surface/fg10/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[8]) for f in files], years)],\
		combine='by_coords')
	ds = ds.sel({"latitude":slice(lat_bounds[1], lat_bounds[0]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).\
		    isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = ds["fg10"].resample({"time":"1D"}).max("time")
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")

	if djf:
		djf_str="djf"
	else:
		djf_str="annual"
	da_annmax.to_netcdf("/g/data/eg3/ab4502/gev/era5_annmax_"+djf_str+"_"+str(y1)+"_"+str(y2)+".nc")

	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	lsm = get_era5_lsm()

	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def load_barra(y1, y2, shapes, nrm=[0,1,2,3], djf=False):

	'''Load BARRA 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Resample to daily maximum, and return a numpy array, loaded into memory. Also return a daily
		time series of the maximum within a domain, specified by nrm super-clusters 0-3. If a list
		of super-cluster lists is given, compute for the combined area of each super-cluster list.
	   Returns "da" (daily maximum gridded wind gusts), "da_dmax" (domain-maximum daily maximum gusts), 
		"da_annmax" (annual maximum gridded wind gusts), "da_dmax_annmax" (domain and annual maximum gust)
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ma05/BARRA_R/v1/forecast/spec/max_wndgust10m/*/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[9]) for f in files], years)], concat_dim="time", combine='by_coords')
	ds = ds.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).\
		    isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = drop_duplicates(ds["max_wndgust10m"]).resample({"time":"1D"}).max("time")
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")
	if djf:
		djf_str="djf"
	else:
		djf_str="annual"
	da_annmax.to_netcdf("/g/data/eg3/ab4502/gev/barra_annmax_"+djf_str+"_"+str(y1)+"_"+str(y2)+".nc")

	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	lsm = get_barra_mask(ds.longitude.values, ds.latitude.values)

	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def load_barpa(y1, y2, shapes, forcing_mdl="ACCESS1-0", nrm=[0,1,2,3], djf=False):

	'''Load BARPA 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Forcing model either "ACCESS1-0" or "erai"
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	query_dates = date_seq([dt.datetime(y1,1,1,12), dt.datetime(y2,12,31,12)], "hours", 24)
	files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/*/"+forcing_mdl+"/r*/*/*/pp0/max_wndgust10m*"))
	files = files[file_dates(files, query_dates)]
	ds = xr.open_mfdataset(files, concat_dim="time", combine='nested')
	ds = ds.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).\
		    isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = drop_duplicates(ds["max_wndgust10m"])
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")
	if djf:
		djf_str="djf"
	else:
		djf_str="annual"
	da_annmax.to_netcdf("/g/data/eg3/ab4502/gev/barpa_"+forcing_mdl+"_annmax_"+djf_str+"_"+str(y1)+"_"+str(y2)+".nc")

	#Grab the values from the daily max array    
	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	#Get the time of the annual maximum wind gust at each spatial point, and save
	g = da_da.groupby("time.year")
	annmax_times = []
	for n, gr in g:
		annmax_times.append((gr["time"].values[gr.argmax("time")]))
	annmax_times_ds = xr.Dataset(data_vars = {"annmax_times": (("time","lat","lon"), np.stack(annmax_times))}, coords={"time":ann_times, "lat":ds.latitude.values, "lon":ds.longitude.values})
	annmax_times_ds.to_netcdf("/g/data/eg3/ab4502/gev/barpa_"+forcing_mdl+"_annmax_times_"+djf_str+"_"+str(y1)+"_"+str(y2)+".nc")
        
	#Get NRM shapefile details as a grid        
	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

    #Load the land-sea mask
	lsm = xr.open_dataset("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc")["lnd_mask"]
	lsm = lsm.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).values

    #For each NRM region, get the domain-maximum annual maximum
	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def load_barpa_sy(y1, y2, shapes, nrm=[0,1,2,3], djf=False):

	'''Load BARPA 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Forcing model either "ACCESS1-0" or "erai"
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	query_dates = date_seq([dt.datetime(y1,1,1,12), dt.datetime(y2,12,31,12)], "hours", 24)
	files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-SY_1p5km/cmip/ACCESS1-0/r1i1p1/*/*/pp0/max_wndgust10m*"))
	files = files[file_dates(files, query_dates)]
	ds = xr.open_mfdataset(files, concat_dim="time", combine='nested')
	ds = ds.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).\
		    isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = drop_duplicates(ds["max_wndgust10m"])
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")

	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	lsm = xr.open_dataset("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc")["lnd_mask"]
	lsm = lsm.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])}).values

	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]


def load_barra_sy(y1, y2, shapes, nrm=[0,1,2,3], djf=False):

	'''Same as above but for the CRM domain over Sydney
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ma05/BARRA_SY/v1/forecast/spec/max_wndgust10m/*/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[9]) for f in files], years)], concat_dim="time", combine='by_coords')
	ds = ds.isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = drop_duplicates(ds["max_wndgust10m"]).resample({"time":"1D"}).max("time")
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")
	if djf:
		djf_str="djf"
	else:
		djf_str="annual"
	da_annmax.to_netcdf("/g/data/eg3/ab4502/gev/barra_sy_annmax_"+djf_str+"_"+str(y1)+"_"+str(y2)+".nc")

	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	lsm = ((xr.open_dataset("/g/data/ma05/BARRA_SY/v1/static/topog-fc-slv-PT0H-BARRA_SY-v1.nc")["topog"].values) > 0) * 1

	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def load_barra_ad(y1, y2, shapes, nrm=[0,1,2,3], djf=False):

	'''Same as above but for the CRM domain over Adelaide
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/max_wndgust10m/*/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[9]) for f in files], years)], concat_dim="time", combine='by_coords')
	ds = ds.isel({"time":(ds["time.year"]>=y1) & (ds["time.year"]<=y2)})
	da = drop_duplicates(ds["max_wndgust10m"]).resample({"time":"1D"}).max("time")
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
	times = da.time.values
	da = fix_wg_spikes(da.values)
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	#Turn the numpy array, da, into a dataarray, so that it can be resampled to annual maximum frequency.
	da_da = xr.Dataset(data_vars = {"da": (("time","lat","lon"), da)}, coords={"time":times, "lat":ds.latitude.values, "lon":ds.longitude.values})["da"]
	da_annmax = da_da.resample({"time":"1Y"}).max("time")

	ann_times = da_annmax.time.values
	da_annmax = da_annmax.values

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	lsm = ((xr.open_dataset("/g/data/ma05/BARRA_AD/v1/static/topog-fc-slv-PT0H-BARRA_AD-v1.nc")["topog"].values) > 0) * 1

	da_dmax = []
	da_dmax_annmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where((~mask) | (lsm==0), da ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), da_annmax ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax, lat, lon, outname):

	'''
	For a given return period (ari), return the corresponding wind gust speed at each grid point.
	    These gusts represent, on average, the speed which is exceeded every "ari" years.
	    Probabilities are determined by fitting an empirical CDF function to the data
	'''

	#For 5, 10, 20 year retuen periods, return a grid of values based on an ECDF function.
	ari_out5 = np.zeros((da.shape[1], da.shape[2]))
	ari_out10 = np.zeros((da.shape[1], da.shape[2]))
	ari_out20 = np.zeros((da.shape[1], da.shape[2]))
	ari_annmax_out5 = np.zeros((da.shape[1], da.shape[2]))
	ari_annmax_out10 = np.zeros((da.shape[1], da.shape[2]))
	ari_annmax_out20 = np.zeros((da.shape[1], da.shape[2]))
	ari_max = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			cdf = ECDF(da[:,i,j])
			cdf_annmax = ECDF(da_annmax[:,i,j])
			p = cdf.y
			p_annmax = cdf_annmax.y
			T = (1/(1-p)) / 365.25
			T_annmax = (1/(1-p_annmax))
			ari_out5[i,j] = np.interp(5, T, cdf.x)
			ari_out10[i,j] = np.interp(10, T, cdf.x)
			ari_out20[i,j] = np.interp(20, T, cdf.x)
			ari_annmax_out5[i,j] = np.interp(5, T_annmax, cdf_annmax.x)
			ari_annmax_out10[i,j] = np.interp(10, T_annmax, cdf_annmax.x)
			ari_annmax_out20[i,j] = np.interp(20, T_annmax, cdf_annmax.x)
			ari_max[i,j] = da[:,i,j].max()

	#Save the gridded ARI values
	xr.Dataset(data_vars = {"ari5": (("lat","lon"), ari_out5), "ari10":(("lat","lon"), ari_out10),\
		    "ari20":(("lat","lon"), ari_out20), "max":(("lat","lon"), ari_max)},\
		    coords={"lat":lat, "lon":lon}).\
	    to_netcdf("/g/data/eg3/ab4502/gev/"+outname+".nc")
	xr.Dataset(data_vars = {"ari5": (("lat","lon"), ari_annmax_out5), "ari10":(("lat","lon"), ari_annmax_out10),\
		    "ari20":(("lat","lon"), ari_annmax_out20)},\
		    coords={"lat":lat, "lon":lon}).\
	    to_netcdf("/g/data/eg3/ab4502/gev/"+outname+"_annmax.nc")

	#For the domain maximum, save the CDF as a .csv file
	for n in np.arange(4):
		cdf = ECDF(da_dmax[n])
		cdf_annmax = ECDF(da_dmax_annmax[n])
		p = cdf.y
		p_annmax = cdf_annmax.y
		T = 1 / (1-p) / 365.25
		T_annmax = 1 / (1-p_annmax)
		pd.DataFrame({"T":T, "wind_speed":cdf.x}).\
			to_csv("/g/data/eg3/ab4502/gev/"+outname+"_nrm"+str(n)+".csv")
		pd.DataFrame({"T":T_annmax, "wind_speed":cdf_annmax.x}).\
			to_csv("/g/data/eg3/ab4502/gev/"+outname+"_nrm"+str(n)+"_annmax.csv")

def ari_gev(da, da_dmax_annmax, lat, lon, outname):

	'''
	As in ari_cdf(), but probabilites are determined by fitting a GEV and Weibull EV
		function to all data points.
	'''
	
	#For each spatial point, fit a GEV distribution to the annual maximum
	c_out = np.zeros((da.shape[1], da.shape[2]))
	loc_out = np.zeros((da.shape[1], da.shape[2]))
	scale_out = np.zeros((da.shape[1], da.shape[2]))
	cl_out = np.zeros((da.shape[1], da.shape[2]))
	locl_out = np.zeros((da.shape[1], da.shape[2]))
	scalel_out = np.zeros((da.shape[1], da.shape[2]))
	lmom_fail = np.zeros((da.shape[1], da.shape[2]))
	cw_out = np.zeros((da.shape[1], da.shape[2]))
	locw_out = np.zeros((da.shape[1], da.shape[2]))
	scalew_out = np.zeros((da.shape[1], da.shape[2]))
	cw_lmom_out = np.zeros((da.shape[1], da.shape[2]))
	locw_lmom_out = np.zeros((da.shape[1], da.shape[2]))
	scalew_lmom_out = np.zeros((da.shape[1], da.shape[2]))
	lmomw_fail = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			#Fit a GEV
			c, loc, scale = genextreme.fit(da[:,i,j])
			c_out[i, j] = c
			loc_out[i, j] = loc
			scale_out[i, j] = scale
			#Fit a Weibell EV
			cw, locw, scalew = wb.fit(da[:,i,j])
			cw_out[i, j] = cw
			locw_out[i, j] = locw
			scalew_out[i, j] = scalew
			#Try and fit a GEV using l-moments, or else just use the MLE fit (scipy)
			try:
				lmom = distr.gev.lmom_fit(da[:,i,j])
				cl_out[i, j] = lmom["c"]
				locl_out[i, j] = lmom["loc"]
				scalel_out[i, j] = lmom["scale"]
			except:
				cl_out[i,j] = c
				locl_out[i,j] = loc
				scalel_out[i,j] = scale
				lmom_fail[i,j] = 1
			#Try and fit a Weibell EV using l-moments, or else just use the MLE fit (scipy)
			try:
				lmomw = distr.wei.lmom_fit(da[:,i,j])
				cw_lmom_out[i, j] = lmomw["c"]
				locw_lmom_out[i, j] = lmomw["loc"]
				scalew_lmom_out[i, j] = lmomw["scale"]
			except:
				cw_lmom_out[i,j] = cw
				locw_lmom_out[i,j] = locw
				scalew_lmom_out[i,j] = scalew
				lmomw_fail[i,j] = 1

	#For each NRM region (considering only land points in ERA5), fit a GEV to the domain maximum
	c_nrm = []
	loc_nrm = []
	scale_nrm = []
	cl_nrm = []
	locl_nrm = []
	scalel_nrm = []
	cw_nrm = []
	locw_nrm = []
	scalew_nrm = []
	cw_lmom_nrm = []
	locw_lmom_nrm = []
	scalew_lmom_nrm = []
	for n in np.arange(4):
		c, loc, scale = genextreme.fit(da_dmax_annmax[n])
		c_nrm.append(c)
		loc_nrm.append(loc)
		scale_nrm.append(scale)
		cw, locw, scalew = wb.fit(da_dmax_annmax[n])
		cw_nrm.append(cw)
		locw_nrm.append(locw)
		scalew_nrm.append(scalew)
		try:
			lmom = distr.gev.lmom_fit(da_dmax_annmax[n])
			cl_nrm.append(lmom["c"])
			locl_nrm.append(lmom["loc"])
			scalel_nrm.append(lmom["scale"])
		except:
			cl_nrm.append(np.nan)
			locl_nrm.append(np.nan)
			scalel_nrm.append(np.nan)
		try:
			lmomw = distr.wei.lmom_fit(da_dmax_annmax[n])
			cw_lmom_nrm.append(lmomw["c"])
			locw_lmom_nrm.append(lmomw["loc"])
			scalew_lmom_nrm.append(lmomw["scale"])
		except:
			cw_lmom_nrm.append(np.nan)
			locw_lmom_nrm.append(np.nan)
			scalew_lmom_nrm.append(np.nan)

	#Save the grid of GEV fitted parameters to netcdf. Save the domain-max (NRM region) parameters as attrs
	xr.Dataset(data_vars = {"c": (("lat","lon"), c_out), "loc":(("lat","lon"), loc_out),\
		    "scale":(("lat","lon"), scale_out), "c_lmom":(("lat","lon"), cl_out), \
		    "loc_lmom":(("lat","lon"), locl_out), "scale_lmom":(("lat","lon"), scalel_out),\
		    "cw":(("lat","lon"), cw_out), "locw":(("lat","lon"), locw_out),\
		    "scalew":(("lat","lon"), scalew_out), "cw_lmom":(("lat","lon"), cw_lmom_out),\
		    "locw_lmom":(("lat","lon"), locw_lmom_out), "scalew_lmom":(("lat","lon"), scalew_lmom_out),\
		    "lmom_fail":(("lat","lon"), lmom_fail), "lmomw_fail":(("lat","lon"), lmomw_fail)},\
		    coords={"lat":lat, "lon":lon},\
		    attrs={"nrm_c":c_nrm, "nrm_loc":loc_nrm, "nrm_scale":scale_nrm, "c_lmom_nrm":cl_nrm,\
			"loc_lmom_nrm":locl_nrm, "scale_lmom_nrm":scalel_nrm, "nrm_cw":cw_nrm, "nrm_locw":locw_nrm,\
			"nrm_scalew":scalew_nrm, "cw_lmom_nrm":cw_lmom_nrm, "locw_lmom_nrm":locw_lmom_nrm,\
			"scalew_lmom_nrm":scalew_lmom_nrm}).to_netcdf("/g/data/eg3/ab4502/gev/"+outname+".nc")

def fix_wg_spikes(da_vals):

	'''
	Take a lat, lon, time numpy array of daily maximum max_wndgust10m, and identify/smooth wind gust spikes
	Identify spikes by considering adjacent points. Spikes are where there is at least one adjacent point with 
	    a gust less than 50% of the potential spike. Potential spikes are gusts above 40 m/s.
	Replace spikes using the mean of adjacent points.
	'''
	
	ind = np.where(da_vals >= 25)
	for i in tqdm(np.arange(len(ind[0]))):
		pot_spike = da_vals[ind[0][i], ind[1][i], ind[2][i]]
		adj_gusts = []
		for ii in [-1, 1]:
			for jj in [-1, 1]:
				try:
					adj_gusts.append( da_vals[ind[0][i], ind[1][i]+ii, ind[2][i]+jj])
				except:
					pass
		if (np.array(adj_gusts) < (0.5*pot_spike)).any():
			pot_spike = np.median(adj_gusts)
		da_vals[ind[0][i], ind[1][i], ind[2][i]] = pot_spike
	return da_vals

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="From reanalysis model data, fit GEV and ECDF to 10 m wind gusts, saving the output")
	parser.add_argument("-m", help="Models", required=True, nargs="+")
	parser.add_argument("-y1", help="Start year (yyyy)", default=1990, type=int)                           
	parser.add_argument("-y2", help="End year (yyyy)", default=2018, type=int)
	args = parser.parse_args()
	y1 = args.y1
	y2 = args.y2
	models = args.m
	nrm = [0,1,2,3]       #0: Northern Australia;  1: Rangelands ; 2: Eastern Australia 3: Southern Australia
	test=False
	print(models, y1, y2, "TEST: "+str(test))

	#Get NRM shapes with geopandas
	f, f2, shapes = get_shapes()

	if "ERA5" in models:
		#Compute GEV/CDF for ERA5
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times  = load_era5(y1,y2,shapes,nrm,djf=False)
		if test:
			lons = lons[0:10,0:10]; lats=lats[0:10,0:10]
			da = da[:,0:10,0:10]; da_annmax = da_annmax[:,0:10,0:10]
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "era5_cdf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "era5_gev_"+str(y1)+"_"+str(y2))
	
		#Compute GEV/CDF for ERA5 for summer only
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times  = load_era5(y1,y2,shapes,nrm,djf=True)
		if test:
			lons = lons[0:10,0:10]; lats=lats[0:10,0:10]
			da = da[:,0:10,0:10]; da_annmax = da_annmax[:,0:10,0:10]
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:], "era5_cdf_djf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "era5_gev_djf_"+str(y1)+"_"+str(y2))

	if "BARRA" in models:
		#Compute GEV/CDF for BARRA
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra(y1,y2,shapes,nrm,djf=False)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_cdf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_gev_"+str(y1)+"_"+str(y2))

		#Compute GEV/CDF for BARRA for summer only
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra(y1,y2,shapes,nrm,djf=True)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_cdf_djf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_gev_djf_"+str(y1)+"_"+str(y2))

	if "BARPA" in models:
		#Compute GEV/CDF for BARRA
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barpa(y1,y2,shapes,nrm=nrm,djf=False,forcing_mdl="ACCESS1-0")
		if test:
			lons = lons[0:10,0:10]; lats=lats[0:10,0:10]
			da = da[:,0:10,0:10]; da_annmax = da_annmax[:,0:10,0:10]
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barpa_access_cdf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barpa_access_gev_"+str(y1)+"_"+str(y2))

		#Compute GEV/CDF for BARRA for summer only
		#lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barpa(y1,y2,shapes,nrm=nrm,djf=True,forcing_mdl="ACCESS1-0")
		#ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barpa_access_cdf_djf_"+str(y1)+"_"+str(y2))
		#ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barpa_access_gev_djf_"+str(y1)+"_"+str(y2))

	if "BARRA-SY" in models:
	
		#Compute GEV/CDF for BARRA-SY
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra_sy(y1,y2,shapes,nrm,djf=False)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_sy_cdf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_sy_gev_"+str(y1)+"_"+str(y2))
		
		#Compute GEV/CDF for BARRA-SY for summer only
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra_sy(y1,y2,shapes,nrm,djf=True)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_sy_cdf_djf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_sy_gev_djf_"+str(y1)+"_"+str(y2))
	
	if "BARRA-AD" in models:
	
		#Compute GEV/CDF for BARRA-AD
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra_ad(y1,y2,shapes,nrm,djf=False)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_ad_cdf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_ad_gev_"+str(y1)+"_"+str(y2))
		
		#Compute GEV/CDF for BARRA-AD for summer only
		lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra_ad(y1,y2,shapes,nrm,djf=True)
		ari_out = ari_cdf(da, da_dmax, da_annmax, da_dmax_annmax,lats[:,0], lons[0,:],"barra_ad_cdf_djf_"+str(y1)+"_"+str(y2))
		ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_ad_gev_djf_"+str(y1)+"_"+str(y2))
