#Calculate a grid of ERA5 Average Reccurence Intervals, based on a given interval.
#Options for which years base the calculation on, as well as whether to use EV theory or an empirical CDF

from scw_compare import get_era5_lsm
from percent_mean_change import transform_from_latlon
import matplotlib.pyplot as plt
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
from tqdm import tqdm
import netCDF4 as nc

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

def get_barra_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2012"+"/"+"12"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2012"+"12"+"01"+"T"+"00"+"*.nc")[0])
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def plot_domain_max_ari(da_dmax, y1, y2, label, legend=False, color="", ls="-"):

	cdf = ECDF(da_dmax.flatten())
	p = cdf.y
	T = 1 / (1-p) / (365.24)
	if color == "":
		plt.plot(cdf.x, T, color, linestyle=ls, label=label)
	else:
		plt.plot(cdf.x, T, label=label, linestyle=ls)
	plt.yscale("log")
	if legend:
		plt.legend()

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
		da_annmax = ds.sel({"time":np.in1d(ds["time.month"], [12,1,2])})["fg10"].resample({"time":"1Y"}).max("time")
	else:
		da_annmax = ds["fg10"].resample({"time":"1Y"}).max("time")
	times = da.time.values
	da = da.values
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

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
		np.ma.masked_where((~mask) | (lsm==0), fix_wg_spikes(da) ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), fix_wg_spikes(da_annmax) ).max(axis=(1,2)).data )

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
	ds = drop_duplicates(ds)
	pd.DataFrame({"times":ds.time.values}).to_csv("/g/data/eg3/ab4502/times.csv")
	da = ds["max_wndgust10m"].resample({"time":"1D"}).max("time")
	if djf:
		da = da.sel({"time":np.in1d(da["time.month"], [12,1,2])})
		da_annmax = ds.sel({"time":np.in1d(ds["time.month"], [12,1,2])})["max_wndgust10m"].resample({"time":"1Y"}).max("time")
	else:
		da_annmax = ds["max_wndgust10m"].resample({"time":"1Y"}).max("time")
	times = da.time.values
	da = da.values
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

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
		np.ma.masked_where((~mask) | (lsm==0), fix_wg_spikes(da) ).max(axis=(1,2)).data )

	    mask_annmax = np.zeros(da_annmax.shape, dtype=bool)
	    mask_annmax[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax_annmax.append(\
		np.ma.masked_where((~mask_annmax) | (lsm==0), fix_wg_spikes(da_annmax) ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax, da_annmax, da_dmax_annmax, ann_times]

def ari_cdf(da, lat, lon, outname):

	'''
	For a given return period (ari), return the corresponding wind gust speed at each grid point.
	    These gusts represent, on average, the speed which is exceeded every "ari" years.
	    Probabilities are determined by fitting an empirical CDF function to the data
	'''

	#For 5, 10, 20 year retuen periods, return a grid of values based on an ECDF function.
	ari_out5 = np.zeros((da.shape[1], da.shape[2]))
	ari_out10 = np.zeros((da.shape[1], da.shape[2]))
	ari_out20 = np.zeros((da.shape[1], da.shape[2]))
	ari_max = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			cdf = ECDF(da[:,i,j])
			p = cdf.y
			T = (1/(1-p)) / 365.25
			ari_out5[i,j] = np.interp(5, T, cdf.x)
			ari_out10[i,j] = np.interp(10, T, cdf.x)
			ari_out20[i,j] = np.interp(20, T, cdf.x)
			ari_max[i,j] = da[:,i,j].max()

	#Save the gridded ARI values
	xr.Dataset(data_vars = {"ari5": (("lat","lon"), ari_out5), "ari10":(("lat","lon"), ari_out10),\
		    "ari20":(("lat","lon"), ari_out20), "max":(("lat","lon"), ari_max)},\
		    coords={"lat":lat, "lon":lon}).\
	    to_netcdf("/g/data/eg3/ab4502/gev/"+outname+".nc")

	#For the domain maximum, save the CDF as a .csv file
	for n in np.arange(4):
		cdf = ECDF(da_dmax[n])
		p = cdf.y
		T = 1 / (1-p) / (365.24)
		x = np.interp(5, T, cdf.x)
		pd.DataFrame({"T":T, "wind_speed":cdf.x}).to_csv("/g/data/eg3/ab4502/gev/"+outname+"_nrm"+str(n)+".csv")

def ari_gev(da, da_dmax_annmax, lat, lon, outname):

	'''
	As in ari_cdf(), but probabilites are determined by fitting a GEV function to all
		data points.
	'''
	
	#days = len(t)
	#T = ari
	#p = 1 - (1 / T)
    
	#For each spatial point, fit a GEV distribution to the annual maximum
	c_out = np.zeros((da.shape[1], da.shape[2]))
	loc_out = np.zeros((da.shape[1], da.shape[2]))
	scale_out = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			c, loc, scale = genextreme.fit(da[:,i,j])
			c_out[i, j] = c
			loc_out[i, j] = loc
			scale_out[i, j] = scale
			#ari_out[i,j] = genextreme.isf(q=1-p, c=c, loc=loc, scale=scale)

	#For each NRM region (considering only land points in ERA5), fit a GEV to the domain maximum
	c_nrm = []
	loc_nrm = []
	scale_nrm = []
	for n in np.arange(4):
		c, loc, scale = genextreme.fit(da_dmax_annmax[n])
		c_nrm.append(c)
		loc_nrm.append(loc)
		scale_nrm.append(scale)
		#x.append([genextreme.isf(q=1-p_arr, c=c, loc=loc, scale=scale)])
		#x_obs.append([da_dmax_annmax[n]])

	#Save the grid of GEV fitted parameters to netcdf. Save the domain-max (NRM region) parameters as attrs
	xr.Dataset(data_vars = {"c": (("lat","lon"), c_out), "loc":(("lat","lon"), loc_out),\
		    "scale":(("lat","lon"), scale_out)}, coords={"lat":lat, "lon":lon},\
		    attrs={"nrm_c":c_nrm, "nrm_loc":loc_nrm, "nrm_scale":scale_nrm}).\
	    to_netcdf("/g/data/eg3/ab4502/gev/"+outname+".nc")

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

	y1=1990
	y2=2018
	nrm = [0,1,2,3]       #0: Northern Australia;  1: Rangelands ; 2: Eastern Australia 3: Southern Australia
	djf = False

	#Get NRM shapes with geopandas
	f, f2, shapes = get_shapes()

	#TODO Perform analysis for all days, as well as DJF. Will require converting "t" array
	#lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times  = load_era5(y1,y2,shapes,nrm,djf=djf)
	#ari_out = ari_cdf(da,lats[:,0], lons[0,:], "era5_cdf_"+str(y1)+"_"+str(y2))
	#ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "era5_gev_"+str(y1)+"_"+str(y2))

	lons, lats, t, da, da_dmax, da_annmax, da_dmax_annmax, ann_times = load_barra(y1,y2,shapes,nrm,djf=djf)
	ari_out = ari_cdf(da,lats[:,0], lons[0,:],"barra_cdf_"+str(y1)+"_"+str(y2))
	ari_out = ari_gev(da_annmax, da_dmax_annmax, lats[:,0], lons[0,:], "barra_gev_"+str(y1)+"_"+str(y2))
	
	#TODO move plotting to a different function, based on loading netcdf output
	#ed0, ed1, ed2, ed3 = da_dmax_era5
	#bd0, bd1, bd2, bd3 = da_dmax_barra
	#plot_domain_max_ari(ed0, y1, y2, "Northern Aus.", legend=False, color="tab:blue")
	#plot_domain_max_ari(ed1, y1, y2, "Rangelands", legend=False, color="tab:red")
	#plot_domain_max_ari(ed2, y1, y2, "Southern Aus.", legend=False, color="tab:orange")
	#plot_domain_max_ari(ed3, y1, y2, "Eastern Aus.", legend=True, color="tab:green")
	#plot_domain_max_ari(bd0, y1, y2, "Northern Aus.", legend=False, color="tab:blue", ls="--")
	#plot_domain_max_ari(bd1, y1, y2, "Rangelands", legend=False, color="tab:red", ls="--")
	#plot_domain_max_ari(bd2, y1, y2, "Southern Aus.", legend=False, color="tab:orange", ls="--")
	#plot_domain_max_ari(bd3, y1, y2, "Eastern Aus.", legend=False, color="tab:green", ls="--")
