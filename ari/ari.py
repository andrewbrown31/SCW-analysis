#Calculate a grid of ERA5 Average Reccurence Intervals, based on a given interval.
#Options for which years base the calculation on, as well as whether to use EV theory or an empirical CDF

from percent_mean_change import transform_from_latlon
import matplotlib.pyplot as plt
import geopandas
from rasterio import features
from affine import Affine
import glob
from dask.diagnostics import ProgressBar
import xarray as xr
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF 
from scipy.stats import genextreme
from tqdm import tqdm

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

def load_era5(y1, y2, shapes, nrm=[[0,1,2,3]]):

	'''Load ERA5 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Resample to daily maximum, and return a numpy array, loaded into memory.
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ub4/era5/netcdf/surface/fg10/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[8]) for f in files], years)])
	ds = ds.sel({"latitude":slice(lat_bounds[1], lat_bounds[0]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])})
	da = ds["fg10"].resample({"time":"1D"}).max("time")
	times = da.time.values
	da = da.values
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	da_dmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where(~mask, fix_wg_spikes(da) ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax]

def load_barra(y1, y2, shapes, nrm=[[0,1,2,3]]):

	'''Load BARRA 10-m wind gust data (max in previous hour), for years between y1 and y2.
	    Resample to daily maximum, and return a numpy array, loaded into memory. Also return a daily
		time series of the maximum within a domain, specified by nrm super-clusters 0-3. If a list
		of super-cluster lists is given, compute for the combined area of each super-cluster list.
	'''

	ProgressBar().register()
	lat_bounds = [-44.525, -9.975]
	lon_bounds = [111.975, 156.275]
	years = np.arange(y1, y2+1)
	files = np.sort(glob.glob("/g/data/ma05/BARRA_R/v1/forecast/spec/max_wndgust10m/*/*/*.nc"))
	ds = xr.open_mfdataset(files[np.in1d([int(f.split("/")[9]) for f in files], years)], concat_dim="time")
	ds = ds.sel({"latitude":slice(lat_bounds[0], lat_bounds[1]), \
		    "longitude":slice(lon_bounds[0], lon_bounds[1])})
	da = ds["max_wndgust10m"].resample({"time":"1D"}).max("time")
	times = da.time.values
	da = da.values
	x,y = np.meshgrid(ds.longitude.values, ds.latitude.latitude.values)

	transform = transform_from_latlon(ds.coords['latitude'], ds.coords['longitude'])
	out_shape = (len(ds.coords['latitude']), len(ds.coords['longitude']))
	raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=np.nan, transform=transform,
                                dtype=float)

	da_dmax = []
	for n in nrm:
	    mask = np.zeros(da.shape, dtype=bool)
	    mask[:,:,:] = np.isin(raster[np.newaxis,:,:], n)
	    da_dmax.append(\
		np.ma.masked_where(~mask, fix_wg_spikes(da) ).max(axis=(1,2)).data )

	return [x, y, times, da, da_dmax]

def ari_cdf(da, ari):

	'''
	For a given return period (ari), return the corresponding wind gust speed at each grid point.
	    These gusts represent, on average, the speed which is exceeded every "ari" years.
	    Probabilities are determined by fitting an empirical CDF function to the data
	'''

	ari_out = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			cdf = ECDF(da[:,i,j])
			p = cdf.y
			T = (1/(1-p)) / 365.25
			ari_out[i,j] = np.interp(ari, T, cdf.x)

	return ari_out

def ari_gev(da, t, ari):

	'''
	As in ari_cdf(), but probabilites are determined by fitting a GEV function to all
		data points.
	'''
	
	days = len(t)
	T = 365.24 * ari
	p = 1 - (1 / T)
    
	ari_out = np.zeros((da.shape[1], da.shape[2]))
	for i in tqdm(np.arange(da.shape[1])):
		for j in np.arange(da.shape[2]):
			c, loc, scale = genextreme.fit(da[:,i,j])
			ari_out[i,j] = genextreme.isf(q=1-p, c=c, loc=loc, scale=scale)

	return ari_out

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

	y1=2000
	y2=2005
	y3=0		    #Set to 0 if not required
	y4=0
	ari=10
	method = "cdf"	    #"gev" or "cdf"
	spatial_extent = "nrm"	    #"nrm" or "point"
	nrm = [2,3]	    #0: Northern Australia;  1: Rangelands ; 2: Southern Australia; 3: Eastern Australia

	#Get NRM shapes with geopandas
	f, f2, shapes = get_shapes()

	x, y, t, da, da_dmax_barra = load_barra(y1,y2,shapes,[[2]])
	x, y, t, da, da_dmax_era5 = load_era5(y1,y2,shapes,[[0], [1], [2], [3]])
	ed0, ed1, ed2, ed3 = da_dmax_era5
	bd0, bd1, bd2, bd3 = da_dmax_barra
	plot_domain_max_ari(ed0, y1, y2, "Northern Aus.", legend=False, color="tab:blue")
	plot_domain_max_ari(ed1, y1, y2, "Rangelands", legend=False, color="tab:red")
	plot_domain_max_ari(ed2, y1, y2, "Southern Aus.", legend=False, color="tab:orange")
	plot_domain_max_ari(ed3, y1, y2, "Eastern Aus.", legend=True, color="tab:green")
	plot_domain_max_ari(bd0, y1, y2, "Northern Aus.", legend=False, color="tab:blue", ls="--")
	plot_domain_max_ari(bd1, y1, y2, "Rangelands", legend=False, color="tab:red", ls="--")
	plot_domain_max_ari(bd2, y1, y2, "Southern Aus.", legend=False, color="tab:orange", ls="--")
	plot_domain_max_ari(bd3, y1, y2, "Eastern Aus.", legend=False, color="tab:green", ls="--")

	if spatial_extent == "point":
		#For the entire 3d dataset, fit an empirical CDF at each grid point, and return the ARI
		if method == "cdf":
			assert ( ( (y2+1)-y1)>=ari)
			ari_out = ari_cdf(da, ari)
			if y3 > 0:
				x, y, t2, da2, da_dmax2 = load_era5(y3,y4)
				ari_out2 = ari_cdf(da2, ari) 
		elif method == "gev":
			ari_out = ari_gev(da, t, ari)

	
