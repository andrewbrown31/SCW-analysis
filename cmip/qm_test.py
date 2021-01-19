from dask.diagnostics import ProgressBar
import time
from numba import jit, njit, prange
from statsmodels.distributions.empirical_distribution import ECDF 
import xarray as xr
import numpy as np

@jit
def ecdf_custom(x, unique):

	y = np.linspace(1./len(np.sort(x)), 1, len(np.sort(x)))
	x = np.concatenate((np.array([-np.inf]), np.sort(x)))
	y = np.concatenate((np.array([0]), y))
	if unique:
		x, inds = np.unique(x, return_index=True)
		y = y[inds]
	return x,y

def ecdf_to_unique(ecdf):

	y = ecdf.y
	x = ecdf.x
	x, inds = np.unique(x, return_index=True)
	y = y[inds]
	return x,y

#@jit
def qm_cmip_combined_loop_v1(obs, vals1, vals2, replace_zeros, mask):

		for i in np.arange(vals1.shape[1]):
			for j in np.arange(vals1.shape[2]):
				if mask[i,j] == 1:
				
					#Create the observed CDF
					obs_cdf = ECDF(obs[:,i,j])
					obs_invcdf, obs_p = ecdf_to_unique(obs_cdf)
	
					#Create the model CDF based on the historical experiment
					model_cdf = ECDF(vals1[:,i,j])
					model_invcdf = model_cdf.x
					model_p = model_cdf.y
	    
					#Interpolate model values onto the historical model CDF probabilities
					model_p1 = np.interp(vals1[:,i,j],\
						model_invcdf,model_p)
					model_p2 = np.interp(vals2[:,i,j],\
						model_invcdf,model_p)

					#Interpolate the model CDF probabilities onto the observed CDF values
					model_xhat1 = \
						np.interp(model_p1,obs_p,obs_invcdf)
					model_xhat2 = \
						np.interp(model_p2,obs_p,obs_invcdf)

					if replace_zeros:
						model_xhat1[vals1[:,i,j] == 0] = 0
						model_xhat2[vals2[:,i,j] == 0] = 0
		model_xhat1[(model_xhat1>0) & (np.isinf(model_xhat1))] = np.max(model_xhat1)
		model_xhat1[(model_xhat1<0) & (np.isinf(model_xhat1))] = np.min(model_xhat1)
		model_xhat2[(model_xhat2>0) & (np.isinf(model_xhat2))] = np.max(model_xhat2)
		model_xhat2[(model_xhat2<0) & (np.isinf(model_xhat2))] = np.min(model_xhat2)

		return model_xhat1, model_xhat2

#@njit(parallel=True)
def qm_cmip_combined_loop_v2(obs, vals1, vals2, replace_zeros, mask):

		for i in np.arange(vals1.shape[1]):
			print(i)
			for j in np.arange(vals1.shape[2]):
				if mask[i,j] == 1:
				
					#Using numpy only for observed CDF
					x = obs[:,i,j]
					y = np.linspace(1./len(np.sort(x)), 1, len(np.sort(x)))
					x = np.concatenate((np.array([-np.inf],dtype=np.float32), np.sort(x)))
					y = np.concatenate((np.array([0],dtype=np.float32), y))
					obs_invcdf, inds = np.unique(x, return_index=True)
					obs_p = y[inds]

					#Create model CDF using numpy only
					x1 = vals1[:,i,j]
					y1 = np.linspace(1./len(np.sort(x1)), 1, len(np.sort(x1)))
					model_invcdf = np.concatenate((np.array([-np.inf]), np.sort(x1)))
					model_p = np.concatenate((np.array([0]), y1))

					#Interpolate model values onto the historical model CDF probabilities
					model_p1 = np.interp(vals1[:,i,j],\
						model_invcdf,model_p)
					model_p2 = np.interp(vals2[:,i,j],\
						model_invcdf,model_p)

					#Interpolate the model CDF probabilities onto the observed CDF values
					model_xhat1 = \
						np.interp(model_p1,obs_p,obs_invcdf)
					model_xhat2 = \
						np.interp(model_p2,obs_p,obs_invcdf)

					if replace_zeros:
						model_xhat1[vals1[:,i,j] == 0] = 0
						model_xhat2[vals2[:,i,j] == 0] = 0

		model_xhat1[(model_xhat1>0) & (np.isinf(model_xhat1))] = np.max(model_xhat1)
		model_xhat1[(model_xhat1<0) & (np.isinf(model_xhat1))] = np.min(model_xhat1)
		model_xhat2[(model_xhat2>0) & (np.isinf(model_xhat2))] = np.max(model_xhat2)
		model_xhat2[(model_xhat2<0) & (np.isinf(model_xhat2))] = np.min(model_xhat2)

		return model_xhat1, model_xhat2

if __name__ == "__main__":
	ProgressBar().register()

	p="mu_cape"; replace_zeros=True
	lats = slice(-90,90)
	lons = slice(-190,190)
	era5_da = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/global/era5/era5_2016*.nc", combine="by_coords")[p].\
		sel({"lat":slice(lats.stop-2,lats.start+2),"lon":slice(lons.start+2, lons.stop-2)})
	model_da1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/global/ACCESS1-0/ACCESS1-0_historical_r1i1p1_19990101_20060101.nc")[p].\
		sel({"lat":lats,"lon":lons}).interp({"lat":era5_da.lat.values, "lon":era5_da.lon.values},method="linear")
	model_da2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/global/ACCESS1-0/ACCESS1-0_rcp85_r1i1p1_20950101_21010101.nc")[p]\
		.sel({"lat":lats,"lon":lons}).interp({"lat":era5_da.lat.values, "lon":era5_da.lon.values},method="linear")
	mask = 	np.ones((era5_da.lat.values.shape[0], era5_da.lon.values.shape[0]))
	print(mask.shape)

	start = time.process_time()
	qm1, qm2 = qm_cmip_combined_loop_v1(era5_da.isel({"time":era5_da["time.month"] == 1}).values,\
		model_da1.isel({"time":model_da1["time.month"] == 1}).values, model_da2.isel({"time":model_da2["time.month"] == 1}).values,\
		replace_zeros, mask)
	print(time.process_time() - start)

	#start = time.process_time()
	#qm3, qm4 = qm_cmip_combined_loop_v2(era5_da.isel({"time":era5_da["time.month"] == 1}).values,\
	#	model_da1.isel({"time":model_da1["time.month"] == 1}).values, model_da2.isel({"time":model_da2["time.month"] == 1}).values,\
	#	replace_zeros, mask)
	#print(time.process_time() - start)
