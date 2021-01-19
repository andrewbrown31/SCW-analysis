import glob
from erai_read import get_mask
from numba import jit
import argparse
from mpl_toolkits.basemap import Basemap
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from dask.diagnostics import ProgressBar
from read_cmip import get_lsm

def drop_duplicates(da):

	#Drop time duplicates

	a, ind = np.unique(da.time.values, return_index=True)
	return(da[ind])

def daily_max_qm(data, da, models, p, lsm):

	#For a list of quantile-mapped 3d arrays (data), create a DataArray using metadata from da, and 
	# resample to daily maximum. Save data.

	print("Resampling to daily max...")
	out = []
	for i in np.arange(len(models)):

		model_name = models[i][0]
		ensemble = models[i][1]
		print(model_name)

		out.append(xr.Dataset(data_vars={p:(("time", "lat", "lon"), data[i])},\
		    coords={"time":da[i].time, "lat":da[i].lat, "lon":da[i].lon}).resample({"time":"1D"}).max())
		
	return out

def calc_logit(models, p, lsm):

	#Load the regridded, quantile mapped model data, and calculate logistic model equations
	erai_lsm = get_erai_lsm()

	if p == "logit_aws":
		vars = ["lr36","mhgt","ml_el","qmean01","srhe_left","Umean06"]
	elif p == "logit_sta":
		vars = ["lr36","ml_cape","srhe_left","Umean06"]

	var_list = []
	print("Loading all variables for "+p+"...")
	for v in vars:
		print(v)
		out = []
		for m in np.arange(len(models)):
			if m == 0:
				erai = regrid_erai(v)
				if lsm:
					out.append(xr.where(erai_lsm, \
							erai.sel({"time":(erai["time.year"] <= 2005) & \
							(erai["time.year"] >= 1979)}), np.nan).values)
				else:
					out.append(erai = erai.sel({"time":(erai["time.year"] <= 2005) & \
						(erai["time.year"] >= 1979)}).values)
			else:
				try:
					out.append(load_qm(models[m][0], models[m][1], v, lsm))
				except:
					raise ValueError(v+" HAS NOT BEEN QUANTILE MAPPED YET FOR "+models[m][0])
		var_list.append(out)

	out_logit = []
	for i in np.arange(len(models)):
		if p == "logit_aws":
			z = 6.4e-1*var_list[0][i] - 1.2e-4*var_list[1][i] + 4.4e-4*var_list[2][i] \
			    -1.0e-1*var_list[3][i] + 1.7e-2*var_list[4][i] + 1.8e-1*var_list[5][i] - 7.4
		elif p == "logit_sta":
			z = 3.3e-1*var_list[0][i] + 1.6e-3*var_list[1][i] + 2.9e-2*var_list[2][i] \
			    +1.6e-1*var_list[3][i] - 4.5
		out_logit.append( 1 / (1 + np.exp(-z)))

	return out_logit

def plot_monthly_threshold(data, models, p, threshold, outname):

	plt.figure(figsize=[12,7])
	ax=plt.gca()
	monthly_thresh = pd.DataFrame()

	spatial_points = (~np.isnan(data[0][p][0]).values).sum()

	for i in np.arange(len(models)):
		thresh = []
		for m in np.arange(1,13):
			thresh.append(\
				( data[i][p][data[i]["time.month"]==m] >= threshold ).sum().values \
				    / (( data[i]["time.month"]==m).values.sum() * spatial_points)  )
		if i >= 1:
			r = str(np.corrcoef(thresh, monthly_thresh["ERAI r=1"])[1,0].round(3))
			monthly_thresh = pd.concat([monthly_thresh, pd.DataFrame({models[i][0]+" r="+r:thresh})],\
				axis=1)
		else:
			monthly_thresh = pd.concat([monthly_thresh, \
				pd.DataFrame({models[i][0]+" r=1":thresh})], axis=1)
	monthly_thresh.plot(color=plt.get_cmap("tab20")(np.linspace(0,1,len(models))), ax=ax)
	plt.xticks(np.arange(12), ["J","F","M","A","M","J","J","A","S","O","N","D"])
	plt.ylabel("Monthly diagnostic frequency over Aus.\n"+p)
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.subplots_adjust(right=0.8)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_monthly_mean(data, data_da, models, p, outname):

	plt.figure(figsize=[12,7])
	ax=plt.gca()
	monthly_mean = pd.DataFrame()
	for i in np.arange(len(models)):
		mean = []
		for m in np.arange(1,13):
			mean.append(np.nanmean(data[i][data_da[i]["time.month"]==m]))
		if i >= 1:
			r = str(np.corrcoef(mean, monthly_mean["ERAI r=1"])[1,0].round(3))
			monthly_mean = pd.concat([monthly_mean, pd.DataFrame({models[i][0]+" r="+r:mean})],\
				axis=1)
		else:
			monthly_mean = pd.concat([monthly_mean, pd.DataFrame({models[i][0]+" r=1":mean})], axis=1)
	monthly_mean.plot(color=plt.get_cmap("tab20")(np.linspace(0,1,len(models))), ax=ax)
	plt.xticks(np.arange(12), ["J","F","M","A","M","J","J","A","S","O","N","D"])
	plt.ylabel("Monthly mean over Aus.\n"+p)
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.subplots_adjust(right=0.8)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_hist(data, models, log, outname):
	#Plot histogram of a list of model data
	plt.figure(figsize=[12,7])
	try:
		hists = [data[i].values.flatten() for i in np.arange(len(data))]
	except:
		hists = [data[i].flatten() for i in np.arange(len(data))]
	plt.hist(hists, \
		density=False, log=log, label = models,\
		bins = 50, histtype="step", lw=2, \
		color=plt.get_cmap("tab20")(np.linspace(0,1,len(models))))
	plt.legend()
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_mean_spatial_diff(data, models, subplots, rel_diff, vmax, vmin, geo_plot, lon, lat, outname):
	if geo_plot:
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
				urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])
	x,y=np.meshgrid(lon,lat)
	for i in np.arange(1, len(models)):
		plt.subplot(subplots[0],subplots[1],i)
		if rel_diff:
			vals = (np.nanmean(data[i],axis=0)-(np.nanmean(data[0],axis=0)))/\
					np.nanmean(data[0],axis=0) * 100
		else:
			vals = np.nanmean(data[i],axis=0)-np.nanmean(data[0],axis=0)
		if geo_plot:
		        m.pcolormesh(x,y,vals,\
			cmap=plt.get_cmap("RdBu_r"),vmin=vmin,vmax=vmax)
		else:
		        plt.pcolormesh(x,y,vals,\
			cmap=plt.get_cmap("RdBu_r"),vmin=vmin,vmax=vmax)
		plt.title(models[i][0])
		if geo_plot:
			m.drawcoastlines()
		c=plt.colorbar()
		if rel_diff:
			c.set_label("%")
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_threshold_spatial_dist(data, models, p, threshold, subplots, log, geo_plot, lon, lat, lsm, outname, \
		vmin=None, vmax=None):

	#For a list of model data, plot the frequency of threshold exceedence
	if geo_plot:
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
				urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])

	erai_freq = ( (data[0][p] >= threshold).sum("time") / data[0][p].shape[0]).values.flatten()
	if lsm:
		erai_freq = erai_freq[~np.isnan(erai_freq)]

	x,y = np.meshgrid(lon,lat)
	for i in np.arange(len(models)):
			plt.subplot(subplots[0],subplots[1],i+1)
			if geo_plot:
				m.drawcoastlines()

    
			mod_freq = ( (data[i][p] >= threshold).sum("time") / data[i][p].shape[0] ).values
			mod_freq1d = mod_freq.flatten()
			if lsm:
				mod_freq1d = mod_freq1d[~np.isnan(mod_freq1d)]
			r = str(np.corrcoef(erai_freq, mod_freq1d)[1,0].round(3))

			if log:
				m.pcolormesh(x, y, mod_freq, norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)
			else:
				m.pcolormesh(x, y, mod_freq, vmin=vmin, vmax=vmax)
			plt.colorbar()
			plt.title(models[i][0] + " r="+r)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_mean_spatial_dist(data, models, subplots, log, geo_plot, lon, lat, lsm, outname, vmin=None, vmax=None):

	#For a list of model data, plot the mean spatial distribution
	if geo_plot:
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
				urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])
	try:
		erai_mean = data[0].mean("time").values.flatten()
	except:
		try:
			erai_mean = data[0].mean(axis=0).flatten()
		except:
			erai_mean = data[0]
	if lsm:
		erai_mean = erai_mean[~np.isnan(erai_mean)]

	x,y = np.meshgrid(lon,lat)
	for i in np.arange(len(models)):
			plt.subplot(subplots[0],subplots[1],i+1)
			if geo_plot:
				m.drawcoastlines()
			try:
				mod_mean = data[i].mean("time").values
			except:
				try:
					mod_mean = np.nanmean(data[i],axis=0)
				except:
					mod_mean = data[i]
			mod_mean1d = mod_mean.flatten()
			if lsm:
				mod_mean1d = mod_mean1d[~np.isnan(mod_mean1d)]
			r = str(np.corrcoef(erai_mean, mod_mean1d)[1,0].round(3))

			if log:
				m.pcolormesh(x, y, mod_mean, norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)
			else:
				m.pcolormesh(x, y, mod_mean, vmin=vmin, vmax=vmax)
			plt.colorbar()
			plt.title(models[i][0] + " r="+r)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def ecdf_to_unique(ecdf):

	y = ecdf.y
	x = ecdf.x
	x, inds = np.unique(x, return_index=True)
	y = y[inds]
	return x,y

def load_model_data(models, p, erai_data=None, save=True, \
		erai_regrid=False, lsm=True, force_cmip_regrid=False,\
		experiment="historical", y1=1979, y2=2005, erai_y1=1979, erai_y2=2005):

	#For each model in the list "models" (including ERAI), load regridded data and slice in time.
	#If lsm is True, then mask ocean data based on ERAI lsm

	assert y2 >= y1
	assert erai_y2 >= erai_y1

	erai_lsm = get_erai_lsm()
	if models[0][0] == "ERAI":
		if erai_data is None:
			erai = load_erai(p, erai_regrid)
			erai_data = erai.sel({"time":(erai["time.year"] <= erai_y2) & \
					    (erai["time.year"] >= erai_y1)})
			if lsm:
				erai_data = xr.where(erai_lsm, \
				    erai_data, np.nan)

	out = []
	for i in np.arange(len(models)):
		if models[i][0] == "ERAI":
			e = erai_data
			if lsm:
				e = xr.where(erai_lsm, \
					e, np.nan)
		elif models[i][0] == "BARPA":
			e = load_barpa(p, y1, y2)
		else:
			e = regrid_cmip(erai_data, models[i][0],\
				models[i][1], p, y1, y2, \
				force_cmip_regrid, experiment=experiment, save=save)
			if lsm:
				e = xr.where(erai_lsm, \
					e, np.nan)

		e = drop_duplicates(e)
		e.close()
		out.append(e)

	return out

def load_all_qm(data, models, p, lsm, replace_zeros, force_compute=False, \
	    experiment="historical", loop=True):

	#Loop over all models, and use either load_qm() to load, or qm_cmip_erai() to calculate

	out_qm = []
	for i in np.arange(len(models)):
		if i == 0:
			out_qm.append(data[i].values)
		else:
			if force_compute:
				print(models[i])
				model_xhat = create_qm(data[0], data[i], models[i][0],\
					    models[i][1], p, lsm, replace_zeros,\
					    experiment=experiment, loop=loop)
			else:
				try:
					model_xhat = load_qm(models[i][0], models[i][1], p, lsm, experiment=experiment)
				except:
					print(models[i])
					model_xhat = create_qm(data[0], data[i], models[i][0],\
					    models[i][1], p, lsm, replace_zeros,\
					    loop, experiment=experiment)
			out_qm.append(model_xhat)

	return out_qm

@jit
def qm_cmip_erai_loop(erai_da, model_da, replace_zeros, mask):

	#As in qm_cmip_erai(), but with treating each spatial point as a unique distribution, 
	# and  matching independantly, rather than treating all spatial points as one distribution

	vals = model_da.values
	obs = erai_da.values
	model_xhat = np.zeros(vals.shape) * np.nan
	for m in np.arange(1,13):
		print(m)
		model_m_inds = (model_da["time.month"] == m)
		obs_m_inds = (erai_da["time.month"] == m)
		for i in np.arange(vals.shape[1]):
			for j in np.arange(vals.shape[2]):
				if mask[i,j] == 1:
					obs_cdf = ECDF(obs[obs_m_inds,i,j])
					obs_invcdf, obs_p = ecdf_to_unique(obs_cdf)

					model_cdf = ECDF(vals[model_m_inds,i,j])
					model_invcdf = model_cdf.x
					model_p = np.interp(vals[model_m_inds,i,j], model_invcdf, model_cdf.y)

					model_xhat[model_m_inds,i,j] = np.interp(model_p,obs_p,obs_invcdf)
	if replace_zeros:
		model_xhat[vals == 0] = 0

	return model_xhat

def qm_cmip_erai(erai_da, model_da, replace_zeros):

	#Quantile match a 3d (time-lat-lon) DataArray of CMIP model data (model_da) to an xarray
	# DataArray of ERAI data, which are on the same spatial grid.

	vals = model_da.values.flatten()
	vals = vals[~np.isnan(vals)]

	obs = erai_da.values.flatten()
	obs = obs[~np.isnan(obs)]
	obs_cdf = ECDF(obs)
	obs_invcdf = obs_cdf.x
	
	#Fit CDF to model
	model_cdf = ECDF(vals)
	#Convert model data to percentiles using the CDF
	model_p = np.interp(model_da.values,\
		model_cdf.x,model_cdf.y)
	#Convert model percentiles to ERAI values
	model_xhat = np.interp(model_p,obs_cdf.y,obs_invcdf)

	#For effective SRH and MLCAPE, a CMIP model value of zero can be matched with a significant non-zero
	# value, if the equivalent ERAI percentile is non-zero. To avoid, manually replace with zeros. This 
	# makes physical sense (can't create CAPE where there is none)
	if replace_zeros:
		model_xhat[model_da.values == 0] = 0

	return model_xhat

def load_qm(model_name, ensemble, p, lsm, experiment="historical"):

	if lsm:
		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm_lsm.nc"
	else:
		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm.nc"
	mod_xhat = xr.open_dataset(fname)[p+"_qm"].values

	return mod_xhat

def save_logit(models, data, da, lsm, p):

	for i in np.arange(len(models)):
		if lsm:
			fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_"+p+"_lsm.nc"
		else:
			fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_"+p+".nc"
		xr.Dataset(data_vars={p:(("time", "lat", "lon"), data[i])},\
		    coords={"time":da[i].time, "lat":da[i].lat, "lon":da[i].lon}).\
		    to_netcdf(fname, mode="w", engine="h5netcdf",\
			    encoding={p:{"zlib":True, "complevel":9}})

def create_qm(erai_da, model_da, model_name, ensemble, p, lsm, replace_zeros, \
	    experiment="historical", loop=True):

	if lsm:
		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm_lsm.nc"
	else:
		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm.nc"
	if loop:
		mod_xhat = qm_cmip_erai_loop(erai_da, model_da, replace_zeros, \
		    get_mask(erai_da.lon.values, erai_da.lat.values))
	else:
		mod_xhat = qm_cmip_erai(erai_da, model_da, replace_zeros)
	xr.Dataset(data_vars={p+"_qm":(("time", "lat", "lon"), mod_xhat)},\
		    coords={"time":model_da.time, "lat":model_da.lat, "lon":model_da.lon}).\
		    to_netcdf(fname, mode="w", engine="h5netcdf",\
			    encoding={p+"_qm":{"zlib":True, "complevel":9}})
	
	return mod_xhat

def get_cmip_lsm(erai, model, experiment, cmip_ver, group):

	lsm = get_lsm(model, experiment, cmip_ver=cmip_ver, group=group)
	lsm_regrid = lsm.interp({"lat":erai.lat, "lon":erai.lon}, method="nearest")
	return np.flipud(np.where(lsm_regrid.values >= 50, 1, 0))
	

def get_erai_lsm(regrid=False):

	#load the erai lsm, subset to Australia, and coarsen as in regrid_erai(). Consider the coarse
	# lsm to be land if greater than 50% of the points are land

    
	f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/global/erai/erai_19790101_19790131.nc")
	lsm = get_mask(f.lon.values,f.lat.values)
	if regrid:
		lsm_coarse = xr.DataArray(lsm, dims=("lat","lon")).coarsen({"lat":6,"lon":6},\
			    boundary="trim", side="left").sum()
		return xr.where(lsm_coarse>=18, 1, 0).values
	else:
		return lsm

def load_barpa(p, y1, y2, regrid=False):

	#Load BARPA data between two years.

	print("Loading BARPA...")
	files = np.sort(glob.glob("/g/data/eg3/ab4502/ExtremeWind/global/barpa_access/barpa_access_*"))
	years = np.array([int(file.split("/")[8].split("_")[2][0:4]) for file in files])
	barpa = xr.open_mfdataset(files[(years>=y1) & (years <= y2)])[p]
	lsm = xr.open_dataset("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc").interp({"longitude":barpa.lon, "latitude":barpa.lat}, "nearest")
	barpa_lsm = barpa.where(lsm["lnd_mask"].values==1, np.nan)
	barpa_lsm = barpa_lsm.sel({"time":(barpa_lsm["time.year"] >= y1) & (barpa_lsm["time.year"] <= y2)})
	print(barpa_lsm)

	return barpa_lsm

def load_erai(p, regrid=False):

	#Regrid ERAI convective diagnostics to a 1.5 degree, 6 hourly grid over the Aus region, by
	# taking the mean over each 1.5 degree region.

	#If this has already been done for parameter "p", then just load the data. Or else, 
	# compute and save

	#(Takes about an hour for all years and one diagnostic)

	fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		"erai_"+p+".nc"

	if regrid:
		if (os.path.isfile(fname)):
			erai_coarse = xr.open_dataset(fname)["__xarray_dataarray_variable__"]
		else:
			print("Regridding ERAI to a 1.5 degree grid...")
			ProgressBar().register()
			erai = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/global/erai/erai_*")
			erai_sub = erai[p].sel({"time":np.in1d(erai["time.hour"], [0,6,12,18])})
			erai_coarse = erai_sub.coarsen({"lat":6, "lon":6}, boundary="trim", side="left").\
					mean()
	else:
			print("Loading ERAI...")
			erai = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/global/erai/erai_*")
			erai_coarse = erai[p].sel({"time":np.in1d(erai["time.hour"], [0,6,12,18])})

	return erai_coarse

def regrid_cmip(erai, model_name, ensemble, p, y1, y2, force_regrid,\
	experiment="historical", save=True):

	#Regrid convective diagnostics from CMIP models to the coarsened ERAI grid over Aus (1.5 
	# degree spacing)
	
	fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+\
		model_name+"_"+experiment+"_"+ensemble+"_"+p+".nc"

	if (os.path.isfile(fname)) & (~force_regrid):
		print("Loading regridded "+model_name+" data...")
		f = xr.open_dataset(fname)
		f = f.sel({"time":(f["time.year"] <= y2) & (f["time.year"] >= y1)})
		mod_regrid = f[p]
		f.close()
	else:
		spacing = str(erai.lon.values[-1] - erai.lon.values[-2])
		print("Regridding "+experiment+" "+\
		    model_name+" to a "+spacing+" degree grid...")
		mod = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/global/"+\
			model_name+"/"+model_name+"_"+experiment+"_"+ensemble+"*.nc")
		mod = mod.sel({"time":(mod["time.year"] <= y2) & (mod["time.year"] >= y1)})
		mod_regrid = mod[p].interp({"lat":erai.lat, "lon":erai.lon}).persist()
		if save:
			mod_regrid.to_netcdf(fname,format="NETCDF4_CLASSIC",\
				encoding={p:{"zlib":True, "complevel":9}})
		mod_regrid.close()

	return mod_regrid

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

	#Parse command line arguments

	parser = argparse.ArgumentParser(description='Post-processing of CMIP convective diagnostics, including'\
	    +" quantile mapping to ERAI and plotting")
	parser.add_argument("-p",help="Parameter",required=True)
	parser.add_argument("--y1",help="Start year for CMIP models",default=1979,\
		type=float)
	parser.add_argument("--y2",help="End year for CMIP models",default=2005,\
		type=float)
	parser.add_argument("--erai_y1",help="Start year for ERAI",default=1979,\
		type=float)
	parser.add_argument("--erai_y2",help="End year for ERAI",default=2005,\
		type=float)
	parser.add_argument("--loop",help="If True, then loop over each spatial point when QQ-matching",default=True,\
		type=str2bool)
	parser.add_argument("--force_compute",help="Force quantile mapping of CMIP data?",default=False,\
		type=str2bool)
	parser.add_argument("--force_cmip_regrid",help="Force regridding of CMIP data?",default=False,\
		type=str2bool)
	parser.add_argument("--lsm",help="Mask ocean values using the ERAI lsm?",default=True,\
		type=str2bool)
	parser.add_argument("--log",help="Make plots with a LogNorm color-scale, and log y-axis?",default=False,\
		type=str2bool)
	parser.add_argument("--plot",help="Make plots and save?",default=True,\
		type=str2bool)
	parser.add_argument("--geo_plot",help="Spatial plots on a geo-grid with coastlines?",default=True,\
		type=str2bool)
	parser.add_argument("--dmax",help="Resample to daily max? If so, threshold exceedence "\
		+"frequencies will be plotted, instead of mean plots" ,default=False, type=str2bool)
	parser.add_argument("--rv1",help="Vmin for relative spatial difference plots",default=None,\
		type=float)
	parser.add_argument("--rv2",help="Vmax for relative spatial difference plots",default=None,\
		type=float)
	parser.add_argument("--av1",help="Vmin for relative spatial difference plots",default=None,\
		type=float)
	parser.add_argument("--av2",help="Vmax for relative spatial difference plots",default=None,\
		type=float)
	parser.add_argument("--v1",help="Vmin for spatial mean plots",default=None,\
		type=float)
	parser.add_argument("--v2",help="Vmax for spatial mean plots",default=None,\
		type=float)
	parser.add_argument("--threshold",help="Threshold for dmax plots",default=0,\
		type=float)
	args = parser.parse_args()
	p = args.p
	y1 = args.y1
	y2 = args.y2
	erai_y1 = args.erai_y1
	erai_y2 = args.erai_y2
	loop = args.loop
	force_compute = args.force_compute
	force_cmip_regrid = args.force_cmip_regrid
	lsm = args.lsm
	log = args.log
	plot = args.plot
	geo_plot = args.geo_plot
	dmax = args.dmax
	rel_vmin = args.rv1
	rel_vmax = args.rv2
	abs_vmin = args.av1
	abs_vmax = args.av2
	vmin = args.v1
	vmax = args.v2
	threshold = args.threshold

	#Implicitly set some settings
	ProgressBar().register()
	if p in ["srhe_left","ml_cape"]:
		replace_zeros=True
	else:
		replace_zeros=False
	#subplots = [5,3]
	subplots = [1,2]
	models = [ ["ERAI",""] ,\
	#		["ACCESS1-3","r1i1p1",5,""] ,\
	#		["ACCESS1-0","r1i1p1",5,""] , \
	#		["ACCESS-ESM1-5","r1i1p1f1",6,"CSIRO"] ,\
	#		["BNU-ESM","r1i1p1",5,""] , \
	#		["CNRM-CM5","r1i1p1",5,""] ,\
			["GFDL-CM3","r1i1p1",5,""] , \
	#		["GFDL-ESM2G","r1i1p1",5,""] , \
	#		["GFDL-ESM2M","r1i1p1",5,""] , \
	#		["IPSL-CM5A-LR","r1i1p1",5,""] ,\
	#		["IPSL-CM5A-MR","r1i1p1",5,""] , \
	#		["MIROC5","r1i1p1",5,""] ,\
	#		["MRI-CGCM3","r1i1p1",5,""], \
	#		["bcc-csm1-1","r1i1p1",5,""], \
			]

	#For all diagnostics (except logit models), load/coarsen data and quantile-map
	if p not in ["logit_sta","logit_aws"]:
		out = load_model_data(models, p, lsm=lsm, force_cmip_regrid=force_cmip_regrid,\
			y1=y1, y2=y2, erai_y1=erai_y1, erai_y2=erai_y2) 
		print("Quantile mapping to ERAI...")
		#out_qm = load_all_qm(out, models, p, lsm, replace_zeros, force_compute=force_compute)
		out_qm = load_all_qm(out, models, p, lsm, replace_zeros, force_compute=force_compute, loop=loop)
	#Or else, Load quantile mapped variables and calculate logit model
	else:
		out_qm = calc_logit(models, p, lsm)
		out = load_model_data(models, "srhe_left", lsm=lsm)
		save_logit(models, out_qm, out, lsm, p)

	#Resample to daily max if option is set
	if dmax:
		if threshold == 0:
			raise ValueError("\nMUST SPECIFY THRESHOLD FOR DMAX ANALYSIS\n")
		out_dmax = daily_max_qm(out_qm, out, models, p, lsm)

	#Make plots
	if plot:
		if p not in ["logit_sta","logit_aws"]:
			plot_hist(out, models, log, p+"_hist")
			plot_mean_spatial_dist(out, models, subplots, log, geo_plot, \
				    out[0].lon.values, out[0].lat.values, lsm, p+"_mean_spatial_dist")

		plot_hist(out_qm, models, log, p+"_hist_qm")
		if dmax:
			plot_threshold_spatial_dist(out_dmax, models, p, threshold, \
                            subplots, log, geo_plot, \
                            out[0].lon.values, out[0].lat.values, lsm, p+"_threshold_spatial_dist",\
			    vmin=vmin, vmax=vmax)
			plot_monthly_threshold(out_dmax, models, p, threshold, p+"_monthly_threshold")
		else:
			plot_mean_spatial_dist(out_qm, models, subplots, log, geo_plot, \
			    out[0].lon.values, out[0].lat.values, lsm, p+"_mean_qm_spatial_dist",\
			    vmin=vmin, vmax=vmax)
			plot_mean_spatial_diff(out_qm, models, subplots, True, rel_vmin, rel_vmax, geo_plot,\
			    out[0].lon.values, out[0].lat.values, p+"_mean_spatial_rel_diff")
			plot_mean_spatial_diff(out_qm, models, subplots, False, abs_vmin, abs_vmax, geo_plot,\
			    out[0].lon.values, out[0].lat.values, p+"_mean_spatial_abs_diff")
			plot_monthly_mean(out_qm, out, models, p, p+"_monthly_mean")

