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

def qm_cmip_era5(era5_da, model_da):

	#Quantile match a 3d (time-lat-lon) DataArray of CMIP model data (model_da) to an xarray
	# DataArray of ERA5 data, which are on the same spatial grid.

	vals = model_da.values.flatten()
	vals = vals[~np.isnan(vals)]

	#obs_cdf = ECDF(out[0][out[0]["time.year"]==1979].values.flatten())
	obs_cdf = ECDF(era5_da.values.flatten())
	#obs_invcdf = np.percentile(obs_cdf.x,obs_cdf.y*100)
	obs_invcdf = obs_cdf.x
	
	#Fit CDF to model
	model_cdf = ECDF(vals)
	#Convert model data to percentiles using the CDF
	model_p = np.interp(model_da.values,\
		model_cdf.x,model_cdf.y)
	#Convert model percentiles to ERA5 values
	model_xhat = np.interp(model_p,obs_cdf.y,obs_invcdf)
	#Handle data outside of interpolation range
	#model_xhat[np.isnan(model_xhat)] = obs_cdf.x[1]

	return model_xhat

def load_qm(model_name, ensemble, p):

	fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		model_name+"_"+ensemble+"_"+p+"_qm.nc"
	mod_xhat = xr.open_dataset(fname)[p+"_qm"].values

	return mod_xhat

def create_qm(era5_da, model_da, model_name, ensemble, p):

	fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		model_name+"_"+ensemble+"_"+p+"_qm.nc"
	mod_xhat = qm_cmip_era5(era5_da, model_da)
	xr.Dataset(data_vars={p+"_qm":(("time", "lat", "lon"), mod_xhat)},\
		    coords={"time":model_da.time, "lat":model_da.lat, "lon":model_da.lon}).\
		    to_netcdf(fname, mode="w")
	
	return mod_xhat

def get_cmip_lsm(era5, model, experiment, cmip_ver, group):

	lsm = get_lsm(model, experiment, cmip_ver=cmip_ver, group=group)
	lsm_regrid = lsm.interp({"lat":era5.lat, "lon":era5.lon}, method="nearest")
	return np.flipud(np.where(lsm_regrid.values >= 50, 1, 0))
	

def get_era5_lsm():

	#load the era5 lsm, subset to Australia, and coarsen as in regrid_era5(). Consider the coarse
	# lsm to be land if greater than 50% of the points are land

	from era5_read import get_mask
    
	f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_19790101_19790131.nc")
	lsm = get_mask(f.lon.values,f.lat.values)
	lsm_coarse = xr.DataArray(lsm, dims=("lat","lon")).coarsen({"lat":6,"lon":6},\
			    boundary="trim", side="left").sum()
	return xr.where(lsm_coarse>=18, 1, 0).values

def regrid_era5(p):

	#Regrid ERA5 convective diagnostics to a 1.5 degree, 6 hourly grid over the Aus region, by
	# taking the mean over each 1.5 degree region.

	#If this has already been done for parameter "p", then just load the data. Or else, 
	# compute and save

	#(Takes about an hour for all years and one diagnostic)

	fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		"era5_"+p+".nc"

	if (os.path.isfile(fname)):
		era5_coarse = xr.open_dataset(fname)["__xarray_dataarray_variable__"]
	else:
		print("Regridding ERA5 to a 1.5 degree grid...")
		ProgressBar().register()
		era5 = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_*")
		era5_sub = era5[p].sel({"time":np.in1d(era5["time.hour"], [0,6,12,18])})
		era5_coarse = era5_sub.coarsen({"lat":6, "lon":6}, boundary="trim", side="left").\
				mean()
		era5_coarse.to_netcdf(fname,format="NETCDF4_CLASSIC")

	return era5_coarse

def regrid_cmip(era5, model_name, ensemble, p):

	#Regrid convective diagnostics from CMIP models to the coarsened ERA5 grid over Aus (1.5 
	# degree spacing)
	
	fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		model_name+"_"+ensemble+"_"+p+".nc"

	if (os.path.isfile(fname)):
		f = xr.open_dataset(fname)
		mod_regrid = f[p]
		f.close()
	else:
		print("Regridding "+model_name+" to a 1.5 degree grid...")
		#ProgressBar().register()
		mod = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/"+\
			model_name+"/"+model_name+"*"+ensemble+"*.nc")
		mod_regrid = mod[p].interp({"lat":era5.lat, "lon":era5.lon})
		mod_regrid.to_netcdf(fname,format="NETCDF4_CLASSIC")
		mod_regrid.close()

	return mod_regrid

if __name__ == "__main__":

	#Settings

	p = "srhe_left"
	log = True
	plot = True
	geo_plot = True
	rel_diff = True
	rel_vmin = -100; rel_vmax = 100	    #For rel. difference plots
	abs_vmin = -5; abs_vmax = 5	    #For abs. difference plots
	vmin = 0; vmax = 100	    #For qunatile-mapped mean plots
	if plot:
		if geo_plot:
			from mpl_toolkits.basemap import Basemap
			m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
				urcrnrlat=-10,projection="cyl")
		ProgressBar().register()
		plt.figure(figsize=[11,9])

	models = [ ["ERA5",""] , ["ACCESS1-3","r1i1p1",5,""] , ["ACCESS1-0","r1i1p1",5,""] , \
			["ACCESS-ESM1-5","r1i1p1f1",6,"CSIRO"] , ["BNU-ESM","r1i1p1",5,""] , \
			["CNRM-CM5","r1i1p1",5,""] , ["GFDL-CM3","r1i1p1",5,""] , \
			["GFDL-ESM2G","r1i1p1",5,""] , ["GFDL-ESM2M","r1i1p1",5,""] , \
			["IPSL-CM5A-LR","r1i1p1",5,""] , ["IPSL-CM5A-MR","r1i1p1",5,""] , \
			["MIROC5","r1i1p1",5,""] , ["MRI-CGCM3","r1i1p1",5,""], \
			["bcc-csm1-1","r1i1p1",5,""], \
			]


	#LOAD EACH MODEL AND PLOT MEAN SPATIAL DISTRIBUTION
	print("Loading models, plotting data...")
	subplots = [5,3]
	out = []
	era5 = regrid_era5(p)
	era5 = era5.sel({"time":(era5["time.year"] <= 2005) & \
				    (era5["time.year"] >= 1979)})
	era5_lsm = get_era5_lsm()
	for i in np.arange(len(models)):
		if models[i][0] == "ERA5":
			e = era5
		else:
			e = regrid_cmip(era5, models[i][0], models[i][1], p)

		if plot:
		
			if lsm:
				e = xr.where(era5_lsm, e.sel({"time":(e["time.year"] <= 2005) & \
				    (e["time.year"] >= 1979)}), np.nan)
				era5_mean = xr.where(era5_lsm,era5.mean("time").values,np.nan).\
					flatten()
				era5_mean = era5_mean[~np.isnan(era5_mean)]
				mod_mean = xr.where(era5_lsm,e.mean("time").values,np.nan).\
					flatten()
				mod_mean = mod_mean[~np.isnan(mod_mean)]
				r = str(np.corrcoef(era5_mean, mod_mean)[1,0].round(3))
			else:
				e = e.sel({"time":(e["time.year"] <= 2005) & \
				    (e["time.year"] >= 1979)})
				r = str(np.corrcoef(era5.mean("time").values.flatten(), \
					    e.mean("time").values.flatten())[1,0].round(3))

			plt.subplot(subplots[0],subplots[1],i+1)
			if log:
				e.mean("time").plot(norm=mpl.colors.LogNorm(), add_labels=False)
			else:
				e.mean("time").plot(add_labels=False)
			plt.title(models[i][0] + " r="+r)
			if geo_plot:
				m.drawcoastlines()

			e.close()

			out.append(e)

	#Quantile map each GCM to coarsened ERA5 
	print("Quantile mapping to ERA5...")
	out_qm = []
	for i in np.arange(len(models)):
		if i == 0:
			out_qm.append(out[i].values)
		else:
			#model_xhat = qm_cmip_era5(out[0], out[i])
			try:
				model_xhat = load_qm(models[i][0], models[i][1], p)
			except:
				print(models[i])
				model_xhat = create_qm(out[0], out[i], models[i][0],\
					    models[i][1], p)
			out_qm.append(model_xhat)


	if plot:
		#Plot histogram of raw data
		plt.figure()
		hists = [out[i].values.flatten() for i in np.arange(len(out))]
		plt.hist(hists, \
			    density=False, log=log, label = models,\
			    bins = 50, histtype="step", lw=2, \
			    color=plt.get_cmap("tab20")(np.linspace(0,1,len(models))))
		plt.legend()

		#Plot quantile mapped histograms
		print("Plotting quantile-matched data...")
		plt.figure()
		hists = [out_qm[i].flatten() for i in np.arange(len(out))]
		plt.hist(hists, \
			    density=False, log=log, label = models,\
			    bins = 50, histtype="step", lw=2, \
			    color=plt.get_cmap("tab20")(np.linspace(0,1,len(models))))
		#[sb.distplot(hists[i], \
		#	    hist=False, kde=True, norm_hist=False, \
		#	    bins = 50, \
		#	    ) for i in np.arange(len(out_qm))]
		plt.legend()

		#Plot quantile mapped mean spatial distributions
		plt.figure(figsize=[11,9])
		x,y=np.meshgrid(e.lon,e.lat)
		for i in np.arange(len(models)):
			plt.subplot(subplots[0],subplots[1],i+1)
			if log:
				if geo_plot:
				    m.pcolormesh(x,y,np.nanmean(out_qm[i],axis=0),\
					    norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
				else:
				    plt.pcolormesh(x,y,np.nanmean(out_qm[i],axis=0),\
					    norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax))
			else:
				if geo_plot:
				    m.pcolormesh(x,y,np.nanmean(out_qm[i],axis=0),vmin=vmin,vmax=vmax)
				else:
				    plt.pcolormesh(x,y,np.nanmean(out_qm[i],axis=0),\
					    vmin=vmin,vmax=vmax)
			r = str(np.corrcoef(era5.mean("time").values.flatten(), \
					    np.nanmean(out_qm[i],axis=0).flatten())[1,0].round(3))
			plt.title(models[i][0] + " r="+r)
			plt.colorbar()
			if geo_plot:
				m.drawcoastlines()

		#Plot quantile mapped diff in spatial mean with ERA5
		plt.figure(figsize=[11,9])
		x,y=np.meshgrid(e.lon,e.lat)
		for i in np.arange(1, len(models)):
			plt.subplot(subplots[0],subplots[1],i)
			if rel_diff:
				vals = (np.nanmean(out_qm[i],axis=0)-(np.nanmean(out_qm[0],axis=0)))/\
						np.nanmean(out_qm[0],axis=0) * 100
				vmax = rel_vmax; vmin=rel_vmin
			else:
				vals = np.nanmean(out_qm[i],axis=0)-np.nanmean(out_qm[0],axis=0)
				vmax = abs_vmax; vmin=abs_vmin
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

		plt.show()

