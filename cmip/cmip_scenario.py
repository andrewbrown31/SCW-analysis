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
from cmip_analysis import load_model_data, regrid_era5

#Load the historical and scenario data for a range of CMIP models. 
#Join into one distribution, and quantile map onto ERA5 data.
#Compare the scenario run to the historical run.

def calc_logit(models, p, lsm, experiment):

        #Load the regridded, quantile mapped model data, and calculate logistic model equations
	era5_lsm = get_era5_lsm()

	if p == "logit_aws":
		vars = ["lr36","mhgt","ml_el",\
			    "qmean01","srhe_left","Umean06"]
	elif p == "logit_sta":
		vars = ["lr36","ml_cape","srhe_left","Umean06"]

	var_list_hist = []
	var_list_scenario = []
	print("Loading all variables for "+p+"...")
	for v in vars:
		print(v)
		out_hist = []
		out_scenario = []
		for m in np.arange(len(models)):
			if m == 0:
				era5 = regrid_era5(v)
				if lsm:
					era5_trim = xr.where(era5_lsm,\
                                            era5.sel({"time":\
					    (era5["time.year"] <= 2005) & \
                                            (era5["time.year"] >= 1979)}),\
						 np.nan).values
				else:
					out_trim = era5.sel({"time":\
					    (era5["time.year"] <= 2005) & \
                                            (era5["time.year"] >= 1979)}).\
					    values
				out_hist.append(era5_trim)
				out_scenario.append(era5_trim)
			else:
				try:
					qm_hist, qm_scenario = load_qm(\
						models[m][0], models[m][1], v, lsm,\
						experiment=experiment)
					out_hist.append(qm_hist)
					out_scenario.append(qm_hist)
				except:
					raise ValueError(v+" HAS NOT BEEN"+\
					    " QUANTILE MAPPED YET FOR "+\
					    models[m][0])
		var_list_hist.append(out_hist)
		var_list_scenario.append(out_scenario)

	out = []
	for var_list in [var_list_hist, var_list_scenario]:
		out_logit = []
		for i in np.arange(len(models)):
			if p == "logit_aws":
				z = 6.4e-1*var_list[0][i] - 1.2e-4*var_list[1][i] +\
				     4.4e-4*var_list[2][i] \
				    -1.0e-1*var_list[3][i] \
				    + 1.7e-2*var_list[4][i] \
				    + 1.8e-1*var_list[5][i] - 7.4
			elif p == "logit_sta":
				z = 3.3e-1*var_list[0][i] + 1.6e-3*var_list[1][i] +\
				     2.9e-2*var_list[2][i] \
				    +1.6e-1*var_list[3][i] - 4.5
			out_logit.append( 1 / (1 + np.exp(-z)))
		out.append(out_logit)

	return out, out_hist[0].lat.values

def plot_monthly_mean(historical, scenario, historical_da, scenario_da, \
	    subplots, models, p, outname):

        fig = plt.figure(figsize=[12,7])
        ax=plt.gca()
        era5_mean = []
        for m in np.arange(1,13):
                era5_mean.append(np.nanmean(\
			historical[0][historical_da[0]["time.month"]==m]))
        for i in np.arange(1,len(models)):
                plt.subplot(subplots[0], subplots[1], i)
                mean_hist = []
                mean_scenario = []
                for m in np.arange(1,13):
                        mean_hist.append(np.nanmean(\
				historical[i][historical_da[i]["time.month"]==m]))
                        mean_scenario.append(np.nanmean(\
				scenario[i][scenario_da[i]["time.month"]==m]))
                plt.plot(era5_mean, "k")
                plt.plot(mean_hist, "b")
                plt.plot(mean_scenario, "r")
	    
                plt.xticks(np.arange(12), \
			["J","F","M","A","M","J","J","A","S","O","N","D"])
                plt.xlim([0,11])
                plt.title(models[i][0])
        #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.legend(["ERA5","Historical",\
			"RCP8.5"], bbox_to_anchor=(0.5, 0.05) , ncol=3, loc=10)
        plt.subplots_adjust(top=0.95, hspace=0.5, bottom=0.15)
        plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")

def plot_scenario_diff(historical, scenario, lat, lon, \
	    subplots, models, log, rel_diff, outname,\
	    vmin=None, vmax=None):

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
	        urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])
	x,y=np.meshgrid(lon,lat)
	for i in np.arange(1, len(models)): 
		plt.subplot(subplots[0],subplots[1],i) 
		if rel_diff: 
			vals = (np.nanmean(scenario[i],axis=0)-\
					(np.nanmean(historical[i],axis=0)))/\
                                        np.nanmean(historical[i],axis=0) * 100 
		else: 
			vals = np.nanmean(scenario[i],axis=0)-\
				    np.nanmean(historical[i],axis=0) 
		m.pcolormesh(x,y,vals,\
		cmap=plt.get_cmap("RdBu_r"),vmin=vmin,vmax=vmax) 
		plt.title(models[i][0]) 
		m.drawcoastlines() 
		c=plt.colorbar() 
		if rel_diff: 
			c.set_label("%") 
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")


def plot_scenario_hist(historical, scenario, subplots, models, log, outname):
	
	fig=plt.figure(figsize=[12,7])
	for i in np.arange(1, len(models)):
		plt.subplot(subplots[0], subplots[1], i)
		plt.hist([ historical[0].flatten(),\
		    historical[i].flatten(),\
		    scenario[i].flatten()], color=["k","b","r"],\
		    normed=True, log=log)
		plt.title(models[i][0])
	fig.legend(["ERA5","Historical",\
			"RCP8.5"], bbox_to_anchor=(0.5, 0.05) , ncol=3, loc=10)
	plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.5)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png")
	

def load_qm(model_name, ensemble, p, lsm, experiment="historical"):

        if lsm:
                fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm_lsm.nc"
        else:
                fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm.nc"
        mod_xhat_hist = xr.open_dataset(fname)[p+"_qm_hist"].values
        mod_xhat_scenario = xr.open_dataset(fname)[p+"_qm_"+experiment].values

        return mod_xhat_hist, mod_xhat_scenario


def create_qm_combined(era5_da, model_da_hist, model_da_scenario, \
	    model_name, ensemble, p, lsm, replace_zeros, experiment="historical"):

        if lsm:
                fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm_lsm.nc"
        else:
                fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+experiment+"_"+ensemble+"_"+p+"_qm.nc"
        mod_xhat_hist, mod_xhat_scenario = \
		    qm_cmip_combined(era5_da, model_da_hist, model_da_scenario,\
		    replace_zeros)
        xr.Dataset(data_vars={p+"_qm_hist":\
			(("time_hist", "lat", "lon"), mod_xhat_hist),\
		    p+"_qm_"+experiment:\
			(("time_scenario","lat","lon"), mod_xhat_scenario)},\
                    coords={"time_hist":model_da_hist.time.values,\
			"time_scenario":model_da_scenario.time.values, \
			"lat":model_da_hist.lat, "lon":model_da_hist.lon}).\
                    to_netcdf(fname, mode="w")

        return mod_xhat_hist, mod_xhat_scenario

def load_all_qm_combined(data_hist, data_scenario, models, p, lsm, replace_zeros,\
	    experiment, force_compute=False):
                                
        out_qm_hist = []                         
        out_qm_scenario = []                         
        for i in np.arange(len(models)):    
                if i == 0:
                        out_qm_hist.append(data_hist[i].values)
                        out_qm_scenario.append(data_scenario[i].values)
                else:                   
                        if force_compute:
                                print(models[i][0])
                                model_xhat_hist, model_xhat_scenario\
					    = create_qm_combined(data_hist[0], \
					    data_hist[i], data_scenario[i],\
					    models[i][0],\
                                            models[i][1], p, lsm, replace_zeros,\
                                            experiment=experiment)
                        else:               
                                try:
                                        model_xhat_hist, model_xhat_scenario = \
						load_qm(models[i][0], models[i][1],\
						p, lsm, experiment=experiment)
                                except:
                                        print(models[i][0])
                                        model_xhat_hist, model_xhat_scenario = \
					    create_qm_combined(data_hist[0],\
					    data_hist[i],\
					    data_scenario[i], models[i][0],\
                                            models[i][1], p, lsm, replace_zeros, \
					    experiment=experiment)
                        out_qm_hist.append(model_xhat_hist)
                        out_qm_scenario.append(model_xhat_scenario)
        
        return out_qm_hist, out_qm_scenario

def qm_cmip_combined(era5_da, model_da1, model_da2, replace_zeros):

	#Take two model DataArrays (corresponding to a historical and scenario
	# period), combine them, and create a CDF function. Then, match values to 
	# an ERA5 CDF for each DataArray, individually. Taken from cmip_analysis()

	model_da = xr.concat([model_da1, model_da2], dim="time") 

	vals = model_da.values.flatten()
	vals = vals[~np.isnan(vals)]

	obs = era5_da.values.flatten()
	obs = obs[~np.isnan(obs)]
	obs_cdf = ECDF(obs)
	obs_invcdf = obs_cdf.x
        
        #Fit CDF to model
	model_cdf = ECDF(vals)
	model_p1 = np.interp(model_da1.values,\
                model_cdf.x,model_cdf.y)
	model_p2 = np.interp(model_da2.values,\
                model_cdf.x,model_cdf.y)
	model_xhat1 = np.interp(model_p1,obs_cdf.y,obs_invcdf)
	model_xhat2 = np.interp(model_p2,obs_cdf.y,obs_invcdf)
        
	if replace_zeros:
                model_xhat1[model_da1.values == 0] = 0
                model_xhat2[model_da2.values == 0] = 0

	return model_xhat1, model_xhat2

if __name__ == "__main__":

	#Set up models

	subplots=[4,2]
	p = "lr36"
	log = False
	experiment = "rcp85"
	force_cmip_regrid=False
	force_compute=False
	if p in ["srhe_left","ml_cape"]:
	        replace_zeros=True
	else:
                replace_zeros=False
	lsm=True
	rel_vmin=-20; rel_vmax=20
	abs_vmin=-1; abs_vmax=1
	models = [ ["ERA5",""] ,\
        #               ["ACCESS1-3","r1i1p1",5,""] ,\
        #               ["ACCESS1-0","r1i1p1",5,""] , \
        #               ["ACCESS-ESM1-5","r1i1p1f1",6,"CSIRO"] ,\
                        ["BNU-ESM","r1i1p1",5,""] , \
                        ["CNRM-CM5","r1i1p1",5,""] ,\
                        ["GFDL-CM3","r1i1p1",5,""] , \
                        ["GFDL-ESM2G","r1i1p1",5,""] , \
                        ["GFDL-ESM2M","r1i1p1",5,""] , \
                        ["IPSL-CM5A-LR","r1i1p1",5,""] ,\
        #                ["IPSL-CM5A-MR","r1i1p1",5,""] , \
                        ["MIROC5","r1i1p1",5,""] ,\
        #               ["MRI-CGCM3","r1i1p1",5,""], \
                        ["bcc-csm1-1","r1i1p1",5,""], \
                        ]

	#Load all the models for the historical period and the scenario given by "experiment"
	if p not in ["logit_sta","logit_aws"]:
                print("Loading re-gridded historical model data...")
                out_hist = load_model_data(models, p, lsm=lsm,\
			force_cmip_regrid=force_cmip_regrid,\
			experiment="historical") 
                print("Loading re-gridded scenario model data...")
                out_scenario = load_model_data(models, p, lsm=lsm,\
			force_cmip_regrid=force_cmip_regrid, \
			experiment=experiment, y2=2100, y1=2081) 
                print("Quantile mapping to ERA5...")
                out_qm_hist, out_qm_scenario = load_all_qm_combined(\
			out_hist, out_scenario, models, p, lsm, \
			replace_zeros, force_compute=force_compute, \
			experiment=experiment)

	#Compare the historical run to the scenario run
	print("Plotting...")
	plot_scenario_hist(out_qm_hist, out_qm_scenario, subplots, models, log, experiment+"/"+p+"_hist")
	plot_scenario_diff(out_qm_hist, out_qm_scenario, out_hist[0].lat.values,\
		out_hist[0].lon.values, subplots, models, log,\
		False, experiment+"/"+p+"_mean_spatial_abs_diff", vmin=abs_vmin, vmax=abs_vmax)
	plot_scenario_diff(out_qm_hist, out_qm_scenario, out_hist[0].lat.values,\
		out_hist[0].lon.values, subplots, models, log,\
		True, experiment+"/"+p+"_mean_spatial_rel_diff", vmin=rel_vmin, vmax=rel_vmax)
	plot_monthly_mean(out_qm_hist, out_qm_scenario, out_hist, out_scenario,\
		    subplots, models, p, experiment+"/"+p+"_monthly_mean")
