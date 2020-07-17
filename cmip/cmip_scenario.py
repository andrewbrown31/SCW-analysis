from scipy.stats import ttest_ind
from era5_read import get_mask
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
from cmip_analysis import load_model_data, load_era5, plot_mean_spatial_dist, daily_max_qm, str2bool, get_era5_lsm, qm_cmip_era5_loop, load_barpa

#Load the historical and scenario data for a range of CMIP models. 
#Join into one distribution, and quantile map onto ERA5 data.
#Compare the scenario run to the historical run.

def plot_monthly_mean(data_da, models, p, outname):

	#Taken from cmip_analysis(), except this function accepts a list of xarray datasets which consist
	# of 2d variables. Each variable is a mean for each month, computed from 6-hourly, quantile-
	# mapped CMIP data

	plt.figure(figsize=[12,7])
	ax=plt.gca()
	monthly_mean = pd.DataFrame()
	for i in np.arange(len(models)):
		mean = []
		for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]:
			mean.append(np.nanmean(data_da[i][m]))
		if i >= 1:
			try:
				r = str(np.corrcoef(mean, monthly_mean["ERA5 r=1"])[1,0].round(3))
			except:
				r = 0
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

def plot_mean_spatial_dist(data, p, models, subplots, log, lon, lat, lsm, outname, vmin=None, vmax=None):

	#Taken from cmip_analysis. However, this function accepts the dataframe created by save_mean(), which 
	# consists of 2d variables representing the mean for each month (and the total mean). This mean has been 
	# generated from 6-hourly data, quantile matched to ERA5.

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])
	era5_mean = data[0][p].values
	if lsm:
		era5_mean = era5_mean[~np.isnan(era5_mean)]

	x,y = np.meshgrid(lon,lat)
	for i in np.arange(len(models)):
			plt.subplot(subplots[0],subplots[1],i+1)
			m.drawcoastlines()
			mod_mean = data[i][p].values
			mod_mean1d = mod_mean.flatten()
			if lsm:
				mod_mean1d = mod_mean1d[~np.isnan(mod_mean1d)]
			try:
				r = str(np.corrcoef(era5_mean, mod_mean1d)[1,0].round(3))
			except:
				r = str(0)

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

def seasonal_freq_change_significance(hist_da_list, scenario_da_list,\
		threshold, models, p, lon, lat, y1, y2, experiment=""):

	for i in np.arange(len(hist_da_list)):
		hist_da = hist_da_list[i][p]
		scenario_da = scenario_da_list[i][p]

		xr.Dataset(data_vars={\
			    p:(("lat","lon"),\
				ttest_ind(\
				    (hist_da>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Jan":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==1]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==1]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Feb":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==2]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==2]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Mar":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==3]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==3]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Apr":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==4]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==4]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "May":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==5]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==5]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Jun":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==6]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==6]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Jul":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==7]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==7]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Aug":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==8]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==8]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Sep":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==9]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==9]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Oct":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==10]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==10]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Nov":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==11]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==11]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Dec":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[hist_da["time.month"]==12]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[scenario_da["time.month"]==12]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "JJA":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[6,7,8])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[6,7,8])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "SON":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[9,10,11])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[9,10,11])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "DJF":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[12,1,2])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[12,1,2])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "MAM":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[3,4,5])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[3,4,5])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Warm1":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[10,11,12,1,2,3])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[10,11,12,1,2,3])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Cool1":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[4,5,6,7,8,9])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[4,5,6,7,8,9])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Warm2":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[9,10,11,12,1,2,3,4])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[9,10,11,12,1,2,3,4])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    "Cool2":(("lat","lon"),\
				ttest_ind(\
				    (hist_da[np.in1d(hist_da["time.month"],[5,6,7,8])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
				    (scenario_da[np.in1d(scenario_da["time.month"],[5,6,7,8])]>=threshold).\
					    resample({"time":"1Y"}).sum("time"),\
					axis=0, equal_var=False)[1]),\
			    },\
			coords={"lat":lat,"lon":lon}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    models[i][0]+"_"+models[i][1]+"_sig_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc")

def save_seasonal_freq(data_list, da_list, threshold, models, p, lon, lat, y1, y2, experiment=""):

	for i in np.arange(len(data_list)):
		data = data_list[i]

		xr.Dataset(data_vars={\
			    p:(("lat","lon"), np.nansum((data >= threshold), axis=0) / (y2+1-y1)),\
			    "Jan":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==1] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Feb":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==2] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Mar":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==3] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Apr":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==4] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "May":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==5] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Jun":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==6] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Jul":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==7] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Aug":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==8] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Sep":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==9] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Oct":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==10] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Nov":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==11] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Dec":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==12] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "JJA":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[6,7,8])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "SON":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[9,10,11])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "DJF":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[12,1,2])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "MAM":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[3,4,5])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Warm1":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [10,11,12,1,2,3])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Cool1":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [4,5,6,7,8,9])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Warm2":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [9,10,11,12,1,2,3,4])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    "Cool2":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [5,6,7,8])] >= threshold), axis=0) / \
					((y2+1) - y1)),\
			    },\
			coords={"lat":lat,"lon":lon}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc")

def save_daily_freq(data_list, da_list, threshold, models, p, lon, lat, y1, y2, experiment=""):

	for i in np.arange(len(data_list)):
		data = data_list[i]

		xr.Dataset(data_vars={\
			    p:(("lat","lon"), np.nansum((data >= threshold), axis=0) / data.shape[0]),\
			    "Jan":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==1] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==1].shape[0]),\
			    "Feb":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==2] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==2].shape[0]),\
			    "Mar":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==3] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==3].shape[0]),\
			    "Apr":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==4] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==4].shape[0]),\
			    "May":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==5] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==5].shape[0]),\
			    "Jun":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==6] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==6].shape[0]),\
			    "Jul":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==7] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==7].shape[0]),\
			    "Aug":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==8] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==8].shape[0]),\
			    "Sep":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==9] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==9].shape[0]),\
			    "Oct":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==10] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==10].shape[0]),\
			    "Nov":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==11] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==11].shape[0]),\
			    "Dec":(("lat","lon"),\
				np.nansum((data[da_list[i]["time.month"]==12] >= threshold), axis=0) / \
					    data[da_list[i]["time.month"]==12].shape[0]),\
			    "JJA":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[6,7,8])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[6,7,8])].shape[0]),\
			    "SON":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[9,10,11])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[9,10,11])].shape[0]),\
			    "DJF":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[12,1,2])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[12,1,2])].shape[0]),\
			    "MAM":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],[3,4,5])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[3,4,5])].shape[0]),\
			    "Warm1":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [10,11,12,1,2,3])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[10,11,12,1,2,3])].shape[0]),\
			    "Cool1":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [4,5,6,7,8,9])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[4,5,6,7,8,9])].shape[0]),\
			    "Warm2":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [9,10,11,12,1,2,3,4])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[9,10,11,12,1,2,3,4])].shape[0]),\
			    "Cool2":(("lat","lon"),\
				np.nansum((data[np.in1d(da_list[i]["time.month"],\
					    [5,6,7,8])] >= threshold), axis=0) / \
					    data[np.in1d(da_list[i]["time.month"],[45,6,7,8])].shape[0]),\
			    },\
			coords={"lat":lat,"lon":lon}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    models[i][0]+"_"+models[i][1]+"_daily_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc")

def save_mean(data_list, da_list, models, p, lon, lat, y1, y2, experiment=""):

	for i in np.arange(len(data_list)):
		xr.Dataset(data_vars={\
			    p:(("lat","lon"), np.nanmean(data_list[i],axis=0)),\
			    "Jan":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==1],\
					axis=0)),
			    "Feb":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==2],\
					axis=0)),
			    "Mar":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==3],\
					axis=0)),
			    "Apr":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==4],\
					axis=0)),
			    "May":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==5],\
					axis=0)),
			    "Jun":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==6],\
					axis=0)),
			    "Jul":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==7],\
					axis=0)),
			    "Aug":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==8],\
					axis=0)),
			    "Sep":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==9],\
					axis=0)),
			    "Oct":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==10],\
					axis=0)),
			    "Nov":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==11],\
					axis=0)),
			    "Dec":(("lat","lon"),\
				np.nanmean(data_list[i][da_list[i]["time.month"]==12],\
					axis=0)),
			    },\
			coords={"lat":lat,"lon":lon}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    models[i][0]+"_"+models[i][1]+"_mean_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc")

def get_seasonal_sig(models, p, y1, y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(None)
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_sig_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def get_seasonal_freq(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def get_daily_freq(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_daily_freq_"+p+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_daily_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def get_mean(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_mean_"+p+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_mean_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def load_logit(model, ensemble, p, lsm, hist_y1, hist_y2, scenario_y1, scenario_y2,\
	experiment, force_compute, hist_save):

	hist_out = []
	scenario_out = []
	hist_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		model+"_"+ensemble+"_"+p+"_historical_"+\
		str(hist_y1)+"_"+str(hist_y2)+".nc"
	scenario_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		model+"_"+ensemble+"_"+p+"_"+experiment+"_"+\
		str(scenario_y1)+"_"+str(scenario_y2)+".nc"

	if (os.path.isfile(hist_fname)) & (~force_compute):
		print("Loading regridded "+model+" logistic model...")
		logit_hist = xr.open_dataset(hist_fname)
		if model == "ERA5":
			logit_scenario = logit_hist
		else:
			logit_scenario = xr.open_dataset(scenario_fname)
	else:
		print("Computing logit regression for "+model+"...")
		logit_hist, logit_scenario = calc_logit(model, ensemble, p, lsm, hist_y1,\
			hist_y2, scenario_y1, scenario_y2, experiment)
		if hist_save:
			xr.Dataset({p:logit_hist}).to_netcdf(hist_fname,\
				encoding={p:{"zlib":True, "complevel":1}})
		if model != "ERA5":
			xr.Dataset({p:logit_scenario}).to_netcdf(scenario_fname,\
				encoding={p:{"zlib":True, "complevel":1}})
	hist_out.append(logit_hist)
	scenario_out.append(logit_scenario)
	return hist_out, scenario_out

def calc_logit(model, ensemble, p, lsm, hist_y1, hist_y2, scenario_y1, scenario_y2,\
	experiment):

        #Load the regridded, quantile mapped model data, and calculate logistic model equations
	era5_lsm = get_era5_lsm()

	if p == "logit_aws":
		vars = ["lr36","mhgt","ml_el",\
			    "qmean01","srhe_left","Umean06"]
	elif p == "logit_sta":
		vars = ["lr36","ml_cape","srhe_left","Umean06"]
	elif p == "logit_aws_barra":
		vars = ["lr36","lr_freezing","ml_el",\
			    "s06","srhe_left","Umean06"]

	var_list_hist = []
	var_list_scenario = []
	print("Loading all variables for "+p+"...")
	for v in vars:
		print(v)
		if model == "ERA5":
			era5 = load_era5(v)
			if lsm:
				era5_trim = xr.where(era5_lsm,\
                                            era5.sel({"time":\
					    (era5["time.year"] <= hist_y2)&\
                                            (era5["time.year"] >= hist_y1)}),\
						 np.nan)
			else:
				era5_trim = era5.sel({"time":\
					    (era5["time.year"] <= hist_y2) & \
                                            (era5["time.year"] >= hist_y1)})
			out_hist = era5_trim
			out_scenario = era5_trim
		elif model == "BARPA":
			out_hist = load_barpa(v, hist_y1, hist_y2)
			out_scenario = load_barpa(v, scenario_y1, scenario_y2)
		else:
			try:
				out_hist, out_scenario = load_qm(\
					    model, ensemble, \
					    v, lsm, hist_y1, hist_y2,\
					    scenario_y1, scenario_y2,\
					    experiment=experiment)
			except:
				raise ValueError(v+" HAS NOT BEEN"+\
					    " QUANTILE MAPPED YET FOR "+\
					    model)
		var_list_hist.append(out_hist)
		var_list_scenario.append(out_scenario)

	out_logit = []
	for var_list in [var_list_hist, var_list_scenario]:
		if p == "logit_aws":
			z = 6.4e-1*var_list[0] - 1.2e-4*var_list[1] +\
				     4.4e-4*var_list[2] \
				    -1.0e-1*var_list[3] \
				    + 1.7e-2*var_list[4] \
				    + 1.8e-1*var_list[5] - 7.4
		elif p == "logit_sta":
			z = 3.3e-1*var_list[0] + 1.6e-3*var_list[1] +\
				     2.9e-2*var_list[2] \
				    +1.6e-1*var_list[3] - 4.5
		elif p == "logit_aws_barra":
			z = 8.5e-1*var_list[0] + 6.2e-1*var_list[1] +\
				     3.9e-4*var_list[2] \
				    + 3.8e-2*var_list[3] \
				    + 1.5e-2*var_list[4] \
				    + 1.6e-1*var_list[5] - 14.8
		out_logit.append( 1 / (1 + np.exp(-z)))
	logit_hist = out_logit[0]
	logit_scenario = out_logit[1]

	return logit_hist, logit_scenario

def plot_monthly_mean_scenario(historical_da, scenario_da, \
	    subplots, models, p, outname):

        fig = plt.figure(figsize=[12,7])
        ax=plt.gca()
        era5_mean = []
        for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]:
                era5_mean.append(np.nanmean(\
			historical_da[0][m]))
        for i in np.arange(1,len(models)):
                plt.subplot(subplots[0], subplots[1], i)
                mean_hist = []
                mean_scenario = []
                for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]:
                        mean_hist.append(np.nanmean(\
				historical_da[i][m]))
                        mean_scenario.append(np.nanmean(\
				scenario_da[i][m]))
                plt.plot(era5_mean, "k", marker="x")
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

def plot_scenario_diff(historical, scenario, season, lat, lon, \
	    subplots, models, log, rel_diff, outname,\
	    vmin=None, vmax=None, sig=None):

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
	        urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[11,9])
	x,y=np.meshgrid(lon,lat)
	for i in np.arange(1, len(models)): 
		plt.subplot(subplots[0],subplots[1],i) 
		if rel_diff: 
			vals = (scenario[i][season]-\
					historical[i][season])/\
                                        (historical[i][season]) * 100 
		else: 
			vals = (scenario[i][season])-\
				    (historical[i][season]) 
		p = m.pcolormesh(x,y,vals,\
			cmap=plt.get_cmap("RdBu_r"),vmin=vmin,vmax=vmax) 
		plt.title(models[i][0]) 
		m.drawcoastlines() 
		try:
			m.contourf(x,y,np.where(sig[i][season]<=.05, 1, 0), levels=[.5,1.5], \
			    colors=["none", "grey"], hatches=["////"], alpha=0)
		except:
			pass
	cax = plt.axes([0.2,0.15,0.6,0.025])
	c=plt.colorbar(p, cax=cax, orientation="horizontal") 
	if rel_diff: 
		c.set_label("%") 
	plt.subplots_adjust(bottom=0.2, top=0.7, hspace=0.075)
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
	

def load_qm(model_name, ensemble, p, lsm, hist_y1, hist_y2, scenario_y1, scenario_y2, experiment="historical"):

        if lsm:
                hist_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_historical_"+p+"_qm_lsm_"+str(hist_y1)+"_"+str(hist_y2)+".nc"
                scenario_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_"+experiment+"_"+p+"_qm_lsm_"+str(scenario_y1)+"_"+str(scenario_y2)+".nc"
        else:
                hist_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_historical_"+p+"_qm_"+str(hist_y1)+"_"+str(hist_y2)+".nc"
                scenario_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_"+experiment+"_"+p+"_qm_"+str(scenario_y1)+"_"+str(scenario_y2)+".nc"
        mod_xhat_hist = xr.open_dataset(hist_fname)[p]
        mod_xhat_scenario = xr.open_dataset(scenario_fname)[p]

        return mod_xhat_hist, mod_xhat_scenario


def create_qm_combined(era5_da, model_da_hist, model_da_scenario, \
	    model_name, ensemble, p, lsm, replace_zeros, hist_y1, hist_y2, scenario_y1, scenario_y2,\
	    save_hist_qm, experiment="historical",loop=True):

        if lsm:
                hist_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_historical_"+p+"_qm_lsm_"+str(hist_y1)+"_"+str(hist_y2)+".nc"
                scenario_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_"+experiment+"_"+p+"_qm_lsm_"+str(scenario_y1)+"_"+str(scenario_y2)+".nc"
        else:
                hist_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_historical_"+p+"_qm_"+str(hist_y1)+"_"+str(hist_y2)+".nc"
                scenario_fname = "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model_name+"_"+ensemble+"_"+experiment+"_"+p+"_qm_"+str(scenario_y1)+"_"+str(scenario_y2)+".nc"
        if loop:
                mod_xhat_hist, mod_xhat_scenario = \
		    qm_cmip_combined_loop(era5_da, model_da_hist, model_da_scenario,\
		    replace_zeros, get_mask(era5_da.lon.values, era5_da.lat.values))
        else:
                mod_xhat_hist, mod_xhat_scenario = \
		    qm_cmip_combined(era5_da, model_da_hist, model_da_scenario,\
		    replace_zeros)
        if save_hist_qm:
                xr.Dataset(data_vars={p:\
			(("time", "lat", "lon"), mod_xhat_hist)},\
                    coords={"time":model_da_hist.time.values,\
			"lat":model_da_hist.lat, "lon":model_da_hist.lon}).\
                    to_netcdf(hist_fname, mode="w",engine="h5netcdf",\
			    encoding={p:{"zlib":True, "complevel":9}})
        xr.Dataset(data_vars={p:\
	    (("time", "lat", "lon"), mod_xhat_scenario)},\
		    coords={"time":model_da_scenario.time.values,\
			"lat":model_da_scenario.lat, "lon":model_da_scenario.lon}).\
		    to_netcdf(scenario_fname, mode="w",engine="h5netcdf",\
			encoding={p:{"zlib":True, "complevel":9}})

        return mod_xhat_hist, mod_xhat_scenario

def load_all_qm_combined(data_hist, data_scenario, models, p, lsm, replace_zeros,\
	    experiment, hist_y1, hist_y2, scenario_y1, scenario_y2, force_compute=False, loop=True, save_hist_qm=False):
                                
        for i in np.arange(len(models)):    
                if models[i][0] == "ERA5":
                        save_mean([data_hist[i].values], [data_hist[i]], [models[i]], p,\
				data_hist[0].lon.values, data_hist[0].lat.values,\
				hist_y1, hist_y2, experiment="historical")
                elif models[i][0] == "BARPA":
                        save_mean([data_hist[i].values], [data_hist[i]], [models[i]], p,\
				data_hist[0].lon.values, data_hist[0].lat.values,\
				hist_y1, hist_y2, experiment="historical")
                        save_mean([data_scenario[i].values], [data_scenario[i]], [models[i]], p,\
				data_scenario[0].lon.values, data_scenario[0].lat.values,\
				scenario_y1, scenario_y2, experiment=experiment)
                else:                   
                        if force_compute:
                                print("Forcing QQ-mapping of "+models[i][0]+"...")
                                model_xhat_hist, model_xhat_scenario = \
					    create_qm_combined(data_hist[0], \
					    data_hist[i], data_scenario[i],\
					    models[i][0],\
                                            models[i][1], p, lsm, replace_zeros,\
					    hist_y1, hist_y2, scenario_y1, scenario_y2, save_hist_qm,\
                                            experiment=experiment, loop=loop)
                        else:               
                                try:
                                        print("Trying to load QQ-mapped model data from file...")
                                        model_xhat_hist, model_xhat_scenario = \
						load_qm(models[i][0], models[i][1],\
						p, lsm, hist_y1, hist_y2, scenario_y1, scenario_y2,\
						experiment=experiment)
                                        model_xhat_hist = model_xhat_hist.values
                                        model_xhat_scenario = model_xhat_scenario.values
                                except:
                                        print("Couldn't load, quantile-mapping "+models[i][0]+"...")
                                        model_xhat_hist, model_xhat_scenario = \
					    create_qm_combined(data_hist[0],\
					    data_hist[i],\
					    data_scenario[i], models[i][0],\
                                            models[i][1], p, lsm, replace_zeros,\
					    hist_y1, hist_y2, scenario_y1, scenario_y2, save_hist_qm,\
					    experiment=experiment)
                        save_mean([model_xhat_hist], [data_hist[i]], [models[i]], p,\
				data_hist[0].lon.values, data_hist[0].lat.values,\
				hist_y1, hist_y2, experiment="historical")
                        save_mean([model_xhat_scenario], [data_scenario[i]], [models[i]], p,\
				data_hist[0].lon.values, data_hist[0].lat.values,\
				scenario_y1, scenario_y2, experiment=experiment)

@jit
def qm_cmip_combined_loop(era5_da, model_da1, model_da2, replace_zeros, mask):

	vals1 = model_da1.values
	vals2 = model_da2.values
	obs = era5_da.values

	model_xhat1 = np.zeros(vals1.shape) * np.nan
	model_xhat2 = np.zeros(vals2.shape) * np.nan
	for m in np.arange(1,13):
		print(m)
		model_m_inds1 = (model_da1["time.month"] == m)
		model_m_inds2 = (model_da2["time.month"] == m)
		obs_m_inds = (era5_da["time.month"] == m)
		for i in np.arange(vals1.shape[1]):
			for j in np.arange(vals1.shape[2]):
				if mask[i,j] == 1:
				
					#Create the observed CDF
					obs_cdf = ECDF(obs[obs_m_inds,i,j])
					obs_invcdf, obs_p = ecdf_to_unique(obs_cdf)

					#Create the model CDF based on the historical experiment
					model_cdf = ECDF(vals1[model_m_inds1,i,j])
					model_invcdf = model_cdf.x
					model_p = model_cdf.y

					#Interpolate model values onto the historical model CDF probabilities
					model_p1 = np.interp(vals1[model_m_inds1,i,j],\
						model_invcdf,model_p)
					model_p2 = np.interp(vals2[model_m_inds2,i,j],\
						model_invcdf,model_p)

					#Interpolate the model CDF probabilities onto the observed CDF values
					model_xhat1[model_m_inds1,i,j] = \
						np.interp(model_p1,obs_p,obs_invcdf)
					model_xhat2[model_m_inds2,i,j] = \
						np.interp(model_p2,obs_p,obs_invcdf)
	if replace_zeros:
                model_xhat1[vals1 == 0] = 0
                model_xhat2[vals2 == 0] = 0
	model_xhat1[(model_xhat1>0) & (np.isinf(model_xhat1))] = np.max(model_xhat1)
	model_xhat1[(model_xhat1<0) & (np.isinf(model_xhat1))] = np.min(model_xhat1)
	model_xhat2[(model_xhat2>0) & (np.isinf(model_xhat2))] = np.max(model_xhat2)
	model_xhat2[(model_xhat2<0) & (np.isinf(model_xhat2))] = np.min(model_xhat2)

	return model_xhat1, model_xhat2

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

	parser = argparse.ArgumentParser(description='Post-processing of CMIP convective diagnostics, including'\
	    +" quantile mapping to ERA5 for historical and scenario runs")
	parser.add_argument("-p",help="Parameter",required=True)
	parser.add_argument("-m",help="Model",default="",nargs="+")
	parser.add_argument("-e",help="Experiment (e.g. rcp85). Note historical is always loaded",required=True)
	parser.add_argument("--threshold",help="If considering diagnostic indices, use a threshold",default=0,\
		type=float)
	parser.add_argument("--era5_y1",help="Start year for ERA5",default=1979,\
		type=int)
	parser.add_argument("--era5_y2",help="End year for ERA5",default=2005,\
		type=int)
	parser.add_argument("--hist_y1",help="Start year for CMIP historical period",default=1979,\
		type=int)
	parser.add_argument("--hist_y2",help="End year for CMIP historical period",default=2005,\
		type=int)
	parser.add_argument("--scenario_y1",help="Start year for CMIP scenario period",default=2081,\
		type=int)
	parser.add_argument("--scenario_y2",help="End year for CMIP scenario period",default=2100,\
		type=int)
	parser.add_argument("--loop",help="If True, then loop over each spatial point when QQ-matching",default=True,\
		type=str2bool)
	parser.add_argument("--force_compute",help="Force quantile mapping of CMIP data?",default=False,\
		type=str2bool)
	parser.add_argument("--force_cmip_regrid",help="Force regridding of CMIP data?",default=True,\
		type=str2bool)
	parser.add_argument("--save_hist_qm",help="Save the historical quantile matched CMIP data",default=False,\
		type=str2bool)
	parser.add_argument("--lsm",help="Mask ocean values using the ERA5 lsm?",default=True,\
		type=str2bool)
	parser.add_argument("--log",help="Make plots with a LogNorm color-scale, and log y-axis?",default=False,\
		type=str2bool)
	parser.add_argument("--mean_only",help="Instead of loading 3d time data, just load in the mean over each period"\
		,default=False, type=str2bool)
	parser.add_argument("--vmin",help="Minimum colour value for spatial distribution plot"\
		,default=None, type=float)
	parser.add_argument("--vmax",help="Maximum colour value for spatial distribution plot"\
		,default=None, type=float)
	parser.add_argument("--rel_vmin",help="Minimum colour value for relative difference plots"\
		,default=None, type=float)
	parser.add_argument("--rel_vmax",help="Maximum colour value for relative difference plots"\
		,default=None, type=float)
	parser.add_argument("--season",help="Season for scenario difference plots"\
		,default="", type=str)

	ProgressBar().register()
	args = parser.parse_args()
	subplots1=[4,4]
	subplots2=[3,4]
	p = args.p
	model = args.m
	experiment = args.e
	threshold = args.threshold
	era5_y1 = args.era5_y1
	era5_y2 = args.era5_y2
	hist_y1 = args.hist_y1
	hist_y2 = args.hist_y2
	scenario_y1 = args.scenario_y1
	scenario_y2 = args.scenario_y2
	log = args.log
	save_hist_qm=args.save_hist_qm
	force_cmip_regrid=args.force_cmip_regrid
	force_compute=args.force_compute
	mean_only=args.mean_only
	if p in ["srhe_left","ml_cape"]:
	        replace_zeros=True
	else:
                replace_zeros=False
	lsm=args.lsm
	vmin = args.vmin
	vmax = args.vmax
	rel_vmin=args.rel_vmin; rel_vmax=args.rel_vmax
	season = args.season
	if season == "":
		season = p
	models = [ ["ERA5",""] ,\
			["ACCESS1-3","r1i1p1",5,""] ,\
			["ACCESS1-0","r1i1p1",5,""] , \
			["BNU-ESM","r1i1p1",5,""] , \
			["CNRM-CM5","r1i1p1",5,""] ,\
			["GFDL-CM3","r1i1p1",5,""] , \
			["GFDL-ESM2G","r1i1p1",5,""] , \
			["GFDL-ESM2M","r1i1p1",5,""] , \
			["IPSL-CM5A-LR","r1i1p1",5,""] ,\
			["IPSL-CM5A-MR","r1i1p1",5,""] , \
			["MIROC5","r1i1p1",5,""] ,\
			["MRI-CGCM3","r1i1p1",5,""], \
			["bcc-csm1-1","r1i1p1",5,""], \
			["ACCESS-ESM1-5", "r1i1p1f1", 6, ""], \
			["ACCESS-CM2", "r1i1p1f1", 6, ""],\
			["BARPA", ""]
                        ]
	if model == "":
		model = ["ERA5", "ACCESS1-3", "ACCESS1-0", "BNU-ESM", "CNRM-CM5", "GFDL-CM3", \
			    "GFDL-ESM2G", "GFDL-ESM2M", "IPSL-CM5A-LR", "IPSL-CM5A-MR", \
			    "MIROC5", "MRI-CGCM3", "bcc-csm1-1"]
	models = list(np.array(models)[np.in1d([m[0] for m in models], model)])

	#Load all the models for the historical period and the scenario given by "experiment"
	if not mean_only:
		if p in ["logit_sta","logit_aws","logit_aws_barra"]:
			out_qm_hist = []; out_qm_scenario = []
			for i in np.arange(len(models)):
				a, b = load_logit(models[i][0], models[i][1], p, lsm,\
				    hist_y1, hist_y2, scenario_y1, scenario_y2,\
				    experiment, force_compute, hist_save=save_hist_qm)
				out_qm_hist.append(a[0])
				out_qm_scenario.append(b[0])
			#From quantile-matched 6-hourly model data, compute the daily maximum
			try:
				out_hist_dmax = daily_max_qm([ds[p] for ds in out_qm_hist], out_qm_hist, models, p, lsm)
			except:
				out_hist_dmax = daily_max_qm([ds for ds in out_qm_hist], out_qm_hist, models, p, lsm)
			save_seasonal_freq([ds[p].values for ds in out_hist_dmax], \
				    out_hist_dmax, threshold, models, p,\
				    out_qm_hist[0].lon.values, out_qm_hist[0].lat.values,\
				    hist_y1, hist_y2, experiment="historical")
			try:
				out_scenario_dmax = daily_max_qm([ds[p] for ds in out_qm_scenario], \
					out_qm_scenario, models, p, lsm)
			except:
				out_scenario_dmax = daily_max_qm([ds for ds in out_qm_scenario], \
					out_qm_scenario, models, p, lsm)
			save_seasonal_freq([ds[p].values for ds in out_scenario_dmax], \
				    out_scenario_dmax, threshold, models, p,\
				    out_qm_scenario[0].lon.values, out_qm_scenario[0].lat.values,\
				    scenario_y1, scenario_y2, experiment=experiment)
			seasonal_freq_change_significance(out_hist_dmax, out_scenario_dmax, threshold, models, p,
				    out_qm_scenario[0].lon.values, out_qm_scenario[0].lat.values,\
				    scenario_y1, scenario_y2, experiment=experiment)
				    
		elif p in ["dcp","t_totals"]:
			print("Loading re-gridded historical model data...")
			out_hist = load_model_data(models, p, lsm=lsm,\
				force_cmip_regrid=force_cmip_regrid,\
				experiment="historical", era5_y1=era5_y1, era5_y2=era5_y2,\
				y1=hist_y1, y2=hist_y2, save=False) 
			print("Loading re-gridded scenario model data...")
			out_scenario = load_model_data(models, p, lsm=lsm,\
				force_cmip_regrid=force_cmip_regrid, \
				experiment=experiment, y2=scenario_y2, \
				y1=scenario_y1, era5_data=out_hist[0], save=False) 
			print("Quantile mapping to ERA5...")
			out_qm_hist, out_qm_scenario = load_all_qm_combined(\
				out_hist, out_scenario, models, p, lsm, \
				replace_zeros, experiment, \
				hist_y1, hist_y2, scenario_y1, scenario_y2, \
				force_compute=force_compute, save_hist_qm=save_hist_qm)
			out_hist_dmax = daily_max_qm( out_qm_hist, out_hist, models, p, lsm)
			save_daily_freq([ds[p].values for ds in out_hist_dmax], \
				    out_hist_dmax, threshold, models, p,\
				    out_hist_dmax[0].lon.values, out_hist_dmax[0].lat.values,\
				    hist_y1, hist_y2, experiment="historical")
			del out_hist_dmax, out_qm_hist
			out_scenario_dmax = daily_max_qm(out_qm_scenario, \
					out_scenario, models, p, lsm)
			save_daily_freq([ds[p].values for ds in out_scenario_dmax], \
				    out_scenario_dmax, threshold, models, p,\
				    out_scenario_dmax[0].lon.values, out_scenario_dmax[0].lat.values,\
				    scenario_y1, scenario_y2, experiment=experiment)
			del out_scenario_dmax, out_qm_scenario
		else:
			if (force_compute) & (["ERA5",""] not in models):
				if models != [["BARPA",""]]:
					models = [ ["ERA5", ""] ]+models
				print(models)
			print("Loading re-gridded historical model data...")
			#Load all models (given my "models"), and regrid to the ERA5 spatial
			# grid, for historical runs only
			out_hist = load_model_data(models, p, lsm=lsm,\
				force_cmip_regrid=force_cmip_regrid,\
				experiment="historical", era5_y1=era5_y1, era5_y2=era5_y2,\
				y1=hist_y1, y2=hist_y2, save=False) 
			[print(models[i], out_hist[i].shape) for i in np.arange(len(models))]
			#For all models, load CMIP data for the run given by "experiment".
			# Parse "era5_data" into this function, instead of loading again
			print("Loading re-gridded scenario model data...")
			out_scenario = load_model_data(models, p, lsm=lsm,\
			    force_cmip_regrid=force_cmip_regrid, \
			    experiment=experiment, y2=scenario_y2, \
			    y1=scenario_y1, era5_data=out_hist[0], save=False) 
			[print(x.shape) for x in out_scenario]
			#For each model, either (a) load quantile-matched CMIP data for the 
			# given experiment/years if it has been done previously or (b) 
			# quantile-match a combined historical-scenario distribution at each
			# spatial point to ERA5. Save the output for the puroposes of computing the 
			# logistic model diagnostic
			print("Quantile mapping to ERA5...")
			load_all_qm_combined(\
				out_hist, out_scenario, models, p, lsm, \
				replace_zeros, experiment, \
				hist_y1, hist_y2, scenario_y1, scenario_y2, \
				force_compute=force_compute, save_hist_qm=save_hist_qm)
	else:

		try:
			hist = get_seasonal_freq(models, p, hist_y1, hist_y2, hist_y1, hist_y2,\
			    experiment="historical")
			scenario = get_seasonal_freq(models, p, scenario_y1, scenario_y2, hist_y1,\
			    hist_y2, experiment=experiment)
			sig = get_seasonal_sig(models, p, scenario_y1, scenario_y2, experiment)
			plot_mean_spatial_dist(hist, p, models, subplots1, log,\
			    hist[0].lon.values, hist[0].lat.values, lsm,\
			    experiment+"/"+p+"_daily_freq", vmin=vmin, vmax=vmax)
		except:
			hist = get_mean(models, p, hist_y1, hist_y2, era5_y1, era5_y2, experiment="historical")
			scenario = get_mean(models, p, scenario_y1, scenario_y2, era5_y1, era5_y2, experiment=experiment)
			sig=None

		plot_monthly_mean_scenario(hist, scenario, subplots2, models, p,\
			experiment+"/"+p+"_monthly_mean_change_"+str(scenario_y1)+"_"+str(scenario_y2))

