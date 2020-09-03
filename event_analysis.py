import sys
import datetime as dt
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib
#from plot_param import contour_properties
#from plot_clim import *
import pandas as pd
import numpy as np
from scipy.stats import spearmanr as spr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import multiprocessing
import glob
import xarray as xr
from obs_read import read_clim_ind

def auc_test():

	#For the ten candidate variables (11 including both S03 and S06) chosen, calculate their ability to separate 1) convective and 
	# non convective gusts (for MLCAPE, MLEL, SRHE) and 2) Severe convective gusts and non-severe convective gusts (Umean06, S03,
	# S06, LR36, LR-Freezing, MHGT, Qmean01, Qmean-subcloud)

	#Do this by calculating the AUC, with confidence intervals attained by bootstrapping.

	era5, df_aws_era5, df_sta_era5 = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",\
                         T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5",\
                         time="floor") 
	barra, df_aws_barra, df_sta_barra = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",\
                         T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc",\
                         time="floor") 
	df_aws_era5["is_lightning"] = (df_aws_era5["lightning"] >= 2)*1
	df_aws_barra["is_lightning"] = (df_aws_barra["lightning"] >= 2)*1
	df_aws_era5.loc[(df_aws_era5["lightning"] >= 2) & ~(df_aws_era5["is_conv_aws"]==1), "is_conv_aws_cond_light"] = 0
	df_aws_barra.loc[(df_aws_barra["lightning"] >= 2) & ~(df_aws_barra["is_conv_aws"]==1), "is_conv_aws_cond_light"] = 0
	
	vlist1 = ["ml_el","eff_el","ml_cape","dcape","srhe_left","Uwindinf","lr13","lr36","rhmin13","U1","Umean06","Umean03","dpd850"]
	vlist2 = ["ml_el","eff_el","ml_cape","dcape","srhe_left","Uwindinf","lr13","lr36","rhmin13","U1","Umean06","Umean03","dpd850"]
	#vlist1 = ["ml_cape", "ml_el", "srhe_left"]
	#vlist2 = ["Umean06", "s03", "s06", "lr36", "lr_freezing", "mhgt", "qmean01", "qmeansubcloud"]
	df1 = pd.DataFrame(columns=["era5_auc","era5_ci","barra_auc","barra_ci"], index=vlist1)
	df2 = pd.DataFrame(columns=["era5_auc","era5_ci","barra_auc","barra_ci"], index=vlist2)

	N=1000

	for v in vlist1:
		print(v)
		a, c1, c2 = calc_auc((df_aws_era5), v, "is_conv_aws", N)
		df1.loc[v, "era5_auc"] = a; df1.loc[v, "era5_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

		a, c1, c2 = calc_auc((df_sta_era5), v, "is_sta", N)
		df2.loc[v, "era5_auc"] = a; df2.loc[v, "era5_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

		a, c1, c2 = calc_auc((df_aws_barra), v, "is_conv_aws", N)
		df1.loc[v, "barra_auc"] = a; df1.loc[v, "barra_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

		a, c1, c2 = calc_auc((df_sta_barra), v, "is_sta", N)
		df2.loc[v, "barra_auc"] = a; df2.loc[v, "barra_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

	#for v in vlist1:
	#	print(v)
	#	a, c1, c2 = calc_auc(df_aws_era5, v, "is_lightning", N)
	#	df1.loc[v, "era5_auc"] = a; df1.loc[v, "era5_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

	#	a, c1, c2 = calc_auc(df_aws_barra, v, "is_lightning", N)
	#	df1.loc[v, "barra_auc"] = a; df1.loc[v, "barra_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

	#for v in vlist2:
	#	print(v)
	#	a, c1, c2 = calc_auc(df_aws_era5.dropna(subset=["is_conv_aws_cond_light"]), v, "is_conv_aws_cond_light", N)
	#	df2.loc[v, "era5_auc"] = a; df2.loc[v, "era5_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

#		a, c1, c2 = calc_auc(df_aws_barra.dropna(subset=["is_conv_aws_cond_light"]), v, "is_conv_aws_cond_light", N)
#		df2.loc[v, "barra_auc"] = a; df2.loc[v, "barra_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

	df1.to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/scw_aws_auc.csv")
	df2.to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/scw_sta_auc.csv")

def calc_auc(df, v, event, N=100):
	if v in ["mhgt","qmean01","qmeansubcloud","dpd850"]:
		df[event] = np.where(df[event]==1, 0, 1)
	fpr, tpr, thresh = roc_curve(df[event], df[v])
	auc_val = auc(fpr, tpr)
	auc_resample = []
	for i in np.arange(N):
		temp_df = resample(df[[event, v]])
		fpr, tpr, _ = roc_curve(temp_df[event], temp_df[v])
		auc_resample.append(auc(fpr, tpr))
	return auc_val, np.percentile(auc_resample, 0.5), np.percentile(auc_resample, 99.5)

def diagnostics_aws_compare():

	#Plot 2-d histogram comparing ~4 diagnostics from BARRA and ERA5 with measured gust speeds

	#Load model data and obs at station locations.
	barra = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl")
	era5 = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl")
	obs = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/convective_wind_gust_aus_2005_2018.pkl")
	obs["hourly_ceil_utc"] = pd.DatetimeIndex(obs["gust_time_utc"]).ceil("1H")

	#Merge model data and obs into same dataframe. For diagnostic wind gust, use the ceiling of the hourly gust time (as diagnostic is defined as maximum in the previous hour). Otherwise, use the floor (environmental diagnostics are essentially forecasting tools).
	barra_sta_wg = pd.merge(obs[["stn_name","wind_gust","hourly_ceil_utc","tc_affected","lightning","is_sta"]],\
                barra, how="left",left_on=["stn_name","hourly_ceil_utc"], right_on=["loc_id","time"]).dropna(subset=["ml_cape"])
	barra_aws_wg = barra_sta_wg.dropna(subset=["wind_gust"])
	barra_sta_p = pd.merge(obs[["stn_name","wind_gust","hourly_floor_utc","tc_affected","lightning","is_sta"]],\
                barra, how="left",left_on=["stn_name","hourly_floor_utc"], right_on=["loc_id","time"]).dropna(subset=["ml_cape"])
	barra_aws_p = barra_sta_p.dropna(subset=["wind_gust"])
	era5_sta_wg = pd.merge(obs[["stn_name","wind_gust","hourly_ceil_utc","tc_affected","lightning","is_sta"]],\
                era5, how="left",left_on=["stn_name","hourly_ceil_utc"], right_on=["loc_id","time"]).dropna(subset=["ml_cape"])
	era5_aws_wg = era5_sta_wg.dropna(subset=["wind_gust"])
	era5_sta_p = pd.merge(obs[["stn_name","wind_gust","hourly_floor_utc","tc_affected","lightning","is_sta"]],\
                era5, how="left",left_on=["stn_name","hourly_floor_utc"], right_on=["loc_id","time"]).dropna(subset=["ml_cape"])
	era5_aws_p = era5_sta_p.dropna(subset=["wind_gust"])

	#Load HSS and thresholds
	barra_hss_floor, temp, temp1 = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"barra_allvars_2005_2018_2.pkl",\
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc") 
	era5_hss_floor, temp, temp1 = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"era5_allvars_2005_2018.pkl",\
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5")
	barra_hss_ceil, temp, temp1 = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"barra_allvars_2005_2018_2.pkl",time="ceil",\
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc_2") 
	era5_hss_ceil, temp, temp1 = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"era5_allvars_2005_2018.pkl",time="ceil",\
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5")

	#Plotting settings
	matplotlib.rcParams.update({'font.size': 14})
	params = ["wg10","t_totals","dcp","mlcape*s06"]
	fancy_names = ["WindGust10", "T-Totals", "DCP", "MLCS6"]
	is_log = [False, False, .01, 10]
	ylims = [ [0,40], [0,65], [0,2], [0,400000] ]
	bins = [ np.linspace(0,40,20), np.linspace(0,65,20), \
		    np.concatenate([ [0], np.logspace(-2,1,19)] ), np.concatenate([ [0], np.logspace(1,6,19)] ) ] 
	label_a = ["a)","c)","e)","g)"]
	label_b = ["b)","d)","f)","h)"]
	plt.figure(figsize=[8,8])

	#Plot for each parameter
	for p in np.arange(len(params)):
		plt.subplot(4,2,2*p+1)
		plt.text(0.05,0.9,label_a[p],transform=plt.gca().transAxes,fontsize=14)
		if params[p] == "wg10":
			plt.hist2d(era5_aws_wg.loc[:, "wind_gust"], era5_aws_wg.loc[:, params[p]], cmap=plt.get_cmap("YlGnBu"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=10000, bins=[20,bins[p]], range = [[0,40],ylims[p]])
			plt.plot(era5_aws_wg.loc[(era5_aws_wg["wind_gust"]>=25) & (era5_aws_wg["lightning"]>=2), "wind_gust"], \
				era5_aws_wg.loc[(era5_aws_wg["wind_gust"]>=25) & (era5_aws_wg["lightning"]>=2), params[p]], "kx",\
				markersize=4)
			plt.gca().axhline(era5_hss_ceil.loc[params[p], "threshold_conv_aws"], color="k", linestyle="--")
			plt.text(15, era5_hss_ceil.loc[params[p], "threshold_conv_aws"], \
				str(round(era5_hss_ceil.loc[params[p], "threshold_conv_aws"], 2) ),\
				va="top", color="k")
		else:
			plt.hist2d(era5_aws_p.loc[:, "wind_gust"], era5_aws_p.loc[:, params[p]], cmap=plt.get_cmap("YlGnBu"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=10000, bins=[20,bins[p]], range = [[0,40],ylims[p]])
			plt.plot(era5_aws_p.loc[(era5_aws_p["wind_gust"]>=25) & (era5_aws_p["lightning"]>=2), "wind_gust"], \
				era5_aws_p.loc[(era5_aws_p["wind_gust"]>=25) & (era5_aws_p["lightning"]>=2), params[p]], "kx",\
				markersize=4)
			plt.gca().axhline(era5_hss_floor.loc[params[p], "threshold_conv_aws"], color="k", linestyle="--")
			if params[p] == "mlcape*s06":
				plt.text(15, era5_hss_floor.loc[params[p], "threshold_conv_aws"], \
					str(int(round(era5_hss_floor.loc[params[p], "threshold_conv_aws"]) )),\
					va="top", color="k")
			else:
				plt.text(15, era5_hss_floor.loc[params[p], "threshold_conv_aws"], \
					str(round(era5_hss_floor.loc[params[p], "threshold_conv_aws"], 2) ),\
					va="top", color="k")
		if is_log[p]:
			plt.yscale("symlog",linthreshy=is_log[p])

		if p < 3:
			plt.gca().set_xticklabels("")
		else:
			plt.xlabel("Measured gust speed ($m.s^{-1}$)")
		plt.ylabel(fancy_names[p])
		if p == 0:
			plt.title("ERA5")

		plt.subplot(4,2,2*p+2)
		plt.text(0.05,0.9,label_b[p],transform=plt.gca().transAxes,fontsize=14)
		if params[p] == "wg10":
			plt.hist2d(barra_aws_wg.loc[:, "wind_gust"], barra_aws_wg.loc[:, params[p]], cmap=plt.get_cmap("YlGnBu"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=10000, bins=[20,bins[p]], range = [[0,40],ylims[p]])
			plt.plot(barra_aws_wg.loc[(barra_aws_wg["wind_gust"]>=25) & (barra_aws_wg["lightning"]>=2), "wind_gust"], \
				barra_aws_wg.loc[(barra_aws_wg["wind_gust"]>=25) & (barra_aws_wg["lightning"]>=2), params[p]], "kx",\
				markersize=4)
			plt.gca().axhline(barra_hss_ceil.loc[params[p], "threshold_conv_aws"], linestyle="--", color="k")
			plt.text(15, barra_hss_ceil.loc[params[p], "threshold_conv_aws"], \
				str(round(barra_hss_ceil.loc[params[p], "threshold_conv_aws"], 2) ),\
				va="top", color="k")
		else:
			plt.hist2d(barra_aws_p.loc[:, "wind_gust"], barra_aws_p.loc[:, params[p]], cmap=plt.get_cmap("YlGnBu"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=10000, bins=[20,bins[p]], range = [[0,40],ylims[p]])
			plt.plot(barra_aws_p.loc[(barra_aws_p["wind_gust"]>=25) & (barra_aws_p["lightning"]>=2), "wind_gust"], \
				barra_aws_p.loc[(barra_aws_p["wind_gust"]>=25) & (barra_aws_p["lightning"]>=2), params[p]], "kx",\
				markersize=4)
			plt.gca().axhline(barra_hss_floor.loc[params[p], "threshold_conv_aws"], linestyle="--", color="k")
			if params[p] == "mlcape*s06":
				plt.text(15, barra_hss_floor.loc[params[p], "threshold_conv_aws"], \
					str(int(round(barra_hss_floor.loc[params[p], "threshold_conv_aws"]) )),\
					va="top", color="k")
			else:
				plt.text(15, barra_hss_floor.loc[params[p], "threshold_conv_aws"], \
					str(round(barra_hss_floor.loc[params[p], "threshold_conv_aws"], 2) ),\
					va="top", color="k")
		if is_log[p]:
			plt.yscale("symlog",linthreshy=is_log[p])
	    
		if p < 3:
			plt.gca().set_xticklabels("")
		else:
			plt.xlabel("Measured gust speed ($m.s^{-1}$)")
		plt.gca().set_yticklabels("")
		if p == 0:
			plt.title("BARRA")
	ax = plt.axes([0.2, 0.08, 0.6, 0.02])
	c = plt.colorbar(cax=ax, orientation="horizontal")
	c.set_label("Days")
	plt.subplots_adjust(top=0.95, bottom=0.2)
	plt.savefig("fig3.png",bbox_inches="tight")


def compare_obs_soundings():

	#Compare observed soundings (2005 - 2015) at Adelaide AP, Woomera, Darwin and Sydney to
	# ERA-Interim (and later, ERA5 and BARRA-R)

	barra = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl")
	era5 = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl")
	obs = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/UA_wrfpython.pkl").\
		reset_index().rename(columns={"index":"time"})
	temp, barra_wg10, barra_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018_2.pkl",
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc") 
	temp, era5_wg10, barra_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",
		T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5") 
	stn_map = {14015:"Darwin",66037:"Sydney",16001:"Woomera",23034:"Adelaide"}
	obs["stn_name"] = obs["stn_id"].map(stn_map)
	mod_cols = ["k_index","t_totals","ml_cape","s06","ml_el","dcape","ml_cin","Umean800_600","Umean01","time","loc_id"]
	df = pd.merge(barra[mod_cols], era5[mod_cols], suffixes=("_barra", "_era5"), on=["time","loc_id"])
	df = pd.merge(df, obs, left_on=["time","loc_id"], right_on=["time","stn_name"]).dropna()
	#df.loc[:,"s06"] = df.loc[:,"s06"] * 1.94384
	#df.loc[:,"Umean800_600"] = df.loc[:,"Umean800_600"] * 1.94384

	locs = ["Adelaide", "Woomera", "Sydney", "Darwin"]
	stn_id = [23034, 16001, 66037, 14015]
	params = ['ml_cape','ml_el','ml_cin','t_totals','dcape','s06','Umean800_600','Umean01']
	
	is_log = {"k_index":False, "ml_cape":True, "ml_cin":True, "mu_cape":True, "ml_el":False,\
			"t_totals":False, "dcape":False, "wg10":False, "s06":False, "Umean800_600":False,"Umean01":False}
	rename_params = {"ml_cape":"MLCAPE", "ml_el":"MLEL", "k_index":"K-index", "t_totals":"T-totals",\
		"dcape":"DCAPE", "ml_cin":"MLCIN", "Umean800_600":"Umean800-600", "s06":"S06", "wg10":"WindGust10","Umean01":"Umean01"}
	pmax = {"ml_cape":10000, "ml_el":17000, "k_index":50, "t_totals":60,\
		"dcape":2500, "ml_cin":1000, "Umean800_600":40, "s06":60, "wg10":45,"Umean01":40}
	rename_mod = {"era5":"ERA5","barra":"BARRA"}
	cnt=1
	fig = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
	plt.figure(figsize=[18,10])
	for i in itertools.product(params, ["_barra","_era5"]):

		ax = plt.subplot(4,4,cnt)
	
		col = "".join(i)
		p, m = i
		m = m[1:]

		print(col)

		if p in ["k_index","t_totals"]:
			df.loc[df[p] < 0,p] = 0
			df.loc[df[col] < 0,col] = 0
		if is_log[p]:
			plt.hist2d(df.loc[:, p], df.loc[:, col], cmap=plt.get_cmap("Greys"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=1000, bins=np.concatenate([[0],np.logspace(0,4,20)]),\
				range = [0,pmax[p]])
			plt.xscale("symlog");plt.yscale("symlog")

		else:
			plt.hist2d(df.loc[:, p], df.loc[:, col], cmap=plt.get_cmap("Greys"), \
				norm=matplotlib.colors.SymLogNorm(1),\
				vmax=1000, bins=20,\
				range = [ [0,pmax[p]], [0,pmax[p]]])
		#plt.colorbar()
		xb, xt = plt.gca().get_xlim()
		yb, yt = plt.gca().get_ylim()
		plt.text(xb, yt, \
			"\n r = "+str(round(spr(df.loc[:, p], df.loc[:, col]).correlation, 3)), \
			horizontalalignment="left", va="center", fontsize=14)
		plt.plot([min(xb, yb), max(xt,yt)], [min(xb, yb), max(xt,yt)], color="r")
		plt.title(fig[cnt-1] + ") " + rename_mod[m] + " " + rename_params[p])
		cnt = cnt+1

	plt.subplots_adjust(hspace=0.5,wspace=0.4)
	ax = plt.axes([0.2, 0.04, 0.6, 0.02])
	plt.text(0.5, 0.0675, "Observed sounding", fontsize=14, transform=plt.gcf().transFigure,\
		horizontalalignment="center")
	plt.text(0.08, 0.5, "Model sounding", fontsize=14, transform=plt.gcf().transFigure,\
		rotation=90, verticalalignment="center")
	plt.colorbar(cax=ax, orientation="horizontal")
	plt.savefig("figA1.png",bbox_inches="tight")

def create_mean_variable(variable, model, native=False, native_dir=None):

	#From sub-daily model data (barra_fc, barpa_erai, era5), compute the monthly mean of a variable.
	# Do this for houry and daily maximum data
	#If native, then get data from the ub4 surface variables directory. Need to then specify the 
	# "native_dir" name for the variable, which is different from the variable within the netcdf 
	# file (e.g. for 2 m dew point, native_dir="2D" but variable="d2m")
	
	if native:
		files = np.array(glob.glob("/g/data/ub4/era5/netcdf/surface/"+native_dir+"/*/*", \
			recursive=True))
		f_years = np.array([int(files[i].split("_")[3][0:4]) for i in np.arange(len(files))])
		files = files[f_years < 2019]
	else:
		files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/"+model+"/"+model+"*")
	files.sort()

	#Initialise output
	ds = xr.open_dataset(files[0])
	if native:
		ds = ds.sel({"latitude":(ds.latitude <= -9.975) & (ds.latitude >= -44.525), \
			"longitude":(ds.longitude <= 156.275) & (ds.longitude >= 111.975)}, "nearest")
		out = np.zeros((len(files),ds.latitude.shape[0], ds.longitude.shape[0]))	
		out_daily = np.zeros((len(files),ds.latitude.shape[0], ds.longitude.shape[0]))	
	else:
		out = np.zeros((len(files),ds.lat.shape[0], ds.lon.shape[0]))	
		out_daily = np.zeros((len(files),ds.lat.shape[0], ds.lon.shape[0]))	
	times = []
	steps = []

	#For each file, calculate the mean
	for f in np.arange(len(files)):
		ds = xr.open_dataset(files[f])
		start = dt.datetime.now()
		print(files[f])
		if native:
			ds = ds.isel({"time":np.in1d(ds["time.hour"].values, [0,6,12,18])})
			ds = ds.sel({"latitude":(ds.latitude <= -9.975) & \
				(ds.latitude >= -44.525)\
				,"longitude":(ds.longitude <= 156.275) & \
				(ds.longitude >= 111.975)}, "nearest")
		out[f] = ds[variable].mean("time").values
		daily_var = ds[variable].resample({"time":"1D"}).max("time")
		out_daily[f] = daily_var.mean("time").values
		steps.append(ds.time.shape[0])
		times.append(ds.time[0].values)
		ds.close()
		print(dt.datetime.now() - start)

	#Save 2d variable (the total number of environments per grid box) as a netcdf, with attribute giving
	# the number of time steps used
	if native:
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out,\
			coords = {"time":times,"lat":ds.latitude.values,"lon":ds.longitude.values},\
			name = variable,\
			attrs = {"steps":steps} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						"era5_"+variable+"_6hr_mean.nc",\
					mode = "w",\
					format = "NETCDF4")
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out_daily,\
			coords = {"time":times,"lat":ds.latitude.values,"lon":ds.longitude.values},\
			name = variable,\
			attrs = {"steps":steps} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						"era5_"+variable+"_6hr_daily_max_mean.nc",\
					mode = "w",\
					format = "NETCDF4")
	else:
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = variable,\
			attrs = {"steps":steps} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_"+variable+"_6hr_mean.nc",\
					mode = "w",\
					format = "NETCDF4")
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out_daily,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = variable,\
			attrs = {"steps":steps} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_"+variable+"_6hr_daily_max_mean.nc",\
					mode = "w",\
					format = "NETCDF4")

def create_annmax_variable(var, start_year, end_year):

	#From ERA5, compute the annual maximum of a variable over between start year and end year. 
	#Do this for each season as well as a total mean, for hourly and daily maximum data

	files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5*")
	files.sort()
	files = np.array(files)
	years = np.array([int(files[i][45:49]) for i in np.arange(len(files))])
	ds = xr.open_mfdataset(files[(years >= start_year) & (years <= end_year)])
	seasons = [np.arange(1,13),[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	for v in var:
		da_out = []
		for s in seasons:
			print(s)
			annmax = ds.sel({"time":np.in1d(ds["time.month"],s)})[v].resample({"time":"1Y"}).max("time")
			da_out.append(annmax)

	xr.Dataset({"ANN":da_out[0], "DJF":da_out[1], "MAM":da_out[2], "JJA":da_out[3], "SON":da_out[4]}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/era5_"+v+"_annmax_"+\
		str(start_year)+"_"+str(end_year)+".nc")

def create_mjo_phase_variable(variable, threshold, model, event=None, predictors=None):

	#For each monthly environmental diagnostic file, count the number of threshold occurrences
	# for each active MJO phase.

	#List BARRA files
	if model == "barra":
		files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc*")
	elif model == "era5":
		files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5*")
	else:
		raise ValueError("INVALID MODEL NAME")
	files.sort()

	#Load MJO data
	mjo = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/clim_ind/mjo_rmm.txt", \
		header=1,  delim_whitespace=True,names=["year","month","day","rmm1","rmm2","phase",\
		"amplitude","a","b","c","d"],index_col=False).iloc[:,0:7]
	mjo.loc[:,"datetime"] = pd.to_datetime({"year":mjo.year,"month":mjo.month,"day":mjo.day})
	mjo = mjo.loc[(mjo["datetime"] >= dt.datetime(1979,1,1)) \
		& (mjo["datetime"]<dt.datetime(2019,1,1)),:]

	#Train logistic equation
	if variable == "logit":
		from event_analysis import optimise_pss 
		if model == "barra":
			pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"barra_allvars_2005_2018.pkl",\
				T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc") 
		elif model == "era5":
			pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"era5_allvars_2005_2018.pkl",\
				T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5") 
		from sklearn.linear_model import LogisticRegression 
		logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
		try:
			if event == "is_conv_aws":
				logit_mod = logit.fit(mod_aws[predictors], mod_aws[event])
			elif event == "is_sta":
				logit_mod = logit.fit(mod_sta[predictors], mod_sta[event])
				
		except:
			raise ValueError("IF TRAINING LOGISTIC MODEL, EVENT AND PREDICTORS MUST BE SPECIFIED")

	#Initialise output
	ds = xr.open_dataset(files[0])
	out = np.zeros((len(files),ds.lat.shape[0], ds.lon.shape[0]))	
	out_mjo = {"1":[], "2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[]}	
	out_times = {"1":[], "2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[]}	
	mjo_days = {"1":[], "2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[]}
	times = {"1":0, "2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0}
	for m in np.arange(1,9):
		times[str(m)] = mjo[(mjo["phase"]==m) & (mjo["amplitude"]>=1)]["datetime"].values

	#For each file, apply the equation
	for f in np.arange(len(files)):
		start = dt.datetime.now()
		print(files[f])
		ds = xr.open_dataset(files[f])
		if variable == "logit":
			if len(predictors) == 6:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 4:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 7:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] \
					+ ds[predictors[6]] * logit_mod.coef_[0][6] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 8:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] \
					+ ds[predictors[6]] * logit_mod.coef_[0][6] \
					+ ds[predictors[7]] * logit_mod.coef_[0][7] +\
					logit_mod.intercept_) ) ) 
			else:
				raise ValueError("FUNCTION ONLY HANDLES 6 OR 8 PREDICTORS")
			daily_logit = (logit >= threshold).resample({"time":"1D"}).max("time")
			for m in np.arange(1,9):
				out_mjo[str(m)].append(daily_logit.sel(\
				    {"time":np.in1d(daily_logit.time, times[str(m)])}).values*1)
				out_times[str(m)].append(daily_logit.time[np.in1d(daily_logit.time, \
					times[str(m)])].values)
		else:
			daily_var = (ds[variable] >= threshold).resample({"time":"1D"}).max("time")
			for m in np.arange(1,9):
				out_mjo[str(m)].append(daily_var.sel(\
				    {"time":np.in1d(daily_var.time, times[str(m)])}).values*1)
				out_times[str(m)].append(daily_var.time[np.in1d(daily_var.time, \
					times[str(m)])].values)

		ds.close()

	#Save the output
	if variable == "logit":
		attrs = {"predictors":predictors,\
				"coefs":logit_mod.coef_[0],\
				"intercept":logit_mod.intercept_, "threshold":threshold}
		path = "/g/data/eg3/ab4502/ExtremeWind/aus/mjo_full_phases_"+model+"_logit_"+\
			event+".nc"
	else:
		attrs = {"threshold":threshold}
		path = "/g/data/eg3/ab4502/ExtremeWind/aus/mjo_full_phases_"+model+"_"+variable+\
			".nc"
	
	xr.Dataset(data_vars = \
		{"phase1": ( ("phase1_times","lat","lon"), np.vstack(out_mjo["1"]) ),\
		"phase2": ( ("phase2_times","lat","lon"), np.vstack(out_mjo["2"]) ),\
		"phase3": ( ("phase3_times","lat","lon"), np.vstack(out_mjo["3"]) ),\
		"phase4": ( ("phase4_times","lat","lon"), np.vstack(out_mjo["4"]) ),\
		"phase5": ( ("phase5_times","lat","lon"), np.vstack(out_mjo["5"]) ),\
		"phase6": ( ("phase6_times","lat","lon"), np.vstack(out_mjo["6"]) ),\
		"phase7": ( ("phase7_times","lat","lon"), np.vstack(out_mjo["7"]) ),\
		"phase8": ( ("phase8_times","lat","lon"), np.vstack(out_mjo["8"]) ) }, \
	    coords = {"phase1_times":np.concatenate(out_times["1"]), \
		"phase2_times":np.concatenate(out_times["2"]), \
		"phase3_times":np.concatenate(out_times["3"]), \
		"phase4_times":np.concatenate(out_times["4"]), \
		"phase5_times":np.concatenate(out_times["5"]), \
		"phase6_times":np.concatenate(out_times["6"]), \
		"phase7_times":np.concatenate(out_times["7"]), \
		"phase8_times":np.concatenate(out_times["8"]), \
		"lat":ds.lat.values, "lon":ds.lon.values} , attrs=attrs).\
		    to_netcdf(\
			path = path,\
			mode = "w",\
			format = "NETCDF4")



def create_threshold_variable(variable, threshold, model, event=None, predictors=None):

	#For a given variable and threshold, create a dataset for BARRA/ERA5 which gives the spatial
	# occurence frequency (hourly and daily). Also save the number of environments on 
	# MJO active and inactive days
	#If the variable is equal to "logit", then train a logistic equation for convective 
	# wind gust event prediction, and then apply the equation to the reanalysis. For logit 
	# with BARRA/convective AWS, have been using a threshold of 0.87 (for POD > 0.67)

	#List BARRA files
	if model == "barra":
		files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc*")
	elif model == "era5":
		files = glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5*")
	else:
		raise ValueError("INVALID MODEL NAME")
	files.sort()

	#Load MJO data
	mjo = read_clim_ind("mjo")
	mjo_active = mjo[mjo["active"]==1].index
	mjo_inactive = mjo[mjo["inactive"]==1].index

	#Train logistic equation
	if variable == "logit":
		from event_analysis import optimise_pss 
		if model == "barra":
			pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"barra_allvars_2005_2018.pkl",\
				T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc") 
		elif model == "era5":
			pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"era5_allvars_2005_2018.pkl",\
				T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5") 
		from sklearn.linear_model import LogisticRegression 
		logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
		try:
			if event == "is_conv_aws":
				logit_mod = logit.fit(mod_aws[predictors], mod_aws[event])
			elif event == "is_sta":
				logit_mod = logit.fit(mod_sta[predictors], mod_sta[event])
				
		except:
			raise ValueError("IF TRAINING LOGISTIC MODEL, EVENT AND PREDICTORS MUST BE SPECIFIED")

	#Initialise output
	ds = xr.open_dataset(files[0])
	out = np.zeros((len(files),ds.lat.shape[0], ds.lon.shape[0]))	
	out_mjo_active = []	
	out_mjo_inactive = []	
	mjo_active_days = 0
	mjo_inactive_days = 0
	out_daily = np.zeros((len(files),ds.lat.shape[0], ds.lon.shape[0]))	
	times = []
	mjo_active_times = []
	mjo_inactive_times = []
	steps = []

	#For each file, apply the equation
	for f in np.arange(len(files)):
		start = dt.datetime.now()
		print(files[f])
		ds = xr.open_dataset(files[f])
		ds = ds.isel({"time":np.in1d(ds["time.hour"].values, [0,6,12,18])})
		if variable == "logit":
			if len(predictors) == 6:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 4:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 7:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] \
					+ ds[predictors[6]] * logit_mod.coef_[0][6] +\
					logit_mod.intercept_) ) ) 
			elif len(predictors) == 8:
				logit = 1 / ( 1 + np.exp( -(\
					ds[predictors[0]] * logit_mod.coef_[0][0] \
					+ ds[predictors[1]] * logit_mod.coef_[0][1] \
					+ ds[predictors[2]] * logit_mod.coef_[0][2] \
					+ ds[predictors[3]] * logit_mod.coef_[0][3] \
					+ ds[predictors[4]] * logit_mod.coef_[0][4] \
					+ ds[predictors[5]] * logit_mod.coef_[0][5] \
					+ ds[predictors[6]] * logit_mod.coef_[0][6] \
					+ ds[predictors[7]] * logit_mod.coef_[0][7] +\
					logit_mod.intercept_) ) ) 
			else:
				raise ValueError("FUNCTION ONLY HANDLES 6 OR 8 PREDICTORS")
			out[f] = ( (logit >= threshold).sum("time") ).values
			daily_logit = (logit >= threshold).resample({"time":"1D"}).max("time")
			out_daily[f] = daily_logit.sum("time").values
			out_mjo_active.append(daily_logit.sel(\
				{"time":np.in1d(daily_logit.time, mjo_active)}).values*1)
			out_mjo_inactive.append(daily_logit.sel(\
				{"time":np.in1d(daily_logit.time, mjo_inactive)}).values*1)
			mjo_active_days = mjo_active_days + np.in1d(daily_logit.time, mjo_active).sum()
			mjo_inactive_days = mjo_inactive_days + np.in1d(daily_logit.time, mjo_inactive).sum()
			mjo_active_times.append(\
				daily_logit.time[np.in1d(daily_logit.time,mjo_active)].values)
			mjo_inactive_times.append(\
				daily_logit.time[np.in1d(daily_logit.time,mjo_inactive)].values)
		else:
			out[f] = ( (ds[variable] >= threshold).sum("time") ).values
			daily_var = (ds[variable] >= threshold).resample({"time":"1D"}).max("time")
			out_daily[f] = daily_var.sum("time").values
			out_mjo_active.append(daily_var.sel(\
				{"time":np.in1d(daily_var.time, mjo_active)}).values*1)
			out_mjo_inactive.append(daily_var.sel(\
				{"time":np.in1d(daily_var.time, mjo_inactive)}).values*1)
			mjo_active_days = mjo_active_days + np.in1d(daily_var.time, mjo_active).sum()
			mjo_inactive_days = mjo_inactive_days + np.in1d(daily_var.time, mjo_inactive).sum()
			mjo_active_times.append(\
				daily_var.time[np.in1d(daily_var.time,mjo_active)].values)
			mjo_inactive_times.append(\
				daily_var.time[np.in1d(daily_var.time,mjo_inactive)].values)
    
		steps.append(ds.time.shape[0])
		times.append(ds.time[0].values)
		ds.close()
		print(dt.datetime.now() - start)

	#Save 2d variable (the total number of environments per grid box) as a netcdf, with attribute giving
	# the number of time steps used
	if variable == "logit":
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = "logit",\
			attrs = {"steps":steps, "predictors":predictors, "coefs":logit_mod.coef_[0],\
				"intercept":logit_mod.intercept_, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+model+"_logit_6hr_"+event+".nc",\
					mode = "w",\
					format = "NETCDF4")
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out_daily,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = "logit",\
			attrs = {"steps":steps, "predictors":predictors, "coefs":logit_mod.coef_[0],\
				"intercept":logit_mod.intercept_, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_logit_6hr_"+event+"_daily.nc",\
					mode = "w",\
					format = "NETCDF4")
		mjo_active_da = xr.DataArray( dims = ("time","lat","lon"),\
			data = np.concatenate(out_mjo_active),\
			coords = {"lat":ds.lat.values,"lon":ds.lon.values,\
				"time":np.concatenate(mjo_active_times)},\
			name = "logit",\
			attrs = {"mjo_days":mjo_active_days, "predictors":predictors,\
				"coefs":logit_mod.coef_[0],\
				"intercept":logit_mod.intercept_, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_logit_6hr_"+event+"_mjo_active.nc",\
					mode = "w",\
					format = "NETCDF4")

		mjo_inactive_da = xr.DataArray( dims = ("time","lat","lon"),\
			data = np.concatenate(out_mjo_inactive),\
			coords = {"lat":ds.lat.values,"lon":ds.lon.values,\
				"time":np.concatenate(mjo_inactive_times)},\
			name = "logit",\
			attrs = {"mjo_days":mjo_inactive_days, "predictors":predictors,\
				"coefs":logit_mod.coef_[0],\
				"intercept":logit_mod.intercept_, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_logit_6hr_"+event+"_mjo_inactive.nc",\
					mode = "w",\
					format = "NETCDF4")
	else:
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = variable,\
			attrs = {"steps":steps, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_"+variable+"_6hr_"+str(threshold)+".nc",\
					mode = "w",\
					format = "NETCDF4")
		xr.DataArray( dims = ("time","lat","lon"),\
			data = out_daily,\
			coords = {"time":times,"lat":ds.lat.values,"lon":ds.lon.values},\
			name = variable,\
			attrs = {"steps":steps, "threshold":threshold} ).to_netcdf(\
					path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						model+"_"+variable+"_6hr_"+str(threshold)+"_daily.nc",\
					mode = "w",\
					format = "NETCDF4")
		#mjo_active_da = xr.DataArray( dims = ("time","lat","lon"),\
			#data = np.concatenate(out_mjo_active),\
			#coords = {"lat":ds.lat.values,"lon":ds.lon.values,\
				#"time":np.concatenate(mjo_active_times)},\
			#name = variable,\
			#attrs = {"mjo_days":mjo_active_days, "threshold":threshold} ).to_netcdf(\
					#path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						#model+"_"+variable+"_6hr_"+str(threshold)+"_mjo_active.nc",\
					#mode = "w",\
					#format = "NETCDF4")

		#mjo_inactive_da = xr.DataArray( dims = ("time","lat","lon"),\
			#data = np.concatenate(out_mjo_inactive),\
			#coords = {"lat":ds.lat.values,"lon":ds.lon.values,\
				#"time":np.concatenate(mjo_inactive_times)},\
			#name = variable,\
			#attrs = {"mjo_days":mjo_inactive_days, "threshold":threshold} ).to_netcdf(\
					#path = "/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/"+\
						#model+"_"+variable+"_6hr_"+str(threshold)+"_mjo_inactive.nc",\
					#mode = "w",\
					#format = "NETCDF4")
		

def plot_hail_envs():

	#Doesn't actually plot - just saves csv files
	#Need to check how to define diurnal data. Is it using the environments for the whole "day" on which 
	# at least one hourly observation of hail or lightning is made? Or is it using the environmnents only for 
	# hourly observations? If using the whole day, then should this be defined as 0-23 UTC? This may not make
	# sense physically.

	df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/hail_mhgt_radar_barra_2005_2018.pkl")
	df.loc[:,"hour"] = df.set_index("time").index.hour
	df.loc[:,"month"] = df.set_index("time").index.month

	wwlln = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/hail_ad/WWLLN_for_hailpaper_100.csv")
	wwlln.loc[:,"datetime"] = pd.to_datetime({"year":wwlln["year"], "month":wwlln["month"], \
		"day":wwlln["day"]})
	wwlln.loc[:,"hourly_dt"] = pd.to_datetime({"year":wwlln["year"], "month":wwlln["month"], \
		"day":wwlln["day"], "hour":wwlln["hour"]})

	path = "/g/data/eg3/ab4502/ExtremeWind/obs/hail_ad/"
	mesh_path = {"Melbourne":path+"MESH_for_hailpaper_02.csv",\
		"Wollongong":path+"MESH_for_hailpaper_03.csv",\
		"Gympie":path+"MESH_for_hailpaper_08.csv",\
		"Grafton":path+"MESH_for_hailpaper_28.csv",\
		"Canberra":path+"MESH_for_hailpaper_40.csv",\
		"Marburg":path+"MESH_for_hailpaper_50.csv",\
		"Adelaide":path+"MESH_for_hailpaper_64.csv",\
		"Namoi":path+"MESH_for_hailpaper_69.csv",\
		"Perth":path+"MESH_for_hailpaper_70.csv",\
		"Hobart":path+"MESH_for_hailpaper_76.csv"}

	for v in ["wbz","mhgt"]:
	
		mesh_out = pd.DataFrame(index=np.arange(0,24), columns=list(mesh_path.keys()))
		wwlln_out = pd.DataFrame(index=np.arange(0,24), columns=list(mesh_path.keys()))

		for loc in np.unique(df.loc_id):
			print(loc)

			#Restrict environment dataframe to the current location
			temp_df = df[df.loc_id==loc][[v, "time", "loc_id", "hour", "month"]]
			temp_df = temp_df.set_index(pd.DatetimeIndex(temp_df.time, freq="1H"))
			dates = pd.to_datetime({"year":temp_df.index.year, "month":temp_df.index.month, \
				"day":temp_df.index.day})

			#Get dates from the hail dataset
			hail = pd.read_csv(mesh_path[loc])
			#hail.loc[:,"datetime"] = pd.to_datetime({"year":hail["year"], "month":hail["month"], \
			#	"day":hail["day"]})
			#hail_dates = hail[(hail[loc]==1) & (hail["year"]>=2005) & (hail["year"] < 2019)].datetime
			#hail_inds = np.in1d(dates, hail_dates)
			hail.loc[:,"hourly_dt"] = pd.to_datetime({"year":hail["year"], "month":hail["month"], \
				"day":hail["day"],"hour":hail["hour"]})
			hail_hourly_dates = hail[(hail[loc]==1) & (hail["year"]>=2005) & (hail["year"] < 2019)].\
						hourly_dt
			hail_hourly_inds = np.in1d(temp_df.time, hail_hourly_dates)

			#Get dates from the WWLLN dataset
			#wwlln_dates = wwlln[(wwlln[loc]==1)].datetime
			wwlln_dates = wwlln[(wwlln[loc]==1)].hourly_dt
			#wwlln_inds = np.in1d(dates, wwlln_dates)
			wwlln_inds = np.in1d(temp_df.time, wwlln_dates)

			#daily_temp_df = temp_df.resample("1D").max()

			#hourly_hail = temp_df.loc[hail_inds,:].groupby(["hour"]).\
			#	agg({v:"mean"})
			#hourly_wwlln = temp_df.loc[wwlln_inds,:].groupby(["hour"]).\
			#	agg({v:"mean"})
			hourly_hail = temp_df.loc[hail_hourly_inds,:]
			hourly_wwlln = temp_df.loc[wwlln_inds,:]
			hours = np.arange(0,24)
			hail_vals = []
			wwlln_vals = []
			for h in hours:
				hail_vals.append(hourly_hail.loc[hourly_hail.hour==h,v].mean())
				wwlln_vals.append(hourly_wwlln.loc[hourly_wwlln.hour==h,v].mean())

			#hourly_hail.loc[:,v].plot(ax=ax1, color=plt.get_cmap("Paired")(0))
			#hourly_wwlln.loc[:,v].plot(ax=ax1, color=plt.get_cmap("Paired")(1))
			#ax1.tick_params(labelcolor=plt.get_cmap("Paired")(1),axis="y")

			#plt.title(loc)

			#cnt = cnt+1

			#mesh_out.loc[:,loc] = hourly_hail.values[:,0]
			#wwlln_out.loc[:,loc] = hourly_wwlln.values[:,0]
			mesh_out.loc[:,loc] = hail_vals
			wwlln_out.loc[:,loc] = wwlln_vals
		mesh_out.to_csv(path+"hail_barra/diurnal_mesh_"+v+".csv")
		wwlln_out.to_csv(path+"hail_barra/diurnal_wwlln100_"+v+".csv")
	
	months = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
	for v in ["mhgt","wbz"]:

		mesh_out = pd.DataFrame(index=np.arange(1,13), columns=list(mesh_path.keys()))
		wwlln_out = pd.DataFrame(index=np.arange(1,13), columns=list(mesh_path.keys()))

		for loc in np.unique(df.loc_id):
			print(loc)

			temp_df = df[df.loc_id==loc][[v, "time", "loc_id", "hour", "month"]]
			temp_df = temp_df.set_index(pd.DatetimeIndex(temp_df.time, freq="1H"))
			dates = pd.to_datetime({"year":temp_df.index.year, "month":temp_df.index.month, \
				"day":temp_df.index.day})

			hail = pd.read_csv(mesh_path[loc])
			hail.loc[:,"datetime"] = pd.to_datetime({"year":hail["year"], "month":hail["month"], \
				"day":hail["day"]})
			hail_dates = hail[(hail[loc]==1) & (hail["year"]>=2005) & (hail["year"] < 2019)].datetime

			wwlln_dates = wwlln[(wwlln[loc]==1)].datetime

			daily_temp_df = temp_df.resample("1D").max()

			monthly_hail = []
			monthly_wwlln = []
			for m in months:
				monthly_hail.append(temp_df.loc[(np.in1d(temp_df.time,hail_dates)) & \
					(temp_df.month==m),:].mean()[v])
				monthly_wwlln.append(temp_df.loc[(np.in1d(temp_df.time,wwlln_dates)) & \
					(temp_df.month==m),:].mean()[v])
			monthly_hail = np.array(monthly_hail)
			monthly_wwlln = np.array(monthly_wwlln)
			monthly_hail[monthly_hail==np.nan] = 0
			monthly_wwlln[monthly_wwlln==np.nan] = 0

			mesh_out.loc[:,loc] = monthly_hail
			wwlln_out.loc[:,loc] = monthly_wwlln
		mesh_out.to_csv(path+"hail_barra/monthly_mesh_"+v+".csv", na_rep="-999")
		wwlln_out.to_csv(path+"hail_barra/monthly_wwlln100_"+v+".csv", na_rep="-999")
	

def plot_multivariate_density(df, event, param1, param2, param3, param4, param5=None, special_cond=None, log=False,\
		param12_cond=None):

	#Plot a density plot with events highlighted, for qunitiles of four parameters

	if special_cond == "deep":
		dfs = [ df[df["ml_el"]>=6000] ]
		conds = ["Deep convective"]
		events_max = 1000
	elif special_cond == "shallow":
		dfs = [ df[df["ml_el"]<6000] ]
		conds = ["Shallow convective"]
		events_max = 3000
	elif special_cond == "mf":
		dfs = [ df[df["Umean800_600"]<26] ]
		conds = ["Mesoscale forced (MF)"]
		events_max = 2000
	elif special_cond == "latitude":
		dfs = [ df[df["lat"].values[:,1]>=-28], df[df["lat"].values[:,1]<-28] ]
		conds = ["Tropical/Sub-tropical (>= 28 S)", "Mid-latitudes (< 28 S)"]
		events_max = 2000
	elif special_cond == "sf":
		dfs = [ df[df["Umean800_600"]>=26] ]
		conds = ["Synoptic forced (SF)"]
		events_max = 100
	elif special_cond == "warm":
		dfs = [ df[np.in1d(df["month"], [11,12,1,2,3,4] )] ]
		conds = ["Warm season (November - April)"]
		events_max = 2000
	elif special_cond == "cool":
		dfs = [ df[np.in1d(df["month"], [5,6,7,8,9,10] )] ]
		conds = ["Cool season (May - October)"]
		events_max = 2000
	else:
		dfs = [df]
		conds = ["All events"]
		events_max = 2500

	cond_cnt = 0
	for df in dfs:

		events = df[event].sum()

		thresh3 = np.quantile(df.loc[:,param3],[.0,.25,.50,.75] )
		thresh4 = np.quantile(df.loc[:,param4],[.0,.25,.50,.75] )

		plt.figure(figsize=[12,8])
		cnt=0
		max_pss = -1
		max_param1 = -1
		max_param2 = -1
		max_param3 = -1
		max_param4 = -1
		for k in np.arange(thresh3.shape[0]):
			print(k)
			for l in np.arange(thresh4.shape[0]):

				plt.subplot(len(thresh3),len(thresh4),cnt+1)

				hits = df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & \
					(df[param4]>=thresh4[l]) ][event].sum()

				if param12_cond is not None:
					hits_cond = df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & \
						(df[param4]>=thresh4[l]) & (df[param1]>=param12_cond[0]) \
						& (df[param2]>=param12_cond[1]) ][event].sum()
					plt.axvline(param12_cond[0], color="b", linestyle="--")
					plt.axhline(param12_cond[1], color="b", linestyle="--")

				if log:
					t1, t2, t3, d = plt.hist2d(df.loc[ (df[event]==0) & (df[param3]>=thresh3[k])\
						 & (df[param4]>=thresh4[l]), param1], \
						df.loc[ (df[event]==0) & (df[param3]>=thresh3[k]) & \
						(df[param4]>=thresh4[l]), param2], \
						cmap = plt.get_cmap("Greys"), vmax = events_max,\
						bins=[np.linspace(0,df[param1].max(),10), \
							np.concatenate(([0],np.logspace(0,4,10)))] )
					plt.yscale("symlog")
					plt.ylim([-.5,10000])
				else:
					t1, t2, t3, d = plt.hist2d(df.loc[ (df[event]==0) & (df[param3]>=thresh3[k])\
						 & (df[param4]>=thresh4[l]), param1], \
						df.loc[ (df[event]==0) & (df[param3]>=thresh3[k]) & \
						(df[param4]>=thresh4[l]), param2], \
						cmap = plt.get_cmap("Greys"), vmax = events_max )
				if param5 is None:
					plt.plot(df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & (df[param4]>=thresh4[l]), param1], \
						df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & (df[param4]>=thresh4[l]), param2] , color="r",\
						label = str(round(hits / float(events), 3)), markeredgecolor="k", \
						linestyle="none", marker="o")
					if param12_cond is not None:
						plt.plot(df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) &\
							(df[param4]>=thresh4[l]) & (df[param1]>=param12_cond[0]) \
							& (df[param2]>=param12_cond[1]), param1], \
							df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) &\
							(df[param4]>=thresh4[l])\
							& (df[param1]>=param12_cond[0]) & \
							(df[param2]>=param12_cond[1]) , param2] , color="r",\
							label = str(round(hits_cond / float(events), 3)),\
							markeredgecolor="b", \
							linestyle="none", marker="o")
				else:
					s = plt.scatter(df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & (df[param4]>=thresh4[l]), param1], \
						df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & (df[param4]>=thresh4[l]), param2] ,\
						c = df.loc[ (df[event]==1) & (df[param3]>=thresh3[k]) & (df[param4]>=thresh4[l]), param5], \
						label = str(round(hits / float(events), 3)), cmap=plt.get_cmap("Reds", 5), vmin=0, vmax=16000,\
						edgecolors="k")
				plt.legend()
				#plt.contourf(thresh1,thresh2,pss,levels=np.linspace(0,0.8,11))
				#plt.clabel(cs,np.linspace(0,0.8,20))
				plt.xlabel(param1)
				plt.ylabel(param2)
				plt.title(param4 +">=" +str(round(thresh4[l],2))+" "+param3 +">=" +str(round(thresh3[k],2)))
				if (k != 3):
					ax = plt.gca()
					ax.set_xticklabels("")
					plt.xlabel("")
					
				if (l != 0):
					ax = plt.gca()
					ax.set_yticklabels("")
					plt.ylabel("")
						
				if (l==3) & (k==3):
					#For the last subplot, set the density count colorbar at the bottom
					cax = plt.axes( (0.2, 0.045, 0.6, 0.01) )
					cb = plt.colorbar(d, cax, orientation="horizontal")
					cb.set_label("Non-events")

					if param5 is None:
						pass
					else:
						cax2 = plt.axes( (0.92, 0.2, 0.01, 0.6) )
						cb2 = plt.colorbar(s, cax2)
						cb2.set_label(param5)

				if param2=="t_totals":
					if plt.ylim()[0] < 0:
						plt.ylim(bottom=0)
	
				cnt=cnt+1

		plt.suptitle(conds[cond_cnt])
		cond_cnt = cond_cnt+1

	#plt.colorbar()

	#return [max_pss, max_param1, max_param2, max_param3, max_param4]

def test_pss(df, pss_df, param_list, param_type, event, T=1000):

	#For a given list of parameters in "df", optimise the PSS score by combining the parameters,
	# either conditionall ("param_type" = "cond") or by multiplication ("multiply"). The PSS will
	# be optimised for the "event" in "df", and appended to "pss_df"	


	import multiprocessing
	import itertools
	pool = multiprocessing.Pool()

	if param_type == "multiply":
	
		new_param = "*".join(param_list)

		df.loc[:,new_param] = df[param_list].product(axis=1)
		test_thresh = np.linspace(df.loc[:,new_param].min(), np.percentile(df.loc[:,new_param],\
			99.95) , T)
		temp_df = df.loc[:, [event, new_param]]
		iterable = itertools.product(test_thresh, [temp_df], [new_param], [event], [True])
		res = pool.map(pss, iterable)
		thresh = [res[i][1] for i in np.arange(len(res))]
		pss_p = [res[i][0] for i in np.arange(len(res))]

		if event == "is_lightning":
			pss_df.loc[new_param, "threshold_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[new_param, "pss_light"] = np.array(pss_p).max()
		if event == "is_conv_aws":
			pss_df.loc[new_param, "threshold_conv_aws"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[new_param, "pss_conv_aws"] = np.array(pss_p).max()
		if event == "is_conv_aws_cond_light":
			pss_df.loc[new_param, "threshold_conv_aws_cond_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[new_param, "pss_conv_aws_cond_light"] = np.array(pss_p).max()
		if event == "is_sta":
			pss_df.loc[new_param, "threshold_sta"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[new_param, "pss_sta"] = np.array(pss_p).max()

	if (param_type == "cond_both") | (param_type == "cond_or"):

		if param_type == "cond_both":
			new_param = "_and_".join(param_list)		
		elif param_type == "cond_or":
			new_param = "_or_".join(param_list)		
		if T >= 50:
			T = 50
	
		conds = list()
		thresholds = list()

		for p in np.arange(0,len(param_list)):

			test_thresh = np.linspace(df.loc[:,param_list[p]].min(), \
				np.percentile(df.loc[:,param_list[p]], 99.95) , T)
			conds.append(np.zeros([df.shape[0], len(test_thresh)], dtype="bool"))
			thresholds.append(test_thresh)
			for t in np.arange(0,len(test_thresh)):
				conds[p][:,t] = ((df[param_list[p]]>=test_thresh[t]))

		conds = np.hstack(conds)
		thresholds = np.hstack(thresholds)
		pss_cur = -1
		threshold_cur = ""
		df.loc[:,new_param] = False
		inds = np.split(np.arange(len(thresholds)), len(param_list))
		iterable = itertools.product(*inds) 

		for i in iterable:
			print(i)
			if param_type == "cond_both":
				temp = np.all(conds[:,i], axis=1)
			elif param_type == "cond_or":
				temp = np.any(conds[:,i], axis=1)
			hits = float(((df[event]==1) & (temp)).sum())
			misses = float(((df[event]==1) & (~temp)).sum())
			fa = float(((df[event]==0) & (temp)).sum())
			cn = float(((df[event]==0) & (~temp)).sum())
			pss_ge = (hits / (hits+misses)) - (fa / (fa + cn))
			print(pss_ge)
			if pss_ge > pss_cur:
				pss_cur = pss_ge
				threshold_cur = ", ".join((thresholds[i,]).round(2).astype(str))
				
		if event == "is_lightning":
			pss_df.loc[new_param, "threshold_light"] = threshold_cur
			pss_df.loc[new_param, "pss_light"] = pss_cur
		if event == "is_conv_aws":
			pss_df.loc[new_param, "threshold_conv_aws"] = threshold_cur
			pss_df.loc[new_param, "pss_conv_aws"] = pss_cur
		if event == "is_conv_aws_cond_light":
			pss_df.loc[new_param, "threshold_conv_aws_cond_light"] = threshold_cur
			pss_df.loc[new_param, "pss_conv_aws_cond_light"] = pss_cur
		if event == "is_sta":
			pss_df.loc[new_param, "threshold_sta"] = threshold_cur
			pss_df.loc[new_param, "pss_sta"] = pss_cur
		
	
	return df, pss_df

def test_pss_location(event, param_list, df=None, pss_df=None, l_thresh=2):

	#Load in optimised PSS thresholds, and test separately for each location

	import itertools

	try:
		df = df.reset_index().rename({"level_0":"date", "level_1":"loc_id"},axis=1)
	except:
		pss_df, df = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl",\
			compute=False, plot=False, l_thresh=l_thresh)
		df = df.reset_index().rename({"level_0":"date", "level_1":"loc_id"},axis=1)
	locs = np.unique(df.loc_id)

	if event == "is_conv_aws":
		pss_colname = "pss_conv_aws"
		threshold_colname = "threshold_conv_aws"
	else:
		pss_colname = ""
		threshold_colname = ""

	pss_df_loc = pd.DataFrame(index=locs, columns=param_list)
	
	for l in locs:
		temp_df = df[(df.loc_id == l)]
		for p in param_list:
			print(l,p)
			if np.nansum(temp_df[event]) > 1:
				temp_df2 = temp_df.loc[:,[event,p]]
				pss_p, thresh = pss([pss_df.loc[p, threshold_colname], temp_df2, p, event])

				pss_df_loc.loc[l, p] = pss_p
				pss_df_loc.loc[l, "events"] = np.nansum(temp_df2[event])
			else:
				pss_df_loc.loc[l, p] = np.nan
				pss_df_loc.loc[l, "events"] = np.nansum(temp_df[event])

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")

	for l in locs:
		pss_df_loc.loc[l, "lon"] = df[(df.loc_id==l)]["lon"].iloc[0].values[0]
		pss_df_loc.loc[l, "lat"] = df[(df.loc_id==l)]["lat"].iloc[0].values[0]
	for p in param_list:
		plt.figure(figsize=[6,6])
		plt.subplot(211)
		m.drawcoastlines()
		m.scatter(pss_df_loc["lon"], pss_df_loc["lat"], c=pss_df_loc[p], s=50, latlon=True,\
			 cmap = plt.get_cmap("Reds"), edgecolors="k")
		cb = plt.colorbar()
		cb.set_label("PSS WITH NATIONAL THRESHOLD")
		plt.title(p+" (threshold = "+str(round(pss_df.loc[p, threshold_colname],3))+")") 		

		plt.subplot(212)
		m.drawcoastlines()
		m.scatter(pss_df_loc["lon"], pss_df_loc["lat"], c=pss_df_loc["events"], s=50, latlon=True, \
			cmap = plt.get_cmap("Greys"), edgecolors="k")
		cb = plt.colorbar()
		cb.set_label("OBSERVED EVENTS")
		plt.title(event)

	return pss_df_loc

def optimise_pss(model_fname,T=1000, compute=True, l_thresh=2, is_pss="pss", model_name="erai", time="floor",\
	exclude_logit_hits=False, logit_mod="era5"):

	#model_fname is the path to the model point data
	#T is the number of linear steps between the max and min of a parameter, used to test the pss
	#compute means that this function will either compute pss (default; True), or just load it from disk (False)
	#l_thresh is the threshold of lightning to consider a gust "convective". Defaults to 2
	#is_pss defines the skill score used. Options are "pss", "hss", and "csi"
	#model_name defines the output file name for skill scores
	#time defines what obs times are to be used. Options are "floor" (relevant for forecast diagnostics) and "ceil" (for other diagnostics)
	#For era5: /g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl
	#For barra: /g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018.pkl

	model = pd.read_pickle(model_fname).dropna()
	param_list = np.delete(np.array(model.columns), \
		np.where((np.array(model.columns)=="lat") | \
		(np.array(model.columns)=="lon") | \
		(np.array(model.columns)=="loc_id") | \
		(np.array(model.columns)=="time")))
	obs = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/convective_wind_gust_aus_2005_2018.pkl")
	obs["hourly_ceil_utc"] = pd.DatetimeIndex(obs["gust_time_utc"]).ceil("1H")
	#There is one missing entry for parameters needing mhgt
	if time == "floor":
	    df_sta = pd.merge(obs[["stn_name","wind_gust","hourly_floor_utc","tc_affected","lightning","is_sta"]],\
		    model, how="left",left_on=["stn_name","hourly_floor_utc"], right_on=["loc_id","time"]).dropna(subset=param_list)
	    df_aws = df_sta.dropna(subset=["wind_gust"])
	elif time == "ceil":
	    df_sta = pd.merge(obs[["stn_name","wind_gust","hourly_ceil_utc","tc_affected","lightning","is_sta"]],\
		    model, how="left",left_on=["stn_name","hourly_ceil_utc"], right_on=["loc_id","time"]).dropna(subset=param_list)
	    df_aws = df_sta.dropna(subset=["wind_gust"])

	dmax=False
	if dmax:
		groups = model.groupby("loc_id")
		dmax_df = pd.DataFrame()
		for name, group in groups:
			group.index = pd.DatetimeIndex(group.index, freq="1H")
			dmax_df = pd.concat([dmax_df, group.resample("1D").max()])
		dmax_df = pd.merge(obs, dmax_df, how="inner", left_on=["daily_date_utc","loc_id"],\
			right_on = ["time","loc_id"])

	df_sta = df_sta[df_sta.tc_affected==0]
	df_aws = df_aws[df_aws.tc_affected==0]

	if exclude_logit_hits:
		if logit_mod == "era5":
			z_aws = 6.4e-1*df_aws["lr36"] - 1.2e-4*df_aws["mhgt"] +\
				     4.4e-4*df_aws["ml_el"] \
				    -1.0e-1*df_aws["qmean01"] \
				    + 1.7e-2*df_aws["srhe_left"] \
				    + 1.8e-1*df_aws["Umean06"] - 7.4
			z_sta = 3.3e-1*df_sta["lr36"] + 1.6e-3*df_sta["ml_cape"] +\
				     2.9e-2*df_sta["srhe_left"] \
				    +1.6e-1*df_sta["Umean06"] - 4.5
			df_aws = df_aws[( 1 / (1 + np.exp(-z_aws))) < 0.72]
			df_sta = df_sta[( 1 / (1 + np.exp(-z_sta))) < 0.62]
		if logit_mod == "barra":
			z_aws = 8.5e-1*df_aws["lr36"] + 6.2e-1*df_aws["lr_freezing"] +\
				     3.9e-4*df_aws["ml_el"] \
				    +3.8e-2*df_aws["s06"] \
				    + 1.5e-2*df_aws["srhe_left"] \
				    + 1.6e-1*df_aws["Umean06"] - 14.8
			z_sta = 4.3e-1*df_sta["lr36"] + 2.0e-1*df_sta["lr_freezing"] \
				    - 9.5e-5*df_sta["mhgt"] \
				    + 3.2e-4*df_aws["ml_el"] \
				    +3.8e-2*df_aws["s06"] \
				    + 1.4e-2*df_aws["srhe_left"] \
				    + 1.5e-1*df_aws["Umean06"] - 7.9
			df_aws = df_aws[( 1 / (1 + np.exp(-z_aws))) < 0.80]
			df_sta = df_sta[( 1 / (1 + np.exp(-z_sta))) < 0.71]

	if compute:

		pss_df = pd.DataFrame(index=param_list, columns=["threshold_light","pss_light",\
			"threshold_conv_aws","pss_conv_aws",\
			"threshold_sta","pss_sta","threshold_conv_aws_cond_light","pss_conv_aws_cond_light"])

		import multiprocessing
		import itertools
		pool = multiprocessing.Pool()

		#Optimise for discriminating lightning and non-lightning
		print("OPTIMISING "+is_pss+" FOR LIGHTNING...")
		df_sta.loc[:,"is_lightning"] = 0
		df_sta.loc[df_sta.lightning >= l_thresh, "is_lightning"] = 1
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df_sta.loc[:,p].min(), np.percentile(df_sta.loc[:,p],99.95) , T)
			temp_df = df_sta.loc[:,["is_lightning",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_lightning"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_light"] = np.array(pss_p).max()

		#Optimise for discriminating convective AWS and non-convective AWS
		print("OPTIMISING "+is_pss+" FOR CONVECTIVE AWS EVENTS...")
		df_aws.loc[:,"is_conv_aws"] = 0
		df_aws.loc[(df_aws.lightning >= l_thresh) & (df_aws.wind_gust >= 25), "is_conv_aws"] = 1
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df_aws.loc[:,p].min(), np.percentile(df_aws.loc[:,p],99.95) , T)
			temp_df = df_aws.loc[:,["is_conv_aws",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_conv_aws"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_conv_aws"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_conv_aws"] = np.array(pss_p).max()

		#Optimise for discriminating STA wind and non-STA wind
		print("OPTIMISING "+is_pss+" FOR STA WIND REPORTS...")
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df_sta.loc[:,p].min(), np.percentile(df_sta.loc[:,p],99.95) , T)
			temp_df = df_sta.loc[:,["is_sta",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_sta"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_sta"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_sta"] = np.array(pss_p).max()

		#Optimise for discriminating convective AWS and lightning but non-convective AWS
		print("OPTIMISING "+is_pss+" FOR CONVECTIVE WIND EVENTS VERSUS CONVECTIVE NON-WIND EVENTS...")
		df_aws.loc[:,"is_conv_aws_cond_light"] = np.nan
		df_aws.loc[(df_aws.lightning >= l_thresh) & (df_aws.wind_gust >= 25), "is_conv_aws_cond_light"] = 1
		df_aws.loc[(df_aws.lightning >= l_thresh) & (df_aws.wind_gust < 25), "is_conv_aws_cond_light"] = 0
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df_aws.loc[:,p].min(), np.percentile(df_aws.loc[:,p],99.95) , T)
			temp_df = df_aws.loc[:,["is_conv_aws_cond_light",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_conv_aws_cond_light"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_conv_aws_cond_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_conv_aws_cond_light"] = np.array(pss_p).max()

		pss_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/skill_scores/"+is_pss+"_"+time+"_df_lightning"+str(l_thresh)+\
				"_"+model_name+".pkl")

	else:

		pss_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/skill_scores/"+is_pss+"_"+time+"_df_lightning"+str(l_thresh)+\
			"_"+model_name+".pkl")

		df_sta.loc[:,"is_lightning"] = 0
		df_sta.loc[df_sta.lightning >= l_thresh, "is_lightning"] = 1
		df_aws.loc[:,"is_conv_aws"] = 0
		df_aws.loc[(df_aws.lightning >= l_thresh) & (df_aws.wind_gust >= 25), "is_conv_aws"] = 1
		df_aws.loc[:,"is_conv_aws_cond_light"] = np.nan
		df_aws.loc[(df_aws.lightning >= l_thresh) & (df_aws.wind_gust >= 25), "is_conv_aws_cond_light"] = 1
		df_sta.loc[(df_sta.lightning >= l_thresh) & (df_sta.wind_gust < 25), "is_conv_aws_cond_light"] = 0

	return pss_df, df_aws, df_sta

def pss(it):

	#Calculate pss with the help of multiprocessing

	t, df, p, event, score = it

	try:

		hits = float(((df[event]==1) & (df[p]>t)).sum())
		misses = float(((df[event]==1) & (df[p]<=t)).sum())
		fa = float(((df[event]==0) & (df[p]>t)).sum())
		if score == "pss":
			cn = float(((df[event]==0) & (df[p]<=t)).sum())
			pss_ge = (hits / (hits+misses)) - (fa / (fa + cn))
		elif score == "csi":
			#This is actually the (CSI) threat score, but call it pss_ge for simplicity
			#Specify that the hit rate must be greater than 0.66
			if (hits / (hits + misses)) > 0.66:
				pss_ge = (hits) / (hits+misses+fa)
			else:
				pss_ge = 0
		elif score == "hss":
			if (hits / (hits + misses)) > 0.66:
				cn = float(((df[event]==0) & (df[p]<=t)).sum())
				pss_ge = ( 2*(hits*cn - misses*fa) ) / \
					( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )
			else:
				pss_ge = 0
		elif score == "edi":
			cn = float(((df[event]==0) & (df[p]<=t)).sum())
			pod = (hits / (hits + misses))
			pofd = (fa / (cn + fa) )
			pss_ge = ( np.log(pofd) - np.log(pod) ) / (np.log(pofd) + np.log(pod) )
		else:
			raise ValueError("SCORE MUST EITHER BE pss, csi, hss, edi")
			

		#Test if param is less than (e.g. cin)
		hits = float(((df[event]==1) & (df[p]<t)).sum())
		misses = float(((df[event]==1) & (df[p]>=t)).sum())
		fa = float(((df[event]==0) & (df[p]<t)).sum())
		if score == "pss":
			cn = float(((df[event]==0) & (df[p]>=t)).sum())
			pss_l = (hits / (hits+misses)) - (fa / (fa + cn))
		elif score == "csi":
			if (hits / (hits + misses)) > 0.66:
				pss_l = (hits) / (hits+misses+fa)
			else:
				pss_l = 0
		elif score == "hss":
			if (hits / (hits + misses)) > 0.66:
				cn = float(((df[event]==0) & (df[p]>=t)).sum())
				pss_l = ( 2*(hits*cn - misses*fa) ) / \
					( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )
			else:
				pss_l = 0
		elif score == "edi":
			cn = float(((df[event]==0) & (df[p]>=t)).sum())
			pod = (hits / (hits + misses))
			pofd = (fa / (cn + fa) )
			pss_l = ( np.log(pofd) - np.log(pod) ) / (np.log(pofd) + np.log(pod) )
		else:
			raise ValueError("SCORE MUST EITHER BE pss, csi, hss, edi")
		
		return [np.array([pss_ge, pss_l]).max(), t]

	except:
		if (score == "pss") | (score == "hss") | (score == "edi"):	
			return [-1, t]
		else:
			return [0,t]


def load_array_points(param,param_out,lon,lat,times,points,loc_id,model,smooth,erai_fc=False,\
		ad_data=False,daily_max=False):
	#Instead of loading data from netcdf files, read numpy arrays. This is so BARRA-AD/
	#BARRA-R fields can be directly loaded from the ma05 g/data directory, rather than
	#being moved to eg3 and saved to monthly files first.
	#If model = barra and smooth = False, the closest point in BARRA to "point" is taken. 
	# Otherwise, smooth = "mean" takes the mean over ~0.75 degrees (same as ERA-Interim),
	# or smooth = "max" takes the max over the same area for all variables 
	
	#Get lat/lon inds to use based on points input, taking in to account the lsm
	if model == "erai":
		from erai_read import get_lat_lon,reform_lsm
		lon_orig,lat_orig = get_lat_lon()
		lsm = reform_lsm(lon_orig,lat_orig)
		smooth = False		#TURN SMOOTHING OFF FOR ERA-I (ALREADY 0.75 DEG)
	elif model == "barra":
		from barra_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
	elif model == "barra_ad":
		from barra_ad_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/lnd_mask-fc-slv-PT0H-BARRA_AD-v1.nc").variables["lnd_mask"][:]
	elif model == "barra_r_fc":
		from barra_r_fc_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
	x,y = np.meshgrid(lon,lat)
	if ad_data:
		lsm_new = lsm[((lat_orig>=lat[0]) & (lat_orig<=lat[-1]))]
		lsm_new = lsm_new[:,((lon_orig>=lon[0]) & (lon_orig<=lon[-1]))]
	else:
		lsm_new = lsm[((lat_orig<=lat[0]) & (lat_orig>=lat[-1]))]
		lsm_new = lsm_new[:,((lon_orig>=lon[0]) & (lon_orig<=lon[-1]))]
		x[lsm_new==0] = np.nan
		y[lsm_new==0] = np.nan
	lat_ind = np.empty(len(points))
	lon_ind = np.empty(len(points))
	lat_used = np.empty(len(points))
	lon_used = np.empty(len(points))
	for point in np.arange(0,len(points)):
		dist = np.sqrt(np.square(x-points[point][0]) + \
				np.square(y-points[point][1]))
		dist_lat,dist_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
		lat_ind[point] = dist_lat
		lon_ind[point] = dist_lon
		lon_used[point] = lon[dist_lon]
		lat_used[point] = lat[dist_lat]

	#Create dataframe the same format as output from calc_param_points
	if ad_data:
		times = [dt.datetime(int(fname[-7:-3]),1,1,0,0,0) + dt.timedelta(hours=6*x) \
			for x in times]
		days = np.unique(np.array([x.day for x in times]))
	else:
		days = (np.array([x.day for x in times]))
		unique_days = np.unique(days)
	var = param
	if daily_max:
		values = np.empty(((len(points)*len(unique_days)),len(var)))
	else:
		values = np.empty((len(points)*len(times),len(var)))
	values_lat = []
	values_lon = []
	values_lon_used = []
	values_lat_used = []
	values_loc_id = []
	values_year = []; values_month = []; values_day = []; values_hour = []; values_minute = []
	values_date = []
	cnt = 0

	if daily_max:
		smooth=False
		for point in np.arange(0,len(points)):
			for t in np.arange(len(unique_days)):
				for v in np.arange(0,len(var)):
					values[cnt,v] = \
						np.nanmax(param_out[v][days==unique_days[t],\
						lat_ind[point],lon_ind[point]],axis=0)
				values_lat.append(points[point][1])
				values_lon.append(points[point][0])
				values_lat_used.append(lat_used[point])
				values_lon_used.append(lon_used[point])
				values_loc_id.append(loc_id[point])
				values_year.append(times[t].year)
				values_month.append(times[t].month)
				values_day.append(unique_days[t])
				values_date.append(dt.datetime(times[t].year,times[t].month,\
					unique_days[t]))
				cnt = cnt+1
	else:
		for point in np.arange(0,len(points)):
			print(lon_used[point],lat_used[point])
			for t in np.arange(len(times)):
				for v in np.arange(0,len(var)):
					if smooth=="mean":
					#SMOOTH OVER ~1 degree
						lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
						lon_points = np.arange(lon_ind[point]-4,lon_ind[point]+5)
						values[cnt,v] = np.nanmean(param_out[v][t,\
						int(lat_points[0]):int(lat_points[-1]),\
						int(lon_points[0]):int(lon_points[-1])])
					elif smooth=="max":
					#Max OVER ~1 degree 
						lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
						lon_points = np.arange(lon_ind[point]-4,lon_ind[point]+5)
						values[cnt,v] = np.nanmax(param_out[v][t,\
						int(lat_points[0]):int(lat_points[-1]),\
						int(lon_points[0]):int(lon_points[-1])])
					elif smooth==False:
						values[cnt,v] = param_out[v][t,int(lat_ind[int(point)]),\
						int(lon_ind[int(point)])]
				values_lat.append(points[point][1])
				values_lon.append(points[point][0])
				values_lat_used.append(lat_used[point])
				values_lon_used.append(lon_used[point])
				values_loc_id.append(loc_id[point])
				values_year.append(times[t].year)
				values_month.append(times[t].month)
				values_day.append(times[t].day)
				values_hour.append(times[t].hour)
				values_minute.append(times[t].minute)
				values_date.append(times[t])
				cnt = cnt+1
	
	df = pd.DataFrame(values,columns=var)
	df["lat"] = values_lat
	df["lon"] = values_lon
	df["lon_used"] = values_lon_used
	df["lat_used"] = values_lat_used
	df["loc_id"] = values_loc_id
	df["year"] = values_year
	df["month"] = values_month
	df["day"] = values_day
	if not erai_fc:
		df["hour"] = values_hour
		df["minute"] = values_minute
	df["date"] = values_date

	return df	

def load_AD_data(param):
	#Load Andrew Dowdy's CAPE/S06 data
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/ad_data/"+param+"/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/ad_data/"+param+"/"+ls[i] \
			for i in np.arange(0,len(ls))]
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(136.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		if int(ls_full[i][-7:-3]) >= 2010:
			print(ls[i])
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
				smooth=False,ad_data=True))
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_ADdata_"+param+"_2010_2015.pkl")
	return df

def hypothesis_test(a,b,B):

	#For two samples (a,b) perform a bootstrap hypothesis test that their mean is different

	if (np.all(np.isnan(a)) ) | (np.all(np.isnan(b)) ):
		return (np.nan)
	else:
		#Difference in each mean
		abs_diff = np.nanmean(b,axis=0) - np.nanmean(a,axis=0)
		#Mean of both datasets combined
		total = np.concatenate((a,b),axis=0)
		tot_mean = np.nanmean(total,axis=0)
		#Shift each dataset to have the same mean
		a_shift = a - np.nanmean(a,axis=0) + tot_mean
		b_shift = b - np.nanmean(b,axis=0) + tot_mean

		#Sample from each shifted array B times
		a_samples = [a_shift[np.random.randint(0,high=a.shape[0],size=a.shape[0])] for temp in np.arange(0,B)]
		b_samples = [b_shift[np.random.randint(0,high=b.shape[0],size=b.shape[0])] for temp in np.arange(0,B)]
		#For each of the B samples, get the mean and compare them
		a_sample_means = np.array( [np.nanmean(a_samples[i],axis=0) for i in np.arange(0,B)] )
		b_sample_means = np.array( [np.nanmean(b_samples[i],axis=0) for i in np.arange(0,B)] )
		sample_diff = b_sample_means - a_sample_means
		#sample_diff = []		
		#for n in np.arange(B):
		#	temp_a = np.nanmean(a_shift[np.random.randint(0,high=a.shape[0],size=a.shape[0])], axis=0)
		#	temp_b = np.nanmean(b_shift[np.random.randint(0,high=b.shape[0],size=b.shape[0])], axis=0)
		#	sample_diff.append(temp_b - temp_a)
		#sample_diff = np.stack(sample_diff)

		#Take the probability that the original mean difference is greater or less than the samples 
		p_up = np.sum(sample_diff >= abs_diff,axis=0) / float(B)
		p_low = np.sum(sample_diff <= abs_diff,axis=0) / float(B)

		out = (2*np.min(np.stack((p_low,p_up)),axis=0))

		#If an area is always masked (e.g. for sst data over the land), then mask the data
		try:
			out[a.sum(axis=0).mask] = np.nan
			out[b.sum(axis=0).mask] = np.nan
		except:
			pass

		return out

def trend_table():

	#For AWS/ERAI-Interim, create csv output for a trend table to make up our final report

	aws = remove_incomplete_aws_years(pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
		"all_daily_max_wind_gusts_sa_1979_2017.pkl"),"Port Augusta").sort_values("date")
	erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_fc_points_1979_2017_daily_max.pkl").\
		sort_values("date")

	ann = np.arange(1,13,1)
	aso = [8,9,10]
	ndj = [11,12,1]
	fma = [2,3,4]
	mjj = [5,6,7]
	times = [ann,aso,ndj,fma,mjj]
	locs = ["Woomera","Port Augusta","Adelaide AP","Mount Gambier"]
	aws_trends = np.empty((4,5))
	erai_trends = np.empty((4,5))
	aws_sig = np.zeros((4,5))
	erai_sig = np.zeros((4,5))
	aws_thresh_trends = np.empty((4,5))
	erai_thresh_trends = np.empty((4,5))
	aws_thresh_sig = np.zeros((4,5))
	erai_thresh_sig = np.zeros((4,5))
	aws_thresh_n = np.zeros((4,5))
	erai_thresh_n = np.zeros((4,5))
	for i in np.arange(0,len(locs)):
		for j in np.arange(0,len(times)):
			#Isolate first and second half of data for location/season
			aws_start = aws[(aws.stn_name==locs[i]) & (np.in1d(aws.month,times[j])) & \
				(aws.year>=1979) & (aws.date<=dt.datetime(1998,12,31))]
			aws_end = aws[(aws.stn_name==locs[i]) & (np.in1d(aws.month,times[j])) & \
				(aws.date>=dt.datetime(1998,1,1))&(aws.year<=2017)]
			erai_start = erai[(erai.loc_id==locs[i]) & (np.in1d(erai.month,times[j])) & \
				(erai.year>=1979) & (erai.date<=dt.datetime(1998,12,31))]
			erai_end = erai[(erai.loc_id==locs[i]) & (np.in1d(erai.month,times[j])) & \
				(erai.date>=dt.datetime(1998,1,1))&(erai.year<=2017)]

			#Get trends for mean gusts
			aws_trends[i,j] = np.mean(aws_end["wind_gust"]) - np.mean(aws_start["wind_gust"])
			erai_trends[i,j] = np.mean(erai_end["wg10"]) - np.mean(erai_start["wg10"])

			if hypothesis_test(aws_start["wind_gust"],aws_end["wind_gust"],1000) <= 0.05:
				aws_sig[i,j] = 1
			if hypothesis_test(erai_start["wg10"],erai_end["wg10"],1000) <= 0.05:
				erai_sig[i,j] = 1

			#Get trends for days exceeding "strong" gust
			aws_start_days = [np.sum((aws_start.wind_gust>=25) & \
				(aws_start.year==y)) for y in aws_start.year.unique()]
			aws_end_days = [np.sum((aws_end.wind_gust>=25) & \
				(aws_end.year==y)) for y in aws_end.year.unique()]
			erai_start_days = [np.sum((erai_start.wg10>=21.5) & \
				(erai_start.year==y)) for y in erai_start.year.unique()]
			erai_end_days = [np.sum((erai_end.wg10>=21.5) & \
				(erai_end.year==y)) for y in erai_end.year.unique()]

			#Get trends in days exceeding "strong" gust
			aws_thresh_trends[i,j] = np.mean(aws_end_days) - np.mean(aws_start_days)
			erai_thresh_trends[i,j] = np.mean(erai_end_days) - np.mean(erai_start_days)

			#Keep count
			aws_thresh_n[i,j] = np.sum(aws_end_days) + np.sum(aws_start_days)
			erai_thresh_n[i,j] = np.sum(erai_end_days) + np.sum(erai_start_days)

			if hypothesis_test(aws_start.wind_gust>=25,aws_end.wind_gust>=25,10000) <= 0.05:
				aws_thresh_sig[i,j] = 1
			if hypothesis_test(erai_start.wg10>=21.5,erai_end.wg10>=21.5,10000) <= 0.05:
				erai_thresh_sig[i,j] = 1

			pd.DataFrame(aws_trends).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"aws_mean_trends.csv")
			pd.DataFrame(erai_trends).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"erai_mean_trends.csv")
			pd.DataFrame(aws_sig).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"aws_mean_sig.csv")
			pd.DataFrame(erai_sig).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"erai_mean_sig.csv")
			pd.DataFrame(aws_thresh_trends).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"aws_thresh_trends.csv")
			pd.DataFrame(erai_thresh_trends).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"erai_thresh_trends.csv")
			pd.DataFrame(aws_thresh_sig).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"aws_thresh_sig.csv")
			pd.DataFrame(erai_thresh_sig).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"erai_thresh_sig.csv")
			pd.DataFrame(aws_thresh_n).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"aws_thresh_n.csv")
			pd.DataFrame(erai_thresh_n).to_csv("/g/data/eg3/ab4502/ExtremeWind/trends/"+\
				"erai_thresh_n.csv")

def far_table():

	#Create a table of False Alarm Rates (FAR) and Thresholds based on a 2/3 hit rate.
	#This is done for identification of three events -> JDH events, strong AWS wind gusts (25-30 m/s) and 
	#	extreme AWS gusts (>30)

	#Load in and combine JDH data (quality controlled), ERA-Interim data and AWS data
	df = analyse_events("jdh","sa_small")
	#Only consider data for time/places where JDH data is available (i.e. where AWS data is available)
	df = df.dropna(axis=0,subset=["wind_gust"])
	df["strong_gust"] = 0;df["extreme_gust"] = 0
	df.loc[(df.wind_gust >= 25) & (df.wind_gust < 30),"strong_gust"] = 1
	df.loc[(df.wind_gust >= 30),"extreme_gust"] = 1

	jdh_far = [];jdh_thresh = []
	strong_gust_far = [];strong_gust_thresh = []
	extreme_gust_far = [];extreme_gust_thresh = []
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","vo10","lr1000","lcl",\
		"relhum1000-700","s06","s0500","s01","s03",\
		"cape*s06","dcp","td850","td800","td950","dcape","mlm","dlm","mlm+dcape",\
		"dcape*cs6","mlm*dcape*cs6","cond"]
	for p in param:
		if p in ["cond","sf","mf"]:
			hits = ((df.jdh==1) & (df[p]==1)).sum()
			misses = ((df.jdh==1) & (df[p]==0)).sum()
			fa = ((df.jdh==0) & (df[p]==1)).sum()
			cn = ((df.jdh==0) & (df[p]==0)).sum()
			jdh_f = fa / float(cn + fa)
		
			hits = ((df.strong_gust==1) & (df[p]==1)).sum()
			misses = ((df.strong_gust==1) & (df[p]==0)).sum()
			fa = ((df.strong_gust==0) & (df[p]==1)).sum()
			cn = ((df.strong_gust==0) & (df[p]==0)).sum()
			sg_f = fa / float(cn + fa)

			hits = ((df.extreme_gust==1) & (df[p]==1)).sum()
			misses = ((df.extreme_gust==1) & (df[p]==0)).sum()
			fa = ((df.extreme_gust==0) & (df[p]==1)).sum()
			cn = ((df.extreme_gust==0) & (df[p]==0)).sum()
			eg_f = fa / float(cn + fa)

			eg_t=jdh_t=sg_t = 1.0
		else:
			temp,jdh_f,jdh_t = get_far66(df,"jdh",p)
			temp,sg_f,sg_t = get_far66(df,"strong_gust",p)
			temp,eg_f,eg_t = get_far66(df,"extreme_gust",p)
		jdh_far.append(jdh_f);jdh_thresh.append(jdh_t)
		strong_gust_far.append(sg_f);strong_gust_thresh.append(sg_t)
		extreme_gust_far.append(eg_f);extreme_gust_thresh.append(eg_t)
	out = pd.DataFrame({"JDH FAR":jdh_far,"Strong Wind Gust FAR":strong_gust_far,"Extreme Wind Gust FAR":\
		extreme_gust_far,"JDH Threshold":jdh_thresh,"Strong Wind Gust Threshold":strong_gust_thresh,\
		"Extreme Wind Gust Threshold":extreme_gust_thresh},index=param)
	out = out.sort_values("JDH FAR")
	out[["JDH FAR","Strong Wind Gust FAR","Extreme Wind Gust FAR","JDH Threshold","Strong Wind Gust Threshold"\
		,"Extreme Wind Gust Threshold"]].to_csv("/home/548/ab4502/working/ExtremeWind/figs/far.csv")

def remove_incomplete_aws_years(df,loc):

	#For an AWS dataframe, remove calendar years for "loc" where there is less than 330 days of data

	df = df.reset_index().sort_values(["stn_name","date"])
	years = df[df.stn_name==loc].year.unique()
	days_per_year = np.array([df[(df.stn_name==loc) & (df.year==y)].shape[0] for y in years])
	remove_years = years[days_per_year<330]
	df = df.drop(df.index[np.in1d(df.year,remove_years) & (df.stn_name==loc)],axis=0)
	print("INFO: REMOVED YEARS FOR "+loc+" ",remove_years)
	return df


def get_far66(df,event,param):
	#For a dataframe containing reanalysis parameters, and columns corresponding to some 
	#deinition of an "event", return the FAR for a 2/3 hit rate

	param_thresh = np.percentile(df[df[event]==1][param],33)
	df["param_thresh"] = (df[param]>=param_thresh)*1
	false_alarms = np.float(((df["param_thresh"]==1) & (df[event]==0)).sum())
	hits = np.float(((df["param_thresh"]==1) & (df[event]==1)).sum())
	correct_negatives = np.float(((df["param_thresh"]==0) & (df[event]==0)).sum())
	fa_ratio =  false_alarms / (hits+false_alarms)
	fa_rate =  false_alarms / (correct_negatives+false_alarms)
	return (fa_ratio,fa_rate,param_thresh)

def get_aus_stn_info():

	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	

	df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
		names=names, header=0)

	#Dict to map station names to
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}

	df = df.replace({"stn_name":renames})

	points = [(df.lon.iloc[i], df.lat.iloc[i]) for i in np.arange(df.shape[0])]

	return [df.stn_name.values,points]

def cewp_spatial_extent():

	from plot_clim import load_ncdf
	from erai_read import reform_lsm, get_lat_lon
	
	f = load_ncdf("sa_small","erai",[1979,2017],var_list=["mf","sf","cond"],exclude_vars=True)
	mf = f.variables["mf"][:]
	sf = f.variables["sf"][:]
	times=nc.num2date(f.variables["time"][:],f.variables["time"].units)
	
	#Mask over the ocean
	lat = f.variables["lat"][:]
	lon = f.variables["lon"][:]
	lon_orig,lat_orig = get_lat_lon()
	lsm = reform_lsm(lon_orig,lat_orig)
	lsm_new = lsm[((lat_orig<=lat[0]) & (lat_orig>=lat[-1]))]
	lsm_new = lsm_new[:,((lon_orig>=lon[0]) & (lon_orig<=lon[-1]))]
	lsm_new = np.repeat(lsm_new[np.newaxis,:,:],mf.shape[0],axis=0)
	mf[lsm_new==0] = np.nan
	sf[lsm_new==0] = np.nan

	#Get a binary time series of if a mf, sf or combined event has been identified in the domain
	mf_occur_sum = np.array([np.nanmax(mf[t]) for t in np.arange(mf.shape[0])]).sum()
	sf_occur_sum = np.array([np.nanmax(sf[t]) for t in np.arange(sf.shape[0])]).sum()
	combined_occur = np.array([(np.nanmax(sf[t])==1) & (np.nanmax(mf[t])==1) \
		for t in np.arange(sf.shape[0])])
	combined_occur_sum = combined_occur.sum()

	#Now get a time series of the number of grid points for each event
	mf_event = np.array([np.nansum(mf[t]) for t in np.arange(mf.shape[0])])
	sf_event = np.array([np.nansum(sf[t]) for t in np.arange(sf.shape[0])])
	combined_event = np.array([np.nansum(sf[t]) + np.nansum(mf[t]) \
		for t in np.arange(sf.shape[0])])
	combined_event[~(combined_occur)] = 0
	
	#Plot time series
	plt.figure(figsize=[10,10]);\
	plt.subplot(311);\
	plt.plot(times[times>=dt.datetime(2010,1,1)],mf_events[times>=dt.datetime(2010,1,1)]);\
	plt.axvline(dt.datetime(2016,9,28,6),color="k",linestyle="--");\
	plt.title("MF")
	plt.ylabel("Number of gridpoints");\
	plt.subplot(312);\
	plt.plot(times[times>=dt.datetime(2010,1,1)],sf_events[times>=dt.datetime(2010,1,1)]);\
	plt.axvline(dt.datetime(2016,9,28,6),color="k",linestyle="--");\
	plt.title("SF");\
	plt.ylabel("Number of gridpoints");\
	plt.subplot(313);\
	plt.plot(times[(times>=dt.datetime(2010,1,1))],combined_event[(times>=dt.datetime(2010,1,1))]);\
	plt.title("SF and MF");\
	plt.ylabel("Number of gridpoints");\
	plt.savefig("/home/548/ab4502/working/test.png");\


def run_logit(it):
	i, df, predictors, predictors_logit, normalised, test_cond, test_param = it
	if normalised:
		X_train, X_test, y_train, y_test = train_test_split(\
			(df[predictors] - (df[predictors].mean())) / (df[predictors].std()), df["is_conv_aws"])
		test_cond = False
	else:
		X_train, X_test, y_train, y_test = train_test_split(df[predictors], df["is_conv_aws"],random_state=i)

	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	logit_mod = logit.fit(X_train[predictors_logit], y_train)
	preds = logit_mod.predict_proba(X_test[predictors_logit])
	preds = (preds[:,1]>=0.6)*1

	hits = ((preds==1) & (y_test==1)).sum()
	misses = ((preds==0) & (y_test==1)).sum()
	fa = ((preds==1) & (y_test==0)).sum()
	cn = ((preds==0) & (y_test==0)).sum()
	#csi_logit = ( hits / (hits+misses+fa) ) 				#CSI
	csi_logit = ( 2*(hits*cn - misses*fa) ) / \
		( misses^2 + fa^2 + 2*hits*cn + (misses + fa) * (hits + cn) ) 	#HSS
	pod_logit = ( hits / (hits+misses) )
	far_logit = (fa / (hits + fa) )

	if test_cond:

		preds_cond = ( (X_test["ml_el"]>=6000) & (X_test["Umean800_600"]>=5) & (X_test["t_totals"]>=46) \
			& (df["k_index"]>=30) ) | \
			( (X_test["ml_el"]<6000) & (X_test["Umean800_600"]>=16) & (X_test["k_index"]>=20) & \
				(X_test["Umean01"]>=10) & (df["s03"]>=10) )
		hits = ((preds_cond==1) & (y_test==1)).sum()
		misses = ((preds_cond==0) & (y_test==1)).sum()
		fa = ((preds_cond==1) & (y_test==0)).sum()
		cn = ((preds_cond==0) & (y_test==0)).sum()
		#csi_cond = ( hits / (hits+misses+fa) ) 					#CSI
		csi_cond = ( 2*(hits*cn - misses*fa) ) / \
			( misses^2 + fa^2 + 2*hits*cn + (misses + fa) * (hits + cn) ) 		#HSS
		pod_cond = ( hits / (hits+misses) )
		far_cond = (fa / (hits + fa) )

		if test_param:

			hits = ((X_test.loc[:,"t_totals"]>=47.4) & (y_test==1)).sum()
			misses = ((X_test.loc[:,"t_totals"]<47.4) & (y_test==1)).sum()
			fa = ((X_test.loc[:,"t_totals"]>=47.4) & (y_test==0)).sum()
			cn = ((X_test.loc[:,"t_totals"]<47.4) & (y_test==0)).sum()
			csi_param = ( hits / (hits+misses+fa) )
			pod_param = ( hits / (hits+misses) )
			far_param = (fa / (hits + fa) )
		
			return [csi_logit, pod_logit, far_logit, csi_cond, pod_cond, far_cond \
					,csi_param, pod_param, far_param]
	
		else:

			return [csi_logit, pod_logit, far_logit, csi_cond, pod_cond, far_logit]

	else:

		return [csi_logit, pod_logit, far_logit, np.nan, np.nan]

def logit_train(it):

	i, df_train, df_test, predictors, event = it
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	logit_mod = logit.fit(df_train[i][list(predictors)], df_train[i][event])
	p = logit_mod.predict_proba(df_test[i][list(predictors)])[:,1]
	df_test_temp = df_test[i]
	df_test_temp.loc[:, "logit"] = p
	#Calculate the HSS/PSS for a range of probabilistic thresholds and save the maximums
	hss_out = []
	pss_out = []
	thresh_hss_out = []
	thresh_pss_out = []
	for t in np.arange(0,1.01,0.01):
		hss_p, thresh_hss = pss([t, df_test_temp, "logit", event, "hss"])
		pss_p, thresh_pss = pss([t, df_test_temp, "logit", event, "pss"])
		hss_out.append(hss_p)
		pss_out.append(pss_p)
		thresh_hss_out.append(thresh_hss)
		thresh_pss_out.append(thresh_pss)

	return [np.max(hss_out), np.max(pss_out), thresh_hss_out[np.argmax(hss_out)], \
			thresh_pss_out[np.argmax(pss_out)] ]

if __name__ == "__main__":

	if len(sys.argv) > 1:
		variable = sys.argv[1]
		threshold = sys.argv[2]
		model = sys.argv[3]
		if len(sys.argv) > 4:
			event = sys.argv[4]
			predictors = sys.argv[5].split(",")
		else:
			event = None
			predictors = None

		print("Creating monthly variables from gridded data...")

		create_mean_variable(variable, model, native=False)
		#create_threshold_variable(variable,float(threshold),model,event=event,\
		#	predictors=predictors)
		#create_mjo_phase_variable(variable,float(threshold),model,event=event,\
		#	predictors=predictors)

	else:	
		#create_mean_variable("cape",native=True, native_dir="cape")
		#compare_obs_soundings()
		diagnostics_aws_compare()
		#auc_test()
