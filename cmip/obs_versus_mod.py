import matplotlib
from plot_param import utc_to_lt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	#Load extra modules
	from event_analysis import optimise_pss
	from sklearn.linear_model import LogisticRegression
	from obs_read import read_aws_half_hourly as read_aws

	names = {"is_conv_aws":"Measured", "is_sta":"Reported", "t_totals":"T-Totals",\
		"dcp":"DCP","mlcape*s06":"MLCS6","mucape*s06":"MUCS6","mu_cape":"MUCAPE"}
	thresh = {"t_totals":46.2,"mlcape*s06":4207, "dcp":0.128, "mucape*s06":30768, "mu_cape":232}
	p_list = ["dcp","mucape*s06","mlcape*s06","t_totals","mu_cape"]

	#Load daily max obs
	l_thresh = 2
	df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"+\
		    "convective_wind_gust_aus_2005_2018.pkl")
	df = df[df.tc_affected==0]
	df.loc[:, "is_conv_aws"] = np.where((df.wind_gust >= 25) & (df.lightning >= l_thresh) &\
		    (df.tc_affected==0), 1, 0)
	df.loc[:, "is_sta"] = np.where((df.is_sta == 1) & (df.tc_affected==0), 1, 0)

	#Load reanalysis diagnostics, fit logistic regression to daily data, and apply to daily and 
	# hourly diagnostics
	pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/"+\
		    "points/era5_allvars_v3_2005_2018.pkl",\
		    T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5_v5")
	mod_hourly = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
	    "era5_allvars_v3_2005_2018.pkl").dropna()


	#Plot seasonal and diurnal cycle using hourly reanalysis data, overwriting the "diurnal_df"
	# and "monthly_df" dataframes
	df["aws_hour"] = pd.DatetimeIndex(df.gust_time_lt).round("H").hour	
	df["sta_hour"] = pd.DatetimeIndex(utc_to_lt(\
	    df.rename(columns={"sta_date":"time","stn_name":"loc_id"})).time).round("H").hour
	df["month"] = pd.DatetimeIndex(df.gust_time_lt).month
	aws_hr = pd.DataFrame(np.unique(df[df["is_conv_aws"]==1]["aws_hour"], \
		    return_counts=True)).T.set_index(0).\
		    rename(columns={1:"Measured"})
	sta_hr = pd.DataFrame(np.unique(df[df["is_sta"]==1]["sta_hour"], \
		    return_counts=True)).T.set_index(0).\
		    rename(columns={1:"Reported"})
	aws_month = pd.DataFrame(np.unique(df[df["is_conv_aws"]==1]["month"], \
		    return_counts=True)).T.set_index(0).\
		    rename(columns={1:"Measured"})
	sta_month = pd.DataFrame(np.unique(df[df["is_sta"]==1]["month"], \
		    return_counts=True)).T.set_index(0).\
		    rename(columns={1:"Reported"})
	month_df = pd.concat([aws_month, sta_month], axis=1)
	diurnal_df = pd.concat([aws_hr, sta_hr], axis=1).replace(np.nan, 0)
	mod_hourly_lt = utc_to_lt(mod_hourly)
	mod_hourly_lt["month"] = pd.DatetimeIndex(mod_hourly_lt.time).month
	mod_hourly_lt["hour"] = pd.DatetimeIndex(mod_hourly_lt.time).hour
	for p in p_list:
		temp_df = mod_hourly_lt.loc[mod_hourly_lt[p] >= thresh[p]]
		month_df.loc[:, names[p]] = \
		    temp_df["month"].value_counts().sort_index()
		diurnal_df.loc[:, names[p]] = \
		    temp_df["hour"].value_counts().sort_index()
	month_data = (month_df / month_df.sum()).reindex([7,8,9,10,11,12,1,2,3,4,5,6])
	month_data["x"] = np.arange(0,12)
	diurnal_df.loc[:,"Measured"] = \
		diurnal_df.loc[:,"Measured"].rolling(3,min_periods=1).mean()
	diurnal_df.loc[:,"Reported"] = \
		diurnal_df.loc[:,"Reported"].rolling(3,min_periods=1).mean()


	#MPL settings
	matplotlib.rcParams.update({'font.size': 14})
	cols = ["k","grey",'#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

	#Plot hourly distributions
	plt.figure(figsize=[12,8])
	ax=plt.subplot(221)
	month_data.plot(x="x", kind="line", color=cols, marker="o",legend=False,\
	    xticks=np.arange(0,12),ax=ax)
	ax.grid()
	ax.set_xticklabels(["J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M", "J"]) 
	plt.xlabel("Month")
	plt.ylabel("Relative frequency")
	plt.text(0.05,0.9,"a)",transform=plt.gca().transAxes,fontsize=14)

	ax=plt.subplot(222)
	ax.grid()
	(diurnal_df/diurnal_df.sum()).plot(kind="line",color=cols,ax=ax,marker="o",legend=False)
	ax.grid()
	plt.xlabel("Time (LST)")
	plt.text(0.05,0.9,"b)",transform=plt.gca().transAxes,fontsize=14)

	ax.legend(\
		bbox_to_anchor=(-0.2,-0.6),loc=8,fancybox=True,edgecolor="k",\
		ncol=4)

	plt.savefig("/g/data/eg3/ab4502/figs/obs_versus_mod_projections_paper.png", bbox_inches="tight")
