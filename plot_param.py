from matplotlib.ticker import FormatStrFormatter
import itertools
from barra_read import date_seq
import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import netCDF4 as nc
#import matplotlib.animation as animation
import os
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from obs_read import load_lightning, analyse_events, get_aus_stn_info
import pandas as pd
from scipy.stats import spearmanr as spr
import matplotlib
from event_analysis import optimise_pss

def utc_to_lt(df):

	#Convert the time in a dataframe from UTC to LT, for multiple locations
	tzs = {"Adelaide": "Australia/Adelaide",\
	    "Alice Springs": "Australia/Darwin","Amberley": "Australia/Brisbane",\
	    "Broome": "Australia/Perth","Cairns": "Australia/Brisbane",\
	    "Carnarvon": "Australia/Perth","Ceduna": "Australia/Adelaide",\
	    "Charleville": "Australia/Brisbane","Cobar": "Australia/Sydney",\
	    "Coffs Harbour": "Australia/Sydney","Darwin": "Australia/Darwin",\
	    "East Sale": "Australia/Melbourne","Esperance": "Australia/Perth",\
	    "Giles": "Australia/Adelaide","Gove": "Australia/Brisbane",\
	    "Halls Creek": "Australia/Darwin","Hobart": "Australia/Hobart",\
	    "Kalgoorlie": "Australia/Perth","Mackay": "Australia/Brisbane",\
	    "Meekatharra": "Australia/Perth","Melbourne": "Australia/Melbourne",\
	    "Mildura": "Australia/Melbourne","Mount Gambier": "Australia/Adelaide",\
	    "Mount Isa": "Australia/Brisbane","Oakey": "Australia/Sydney",\
	    "Perth": "Australia/Perth","Port Hedland": "Australia/Perth",\
	    "Rockhampton": "Australia/Brisbane","Sydney": "Australia/Sydney",\
	    "Tennant Creek": "Australia/Darwin","Townsville": "Australia/Brisbane",\
	    "Wagga Wagga": "Australia/Sydney","Weipa": "Australia/Brisbane",\
	    "Williamtown": "Australia/Sydney", "Woomera": "Australia/Adelaide"}
	tzs2 = {"Adelaide": 9.5,\
	    "Alice Springs": 9.5,"Amberley": 10,\
	    "Broome": 8,"Cairns": 10,\
	    "Carnarvon": 8,"Ceduna": 9.5,\
	    "Charleville": 10,"Cobar": 10,\
	    "Coffs Harbour": 10,"Darwin": 9.5,\
	    "East Sale": 10,"Esperance": 8,\
	    "Giles": 9.5,"Gove": 10,\
	    "Halls Creek": 9.5,"Hobart": 10,\
	    "Kalgoorlie": 8,"Mackay": 10,\
	    "Meekatharra": 8,"Melbourne": 10,\
	    "Mildura": 10,"Mount Gambier": 9.5,\
	    "Mount Isa": 10,"Oakey": 10,\
	    "Perth": 8,"Port Hedland": 8,\
	    "Rockhampton": 10,"Sydney": 10,\
	    "Tennant Creek": 9.5,"Townsville": 10,\
	    "Wagga Wagga": 10,"Weipa": 10,\
	    "Williamtown": 10, "Woomera": 9.5}

	out_df = pd.DataFrame()
	for loc in df.loc_id.unique():
		temp_df = df[df["loc_id"]==loc]
		#temp_df["time"] = pd.DatetimeIndex(temp_df.time.dt.tz_localize(tz="UTC")).\
		#	tz_convert(tzs[loc])
		temp_df["time"] = pd.DatetimeIndex(temp_df.time) + dt.timedelta(hours=tzs2[loc])
		out_df = pd.concat([out_df, temp_df], axis=0)
	return out_df

def plot_candidate_variable_kde():

	#For a list of "candidate variables", defined below, plot kde functions for all times, measured events, and reported events
	matplotlib.rcParams.update({'font.size': 14})
	pss_df, barra_df_aws, barra_df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"barra_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="barra_fc") 
	pss_df, era5_df_aws, era5_df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"era5_allvars_2005_2018.pkl", T=1000, compute=False, l_thresh=2,\
		is_pss="hss", model_name="era5") 
	variables = ["lr36","lr_freezing","mhgt","ml_cape","ml_el","qmean01","qmeansubcloud","s06","srhe_left","Umean06"]
	xlims = {"lr36":[2,10], "lr_freezing":[2,15], "mhgt":[0,6000], "ml_cape":[0,2000], "ml_el":[0,16000], \
		    "qmean01":[0,25], "qmeansubcloud":[0,25], "s06":[0,60], "srhe_left":[0,200], "Umean06":[0,45] }
	titles = {"lr36":"LR36","lr_freezing":"LR-Freezing","mhgt":"MHGT","ml_cape":"MLCAPE","ml_el":"MLEL","qmean01":"Qmean01",\
		    "qmeansubcloud":"Qmean-Subcloud","s06":"S06","srhe_left":"SRHE","Umean06":"Umean06"}
	units = {"lr36":"deg km$^{-1}$","lr_freezing":"deg km$^{-1}$","mhgt":"m","ml_cape":"J kg$^{-1}$","ml_el":"m","qmean01":"g kg$^{-1}$",\
		    "qmeansubcloud":"g kg$^{-1}$","s06":"m s$^{-1}$","srhe_left":"m$^{2}$ s$^{-1}$","Umean06":"m s$^{-1}$"}
	#Fix lr_freezing in era5
	era5_df_sta.loc[era5_df_sta["lr_freezing"]>barra_df_sta["lr_freezing"].max(),"lr_freezing"] = barra_df_sta["lr_freezing"].max()
	era5_df_aws.loc[era5_df_aws["lr_freezing"]>barra_df_aws["lr_freezing"].max(),"lr_freezing"] = barra_df_aws["lr_freezing"].max()

	cnt=1
	plt.figure(figsize=[12,9.5])
	plt.subplots_adjust(top=0.97, bottom=0.15, hspace=0.85, wspace=0.3, right=0.95)
	matplotlib.rcParams.update({'font.size': 14})
	for v in variables:
		print(v)

		#Plot dist for 1) Non-convective gusts 2) Non-severe convective gusts and 3) SCW

		ax1 = plt.subplot(5,2,cnt)
		barra_df_aws[barra_df_aws["lightning"]<2][v].plot(kind="kde", color="r", linestyle=":", ind=np.linspace(xlims[v][0], xlims[v][1], 1000),\
			label="Non-convective gusts (BARRA)")
		barra_df_aws[(barra_df_aws["lightning"]>=2) & (barra_df_aws["wind_gust"]<25)][v].plot(kind="kde", color="r", linestyle="--", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Non-severe convective gust (BARRA)")
		barra_df_aws[barra_df_aws["is_conv_aws"]==1][v].plot(kind="kde", color="r", linestyle="-", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Severe convective gust (BARRA)")

		era5_df_aws[era5_df_aws["lightning"]<2][v].plot(kind="kde", color="b", linestyle=":", ind=np.linspace(xlims[v][0], xlims[v][1], 1000),\
			label="Non-convective gusts (ERA5)")
		era5_df_aws[(era5_df_aws["lightning"]>=2) & (era5_df_aws["wind_gust"]<25)][v].plot(kind="kde", color="b", linestyle="--", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Non-severe convective gust (ERA5)")
		era5_df_aws[era5_df_aws["is_conv_aws"]==1][v].plot(kind="kde", color="b", linestyle="-", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Severe convective gust (ERA5)")


		plt.xlim(xlims[v][0], xlims[v][1])
		if v in ["ml_cape","srhe_left"]:
			plt.yscale("log")

		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

		if (cnt % 2) == 0:
			ax1.set_ylabel("")
		if cnt == 10:
			ax1.legend(loc="lower center", bbox_to_anchor=(-0.2,-1.6), fancybox=True, ncol=2, edgecolor="k")
		plt.title(titles[v], size=14)
		ax1.tick_params(labelsize=12)

		if v in ["ml_cape"]:
			ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2])
		elif v in ["srhe_left"]:
			ax1.set_yticks([1e-6, 1e-4, 1e-2, 1e-0])
		else:
			ax1.locator_params(axis='y', nbins=6)

		plt.xlabel(units[v]+"\n", size=10)

		cnt = cnt + 1
 
	plt.figure(figsize=[12,9])
	cnt=1
	plt.subplots_adjust(top=0.97, bottom=0.15, hspace=0.7, wspace=0.3, right=0.95)
	matplotlib.rcParams.update({'font.size': 14})
	for v in variables:
		#Plot diff between lightning and lightning w/measured gust

		ax2=plt.subplot(5,2,cnt)
		barra_df_aws[barra_df_aws["lightning"] >= 2][v].plot(kind="kde", color="k", linestyle="-", ind=np.linspace(xlims[v][0], xlims[v][1], 1000),  label="Lightning (BARRA)")
		era5_df_aws[era5_df_aws["lightning"] >= 2][v].plot(kind="kde", color="k", linestyle="--", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Lightning (ERA5)")
		barra_df_aws[barra_df_aws["is_conv_aws_cond_light"]==1][v].plot(kind="kde", color="g", linestyle="-", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Measured SCW (BARRA)")
		era5_df_aws[era5_df_aws["is_conv_aws_cond_light"]==1][v].plot(kind="kde", color="g", linestyle="--", ind=np.linspace(xlims[v][0], xlims[v][1], 1000), label="Measured SCW (ERA5)") 
		plt.xlim(xlims[v][0], xlims[v][1])
		ax2.locator_params(axis='y', nbins=3)
		if v in ["ml_cape","srhe_left"]:
			plt.yscale("log")

		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

		if (cnt % 2) == 0:
			ax2.set_ylabel("")
		if cnt == 10:
			ax2.legend(loc="lower center", bbox_to_anchor=(-0.2,-1.2), fancybox=True, ncol=2, edgecolor="k")
		plt.title(titles[v])
		ax2.tick_params(labelsize=12)

		cnt=cnt+1


def plot_pss_box(df, pss_df, param_list, score="PSS"):

	#Visualise the optimial score as a boxplot for four different events. Default score is PSS,
	# but can give any type (including CSI)

	for p in param_list:

		[cmap,mean_levels,extreme_levels,cb_lab,range,log_plot,threshold] = \
			contour_properties(p)

		plt.figure()
		plt.subplot(221)
		box = plt.boxplot( [df[(df.is_lightning==0)][p],\
			df[(df.is_lightning==1)][p] ], whis=1e10, 
			labels=["Non-lightning", \
				"Lightning"])
		if log_plot:
			plt.yscale("symlog")
		ax=plt.gca(); ax.axhline(pss_df.loc[p,"threshold_light"],color="k",linestyle="--")
		plt.text(1.5,pss_df.loc[p,"threshold_light"],score+"="+str(round(pss_df.loc[p,"pss_light"],3)),\
			horizontalalignment='center',verticalalignment="bottom")

		plt.subplot(222)
		box = plt.boxplot( [df[(df.is_conv_aws==0)][p],\
			df[(df.is_conv_aws==1)][p] ], whis=1e10, 
			labels=["Non-SCW", \
				"SCW"])
		if log_plot:
			plt.yscale("symlog")
		ax=plt.gca(); ax.axhline(pss_df.loc[p,"threshold_conv_aws"],color="k",linestyle="--")
		plt.text(1.5,pss_df.loc[p,"threshold_conv_aws"],score+"="+\
			str(round(pss_df.loc[p,"pss_conv_aws"],3)),horizontalalignment='center',\
			verticalalignment="bottom")

		plt.subplot(223)
		box = plt.boxplot( [df[(df.is_conv_aws_cond_light==0)][p],\
			df[(df.is_conv_aws_cond_light==1)][p] ], whis=1e10, 
			labels=["Lightning", \
				"SCW"])
		if log_plot:
			plt.yscale("symlog")
		ax=plt.gca(); ax.axhline(pss_df.loc[p,"threshold_conv_aws_cond_light"],color="k",\
				linestyle="--")
		plt.text(1.5,pss_df.loc[p,"threshold_conv_aws_cond_light"],\
			score+"="+str(round(pss_df.loc[p,"pss_conv_aws_cond_light"],3)),\
			horizontalalignment='center',verticalalignment="bottom")

		plt.subplot(224)
		box = plt.boxplot( [df[(df.is_sta==0)][p],\
			df[(df.is_sta==1)][p] ], whis=1e10, 
			labels=["Non-report", \
				"Report"])
		if log_plot:
			plt.yscale("symlog")
		ax=plt.gca(); ax.axhline(pss_df.loc[p,"threshold_sta"],color="k",linestyle="--")
		plt.text(1.5,pss_df.loc[p,"threshold_sta"],score+"="+str(round(pss_df.loc[p,"pss_sta"],3)),\
			horizontalalignment='center',verticalalignment="bottom")

		plt.suptitle(p)

def plot_ranked_hss():

	#Load in ERA5 and BARRA HSS stats, and plot both
	import matplotlib
	matplotlib.rcParams.update({'font.size': 14})

	ver=2

	era5, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_2005_2018.pkl",\
			T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5",\
			time="floor")
	barra, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018.pkl",\
			T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc",\
			time="floor")
	hss = pd.merge(era5, barra, how="outer", suffixes=("_era5","_barra"), left_index=True, right_index=True)
	hss = hss.rename(index={"ship":"SHIP", "k_index":"K-index", "ml_cape":"MLCAPE", "srhe_left":"SRHE", \
		"ml_el":"MLEL", "mu_cape":"MUCAPE", "eff_cape":"Eff-CAPE", "sb_cape":"SBCAPE", \
		"wmsi_ml":"WMSI", "dcp":"DCP", "mlcape*s06":"MLCS6", "eff_sherb":"SHERBE", "ebwd":"EBWD",\
		"mu_el":"MUEL", "eff_el":"Eff-EL", "sb_el":"SBEL", "mucape*s06":"MUCS6", "sweat":"SWEAT",\
		"Umean800_600":"Umean800-600", "Ust_left":"Ust", "sherb":"SHERB", "t_totals":"T-totals", \
		"scp_fixed":"SCP", "dmgwind_fixed":"DmgWind-Fixed", "lr36":"LR36", "lr_freezing":"LR-Freezing", \
		"srh06_left":"SRH06", "s06":"S06", "wg10":"WindGust10", "srh01_left":"SRH01",\
		"qmeansubcloud":"Qmeansubcloud", "s010":"S010", "effcape*s06":"Eff-CS6", "mmp":"MMP",\
		"sbcape*s06":"SBCS6", "q_melting":"Qmelting", "gustex":"GUSTEX", "pwat":"PWAT",\
		"qmean06":"Qmean06","q3":"Q3","qmean03":"Qmean03","s03":"S03",\
		"convgust_dry":"ConvGust-Dry","cp":"ConvPrcp","dpd700":"DPD700",\
		"v_totals":"V-Totals","c_totals":"C-Totals","qmean01":"Qmean01","mhgt":"MHGT",\
		"wbz":"WBZ","sfc_thetae":"Sfc-ThetaE","q1":"Q1","te_diff":"TED"}) 

	#VERSION 1
	if ver == 1:
		titles = ["Lightning", "Extreme measured gust\nconditioned on lightning", \
				"Measured convective gust", "STA wind report"]
		event = ["pss_light", "pss_conv_aws_cond_light", "pss_conv_aws", "pss_sta"]
		xlims=[[0,0.6],[0,0.06],[0,0.02],[0,0.025]]
		letters=["a","b","c","d"]
		plt.figure(figsize=[12,7])
		for i in np.arange(len(event)):
				ax = plt.subplot(2,2,i+1)
				ax.barh(y=np.arange(1,21), width=hss.sort_values(event[i]+"_barra", na_position="first", ascending=True).iloc[-20:][event[i]+"_barra"].values, color="k", height=0.33)
				plt.yticks(ticks=np.arange(1,21),labels=hss.sort_values(event[i]+"_barra", na_position="first", ascending=True).iloc[-20:].index.values.astype(str),color="k")
				ax.set_xlim(xlims[i])
				ax.set_ylim([0,20.5])
				ax2 = ax.twinx()
				ax2.set_ylim([0,20.5])
				ax2.barh(y=np.arange(0.5,20.5), width=hss.sort_values(event[i]+"_era5", na_position="first", ascending=True).iloc[-20:][event[i]+"_era5"].values, color="grey", height=0.33)
				plt.yticks(ticks=np.arange(0.5,20.5),labels=hss.sort_values(event[i]+"_era5", na_position="first", ascending=True).iloc[-20:].index.values.astype(str),color="grey")
				ax2.set_xlim(xlims[i])
				ax.tick_params("x",rotation=20)
				ax.text(xlims[i][1], 0, letters[i]+")   ", ha="right", va="bottom")
				if i in [2,3]:
					ax.set_xlabel("HSS")
				if i in [0,2]:
					ax.set_ylabel("BARRA",color="k")
				else:
					ax2.set_ylabel("ERA5",color="grey")
				ax.tick_params("y",labelsize=12)
				ax2.tick_params("y",labelsize=12)
		plt.subplots_adjust(wspace=0.75, hspace=0.2, top=0.99, right=0.85, left=0.1)
	#VERSION 2
	elif ver == 2:
		ms = 10		


		#Rank skill for event versus non-event
		plt.figure(figsize=[6,9])
		cnt=0
		cols=['#e41a1c','#377eb8','#4daf4a','#984ea3']
		legend = [ ["Measured ERA5","Measured BARRA"] , ["Reported ERA5","Reported BARRA"] ]
		sorted_vars = hss["pss_conv_aws_era5"].sort_values(ascending=False).iloc[0:10].\
				index.values
		for event in ["pss_conv_aws_era5", "pss_sta_era5"]:
			ax = plt.subplot(1,2,cnt+1)
			temp_hss = hss.loc[list(sorted_vars)+["WindGust10"],event]

			plt.plot(temp_hss, \
				np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
				marker="x", color=cols[cnt], markerfacecolor="none"\
				,markersize=ms,mew=2)

			plt.yticks(np.arange(temp_hss.shape[0],0,-1))
			if cnt == 0:
				plt.gca().set_yticklabels(temp_hss.index.values.astype(str))
			else:
				plt.gca().set_yticklabels("")
			plt.gca().grid(True)
			plt.gca().tick_params(axis="y",labelrotation=45)

			temp_hss = hss[event.replace("era5","barra")].\
				loc[list(sorted_vars)+["WindGust10"]]

			plt.plot(temp_hss, \
				np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
				marker="o", color=cols[cnt], markerfacecolor="none"\
				,markersize=ms,mew=2)

			plt.yticks(np.arange(temp_hss.shape[0],0,-1))
			plt.gca().grid(True)
			plt.axhline(1.5,color="k")
			plt.text(plt.xlim()[0],21.5,[" a)", " b)"][cnt],va="center",ha="left")
			ax.legend(legend[cnt],\
			    bbox_to_anchor=(0.5,-0.25),loc=8,fancybox=True,edgecolor="k")
			plt.xlabel("HSS")
			cnt=cnt+1
		plt.subplots_adjust(wspace=0.1,top=0.99,bottom=0.2,left=0.2,right=0.95)

		#Rank skill for lightning versus non-lightning
		plt.figure(figsize=[6,9])
		ax=plt.subplot(1,2,2)
		cnt=2
		for event in ["pss_light_era5"]:
			temp_hss = hss.loc[\
				["MLCAPE","SRHE","MLEL","Eff-CAPE","MUCAPE","PWAT","SBCAPE","Qmean06",\
				"MUEL","SBEL",\
				"Q3","Qmean03","WBZ","Sfc-ThetaE","Q1","EBWD",\
				"Uwindinf","Qmean01","DPD700","Qmeansubcloud"],\
				event].sort_values(ascending=False)
			temp_hss = hss.loc[temp_hss.index.values,event]

			plt.plot(temp_hss, \
				np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
				marker="x", color=cols[cnt], markerfacecolor="none"\
				,markersize=ms,mew=2)

			temp_hss = hss[event.replace("era5","barra")].loc[temp_hss.index.values]

			plt.plot(temp_hss, \
				np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
				marker="o", color=cols[cnt], markerfacecolor="none"\
				,markersize=ms,mew=2)

			plt.yticks(np.arange(temp_hss.shape[0],0,-1))
			plt.gca().set_yticklabels(temp_hss.index.values.astype(str))
			plt.gca().grid()
			plt.gca().tick_params(axis="y",labelrotation=45)
			cnt=cnt+1
		ax.legend(["Convective versus \nnon-convective gusts ERA5",\
			"Convective versus \nnon-convective gusts BARRA"],\
			bbox_to_anchor=(0.3,-0.18),loc=8,fancybox=True,edgecolor="k")
		plt.xlabel("HSS")
		plt.subplots_adjust(wspace=0.1,top=0.99,bottom=0.2,left=0.2,right=0.95)

		#Rank skill for SCW versus lightning
		plt.figure(figsize=[7,9])
		cnt = 2
		event = "pss_conv_aws_cond_light_era5"

		ax=plt.subplot(1,2,1)
		temp_hss = hss.loc[\
			["Umean06","Umean800-600","U3","U500","U6","Umean03","S03","S06",\
			"U1","Ust"],\
			event].sort_values(ascending=False)
		temp_hss = hss.loc[temp_hss.index.values,event]

		plt.plot(temp_hss, \
			np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
			marker="x", color=cols[cnt], markerfacecolor="none"\
			,markersize=ms,mew=2)

		temp_hss = hss[event.replace("era5","barra")].loc[temp_hss.index.values]

		plt.plot(temp_hss, \
			np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
			marker="o", color=cols[cnt], markerfacecolor="none"\
			,markersize=ms,mew=2)
		plt.text(plt.xlim()[0],10," a)",va="bottom",ha="left")

		plt.yticks(np.arange(temp_hss.shape[0],0,-1))
		plt.gca().set_yticklabels(temp_hss.index.values.astype(str))
		plt.gca().grid()
		plt.gca().tick_params(axis="y",labelrotation=45)
		ax.legend(["SCW ERA5\n(wind variables)",\
			"SCW BARRA\n(wind variables)"],\
			bbox_to_anchor=(0.4,-0.35),loc=8,fancybox=True,edgecolor="k")
		plt.title("Wind variables")
		plt.xlabel("HSS")
		cnt = cnt+1

		ax=plt.subplot(1,2,2)
		temp_hss = hss.loc[\
			["LR36","LR-Freezing","MHGT","Qmeansubcloud","Qmean01","TED",\
			"V-Totals","WBZ","Sfc-ThetaE","Q1"],\
			event].sort_values(ascending=False)
		temp_hss = hss.loc[temp_hss.index.values,event]

		plt.plot(temp_hss, \
			np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
			marker="x", color=cols[cnt], markerfacecolor="none"\
			,markersize=ms,mew=2)

		temp_hss = hss[event.replace("era5","barra")].loc[temp_hss.index.values]

		plt.plot(temp_hss, \
			np.arange(temp_hss.shape[0],0,-1),linestyle="none",\
			marker="o", color=cols[cnt], markerfacecolor="none"\
			,markersize=ms,mew=2)

		plt.yticks(np.arange(temp_hss.shape[0],0,-1))
		plt.gca().set_yticklabels(temp_hss.index.values.astype(str))
		plt.gca().grid()
		plt.gca().tick_params(axis="y",labelrotation=45)
		plt.text(plt.xlim()[0],10," b)",va="bottom",ha="left")
		plt.title("Thermodynmaic variables")
		cnt=cnt+1
		ax.legend(["SCW ERA5\n(thermo. variables)",\
			"SCW BARRA\n(thermo. variables)"],\
			bbox_to_anchor=(0.4,-0.35),loc=8,fancybox=True,edgecolor="k")
		plt.xlabel("HSS")
		plt.subplots_adjust(wspace=0.75,top=0.9,bottom=0.3,left=0.2,right=0.9)

		plt.show()

def plot_roc_curve():

	#Plot a ROC curve for two variables based on BARRA data, and label the maximised
	# HSS and PSS. One variable is a fitted logistic equation
	
	from event_analysis import optimise_pss 
	pss_df, mod = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2015.pkl",
	T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc") 
	from sklearn.linear_model import LogisticRegression 
	predictors = ["ml_el","k_index","dcape","Umean06","t_totals","U1","ml_cin"]
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	logit_mod = logit.fit(mod[predictors], mod["is_conv_aws"])
	p = logit_mod.predict_proba(mod[predictors])[:,1]
	pods = [] 
	fars = [] 
	pss = [] 
	hss = [] 
	hss_free = [] 
	thresh = np.linspace(0,1,100) 
	for t in thresh: 
		hits = ((mod["is_conv_aws"]==1) & (p>=t)).sum() 
		misses = ((mod["is_conv_aws"]==1) & (p<t)).sum() 
		fa = ((mod["is_conv_aws"]==0) & (p>=t)).sum() 
		cn = ((mod["is_conv_aws"]==0) & (p<t)).sum() 
		pods.append(hits / (hits+misses)) 
		fars.append(fa / (fa+cn)) 
		pss.append((hits / (hits+misses)) - (fa / (fa + cn))) 
		hss_free.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
		if (hits / (hits+misses)) > 0.67: 
			hss.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
	plt.plot(fars, pods, color="g", label="logistic equation") 
	plt.text(fars[np.argmax(pss)], pods[np.argmax(pss)]+.01, "PSS="+str(round(np.max(pss),2))+"("+str(round(thresh[np.argmax(pss)],1))+")", color="g") 
	plt.plot(fars[np.argmax(pss)], pods[np.argmax(pss)], "ro") 
	plt.text(fars[np.argmax(hss)], pods[np.argmax(hss)]+.01, "HSS="+str(round(np.max(hss),2))+"("+str(round(thresh[np.argmax(hss)],1))+")", color="g") 
	plt.plot(fars[np.argmax(hss)], pods[np.argmax(hss)], "ro") 
	plt.text(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)]+.01, "HSS (2)="+str(round(np.max(hss_free),2))+"("+str(round(thresh[np.argmax(hss_free)],1))+")", color="g") 
	plt.plot(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)], "ro") 

	predictors = ["ml_el","k_index","dcape","Umean06","t_totals","U1"]
	logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
	logit_mod = logit.fit(mod[predictors], mod["is_conv_aws"])
	p = logit_mod.predict_proba(mod[predictors])[:,1]
	pods = [] 
	fars = [] 
	pss = [] 
	hss = [] 
	hss_free = [] 
	thresh = np.linspace(0,1,100) 
	for t in thresh: 
		hits = ((mod["is_conv_aws"]==1) & (p>=t)).sum() 
		misses = ((mod["is_conv_aws"]==1) & (p<t)).sum() 
		fa = ((mod["is_conv_aws"]==0) & (p>=t)).sum() 
		cn = ((mod["is_conv_aws"]==0) & (p<t)).sum() 
		pods.append(hits / (hits+misses)) 
		fars.append(fa / (fa+cn)) 
		pss.append((hits / (hits+misses)) - (fa / (fa + cn))) 
		hss_free.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
		if (hits / (hits+misses)) > 0.67: 
			hss.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
	plt.plot(fars, pods, color="k", label="logistic equation w ml_cin") 
	plt.text(fars[np.argmax(pss)], pods[np.argmax(pss)]+.01, "PSS="+str(round(np.max(pss),2))+"("+str(round(thresh[np.argmax(pss)],1))+")", color="k") 
	plt.plot(fars[np.argmax(pss)], pods[np.argmax(pss)], "ro") 
	plt.text(fars[np.argmax(hss)], pods[np.argmax(hss)]+.01, "HSS="+str(round(np.max(hss),2))+"("+str(round(thresh[np.argmax(hss)],1))+")", color="k") 
	plt.plot(fars[np.argmax(hss)], pods[np.argmax(hss)], "ro") 
	plt.text(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)]+.01, "HSS (2)="+str(round(np.max(hss_free),2))+"("+str(round(thresh[np.argmax(hss_free)],1))+")", color="k") 
	plt.plot(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)], "ro") 

	pods = []  
	fars = []  
	pss = [] 
	hss = [] 
	hss_free = [] 
	thresh = np.linspace(0,mod["t_totals"].max(),100)  
	for t in thresh:  
		hits = ((mod["is_conv_aws"]==1) & (mod["t_totals"]>=t)).sum()  
		misses = ((mod["is_conv_aws"]==1) & (mod["t_totals"]<t)).sum()  
		fa = ((mod["is_conv_aws"]==0) & (mod["t_totals"]>=t)).sum()  
		cn = ((mod["is_conv_aws"]==0) & (mod["t_totals"]<t)).sum()  
		pods.append(hits / (hits+misses)) 
		fars.append(fa / (fa+cn))  
		pss.append((hits / (hits+misses)) - (fa / (fa + cn))) 
		hss_free.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
		if (hits / (hits+misses)) > 0.67: 
			hss.append(( 2*(hits*cn - misses*fa) ) / ( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )) 
	plt.plot(fars, pods, color="b", label="total totals")
	plt.text(fars[np.argmax(pss)], pods[np.argmax(pss)]-.05, "PSS="+str(round(np.max(pss),2))+"("+str(round(thresh[np.argmax(pss)],1))+")", color="b") 
	plt.plot(fars[np.argmax(pss)], pods[np.argmax(pss)], "ro")  
	plt.text(fars[np.argmax(hss)], pods[np.argmax(hss)]-.05, "HSS="+str(round(np.max(hss),2))+"("+str(round(thresh[np.argmax(hss)],1))+")", color="b") 
	plt.plot(fars[np.argmax(hss)], pods[np.argmax(hss)], "ro") 
	plt.text(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)]+.01, "HSS (2)="+str(round(np.max(hss_free),2))+"("+str(round(thresh[np.argmax(hss_free)],1))+")", color="b") 
	plt.plot(fars[np.argmax(hss_free)], pods[np.argmax(hss_free)], "ro") 
	plt.plot([0,1],[0,1],"k") 
	plt.axhline(0.67, color="k", linestyle="--") 
	plt.ylabel("POD") 
	plt.xlabel("POFD") 
	plt.legend() 
	plt.show() 

def compare_ctk_soundings():

	#Create a merged dataframe of BARRA pressure-level CAPE, model level CAPE and radiosonde derived CAPE

	#Load data
	prs = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2015.pkl")
	#mdl_cin = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ctk_ml_cin_2005_2015.pkl")
	#mdl_cape = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ctk_ml_cape_2005_2015.pkl")
	#mdl = pd.merge(mdl_cin, mdl_cape[["time","loc_id","ml_cape"]], on=["time","loc_id"])
	mdl = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2015.pkl")
	obs_wrf = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/UA_wrfpython_mass_weighted.pkl").reset_index().rename(columns={"index":"time"})
	obs_sharp = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/UA_sharppy.pkl").reset_index().rename(columns={"index":"time"})

	#For cin, flip the sign from the MDL data
	mdl["ml_cin"] = -1 * mdl["ml_cin"]

	#Create station name column in observation data
	stn_map = {14015:"Darwin",66037:"Sydney",16001:"Woomera",23034:"Adelaide"}
	obs_sharp["stn_name"] = obs_sharp["stn_id"].map(stn_map)
	obs_wrf["stn_name"] = obs_wrf["stn_id"].map(stn_map)

	#Merge datasets
	df = pd.merge(prs, mdl, how="inner", on=["time","loc_id"], suffixes=("_prs","_mdl"))
	df = pd.merge(df, obs_wrf, how="inner", left_on=["time","loc_id"], right_on=["time","stn_name"], \
		suffixes=("","_obs_wrf")).\
		rename(columns={"ml_cin":"ml_cin_obs_wrf", "ml_cape":"ml_cape_obs_wrf","dcape":"dcape_obs_wrf"})
	df = pd.merge(df, obs_sharp, how="inner", left_on=["time","loc_id"], right_on=["time","stn_name"], \
		suffixes=("","_obs_sharp")).\
		rename(columns={"ml_cin":"ml_cin_obs_sharp", "ml_cape":"ml_cape_obs_sharp","dcape":"dcape_obs_sharp"})
	df = df[["ml_cin_prs","ml_cin_mdl","ml_cin_obs_wrf","ml_cin_obs_sharp",\
		"ml_cape_prs","ml_cape_mdl","ml_cape_obs_wrf","ml_cape_obs_sharp",\
		"dcape_prs","dcape_obs_sharp","dcape_obs_wrf",\
		"k_index","time","loc_id"]]
	
	#Create a daily max dataframe for comparison with lightning
	df_dmax = df.groupby("loc_id").resample("1D",on="time").max().drop(columns=["time","loc_id"]).reset_index()
	lightning = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_50_smoothed.pkl")
	lightning = lightning.loc[lightning["loc_id"].isin(list(stn_map.values())), :]
	lightning = lightning.groupby("loc_id").resample("1D", on="date").max().drop(columns=["loc_id","date"]).\
			reset_index()
	df_dmax = pd.merge(df_dmax, lightning, left_on=["time","loc_id"], right_on=["date","loc_id"],\
		how="left").dropna()
	df_dmax["is_light"] = (df_dmax["lightning"] >= 2)*1

	#Plot PSS for lightning
	from event_analysis import pss
	p = "dcape_prs"
	#FOR CIN...
	df_dmax.loc[:,"is_conv_aws_cond_light"] = np.where(df_dmax["lightning"]>=2, 0, np.nan)
	#df_dmax.loc[(df_dmax["lightning"]>=2) & (df_, "light_cond_cape"] = 1
	df_dmax.loc[(df_dmax[p.replace("cin","cape")] >= 100) & (df_dmax["lightning"]<2), "light_cond_cape"] = 0
	pss_all = []
	thresh_all = []
	for t in np.linspace(0,1000,100):
		a,b = pss([t, df_dmax, p, "is_conv_aws_cond_light", "pss"])
		pss_all.append(a); thresh_all.append(b)
	plt.figure(); plt.subplot(221)
	box = plt.boxplot( [df_dmax[(df_dmax.light_cond_cape==0)][p],\
		df_dmax[(df_dmax.light_cond_cape==1)][p] ], whis=1e10, 
		labels=["Non-lightning", \
			"Lightning"])
	plt.yscale("symlog")
	ax=plt.gca(); ax.axhline(thresh_all[np.argmax(pss_all)],color="k",linestyle="--")
	plt.text(1.5,thresh_all[np.argmax(pss_all)],"PSS="+str(round(np.max(pss_all),3)),\
		horizontalalignment='center',verticalalignment="bottom")
	plt.title(p)

	#Plot a comparison daily max time series
	loc = "Darwin"
	df[df.loc_id==loc][["ml_cin_prs","ml_cin_mdl","ml_cin_obs_wrf","ml_cin_obs_sharp","time"]]\
		.set_index("time").loc[slice("2014-9-1","2015-4-1"),:].resample("1D").max().\
		plot(marker="o",markersize=4)
	plt.legend(["PRS (wrf-python)", "MDL (ctk code)","OBS (wrf-python)", "OBS (SHARPpy)"])

	#Plot a density plot comparison and compute the Spearman correlation coefficient
	x = "ml_cin_obs_sharp"
	y = "ml_cin_prs"
	plt.hist2d(df.loc[:, x], df.loc[:, y], cmap=plt.get_cmap("Greys"), \
		norm=matplotlib.colors.SymLogNorm(1),\
		vmax=1000, bins=np.concatenate([[0],np.logspace(0,4,20)]))
	plt.xscale("symlog");plt.yscale("symlog")
	plt.colorbar()
	xb, xt = plt.gca().get_xlim()
	yb, yt = plt.gca().get_ylim()
	plt.text(xt, 0, \
		"r = "+str(round(spr(df.dropna().loc[:, x], df.dropna().loc[:, y]).correlation, 3)), \
		horizontalalignment="right", verticalalignment="bottom")
	plt.plot([min(xb, yb), max(xt,yt)], [min(xb, yb), max(xt,yt)], color="r")

	plt.show()


def bootstrap_slope(x,y,n_boot):

	#Return the gradient, and standard deviation for an n_boot bootstrap resamplint

	samples = np.random.choice(np.arange(0,y.shape[0],1),(n_boot,y.shape[0]))
	m,b = np.polyfit(x,y,1)
	m_boot = []
	for i in np.arange(0,samples.shape[0]):
		temp_m,b = np.polyfit(x[samples][i,:],y[samples][i,:],1)
		m_boot.append(temp_m)
	m_boot = np.array(m_boot)
	std = np.std(m_boot)
	return (m,std)

def three_month_average(a):

	#Takes an array of length 12 (months) and returns 3-month rolling avg	
	
	assert (len(a) == 12), "ARRAY MUST BE LENGTH 12"

	rolling = np.zeros(len(a))

	for i in np.arange(12):
		if i == 0:
			rolling[i] = (a[-1] + a[i] + a[i+1]) / 3.
		elif i == 11:
			rolling[i] = (a[i-1] + a[i] + a[0]) / 3.
		else:
			rolling[i] = (a[i-1] + a[i] + a[i+1]) / 3.
	return rolling

def obs_versus_mod(model):

	#This function plots figures of historical SCW occurrence. Namely, a comparison between
	# the modelled and observed seasonal cycle, diurnal cycle, and wind direction frequency
	# distribution

	#Load extra modules
	from event_analysis import optimise_pss
	from sklearn.linear_model import LogisticRegression
	from obs_read import read_aws_half_hourly as read_aws

	#MPL settings
	matplotlib.rcParams.update({'font.size': 14})
	cols = ["k","grey",'#e41a1c','#377eb8','#4daf4a','#984ea3']

	#Other settings
	dir_data = "obs"  #or "obs"
	names = {"is_conv_aws":"Measured", "is_sta":"Reported", "t_totals":"T-Totals",\
		"dcp":"DCP", "model_aws":"logistic eq. (measured)",\
		"model_sta":"logistic eq. (reported)"}
	data = "hourly"

	#Load daily max obs
	l_thresh = 2
	df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"+\
		    "convective_wind_gust_aus_2005_2018.pkl")
	df.loc[:, "is_conv_aws"] = np.where((df.wind_gust >= 25) & (df.lightning >= l_thresh) &\
		    (df.tc_affected==0), 1, 0)
	df.loc[:, "is_sta"] = np.where((df.is_sta == 1) & (df.tc_affected==0), 1, 0)

	#Load reanalysis diagnostics, fit logistic regression to daily data, and apply to daily and 
	# hourly diagnostics
	if model == "ERA5":
	    pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/"+\
		    "points/era5_allvars_2005_2018.pkl",\
		    T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="era5")
	    #CONV AWS
	    aws_predictors = ["lr36","mhgt","ml_el","qmean01","srhe_left","Umean06"]
	    logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	    logit_mod_aws = logit.fit(mod_aws[aws_predictors], mod_aws["is_conv_aws"])
	    p = logit_mod_aws.predict_proba(mod_aws[aws_predictors])[:,1]
	    mod_aws.loc[:, "model"] = ((p >= 0.72)) * 1
	    mod_aws["month"] = pd.DatetimeIndex(mod_aws["time"]).month
	    #STA
	    sta_predictors = ["lr36","ml_cape","srhe_left","Umean06"]
	    logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	    logit_mod_sta = logit.fit(mod_sta[sta_predictors], mod_sta["is_sta"])
	    p = logit_mod_sta.predict_proba(mod_sta[sta_predictors])[:,1]
	    mod_sta.loc[:, "model_sta"] = ((p >= 0.65)) * 1
	    mod_sta["month"] = pd.DatetimeIndex(mod_sta["time"]).month
	    #Hourly data
	    mod_hourly = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
	    "era5_allvars_2005_2018.pkl").dropna()
	    #wind direction data
	    if dir_data == "model":
		    mod_dir = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/era5_wind_dir_2005_2018.pkl")
		    mod_hourly = pd.concat([mod_hourly.reset_index(),\
				    mod_dir["wd"].reset_index()], axis=1).\
				    rename(columns={"wd":"wind_dir"})
		    mod_hourly = mod_hourly.dropna()
	    p_aws = logit_mod_aws.predict_proba(mod_hourly[aws_predictors])[:,1]
	    p_sta = logit_mod_sta.predict_proba(mod_hourly[sta_predictors])[:,1]
	    p_thresh_aws = 0.72
	    p_thresh_sta = 0.65
	    thresh_t_totals = 48.02
	    thresh_dcp = 0.03
	    mod_hourly["model_aws"] = p_aws
	    mod_hourly["model_sta"] = p_sta
	    label = ["d)","e)","f)"]
	elif model == "BARRA":
	    pss_df, mod_aws, mod_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/"+\
		    "points/barra_allvars_2005_2018.pkl",\
		    T=1000, compute=False, l_thresh=2, is_pss="hss", model_name="barra_fc")
	    #CONV AWS
	    aws_predictors = ["lr36","lr_freezing","ml_el","s06","srhe_left","Umean06"]
	    logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	    logit_mod_aws = logit.fit(mod_aws[aws_predictors], mod_aws["is_conv_aws"])
	    p = logit_mod_aws.predict_proba(mod_aws[aws_predictors])[:,1]
	    mod_aws.loc[:, "model"] = ((p >= 0.82)) * 1
	    mod_aws["month"] = pd.DatetimeIndex(mod_aws["time"]).month
	    #STA
	    sta_predictors = ["lr36","lr_freezing","mhgt","ml_el","s06","srhe_left","Umean06"]
	    logit = LogisticRegression(class_weight="balanced", solver="liblinear",\
		max_iter=1000)
	    logit_mod_sta = logit.fit(mod_sta[sta_predictors], mod_sta["is_sta"])
	    p = logit_mod_sta.predict_proba(mod_sta[sta_predictors])[:,1]
	    mod_sta.loc[:, "model_sta"] = ((p >= 0.72)) * 1
	    mod_sta["month"] = pd.DatetimeIndex(mod_sta["time"]).month
	    #Hourly data
	    mod_hourly = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_allvars_2005_2018.pkl").dropna()
	    #wind direction data
	    if dir_data == "model":
		    mod_dir = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_wind_dir_2005_2018.pkl")
		    mod_hourly = pd.concat([mod_hourly.reset_index(),\
				    mod_dir["wd"].reset_index()], axis=1).\
				    rename(columns={"wd":"wind_dir"})
		    mod_hourly = mod_hourly.dropna()
	    p_aws = logit_mod_aws.predict_proba(mod_hourly[aws_predictors])[:,1]
	    p_sta = logit_mod_sta.predict_proba(mod_hourly[sta_predictors])[:,1]
	    p_thresh_aws = 0.82
	    p_thresh_sta = 0.72
	    thresh_t_totals = 48.91
	    thresh_dcp = 0.03
	    mod_hourly["model_aws"] = p_aws
	    mod_hourly["model_sta"] = p_sta
	    label = ["a)","b)","c)"]

	thresh = {"model_aws":p_thresh_aws, "model_sta":p_thresh_sta, "t_totals":thresh_t_totals,\
		"dcp":thresh_dcp, "is_conv_aws":1, "is_sta":1}

	if data == "dmax":

		#Resample hourly model diagnostics to daily max, conserving the hour and day of the dmax
		# for each diagnostic. Split the hourly data up into daily chunks from 06:00 UTC to 05:00 UTC
		# the next day
		locs = np.unique(mod_hourly.loc_id)
		mod_hourly["hour"] = pd.DatetimeIndex(mod_hourly.time).hour
		mod_hourly["month"] = pd.DatetimeIndex(mod_hourly.time).month
		mod_dmax = pd.DataFrame()
		print("Finding daily max values of diagnostics from hourly data...")
		for l in locs:
			print(l)
			l_dmax = pd.DataFrame()
			for p in ["model_aws","model_sta","t_totals","dcp"]:
				temp_df = mod_hourly.loc[mod_hourly.loc_id==l, [p]+["time","hour","month"]].\
					rename(columns={"hour":p+"_"+"hour","month":p+"_"+"month"})
				temp_dmax = temp_df.groupby(pd.Grouper(key="time",freq="24H",base=0)).\
					agg(lambda temp_df: temp_df.loc[temp_df[p].idxmax(), :])
				l_dmax = pd.concat([l_dmax, temp_dmax], axis=1)
			l_dmax.loc[:,"loc_id"] = l
			mod_dmax = pd.concat([mod_dmax, l_dmax], axis=0)

		#Merge daily maximum data with daily maximum observations
		df["is_conv_aws_hour"] = pd.DatetimeIndex(df["gust_time_utc"]).hour
		df["is_sta_hour"] = pd.DatetimeIndex(df["sta_date"]).hour
		df["is_conv_aws_month"] = df["month"]
		df["is_sta_month"] = df["month"]
		mod_dmax = pd.merge(mod_dmax, \
			    df[["stn_name","daily_date_utc","is_conv_aws","is_sta","is_conv_aws_month",\
				"is_conv_aws_hour","is_sta_month","is_sta_hour"]],\
			    left_on=["loc_id","time"],right_on=["stn_name","daily_date_utc"],how="left").\
			    dropna(subset=["is_conv_aws"])

		#Plot seasonal and diurnal cycle using daily max reanalysis data
		month_df = pd.DataFrame()
		diurnal_df = pd.DataFrame()
		for p in ["is_conv_aws","is_sta","model_aws","model_sta","dcp","t_totals"]:
			month_df = pd.concat([month_df,\
				pd.DataFrame({model+" "+names[p]:\
			    mod_dmax.loc[mod_dmax[p] >= thresh[p]][p+"_month"].value_counts().\
				sort_index()})],\
				axis=1)
			diurnal_df = pd.concat([diurnal_df,\
				pd.DataFrame({model+" "+names[p]:\
			    mod_dmax.loc[mod_dmax[p] >= thresh[p]][p+"_hour"].value_counts().sort_index()})],\
				axis=1)
		diurnal_df.loc[:,model+" Measured"] = \
			diurnal_df.loc[:,model+" Measured"].rolling(3,min_periods=1).mean()
		diurnal_df.loc[:,model+" Reported"] = \
			diurnal_df.loc[:,model+" Reported"].rolling(3,min_periods=1).mean()
		plt.figure(figsize=[12,6])
		ax=plt.subplot(121)
		month_data = (month_df / month_df.sum()).reindex([7,8,9,10,11,12,1,2,3,4,5,6])
		month_data["x"] = np.arange(0,12)
		month_data.plot(x="x", kind="line", color=cols, marker="o",legend=False,\
		    xticks=np.arange(0,12),ax=ax)
		ax.grid()
		ax.set_xticklabels(["J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M", "J"]) 
		plt.xlabel("Month")
		ax=plt.subplot(122)
		(diurnal_df/diurnal_df.sum()).plot(kind="line",color=cols,ax=ax,marker="o")
		ax.grid()
		ax.legend(\
			bbox_to_anchor=(-0.1,-0.4),loc=8,fancybox=True,edgecolor="k",\
			ncol=3)
		plt.xlabel("Time (UTC)")
		plt.subplots_adjust(bottom=0.3)

	elif data=="hourly":

		#Plot seasonal and diurnal cycle using hourly reanalysis data, overwriting the "diurnal_df"
		# and "monthly_df" dataframes
		#df["aws_hour"] = pd.DatetimeIndex(df.gust_time_utc).hour
		#df["sta_hour"] = pd.DatetimeIndex(df.sta_date).hour
		#df["month"] = pd.DatetimeIndex(df.gust_time_utc).month
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
		mod_hourly = utc_to_lt(mod_hourly)
		mod_hourly["month"] = pd.DatetimeIndex(mod_hourly.time).month
		mod_hourly["hour"] = pd.DatetimeIndex(mod_hourly.time).hour
		#mod_hourly["hour"] = mod_hourly.time.hour
		for p in ["model_aws","model_sta","dcp","t_totals"]:
			temp_df = mod_hourly.loc[mod_hourly[p] >= thresh[p]]
			month_df.loc[:, model+" "+names[p]] = \
			    temp_df["month"].value_counts().sort_index()
			diurnal_df.loc[:, model+" "+names[p]] = \
			    temp_df["hour"].value_counts().sort_index()
		diurnal_df.loc[:,"Measured"] = \
			diurnal_df.loc[:,"Measured"].rolling(3,min_periods=1).mean()
		diurnal_df.loc[:,"Reported"] = \
			diurnal_df.loc[:,"Reported"].rolling(3,min_periods=1).mean()
		#Distributions of wind direction
		if dir_data == "obs":
			aws = read_aws()
			aws["time"] = pd.DatetimeIndex(aws["time"]) + dt.timedelta(hours=-1)
			mod_hourly = pd.merge(mod_hourly, aws, left_on=["loc_id","time"], \
				right_on=["stn_name","time"], how="right").dropna()
		arr = ["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
		temp = (df[df["is_conv_aws"]==1]["wind_dir"] / 22.5 + 0.5).mod(16).astype(int) 
		aws_wd = [arr[temp.iloc[i]] for i in np.arange(temp.shape[0])]
		aws_wd = pd.DataFrame(np.unique(aws_wd, return_counts=True)).T.set_index(0).\
			    rename(columns={1:"Measured"}).reindex(arr)
		temp = (df[df["is_sta"]==1]["wind_dir"].dropna() / 22.5 + 0.5).mod(16).astype(int) 
		sta_wd = [arr[temp.iloc[i]] for i in np.arange(temp.shape[0])]
		sta_wd = pd.DataFrame(np.unique(sta_wd, return_counts=True)).T.set_index(0).\
			    rename(columns={1:"Reported"}).reindex(arr)
		wind_dir_df = pd.concat([aws_wd, sta_wd], axis=1)
		for p in ["model_aws","model_sta","dcp","t_totals"]:
			temp = (mod_hourly[mod_hourly[p]>=thresh[p]]["wind_dir"] / 22.5 + 0.5).mod(16).\
				dropna().astype(int)
			temp_wd = [arr[temp.iloc[i]] for i in np.arange(temp.shape[0])]
			temp_wd = pd.DataFrame(np.unique(temp_wd, return_counts=True)).T.set_index(0).\
			    rename(columns={1:model+" "+names[p]}).reindex(arr)
			wind_dir_df = pd.concat([wind_dir_df, temp_wd], axis=1)
		temp = (mod_hourly["wind_dir"].dropna() / 22.5 + 0.5).mod(16).astype(int) 
		all_wd = [arr[temp.iloc[i]] for i in np.arange(temp.shape[0])]
		all_wd = pd.DataFrame(np.unique(all_wd, return_counts=True)).T.set_index(0).\
			    rename(columns={1:"All hourly data"}).reindex(arr)
		wind_dir_df = pd.concat([wind_dir_df, all_wd], axis=1)
		wind_dir_data = (wind_dir_df / wind_dir_df.sum())

		#Plot hourly distributions
		plt.figure(figsize=[14,6])
		ax=plt.subplot(131)
		month_data = (month_df / month_df.sum()).reindex([7,8,9,10,11,12,1,2,3,4,5,6])
		month_data["x"] = np.arange(0,12)
		month_data.plot(x="x", kind="line", color=cols, marker="o",legend=False,\
		    xticks=np.arange(0,12),ax=ax)
		ax.grid()
		ax.set_xticklabels(["J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M", "J"]) 
		plt.xlabel("Month")
		plt.text(0.05,0.9,label[0],transform=plt.gca().transAxes,fontsize=14)

		ax=plt.subplot(132)
		(diurnal_df/diurnal_df.sum()).plot(kind="line",color=cols,ax=ax,marker="o",legend=False)
		ax.grid()
		plt.xlabel("Time (LST)")
		plt.subplots_adjust(bottom=0.3)
		plt.text(0.05,0.9,label[1],transform=plt.gca().transAxes,fontsize=14)

		ax=plt.subplot(133)
		wind_dir_data.plot(kind="line", color=cols+["k"], \
			ax=ax, legend=False,xticks=np.arange(0,len(arr),2))
		ax.set_xticklabels(arr[0::2]) 
		ax.tick_params("x",rotation=0)
		ax.set_xlabel("Measured wind direction")
		mss = ["o","o","o","o","o","o","^"]
		for i, l in enumerate(ax.lines):
			plt.setp(l, marker=mss[i])
		ax.legend(\
			bbox_to_anchor=(-0.7,-0.4),loc=8,fancybox=True,edgecolor="k",\
			ncol=4)
		ax.grid()
		plt.text(0.05,0.9,label[2],transform=plt.gca().transAxes,fontsize=14)

def sta_versus_aws():

	#Plot spatially smoothed maps of STA wind reports and AWS + lightning, as well as distributions of 
	# wind speed and direction

	from scipy.ndimage.filters import gaussian_filter as filter
	matplotlib.rcParams.update({'font.size': 14})

	l_thresh = 2

	df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"+\
		    "convective_wind_gust_aus_2005_2018.pkl")
	df.loc[:, "is_conv_aws"] = np.where((df.wind_gust >= 25) & (df.lightning >= l_thresh) &\
		    (df.tc_affected==0), 1, 0)
	df.loc[:, "is_sta"] = np.where((df.is_sta == 1) & (df.tc_affected==0), 1, 0)
	df.loc[:, "is_light"] = np.where((df.lightning >= l_thresh) & (df.tc_affected==0), 1, 0)
	df.loc[:, "is_windy"] = np.where((df.wind_gust >= 25) & (df.tc_affected==0), 1, 0)

	#SPATIAL DISTRIBUTION
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl");\
	
	loc_id, points = get_aus_stn_info()
	lon_stns = [points[i][0] for i in np.arange(len(points))]
	lat_stns = [points[i][1] for i in np.arange(len(points))]
	no_stns,y,x=np.histogram2d( lat_stns, lon_stns, \
			bins=20,range=([-45,-10],[110,160]));\
	no_stns[no_stns==0] = 1
	locs = np.unique(df["stn_name"])

	#Get number of observations for each loc
	obs = []
	g = df.groupby("stn_name")
	for l in locs:
		obs.append((~g.get_group(l).wind_gust.isna()).sum())
	obs=np.array(obs) / 365.	

	import matplotlib	

	#Final fig
	plt.figure(figsize=[8,6])
	plt.subplot(223)
	mind=0
	maxd=10
	scale=10 
	cm = plt.get_cmap("Reds")
	norm = matplotlib.colors.Normalize(vmin=mind,vmax=maxd) 
	#norm = matplotlib.colors.SymLogNorm(0.01,vmin=mind,vmax=maxd) 
	sm1 = plt.cm.ScalarMappable(cmap=cm, norm=norm) 
	for i in np.arange(len(locs)): 
		c = cm(norm((df[(df.is_windy==1) & (df.stn_name==locs[i])].shape[0] / obs[i] )) ) 
		m.plot( df[df["stn_name"]==locs[i]]["lon"].iloc[0], df[df["stn_name"]==locs[i]]["lat"].iloc[0],
		"o", linestyle="none", color = c,
		markersize=scale,
		markeredgecolor="k") 
	plt.gca().tick_params(axis="x", labelrotation=30)
	m.drawcoastlines() 
	m.drawmeridians(np.arange(115, 165, 10), labels=[0,0,0,1],rotation=30)
	m.drawparallels(np.arange(-40, 0, 10), labels=[1,0,0,0])
	plt.text(0,0.9," a)",transform=plt.gca().transAxes,fontsize=14)
	plt.subplot(224)
	mind=0
	maxd=5
	scale=10
	norm = matplotlib.colors.Normalize(vmin=mind,vmax=maxd) 
	sm2 = plt.cm.ScalarMappable(cmap=cm, norm=norm) 
	for i in np.arange(len(locs)): 
		ms = np.log(df[(df.is_sta==1) & (df.stn_name==locs[i])].shape[0] + 2) / 11. * scale 
		c = cm(norm((df[(df.is_sta==1) & (df.stn_name==locs[i])].shape[0] / 11.) ) )
		m.plot( df[df["stn_name"]==locs[i]]["lon"].iloc[0], df[df["stn_name"]==locs[i]]["lat"].iloc[0],
		"o", linestyle="none", color = c,
		markersize = scale,
		markeredgecolor="k") 
	plt.gca().tick_params(axis="x", labelrotation=30)
	m.drawcoastlines()
	m.drawmeridians(np.arange(115, 165, 10), labels=[0,0,0,1],rotation=30)
	m.drawparallels(np.arange(-40, 0, 10), labels=[1,0,0,0])
	plt.text(0,0.9," b)",transform=plt.gca().transAxes,fontsize=14)
	cbax1 = plt.axes([0.08,0.1,0.4,0.01])
	cbax2 = plt.axes([0.52,0.1,0.4,0.01])
	plt.colorbar(sm1, cax=cbax1, orientation="horizontal", extend="max")
	plt.colorbar(sm2, cax=cbax2, orientation="horizontal", extend="max")
	plt.gcf().text(0.5, 0.01, "Events per year", fontsize=14, va="bottom", ha="center")
	plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.3, top=0.95)

	plt.show()
 
def temporal_dist_plots():
	#New function to plot diurnal distribution of extreme convective gusts, using the "time of max gust" 
	# daily data in combination with 6 hourly lightning data

	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_6hrly_aus_1979_2017.pkl")
	lightning = load_lightning(daily=False,smoothing=True).reset_index()
	lightning["lightning_hour"] = [lightning.date[i].hour for i in np.arange(lightning.shape[0])]

	aws = aws.set_index(["an_gust_time_utc","stn_name"])
	lightning = lightning.set_index(["date","loc_id"])

	df = pd.concat([aws,lightning.lightning],axis=1).dropna(subset=["wind_gust"])
	df_lightning = df.dropna(subset=["lightning"])

	l = 30
	ymax=50
	plt.figure()
	plt.subplot(221);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat > -l) & (df_lightning.tc_affected==0)].month.hist();plt.ylim([0,ymax]);\
		plt.title("Convective gusts greater than 25 m/s \nfor stations north of "+str(l)+"$^{o}$S") 
	plt.subplot(222);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat < -l) & (df_lightning.tc_affected==0)].month.hist();plt.ylim([0,ymax]);\
		plt.title("Convective gusts greater than 25 m/s \nfor stations south of "+str(l)+"$^{o}$S") 
	plt.subplot(223);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat > -l) & (df_lightning.tc_affected==0)].hour.hist();plt.ylim([0,ymax]);\
	plt.subplot(224);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat < -l) & (df_lightning.tc_affected==0)].hour.hist();plt.ylim([0,ymax]);\

def probability_plot(param1,param2):

	#Plot a 2d probability plot for two parameters, as in Taszarek (2017) Fig 10 and Brooks 2013 (Fig 5)
	#Currently just for ERA-Interim on aus domain

	df = analyse_events(domain="aus",event_type="aws",model="erai")

	#Convert CAPE to WMAX
	if param1 in ["ml_cape","mu_cape","dcape"]:
		df[param1] = np.power( df[param1] * 2, 0.5)
	if param2 in ["ml_cape","mu_cape","dcape"]:
		df[param2] = np.power( df[param2] * 2, 0.5)

	#Define "convective" or "thunderstorm" events by when daily maximum lightning over a ~1 degree area around each 
	# AWS wind station is greater than or equal to 2. Define convective wind events as above but when a gust over 25 m/s is 
	# recorded
	conv_df = df[(df["lightning"]>=2)].dropna(subset=["lightning","wind_gust",param1])
	conv_wind_df = df[(df["lightning"]>=2) & (df["wind_gust"]>=25)].dropna(subset=["lightning","wind_gust",param1])

	#Bin data and get centre points
	xbins = np.linspace(df[param1].min(),df[param1].max(),20)
	ybins = np.linspace(df[param2].min(),df[param2].max(),20)
	h_wind, xe, ye = np.histogram2d(conv_wind_df[param1], conv_wind_df[param2], bins = [xbins,ybins])
	h_conv, xe, ye = np.histogram2d(conv_df[param1], conv_df[param2], bins = [xbins, ybins])
	h_all, xe, ye = np.histogram2d(df[param1], df[param2], bins = [xbins, ybins])
	x = [(xe[i] + xe[i]) /2. for i in np.arange(0,len(xe)-1)] 
	y = [(ye[i] + ye[i]) /2. for i in np.arange(0,len(ye)-1)] 
	xg,yg = np.meshgrid(x,y)

	#Smooth binned data converted to probability, using a gaussian filter
	from scipy.ndimage.filters import gaussian_filter
	#sigx = int((conv_df[param1].max() - conv_df[param1].min()) / 20.)
	#sigy = int((conv_df[param2].max() - conv_df[param2].min()) / 20.)
	sigx = 1
	sigy = 1
	h_wind_smooth = gaussian_filter(h_wind.T, (sigy, sigx))
	h_conv_smooth = gaussian_filter(h_conv.T, (sigx, sigy))
	h_all_smooth = gaussian_filter(h_all.T, (sigx, sigy))

	#Plot
	plt.subplot(221);
	plt.contourf(xg,yg,h_conv_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar();
	plt.title("Convective events")
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r",s=4)
	plt.ylabel(param2)

	plt.subplot(222);
	plt.contourf(xg,yg,h_wind_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar();plt.title("Convective wind events (>25m/s)");
	plt.xlabel(param1);
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r",s=4)

	plt.subplot(223);
	plt.contourf(xg,yg,(h_wind_smooth / h_conv_smooth),\
		levels=np.linspace(0,1,11),extend="max",cmap=plt.get_cmap("Greys"));
	plt.colorbar()
	plt.title("\nProbability of convective wind event given convective event");plt.xlabel(param1)

	plt.subplot(224);
	plt.contourf(xg,yg,h_all_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar()
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r", s=4)
	plt.title("All (convective and non convective) events")
	
	plt.show()
	

def var_trends(df,var,loc,method,dataset,year_range=[1979,2017],months=np.arange(1,13,1),percentile=0,threshold=0):

	#Load long-term daily max data, and look for trends in "var"
	#Df should have a reset index upon parsing
	#Df should have column name for station name as "stn_name"

	#method is either:
	#	- percebtile: Plot both the monthly mean time series, and model a linear best fit of
	#		annual frequency exceeding 99th percentile
	#	- amean: Fit a linear best fit model to the annual mean
	#	- amax: Fit a linear best fit model to the annual max
	#	- threshold: Same as percentile, except plot monthly max time series, and linear 
	#		best fit/counts of being within a range of "thresholds". 
	#		"thresholds" is a list of lists, where each list is either one integer 
	#		(to be exceeded) or two	increasing integers (to be between)
	#	- threshold_only: Same as threshold, except don't plot monthly max time series

	df = df.dropna(subset=[var])

	if (loc == "Adelaide AP") & (var == "wind_gust"):
		vplot = [dt.datetime(1988,10,21),dt.datetime(2002,12,17)]
	elif (loc == "Woomera") & (var == "wind_gust"):
		vplot = [dt.datetime(1991,5,16)]
	elif (loc == "Mount Gambier") & (var == "wind_gust"):
		#Note that trees and bushes have been growing on S side of anemometer for a 20 yr
		# period
		vplot = [dt.datetime(1993,7,5),dt.datetime(1995,5,25)]
	elif (loc == "Port Augusta") & (var == "wind_gust"):
		vplot = [dt.datetime(1997,6,12),dt.datetime(2001,7,2),dt.datetime(2014,2,20)]
	elif (loc == "Edinburgh") & (var == "wind_gust"):
		vplot = [dt.datetime(1993,5,4)]
	elif (loc == "Parafield") & (var == "wind_gust"):
		vplot = [dt.datetime(1992,7,22),dt.datetime(2006,6,27)]
	else:
		year_range = [1979,2017]
		vplot = []

	df = df[np.in1d(df.month,months) & (df.stn_name==loc) & (df.year>=year_range[0]) & \
		(df.year<=year_range[1])].sort_values("date").reset_index()

	if method=="threshold":
		years = df.year.unique()
		fig,[ax1,ax3]=plt.subplots(figsize=[10,8],nrows=2,ncols=1)
		ax1.set_ylabel("Wind gust (m.s$^{-1}$)",fontsize="xx-large")
		df_monthly = df.set_index("date").resample("1M").max()
		y = np.array([dt.datetime(t.year,t.month,t.day) for t in df_monthly.index])
		ax1.plot(y,df_monthly[var])
		ax2 = ax1.twinx()
		cnt = [df[(df.stn_name==loc)&(df.year==y)].shape[0] for y in np.arange(1979,2018,1)]
		y = np.array([dt.datetime(y,6,1) for y in np.arange(1979,2018,1)])
		ax2.plot(y,cnt,color="k",marker="s",linestyle="none",markersize=8)
		ax2.set_ylabel("Obs. per year",fontsize="xx-large")
		[plt.axvline(v,color="k",linestyle="--") for v in vplot]
		plt.xlabel("")
		cnt=0
		for thresh in threshold:
			if (len(thresh) == 1):
				event_years = [df[(df[var]>=thresh[0]) & (df.year == y)].shape[0] \
					for y in years]
				lab1 = "Over "+str(thresh[0])+" m/s: "
			elif (len(thresh) == 2):
				event_years = [df[(df[var]>=thresh[0]) & (df[var]<thresh[1]) & \
					(df.year == y)].shape[0] for y in years]
				lab1 = "Between "+str(thresh[0])+" - "+str(thresh[1])+" m/s: "

			events = pd.DataFrame({"years":years,"event_years":event_years})
			df_monthly = df.set_index("date").resample("1M").mean()

			x = events.years
			y = events.event_years
			m,std = bootstrap_slope(x.values,y.values,1000)

			lab = lab1+" "+str(round(m,3))+" +/- "+\
				str(round(std*2,3))+" d y^-1"
			if cnt == 2:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,color="k",\
					n_boot=10000,order=1,label=lab,ax=ax3,marker="x",\
					scatter_kws={"s":150,"linewidth":1.5})
			else:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax3,marker="s",\
					scatter_kws={"s":75})
			cnt=cnt+1
		ax1.set_xlim([dt.datetime(years.min(),1,1),dt.datetime(years.max(),12,31)])
		ax3.set_xlim([events.years.min()-1,events.years.max()+1])
		ax3.set_ylim([0.5,1000])
		ax3.set_ylabel("Days",fontsize=fs)
		ax3.set_xlabel("Year",fontsize=fs)
		ax3.set_yscale('log')
		ax1.set_title(loc,fontsize=fs)
		ax3.tick_params(labelsize=fs)
		ax2.tick_params(labelsize=fs)
		ax1.tick_params(labelsize=fs)
	if method=="threshold_only":
		fs=35
		if dataset=="ERA-Interim":
			ssize=150
			xticklabs = np.arange(1980, 2020, 5)
		elif dataset=="BARRA-R":
			ssize=500
			xticklabs = np.array([2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016])
		elif dataset=="BARRA-AD":
			ssize=500
			xticklabs = np.array([2006, 2008, 2010, 2012, 2014, 2016])
		else:
			raise ValueError("Incorrect Dataset")
		years = df.year.unique()
		fig,ax=plt.subplots(figsize=[12,5])
		cnt = [df[(df.stn_name==loc)&(df.year==y)].shape[0] for y in np.arange(1979,2018,1)]
		y = [dt.datetime(y,6,1) for y in np.arange(1979,2018,1)]
		cnt=0
		for thresh in threshold:
			if (len(thresh) == 1):
				event_years = [df[(df[var]>=thresh[0]) & (df.year == y)].shape[0] \
					for y in years]
				lab1 = "Over "+str(thresh[0])+" m/s: "
			elif (len(thresh) == 2):
				event_years = [df[(df[var]>=thresh[0]) & (df[var]<thresh[1]) & \
					(df.year == y)].shape[0] for y in years]
				lab1 = "Between "+str(thresh[0])+" - "+str(thresh[1])+" m/s: "

			events = pd.DataFrame({"years":years,"event_years":event_years})
			df_monthly = df.set_index("date").resample("1M").mean()

			x = events.years
			y = events.event_years
			m,std = bootstrap_slope(x.values,y.values,1000)

			lab = lab1+" "+str(round(m,3))+" +/- "+\
				str(round(std*2,3))+" d y^-1"
			if cnt==2:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax,marker="x",color="k",\
					scatter_kws={"s":ssize,"linewidth":2})
			else:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax,marker="s",\
					scatter_kws={"s":ssize})
			cnt=cnt+1
		if dataset == "BARRA-AD":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		elif dataset == "BARRA-R":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		else:
			ax.set_xlim([events.years.min()-1,events.years.max()+1])
		ax.set_ylim([0.5,1000])
		ax.set_ylabel("Days",fontsize=fs)
		ax.set_yscale('log')
		ax.tick_params(labelsize=fs)
		ax.set_title(loc,fontsize=fs)
		ax.set_xlabel("")
		for label in ax.get_xticklabels():
        		label.set_rotation(90) 
		plt.grid()
		plt.xticks(xticklabs, xticklabs.astype(str))
	if method=="percentile":
		years = df.year.unique()
		thresh = np.percentile(df[var],percentile)
		event_years = [df[(df[var]>=thresh) & (df.year == y)].shape[0] for y in\
			 years]

		events = pd.DataFrame({"years":years,"event_years":event_years})
		df_monthly = df.set_index("date").resample("1M").mean()

		x = events.years
		y = events.event_years
		m,std = bootstrap_slope(x.values,y.values,1000)

		plt.figure(figsize=[12,8]);plt.subplot(211)
		df_monthly[~df_monthly[var].isna()][var].plot()
		plt.ylabel("Monthly mean wind gust (m/s)")
		plt.subplot(212)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" d y^-1"
		sb.regplot("years","event_years",events,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.xlim([events.years.min(),events.years.max()])
		plt.ylabel("Days exceeding "+str(percentile)+" percentile")
		plt.suptitle(loc)
		plt.legend(fontsize=10)
			
	elif method=="amean":
		plt.figure(figsize=[12,8])
		df_yearly = df.set_index("date").resample("1Y").mean()
		y = df_yearly[var]
		x = df_yearly.year
		m,std = bootstrap_slope(x,y,1000)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" units y^-1"
		sb.regplot("year",var,df_yearly,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.legend(fontsize=10)
	elif method=="amax":
		plt.figure(figsize=[12,8])
		df_yearly = df.set_index("date").resample("1Y").max()
		y = df_yearly[var]
		x = df_yearly.year
		m,std = bootstrap_slope(x,y,1000)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" units y^-1"
		sb.regplot("year",var,df_yearly,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.legend(fontsize=10)
	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+\
		dataset+"_"+loc+"_"+var+"_"+method+"_"+str(months[0])+"_"+str(months[-1])+".tiff",\
		bbox_inches="tight")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/trends/"+\
	#	dataset+"_"+loc+"_"+var+"_"+method+"_"+str(months[0])+"_"+str(months[-1])+".png",\
	#	bbox_inches="tight")
	plt.close()

def interannual_time_series(df_list,var,names,loc,year_range,method,months=np.arange(1,13,1),\
		percentile=0):

	#Takes a list of dataframes with reset index, and plots an interannual time series for
	# each dataset at a location

	#Depending on the method, the time series is either:
	#	- Monthly mean
	#	- Days exceeding 99th percentile annually

	#"months" defines the months used for this analysis

	plt.figure()

	for i in np.arange(0,len(df_list)):
		temp = df_list[i][(df_list[i].year>=year_range[0]) & \
				(df_list[i].year<=year_range[1]) & \
				(df_list[i].stn_name==loc) & (np.in1d(df_list[i].month,months))]\
				.sort_values("date").set_index("date")
		if method=="am":
			temp = temp.resample("1Y").mean()
			temp[var[i]].plot(label=names[i])
		elif method=="percentile":
			p99 = np.percentile(temp[var[i]],percentile)
			years = temp.year.unique()
			event_years = [temp[(temp[var[i]]>=p99) & (temp.year == y)].shape[0] \
				for y in years]
			plt.plot(years,event_years,label=names[i]+" "+str(round(p99,3)),marker="s")
	plt.legend()
	plt.show()
		
def wind_gust_boxplot(df,aws,var,loc=False,season=np.arange(1,13,1),two_thirds=True):

	#For a reanalysis dataframe (df), create a boxplot for parameter "var" for each wind speed gust
	# category

	if loc==False:
		df = df.reset_index().set_index(["date","stn_name"])
		aws = aws.reset_index().set_index(["date","stn_name"])
	else:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		aws = aws[(aws.stn_name == loc)].reset_index().set_index("date")

	df = pd.concat([df,aws.wind_gust],axis=1)

	#if "mlcape*s06" == var:
	#	df["mlcape*s06"] = df["ml_cape"] * np.power(df["s06"],1.67)

	df1 = df[(df.wind_gust >= 5) & (df.wind_gust<15) & (np.in1d(df.month,np.array(season)))]
	df2 = df[(df.wind_gust >= 15) & (df.wind_gust<25) & (np.in1d(df.month,np.array(season)))]
	df3 = df[(df.wind_gust >= 25) & (df.wind_gust<30) & (np.in1d(df.month,np.array(season)))]
	df4 = df[(df.wind_gust >= 30) & (np.in1d(df.month,np.array(season)))]
	#df5 = df[(df.wind_gust >= 35) & (df.wind_gust<40) & (np.in1d(df.month,np.array(season)))]
	#df6 = df[(df.wind_gust >= 40) & (np.in1d(df.month,np.array(season)))]

	plt.figure(figsize=[8,5])

	if two_thirds:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var],df5[var],df6[var]],whis=[33,100],\
			labels=["5-15 m/s\n("+str(df1.shape[0])+")","15-25 m/s\n("+str(df2.shape[0])+")",\
			"25-30 m/s\n("+str(df3.shape[0])+")","30-35 m/s\n("+str(df4.shape[0])+")",\
			"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	else:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var]],
			#df5[var],df6[var]],
			whis="range",\
			labels=["5-15 m.s$^{-1}$\n("+str(df1.shape[0])+")","15-25 m.s$^{-1}$\n("+str(df2.shape[0])+")",\
			"25-30 m.s$^{-1}$\n("+str(df3.shape[0])+")","30+ m.s$^{-1}$\n("+str(df4.shape[0])+")"])
			#"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	plt.xticks(fontsize="xx-large")
	plt.yticks(fontsize="xx-large")

	if loc==False:
		plt.title("All Stations",fontsize="x-large")
	else:
		plt.title(loc,fontsize="xx-large")
	t1,t2,t3,units,t4,log_plot,t6 = contour_properties(var)
	plt.ylabel(units,fontsize="xx-large")

	if log_plot:
		plt.yscale("symlog")
		plt.ylim(ymin=0)

	if var == "mu_cape":
		plt.ylim([0,5000])
	elif var == "s06":
		plt.ylim([0,65])

	if loc==False:
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_AllStations_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".png",\
			bbox_inches="tight")
	else:
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/boxplot_"+loc+"_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".tiff",\
			bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_"+loc+"_"+var+"_"+\
		#	str(season[0])+"_"+str(season[-1])+".png",\
		#	bbox_inches="tight")
	plt.close()

def plot_conv_seasonal_cycle(df,loc,var,trend=False,mf_days=False):

	#For a reanalysis dataframe (df), create a seasonal mean plot for parameter "var" for each wind speed gust
	# category
	
	if len(var) == 2:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		#barra = barra[(barra.stn_name == loc)].reset_index().set_index("date")
		fig,ax = plt.subplots()
		ax.plot(np.arange(1,13,1),[np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)],\
			color="b")
		#ax.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[0]]) for m in np.arange(1,13,1)],\
		#	color="b",linestyle="--")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[0])
		ax.set_ylabel(units,fontsize="xx-large")
		ax.tick_params(labelcolor="b",axis="y",labelsize="xx-large")
		ax2 = ax.twinx()
		ax2.plot(np.arange(1,13,1),[np.mean(df[df.month==m][var[1]]) for m in np.arange(1,13,1)],\
			color="red")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax2.tick_params(labelcolor="red",axis="y",labelsize="xx-large")
		ax2.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.set_xticks(np.arange(1,13))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[1])
		ax2.set_ylabel(units,fontsize="xx-large")
		plt.title(loc,fontsize="xx-large")
		ax.set_xlabel("Months",fontsize="xx-large")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
			"_"+var[0]+"_"+var[1]+".tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
		#	"_"+var[0]+"_"+var[1]+".png",bbox_inches="tight")
	if len(var) == 3:
		df = df[(df.loc_id == loc)].reset_index().set_index("date")
		#barra = barra[(barra.stn_name == loc)].reset_index().set_index("date")
		fig,ax = plt.subplots(figsize=[8,6])

		x = [np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)]
		ax.plot(np.arange(1,13,1), three_month_average(x),\
			color="b")
		#ax.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[0]]) for m in np.arange(1,13,1)],\
		#	color="b",linestyle="--")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[0])
		ax.set_ylabel(units,fontsize="xx-large",color="b")
		ax.tick_params(labelcolor="b",axis="y",labelsize="xx-large")

		ax2 = ax.twinx()
		x = [np.mean(df[df.month==m][var[1]]) for m in np.arange(1,13,1)]
		ax2.plot(np.arange(1,13,1), three_month_average(x),\
			color="red")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax2.tick_params(labelcolor="red",axis="y",labelsize="xx-large")
		ax2.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[1])
		#ax2.set_ylabel(units,fontsize="xx-large")

		ax3 = ax.twinx()
		x = [np.mean(df[df.month==m][var[2]]) for m in np.arange(1,13,1)]
		ax3.plot(np.arange(1,13,1), three_month_average(x),\
			color="green")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax3.tick_params(labelcolor="green",axis="y",labelsize="xx-large")
		ax3.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[2])
		ax3.set_ylabel(units,fontsize="xx-large")

		ax.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.set_xticks(np.arange(1,13))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		plt.title(loc,fontsize="xx-large")
		ax.set_xlabel("Months",fontsize="xx-large")
		ax3.tick_params(pad=30)
		fig.subplots_adjust(right=0.8)
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
			"_"+var[0]+"_"+var[1]+".tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
		#	"_"+var[0]+"_"+var[1]+".png",bbox_inches="tight")
	else:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		if trend:
			if mf_days:
				plt.plot([np.mean(df[(df.month==m) & (df.year<=1998) & (df.mf==1)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1979-1998",color="b")
				plt.plot([np.mean(df[(df.month==m) & (df.year>=1998) & (df.mf==1)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1998-2017",color="b",linestyle="--")
				plt.title(loc)
				plt.legend(fonsize="xx-large")
				ax=plt.gca()
				ax.tick_params(labelsize="xx-large")
				plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
						"_"+var[0]+"_trend_mf_days.png",bbox_inches="tight")
			else:
				plt.plot([np.mean(df[(df.month==m) & (df.year<=1998)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1979-1998",color="b")
				plt.plot([np.mean(df[(df.month==m) & (df.year>=1998)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1998-2017",color="b",linestyle="--")
				plt.title(loc,fontsize="xx-large")
				plt.legend(fontsize="xx-large")
				ax=plt.gca()
				ax.tick_params(labelsize="xx-large")
				plt.ylabel("J.kg$^{-1}$",fontsize="xx-large")
				ax.set_xticks(np.arange(0,12))
				ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
				plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
						"_"+var[0]+"_trend.tiff",bbox_inches="tight")
				#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
				#		"_"+var[0]+"_trend.png",bbox_inches="tight")
		else:
			plt.plot([np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)])
			plt.title(loc)
			plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
					"_"+var[0]+".png",bbox_inches="tight")
	plt.close()

def magnitude_trends(param,hr,param_names):

	#Attempt to identify trends in a range of magnitudes, following the method of Dowdy (2014)
	#Param is a string, hr is an array or percentiles which identify a daignostic threshold

	#Load datasets and combine
	df = analyse_events("jdh","sa_small")
	df = df.dropna(subset=["wind_gust"])
	
	markers=["s","o","^","+","v"]
	cols = ["b","r","g","c","y"]
	plt.figure(figsize=[12,5])
	for i in np.arange(len(hr)):
		#Find the proportion of event days given all days, as a function of wind speed catergory
		speed_cats = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50]]
		speed_labs = ["0-5","5-10","10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50"]
		all_days = np.zeros(len(speed_cats))
		event_days = np.zeros(len(speed_cats))
		for s in np.arange(len(speed_cats)):
			all_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				(df["wind_gust"]<speed_cats[s][1])].shape[0]
			if hr[i] == 1:
				event_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				    (df["wind_gust"]<speed_cats[s][1]) & \
				    (df[param[i]]==hr[i])].shape[0]
				lab = "Days with "+param_names[i]+" = 1"
			else:
				event_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				    (df["wind_gust"]<speed_cats[s][1]) & \
				    (df[param[i]]>=np.percentile(df[df.jdh==1][param[i]],hr[i]))].shape[0]
				lab = "Days with "+param_names[i]+" = "+\
					str(np.percentile(df[df.jdh==1][param[i]],hr[i]).round(3))

		#Plot the relationship
		if i==0:
			plt.plot(np.arange(len(speed_cats)),all_days,"kx",linestyle="none",markersize=12,\
				label = "All days")
		plt.plot(np.arange(len(speed_cats)),event_days,color=cols[i],marker=markers[i],linestyle="none",\
			fillstyle="none",markersize=12,label = lab)
		print(lab+"\n"+str((event_days/all_days.astype(float)).round(3)))

	plt.yscale("log")
	plt.ylabel("Number of Days",fontsize="xx-large")
	plt.xlabel("Wind Speed (m.s$^{-1}$)",fontsize="xx-large")
	ax=plt.gca();ax.set_xticks(np.arange(len(speed_cats)));ax.set_xticklabels(speed_labs)
	ax.tick_params(labelsize="xx-large")
	plt.xlim([-0.5,len(speed_cats)]);plt.ylim([0.5,10e4])
	plt.legend(numpoints=1,fontsize="xx-large")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/magnitude_trends.png",bbox_inches="tight")
	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/magnitude_trends.tiff",bbox_inches="tight")

def plot_aus_station_wind_gusts():

	from adjustText import adjust_text
	
	df = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_aus_1979_2017.pkl")
	plt.figure()
	locs = df.stn_name.unique()
	max_gusts = np.array([df[df.stn_name==l]["wind_gust"].max() for l in locs])
	no_of_gusts = np.array([df[(df.stn_name==l)&(df.wind_gust>=30)].shape[0] for l in locs])
	no_of_points = np.array([df[(df.stn_name==l)].shape[0] for l in locs])
	plt.plot(max_gusts,no_of_gusts/no_of_points.astype(float),"bo")
	texts = [plt.text(max_gusts[i],no_of_gusts[i]/float(no_of_points[i]),locs[i]) for i in np.arange(len(locs))]
	adjust_text(texts,arrowprops={"arrowstyle":"->","color":"red"})
	plt.xlabel("Max wind gust (m/s)")
	plt.ylabel("No. of gusts above 30 m/s")
	plt.xlim([28,60])
	plt.show()

def plot_diurnal_wind_distribution():
	#Plot the diurnal distributions of AWS-defined wind gust events

	#Load unsampled (half-horuly) aws data
	aws_30min = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
			"all_wind_gusts_sa_1985_2017.pkl").reset_index().sort_values(["date","stn_name"])
	#Change to local time
	aws_30min["date_lt"] = aws_30min.date + dt.timedelta(hours = -10.5)
	aws_30min["hour"] = [x.hour for x in aws_30min.date_lt]
	aws_30min["month"] = [x.month for x in aws_30min.date_lt]
	#Attempt to eliminate erroneous values
	inds = aws_30min[aws_30min.wind_gust>=40].index.values
	err_cnt=0
	for i in inds:
		prev = aws_30min.loc[i-1].wind_gust
		next = aws_30min.loc[i+1].wind_gust
		if (prev < 20) & (next < 20):
			aws_30min.wind_gust.iat[i] = np.nan
			err_cnt = err_cnt+1

	aws_30min_warm_inds = np.in1d(aws_30min.month,np.array([10,11,12,1,2,3]))

	#Get diurnal distribution for 30 min data
	for loc in ["Adelaide AP","Port Augusta","Woomera","Mount Gambier"]:
		aws_30min_5_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=5) & (aws_30min.wind_gust<15)])
		aws_30min_15_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=15) & (aws_30min.wind_gust<25)])
		aws_30min_25_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=25) & (aws_30min.wind_gust<30)])
		aws_30min_30_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=30)])

	#Get diurnal distribution for 6 hourly data
		plt.figure()
		plt.plot(hours,aws_30min_15_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_25_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_5_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_30_cnt,marker="x",linestyle="none",color="k",\
				markeredgewidth=1.5,markersize=17.5)
		plt.ylabel("Count",fontsize="large")
		plt.xlabel("Hour (Local Time)",fontsize="large")
		plt.yscale("log")
		plt.title(loc)
		plt.xlim([0,24]);plt.ylim([0.5,plt.ylim()[1]])
		ax = plt.gca()
		ax.tick_params(labelsize=20)
		plt.grid()
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
			"diurnal_"+loc+".png",bbox_inches="tight")
		plt.close()


def plot_daily_data_monthly_dist(aws,stns,outname):

	#Plot seasonal distribution of wind gusts over certain thresholds for each AWS station
	s = 11
	no_years = float(2018-1979)
	for i in np.arange(0,len(stns)):
		fig = plt.figure(figsize=[8,6])
		aws_mth_cnt5,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[5,15])
		aws_mth_cnt15,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[15,25])
		aws_mth_cnt25,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[25,30])
		aws_mth_cnt30,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[30])
		plt.plot(mths,aws_mth_cnt15/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt25/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt5/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt30/no_years,linestyle="none",marker="x",markersize=17.5,color="k",\
			markeredgewidth=1.5)
		plt.title(stns[i],fontsize="xx-large")
		plt.xlabel("Month",fontsize="xx-large");plt.ylabel("Gusts per month",fontsize="xx-large")
		#plt.legend(loc="upper left")
		ax=plt.gca();ax.set_yscale('log')
		ax.set_xticks(np.arange(1,13,1))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		ax.set_xlim([0.01,12.5])
		ax.set_ylim([0.01,ax.get_ylim()[1]])
		ax.tick_params(labelsize="xx-large")
		ax.grid()
		fig.subplots_adjust(bottom=0.2)
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"\
			+outname+"_"+"monthly_"+stns[i]+"_1979_2017.tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"\
		#	+outname+"_"+"monthly_"+stns[i]+"_1979_2017.png")
		plt.close()

def get_diurnal_dist(aws):
	hours = np.arange(0,24,1)
	hour_counts = np.empty(len(hours))
	for h in np.arange(0,len(hours)):
		hour_counts[h] = (aws.hour==hours[h]).sum()
	return hour_counts,hours

def get_monthly_dist(aws,threshold=0,mean=False):
	#Get the distribution of months in a dataframe. If threshold is specified, restrict the 
	# dataframe to where the field "wind_gust" is above that threshold
	#If mean = "var", then also get the monthly mean for "var"
	months = np.sort(aws.month.unique())
	month_counts = np.empty(len(months))
	month_mean = np.empty(len(months))
	for m in np.arange(0,len(months)):
		if len(threshold)==1:
			month_count = ((aws[aws.wind_gust>=threshold[0]].month==months[m]).sum())
		elif len(threshold)==2:
			month_count = ((aws[(aws.wind_gust>=threshold[0]) & \
				(aws.wind_gust<threshold[1])].month==months[m]).sum())
		total_month = ((aws.month==months[m]).sum())
		#month_counts[m] = month_count / float(total_month)
		#month_counts[m] = month_count / float(aws[aws.wind_gust>=threshold].shape[0])
		month_counts[m] = month_count
		if mean != False:
			month_mean[m] = aws[aws.month==months[m]][mean].mean()
	if mean != False:
		return month_counts,month_mean,months
	else:
		return month_counts,months

def plot_stations():
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
		"all_daily_max_wind_gusts_sa_1979_2017.pkl")
	stns = aws.stn_name.unique()
	lons = aws.lon.unique()
	lats = aws.lat.unique()
	#remove Port Augusta power station coordinates, as it has merged with Port Augusta in aws df
	lats = lats[~(lats==-32.528)]
	lons = lons[~(lons==137.79)]
	
	start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	m = Basemap(llcrnrlon=start_lon, llcrnrlat=start_lat, urcrnrlon=end_lon, urcrnrlat=end_lat,\
		projection="cyl",resolution="i")
	m.drawcoastlines()

	for i in np.arange(0,len(stns)):
		x,y = m(lons[i],lats[i])
		plt.annotate(stns[i],xy=(x,y),color="k",size="small")
		plt.plot(x,y,"ro")
	plt.show()

def plot_netcdf(domain,fname,outname,time,model,vars=False):

#Load a single netcdf file and plot time max if time is a list of [start_time,end_time]
# or else plot for a single time if time is length=1

	f = nc.Dataset(fname)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)

	if vars == False:
		vars = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
		vars = vars[~(vars=="time") & ~(vars=="lat") & ~(vars=="lon")]	
	
	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "i")
	for param in vars:
		if len(time)==2:
			values = np.nanmax(f.variables[param][(times_dt>=time[0]) & \
				(times_dt<=time[-1])],axis=0)
			plt.title(str(time[0]) + "-" + str(time[-1]))
		else:
			if param == "mlcape*s06":
				values = np.squeeze(f.variables["ml_cape"][times_dt == time] * \
					np.power(f.variables["s06"][times_dt == time],1.67))
			elif param == "cond":
				mlcape = np.squeeze(f.variables["ml_cape"][times_dt == time])
				dcape = np.squeeze(f.variables["dcape"][times_dt == time])
				s06 = np.squeeze(f.variables["s06"][times_dt == time])
				mlm = np.squeeze(f.variables["mlm"][times_dt == time])
				sf = ((s06>=30) & (dcape<500) & (mlm>=26) ) 
				wf = ((mlcape>120) & (dcape>350) & (mlm<26) ) 
				values = sf | wf 
				values = values * 1.0; sf = sf * 1.0; wf = wf * 1.0
				values_disc = np.zeros(values.shape)
				#Separate conditions 1 and 2 (strong forcing and weak forcing)
				#Make in to an array with 1's for SF, 2's for WF and 3's for both types
				values_disc[sf==1] = 1
				values_disc[wf==1] = 2
			else:
				values = np.squeeze(f.variables[param][times_dt == time])
			plt.title(str(time[0]))

		print(param)
		[cmap,mean_levels,levels,cb_lab,range,log_plot,threshold] = contour_properties(param)

		plt.figure()

		m.drawcoastlines()
		if (domain[0]==-38) & (domain[1]==-26):
			m.drawmeridians([134,137,140],\
					labels=[True,False,False,True],fontsize="xx-large")
			m.drawparallels([-36,-34,-32,-30,-28],\
					labels=[True,False,True,False],fontsize="xx-large")
		else:
			m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
					labels=[True,False,False,True])
			m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
					labels=[True,False,True,False])
		if param == "cond":
			m.pcolor(x,y,values,latlon=True,cmap=plt.get_cmap("Reds",2))
		else:
			m.contourf(x,y,values,latlon=True,cmap=cmap,levels=levels,extend="max")
		cb = plt.colorbar()
		cb.set_label(cb_lab,fontsize="xx-large")
		cb.ax.tick_params(labelsize="xx-large")
		if param == "cond":
			cb.set_ticks([0,1])
		m.contour(x,y,values,latlon=True,colors="grey",levels=threshold)
		if (outname == "system_black_2016092806"):
			tx = (138.5159,138.3505,138.0996)
			ty = (-33.7804,-32.8809,-32.6487)
			m.plot(tx,ty,color="k",linestyle="none",marker="^",markersize=10)
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+\
		#	"_"+param+".png",bbox_inches="tight")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+model+"_"+outname+\
			"_"+param+".tiff",bbox_inches="tight")
		plt.close()
		#IF COND, ALSO DRAW COND TYPES
		if param == "cond":
			plt.figure()
			m.drawcoastlines()
			if (domain[0]==-38) & (domain[1]==-26):
				m.drawmeridians([134,137,140],\
						labels=[True,False,False,True],fontsize="xx-large")
				m.drawparallels([-36,-34,-32,-30,-28],\
						labels=[True,False,True,False],fontsize="xx-large")
			else:
				m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
						labels=[True,False,False,True])
				m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
						labels=[True,False,True,False])
			m.pcolor(x,y,values_disc,latlon=True,cmap=plt.get_cmap("Accent_r",3),vmin=0,vmax=2)
			cb = plt.colorbar()
			cb.set_label("Forcing type",fontsize="xx-large")
			cb.set_ticks([0,1,2,3])
			cb.set_ticklabels(["None","SF","MF","Both"])
			cb.ax.tick_params(labelsize="xx-large")
			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+model+"_"+outname+\
				"_"+"forcing_types"+".tiff",bbox_inches="tight")
			#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+\
			#	"_"+"forcing_types"+".png",bbox_inches="tight")
			plt.close()

def plot_netcdf_animate(fname,param,outname,domain):

#Load a single netcdf file and plot the time animation

	global values, times_str, x, y, levels, cmap, cb_lab, m

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, bitrate=1800)

	f = nc.Dataset(fname)
	values = f.variables[param][:]
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)
	times_str = [dt.datetime.strftime(t,"%Y-%m-%d %H:%M") for t in times_dt]

	[cmap,levels,cb_lab] = contour_properties(param)

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "l")
	
	fig = plt.figure()
	m.contourf(x,y,np.zeros(x.shape),latlon=True,levels=levels,cmap=cmap,extend="both")
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	cb = plt.colorbar()
	cb.set_label(cb_lab)
	anim = animation.FuncAnimation(fig, animate, frames=values.shape[0],interval=500)
	anim.save("/home/548/ab4502/working/ExtremeWind/figs/mean/"+outname+".mp4",\
			writer=writer)
	plt.show()

def animate(i):
	z = values[i]
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	im = m.contourf(x,y,values[i],latlon=True,levels=levels,cmap=cmap,extend="both")
	plt.title(times_str[i] + " UTC")
	return im
	
def contour_properties(param):
	threshold = [1]
	if param in ["mu_cape","cape","cape700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,2000,11)
		extreme_levels = np.linspace(0,2000,11)
		cb_lab = "J.kg$^{-1}$"
		range = [0,4000]
		log_plot = True
	if param in ["dlm*dcape*cs6","mlm*dcape*cs6"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,4,11)
		extreme_levels = np.linspace(0,40,11)
		cb_lab = ""
		range = [0,40]
		log_plot = True
		threshold = [0.669]
	elif param == "ml_cape":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,60,11)
		extreme_levels = np.linspace(0,2000,11)
		cb_lab = "J.kg$^{-1}$"
		threshold = [127]
		range = [0,4000]
		log_plot = True
	elif param == "mu_cin":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
		range = [0,500]
		log_plot = True
	elif param == "ml_cin":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
		range = [0,600]
		log_plot = True
	elif param in ["s06","ssfc6"]:
		cmap = cm.Reds
		mean_levels = np.linspace(14,18,17)
		extreme_levels = np.linspace(0,50,11)
		cb_lab = "m.s$^{-1}$"
		threshold = [23.83]
		range = [0,60]
		log_plot = False
	elif param in ["ssfc3"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,30,7)
		extreme_levels = np.linspace(0,40,11)
		cb_lab = "m/s"
		range = [0,30]
		log_plot = False
	elif param in ["ssfc850"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,20,7)
		extreme_levels = np.linspace(0,30,11)
		cb_lab = "m/s"
		range = [0,25]
		log_plot = False
	elif param in ["ssfc500","ssfc1"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,20,7)
		extreme_levels = np.linspace(0,30,11)
		cb_lab = "m/s"
		range = [0,20]
		log_plot = False
	elif param in ["dcp", "dcp2"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,4,11)
		cb_lab = "DCP"
		range = [0,3]
		threshold = [0.028,1]
		log_plot = True
	elif param in ["scp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,4,11)
		cb_lab = ""
		range = [0,3]
		threshold = [0.025,1]
		log_plot = True
	elif param in ["stp","ship","non_sc_stp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,2.5,11)
		cb_lab = ""
		threshold = [0.027,1]
		range = [0,3]
		log_plot = True
	elif param in ["mmp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,1,11)
		extreme_levels = np.linspace(0,1,11)
		cb_lab = ""
		range = [0,1]
		log_plot = True
	elif param in ["srh01","srh03","srh06"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,600,11)
		cb_lab = "m^2/s^2"
		range = [0,400]
		log_plot = True
	elif param == "crt":
		cmap = cm.YlGnBu
		mean_levels = [60,120]
		extreme_levels = [60,120]
		cb_lab = "degrees"
		range = [0,180]
		log_plot = False
	elif param in ["relhum850-500","relhum1000-700","hur850","hur700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,100,11)
		extreme_levels = np.linspace(0,100,11)
		cb_lab = "%"
		range = [0,100]
		log_plot = False
	elif param == "vo":
		cmap = cm.RdBu_r
		mean_levels = np.linspace(-8e-5,8e-5,17)
		extreme_levels = np.linspace(-8e-5,8e-5,17)
		cb_lab = "s^-1"
		range = [-8e-5,-8e-5]
		log_plot = False
	elif param == "lr1000":
		cmap = cm.Blues
		mean_levels = np.linspace(2,12,11)
		extreme_levels = np.linspace(2,12,11)
		cb_lab = "deg/km"
		range = [0,12]
		log_plot = False
	elif param == "lcl":
		cmap = cm.YlGnBu_r
		mean_levels = np.linspace(0,8000,9)
		extreme_levels = np.linspace(0,8000,9)
		cb_lab = "m"
		range = [0,8000]
		log_plot = False
	elif param in ["td800","td850","td950"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,30,9)
		extreme_levels = np.linspace(0,30,9)
		cb_lab = "deg"
		range = [0,40]
		log_plot = False
	elif param in ["cape*s06","cape*ssfc6","mlcape*s06"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,500000,11)
		extreme_levels = np.linspace(0,500000,11)
		cb_lab = ""
		range = [0,100000]
		threshold = [10344.542,68000]
		log_plot = True
	elif param in ["cape*td850"]:
		cmap = cm.YlGnBu
		mean_levels = None
		extreme_levels = None
		cb_lab = ""
		range = None
		log_plot = True
	elif param in ["dp850-500","dp1000-700","dp850","dp700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(-10,10,11)
		extreme_levels = np.linspace(-10,10,11)
		cb_lab = "degC"
		range = [-20,20]
		log_plot = False
	elif param in ["dlm","mlm","Umean06","Umean800_600"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(5,15,11)
		extreme_levels = np.linspace(0,50,11)
		threshold = [1]
		cb_lab = "m/s"
		range = [0,50]
		log_plot = False
	elif param in ["dcape"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,400,11)
		extreme_levels = np.linspace(0,2000,11)
		threshold = [500]
		cb_lab = "J/kg"
		range = [0,2000]
		log_plot = False
	elif param in ["mfper"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,0.1,11)
		extreme_levels = np.linspace(0,1,2)
		threshold = [1]
		cb_lab = "% of WF"
		range = [0,1]
		log_plot = False
	elif param in ["cond","sf","mf"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.1,11)
		extreme_levels = np.linspace(0,1,2)
		threshold = [1]
		cb_lab = "CEWP"
		range = [0,1]
		log_plot = False
	elif param in ["max_wg10","wg10"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [12.817,21.5]
		cb_lab = "no. of days"
		range = [0,30]
		log_plot = False
	elif param in ["tas","ta850","ta700","tos"]:
		cmap = cm.Reds
		mean_levels = np.linspace(285,320,5)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [1]
		cb_lab = "K"
		range = [-20,20]
		log_plot = False
	elif param in ["ta2d"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [12.817,21.5]
		cb_lab = "no. of days"
		range = [-20,20]
		log_plot = False
	else:
		cmap = cm.Reds
		mean_levels = None
		extreme_levels = None
		threshold = None
		cb_lab = ""
		range = None
		log_plot = False
	
	return [cmap,mean_levels,extreme_levels,cb_lab,range,log_plot,threshold]

def degToCompass(num):
	if np.isnan(num):
		pass
	else:
		val=int((num/22.5)+.5)
		arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
		return(arr[(val % 16)])

if __name__ == "__main__":

	path = '/g/data/eg3/ab4502/ExtremeWind/aus/'
	f = "erai_wrf_20140622_20140624"
	model = "erai"
	fname = path+f+".nc"
	param = "cape*s06"
	region = "sa_small"

	time = [dt.datetime(2014,10,6,6,0,0),dt.datetime(2014,6,10,12,0,0)]
	outname = param+"_"+dt.datetime.strftime(time[0],"%Y%m%d_%H%M")+"_"+\
			dt.datetime.strftime(time[-1],"%Y%m%d_%H%M")

	#NOTE PUT IN FUNCTION
	if region == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	else:
	    raise NameError("Region must be one of ""aus"", ""sa_small"" or ""sa_large""")
	domain = [start_lat,end_lat,start_lon,end_lon]

#	PLOT SEASONAL DISTRIBUTIONS
	#plot_daily_data_monthly_dist(pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
	#	"erai_fc_points_1979_2017_daily_max.pkl").reset_index().\
	#	rename(columns={"wg10":"wind_gust","loc_id":"stn_name"}),\
	#	["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="ERA-Interim")
	#barra_r_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
	#	"barra_r_fc_points_daily_2003_2016.pkl").reset_index()
	#barra_r_df["month"] = [t.month for t in barra_r_df.date]
	#plot_daily_data_monthly_dist(barra_r_df.rename(columns={"max_wg10":"wind_gust","loc_id":"stn_name"}),\
	#	["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="BARRA-R")
	#plot_daily_data_monthly_dist(remove_incomplete_aws_years(\
	#		pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
	#		"all_daily_max_wind_gusts_sa_1979_2017.pkl"),"Port Augusta").reset_index()\
	#		,["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="AWS")
	#barra_ad_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
	#	"barra_ad_points_daily_2006_2016.pkl").reset_index()
	#barra_ad_df["month"] = [t.month for t in barra_ad_df.date]
	#plot_daily_data_monthly_dist(barra_ad_df.rename(columns={"max_wg10":"wind_gust","loc_id":"stn_name"}),\
		#["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="BARRA-AD")

	#CASE STUDIES
	#for time in date_seq([dt.datetime(1979,11,14,0),dt.datetime(1979,11,14,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"+\
	#		"erai_19791101_19791130.nc"\
	#		,"event_1979_"+time.strftime("%Y%m%d%H"),\
	#		[time],"erai",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"+\
	#		"erai_20160901_20160930.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"erai",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"+\
	#		"barra_20160901_20160930.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"barra",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_ad/"+\
	#		"barra_ad_20160928_20160929.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"barra_ad",vars=["scp"])

#	PLOT OBSERVED DIURNAL DISTRIBUTION
	#plot_diurnal_wind_distribution()

	#probability_plot("ml_cape","mlm")
	
	#INTERANNUAL TIME SERIES
	#erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	#erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_fc_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	#barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/barra_r_fc_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	#barra_r_fc["month"] = [t.month for t in barra_r_fc.date]
	#barra_r_fc["year"] = [t.year for t in barra_r_fc.date]
	#barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	#barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_daily_2006_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	#barra_ad["month"] = [t.month for t in barra_ad.date]
	#barra_ad["year"] = [t.year for t in barra_ad.date]
	#locs = ["Adelaide AP","Woomera","Mount Gambier","Port Augusta"]
	#locs = ["Adelaide AP"]
	#[var_trends(aws,"wind_gust",l,"threshold","AWS",threshold=[[15,25],[25,30],[30]]) \
	#	for l in locs]
	#[var_trends(erai_fc,"wg10",l,"threshold_only","ERA-Interim",threshold=[[15,25],[25,30],[30]]) \
	#	for l in locs]
	#[var_trends(barra_r_fc,"max_wg10",l,"threshold_only","BARRA-R",threshold=[[15,25],[25,30],[30]],year_range=[2003,2016]) \
	#	for l in locs]
	#[var_trends(barra_ad,"max_wg10",l,"threshold_only","BARRA-AD",threshold=[[15,25],[25,30],[30]],year_range=[2006,2016]) \
	#	for l in locs]

	#erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sa_small_1979_2017_daily_max.pkl")
	#locs = ["Woomera", "Port Augusta", "Adelaide AP", "Mount Gambier"]
	#[plot_conv_seasonal_cycle(erai, l, ["ml_cape", "s06", "cape*s06"] ) for l in locs]

	#plot_ranked_hss()
	obs_versus_mod("BARRA")
