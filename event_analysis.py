import os
import matplotlib.pyplot as plt
from plot_param import contour_properties
#from plot_clim import *
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

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


def optimise_pss_location(model_fname, event, T=1000, compute=True, l_thresh=2, param_list=None, test=False,\
		test_param=None):

	#Same as optimise pss, except that optimisation is done separately for each location

	#For erai: /g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl

	#model = pd.read_pickle(model_fname).set_index(["time", "loc_id"]).drop("points", axis=1)
	#obs = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/convective_wind_gust_aus_2005_2015.pkl")
	#There is one missing entry for parameters needing mhgt
	#df = pd.concat([obs, model], axis=1).dropna(subset=["lightning","mhgt"])
	#df = df[df.tc_affected==0]
	pss_df, df = optimise_pss(model_fname, compute=False, plot=False, l_thresh=l_thresh)
	if test:
		df, pss_df = test_pss(df, pss_df, test_param, "multiply", event, T=1000)
	else:
		pass
			
	df = df.reset_index().rename({"level_0":"date", "level_1":"loc_id"},axis=1)
	locs = np.unique(df.loc_id)

	if compute:

		if param_list == None:
			param_list = np.delete(np.array(model.columns), \
					np.where((np.array(model.columns)=="lat") | \
					(np.array(model.columns)=="lon")))

		#if event == "is_lightning":
		#	df.loc[:,"is_lightning"] = 0
		#	df.loc[df.lightning >= l_thresh, "is_lightning"] = 1
		#elif event == "is_conv_aws":
		#	df.loc[:,"is_conv_aws"] = 0
		#	df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws"] = 1
		#elif event == "is_sta":
		#	df.loc[:,"is_sta"] = 0
		#	df.loc[~(df["sta_wind"].isna()), "is_sta"] = 1
		#elif event == "is_conv_aws_cond_light":
		#	df.loc[:,"is_conv_aws_cond_light"] = np.nan
		#	df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws_cond_light"] = 1
		#	df.loc[(df.lightning >= l_thresh) & (df.wind_gust < 25), "is_conv_aws_cond_light"] = 0
		#else:
		#	raise ValueError("INVALID EVENT")

		pss_df = pd.DataFrame(index=locs, columns=param_list)
		threshold_df = pd.DataFrame(index=locs, columns=param_list)

		import multiprocessing
		import itertools
		pool = multiprocessing.Pool()

		#Optimise for discriminating convective AWS and lightning but non-convective AWS
		for l in locs:
			temp_df = df[(df.loc_id == l)]
			for p in param_list:
				print(l,p)
				if np.nansum(temp_df[event]) > 1:
					test_thresh = np.linspace(temp_df.loc[:,p].min(), \
						np.percentile(temp_df.loc[:,p],99.95) , T)
					temp_df2 = temp_df.loc[:,[event,p]]
					iterable = itertools.product(test_thresh, [temp_df2], [p], [event])
					res = pool.map(pss, iterable)
					thresh = [res[i][1] for i in np.arange(len(res))]
					pss_p = [res[i][0] for i in np.arange(len(res))]

					threshold_df.loc[l, p] = thresh[np.argmax(np.array(pss_p))]
					pss_df.loc[l, p] = np.array(pss_p).max()
					threshold_df.loc[l, "events"] = np.nansum(temp_df2[event])
					pss_df.loc[l, "events"] = np.nansum(temp_df2[event])
				else:
					threshold_df.loc[l, p] = np.nan
					pss_df.loc[l, p] = np.nan
					threshold_df.loc[l, "events"] = np.nansum(temp_df[event])
					pss_df.loc[l, "events"] = np.nansum(temp_df[event])


		pss_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/pss_df_locations_"+event+".pkl")
		threshold_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/thresholds_df_locations_"+event+".pkl")

	else:

		pss_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/pss_df_locations_"+event+".pkl")
		threshold_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/thresholds_df_locations_"+event+".pkl")

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")

	for l in locs:
		pss_df.loc[l, "lon"] = df[(df.loc_id==l)]["lon"].iloc[0].values[0]
		pss_df.loc[l, "lat"] = df[(df.loc_id==l)]["lat"].iloc[0].values[0]
		threshold_df.loc[l, "lon"] = df[(df.loc_id==l)]["lon"].iloc[0].values[0]
		threshold_df.loc[l, "lat"] = df[(df.loc_id==l)]["lat"].iloc[0].values[0]
	for p in param_list:
		plt.figure(figsize=[6,8])
		plt.subplot(311)
		m.drawcoastlines()
		m.scatter(pss_df["lon"], pss_df["lat"], c=pss_df[p], s=50, latlon=True,\
			 cmap = plt.get_cmap("Reds"), edgecolors="k")
		cb = plt.colorbar()
		cb.set_label("OPTIMISED PSS")
		plt.title(p) 		

		plt.subplot(312)
		m.drawcoastlines()
		m.scatter(pss_df["lon"], pss_df["lat"], c=threshold_df[p], s=50, latlon=True, \
			cmap = plt.get_cmap("Reds"), edgecolors="k")
		cb = plt.colorbar()
		cb.set_label("OPTIMAL THRESHOLD")
		plt.title(p) 		

		plt.subplot(313)
		m.drawcoastlines()
		m.scatter(pss_df["lon"], pss_df["lat"], c=threshold_df["events"], s=50, latlon=True, \
			cmap = plt.get_cmap("Greys"), edgecolors="k")
		cb = plt.colorbar()
		cb.set_label("OBSERVED EVENTS")
		plt.title(event)

	return [pss_df, threshold_df]


def optimise_pss(model_fname,T=1000, compute=True, plot=False, l_thresh=2, is_pss=True, model_name="erai"):

	#T is the number of linear steps between the max and min of a parameter, used to test the pss

	#For erai: /g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl
	#For barra: /g/data/eg3/ab4502/ExtremeWind/points/barra_points_wrfpython_aus_1979_2017.pkl

	model = pd.read_pickle(model_fname).set_index(["time", "loc_id"])
	obs = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/convective_wind_gust_aus_2005_2015.pkl")
	#There is one missing entry for parameters needing mhgt
	if model_name == "erai":
		df = pd.concat([obs, model], axis=1).dropna(subset=["lightning","mhgt"])
	else:
		df = pd.concat([obs, model], axis=1).dropna(subset=["lightning","ml_cape"])

	df = df[df.tc_affected==0]

	if compute:

		param_list = np.delete(np.array(model.columns), \
				np.where((np.array(model.columns)=="lat") | (np.array(model.columns)=="lon")))
		pss_df = pd.DataFrame(index=param_list, columns=["threshold_light","pss_light",\
			"threshold_conv_aws","pss_conv_aws",\
			"threshold_sta","pss_sta","threshold_conv_aws_cond_light","pss_conv_aws_cond_light"])

		import multiprocessing
		import itertools
		pool = multiprocessing.Pool()

		#Optimise for discriminating lightning and non-lightning
		if is_pss:
			print("OPTIMISING PSS FOR LIGHTNING...")
		else:
			print("OPTIMISING CSI FOR LIGHTNING...")
		df.loc[:,"is_lightning"] = 0
		df.loc[df.lightning >= l_thresh, "is_lightning"] = 1
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df.loc[:,p].min(), np.percentile(df.loc[:,p],99.95) , T)
			temp_df = df.loc[:,["is_lightning",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_lightning"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_light"] = np.array(pss_p).max()

		#Optimise for discriminating convective AWS and non-convective AWS
		if is_pss:
			print("OPTIMISING PSS FOR CONVECTIVE AWS EVENTS...")
		else:
			print("OPTIMISING CSI FOR CONVECTIVE AWS EVENTS...")
		df.loc[:,"is_conv_aws"] = 0
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws"] = 1
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df.loc[:,p].min(), np.percentile(df.loc[:,p],99.95) , T)
			temp_df = df.loc[:,["is_conv_aws",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_conv_aws"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_conv_aws"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_conv_aws"] = np.array(pss_p).max()

		#Optimise for discriminating STA wind and non-STA wind
		if is_pss:
			print("OPTIMISING PSS FOR STA WIND REPORTS...")
		else:
			print("OPTIMISING CSI FOR STA WIND REPORTS...")
		df.loc[:,"is_sta"] = 0
		df.loc[~(df["sta_wind"].isna()), "is_sta"] = 1
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df.loc[:,p].min(), np.percentile(df.loc[:,p],99.95) , T)
			temp_df = df.loc[:,["is_sta",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_sta"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_sta"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_sta"] = np.array(pss_p).max()

		#Optimise for discriminating convective AWS and lightning but non-convective AWS
		if is_pss:
			print("OPTIMISING PSS FOR CONVECTIVE WIND EVENTS VERSUS CONVECTIVE NON-WIND EVENTS...")
		else:
			print("OPTIMISING PSS FOR CONVECTIVE WIND EVENTS VERSUS CONVECTIVE NON-WIND EVENTS...")
		df.loc[:,"is_conv_aws_cond_light"] = np.nan
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws_cond_light"] = 1
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust < 25), "is_conv_aws_cond_light"] = 0
		for p in param_list:
			print(p)
			test_thresh = np.linspace(df.loc[:,p].min(), np.percentile(df.loc[:,p],99.95) , T)
			temp_df = df.loc[:,["is_conv_aws_cond_light",p]]
			iterable = itertools.product(test_thresh, [temp_df], [p], ["is_conv_aws_cond_light"], [is_pss])
			res = pool.map(pss, iterable)
			thresh = [res[i][1] for i in np.arange(len(res))]
			pss_p = [res[i][0] for i in np.arange(len(res))]

			pss_df.loc[p, "threshold_conv_aws_cond_light"] = thresh[np.argmax(np.array(pss_p))]
			pss_df.loc[p, "pss_conv_aws_cond_light"] = np.array(pss_p).max()

		if is_pss:
			pss_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/pss_df_lightning"+str(l_thresh)+\
				"_"+model_name+".pkl")
		else:
			pss_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/csi_df_lightning"+str(l_thresh)+\
				"_"+model_name+".pkl")

	else:

		if is_pss:
			pss_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/pss_df_lightning"+str(l_thresh)+".pkl")
		else:
			pss_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/csi_df_lightning"+str(l_thresh)+".pkl")

		df.loc[:,"is_lightning"] = 0
		df.loc[df.lightning >= l_thresh, "is_lightning"] = 1
		df.loc[:,"is_conv_aws"] = 0
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws"] = 1
		df.loc[:,"is_sta"] = 0
		df.loc[~(df["sta_wind"].isna()), "is_sta"] = 1
		df.loc[:,"is_conv_aws_cond_light"] = np.nan
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust >= 25), "is_conv_aws_cond_light"] = 1
		df.loc[(df.lightning >= l_thresh) & (df.wind_gust < 25), "is_conv_aws_cond_light"] = 0

	return pss_df, df

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
			cn = float(((df[event]==0) & (df[p]<=t)).sum())
			pss_ge = ( 2*(hits*cn - misses*fa) ) / \
				( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )
		elif score == "edi":
			cn = float(((df[event]==0) & (df[p]<=t)).sum())
			pod = (hits / (hits + misses))
			pofd = (fa / (cn + fa) )
			pss_ge = ( np.log(pofd) - np.log(pod) ) / (np.log(pofd) + np.log(pod) )
			

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
			cn = float(((df[event]==0) & (df[p]>=t)).sum())
			pss_l = ( 2*(hits*cn - misses*fa) ) / \
				( misses*misses + fa*fa + 2*hits*cn + (misses + fa) * (hits + cn) )
		elif score == "edi":
			cn = float(((df[event]==0) & (df[p]>=t)).sum())
			pod = (hits / (hits + misses))
			pofd = (fa / (cn + fa) )
			pss_l = ( np.log(pofd) - np.log(pod) ) / (np.log(pofd) + np.log(pod) )
		
		return [np.array([pss_ge, pss_l]).max(), t]

	except:
		if (score == "pss") | (score == "hss") | (score == "edi"):	
			return [-1, t]
		else:
			return [0,t]

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
	#df = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_aus_1979_2017.pkl")
	#loc_id = list(df.stn_name.unique())
	#points = []
	#for loc in loc_id:
	#	lon = df[df.stn_name==loc]["lon"].unique()[0]
	#	lat = df[df.stn_name==loc]["lat"].unique()[0]
	#	points.append((lon,lat))

	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	

	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
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

if __name__ == "__main__":

	#SOUTH AUSTRALIA
#########################################################################################################
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
			(140.7270,-36.9813),(1397164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]
#########################################################################################################
	#AUSTRALIA
#########################################################################################################
	#loc_id,points = get_aus_stn_info()

	#aws_model,model_df = plot_scatter(model)
	#plot_scatter(model,False,False)
	#df = load_AD_data()
	#df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_mean_2010_2015.csv",\
	#	float_format="%.3f")
	#load_jdh_points_barra(smooth=False)
	#df = get_wind_sa("erai")

	#EXTRACT DAILY POINT DATA FROM CONVECTIVE PARAMETER NETCDF FILES
	#load_jdh_points_erai(loc_id,points,fc=False,daily_max=True)
	#load_jdh_points_barra_ad(smooth=False)
	#load_jdh_points_barra_r_fc(daily_max=True,smooth=False)
	#load_netcdf_points_mf(points,loc_id,"sa_small","barra_ad",[2006,2016],"sa_small")

	#aws = remove_incomplete_aws_years(pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
	#	"all_daily_max_wind_gusts_sa_1979_2017.pkl"),"Port Augusta")
	#aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
	#	"all_daily_max_wind_gusts_sa_1979_2017.pkl")
	#erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"erai_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"erai_fc_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
	#	+"barra_r_fc_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_r_fc["month"] = [t.month for t in barra_r_fc.date]
	#barra_r_fc["year"] = [t.year for t in barra_r_fc.date]
	#barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"barra_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"\
	#	+"barra_ad_points_daily_2006_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_ad["month"] = [t.month for t in barra_ad.date]
	#barra_ad["year"] = [t.year for t in barra_ad.date]
	#interannual_time_series([aws,erai_fc],["wind_gust","wg10"],["AWS","ERA-Interim"],\
	#		"Adelaide AP",[1989,2017],"am",[10,11,12,1,2,3])
	#trend_table()
	#seasons = [np.arange(1,13,1),[11,12,1],[2,3,4],[5,6,7],[8,9,10]]
	#for loc in ["Woomera","Adelaide AP","Mount Gambier", "Port Augusta"]:
	  # plot_conv_seasonal_cycle(erai,loc,["ml_cape","s06"],trend=False,mf_days=False)
	   #for var in ["ml_cape","s06"]:
	     #for s in seasons:
		#wind_gust_boxplot(erai,aws,var,loc=loc,two_thirds=False)
	#far_table()
	#magnitude_trends(["cond"],[1],\
	#			["CEWP"])
	#plot_conv_seasonal_cycle(erai,"Adelaide AP",["ml_cape"],trend=True,mf_days=False)

	#pss_df, df = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_wrfpython_aus_1979_2017.pkl", \
	#	compute=True, l_thresh=100, is_pss="pss", model_name="barra")
	pss_df, df = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl", \
		compute=False, l_thresh=100, is_pss="pss")
	#df = df[df["ml_el"]>=6000]
	#plot_multivariate_density(df.dropna(subset=["wbz"]), "is_conv_aws", \
	#		"s06", "ml_cape", "dcape", "Umean800_600", special_cond="warm",log=True)
	#df, pss_df = test_pss(df, pss_df, ["k_index", "Umean800_600", "sherb"], "multiply", "is_conv_aws", T=1000)
	#pss_df, threshold_df = optimise_pss_location("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl", "is_conv_aws", param_list=["k_index*Umean800_600"], test=True, test_param=["k_index","Umean800_600"])

	rfe=False
	test_high_U = False
	try_logit_cv = False
	final_logit = True


	if try_logit_cv:

		#Train a logistic regression model N times, each time cross validating with 25% of the dataset
		# using CSI and POD

		from sklearn.model_selection import train_test_split
		from sklearn.linear_model import LogisticRegression
		from sklearn.preprocessing import MinMaxScaler
		import multiprocessing
		import itertools
		pool = multiprocessing.Pool()
		
		predictors = ["ml_el", "k_index", "t_totals", "Umean800_600", \
				"dcape", "Umean01", "s06", "s03", "U6", "qmean01", "ml_cape"]
		#predictors_logit = ["k_index", "Umean01", "Umean800_600", "t_totals", "s06", "dcape"]
		predictors_logit = ["ml_cape", "Umean01", "Umean800_600", "t_totals", "s06", "dcape"]
		test_cond = True
		test_param = True
		normalised = False
		N = 100
		iterable = itertools.product(np.arange(0,N), \
				[df[np.append(predictors,"is_conv_aws")]], \
				[predictors], [predictors_logit], [normalised], [test_cond], [test_param])
		print("Training Logit...")
		res = pool.map(run_logit, iterable)

		csi_logit = [res[i][0] for i in np.arange(0,len(res))]
		pod_logit = [res[i][1] for i in np.arange(0,len(res))]
		far_logit = [res[i][2] for i in np.arange(0,len(res))]
		csi_cond = [res[i][3] for i in np.arange(0,len(res))]
		pod_cond = [res[i][4] for i in np.arange(0,len(res))]
		far_cond = [res[i][5] for i in np.arange(0,len(res))]
		csi_param = [res[i][6] for i in np.arange(0,len(res))]
		pod_param = [res[i][7] for i in np.arange(0,len(res))]
		far_param = [res[i][8] for i in np.arange(0,len(res))]

		print(np.mean(csi_logit))
		print(np.mean(pod_logit))
		print(np.mean(far_logit))

		plt.figure();plt.boxplot([csi_logit, csi_cond, csi_param],labels=["logit","cond","t_totals"])
		plt.figure();plt.boxplot([pod_logit, pod_cond, pod_param],labels=["logit","cond","t_totals"])
		plt.figure();plt.boxplot([far_logit, far_cond, far_param],labels=["logit","cond","t_totals"])

	if rfe:
		sorted_pss_light = pss_df["pss_light"].sort_values(ascending=False)
		sorted_pss_conv_aws_cond_light = pss_df["pss_conv_aws_cond_light"].sort_values(ascending=False)

		test_predictors = np.unique(np.concatenate([sorted_pss_conv_aws_cond_light.index[0:10], \
			sorted_pss_light.index[0:10]]))
		logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000) 
		from sklearn.feature_selection import RFE, RFECV
		selector = RFE(logit, 5, 1, verbose=True)
		selector = selector.fit(\
			(df[test_predictors]-df[test_predictors].mean()) / (df[test_predictors].std()), \
			df["is_conv_aws"])

	if final_logit:
	
		#Develop a best-fit logistic equation using all available events


		from sklearn.linear_model import LogisticRegression
		#predictors_shallow = ["k_index", "Umean01", "Umean800_600", "t_totals", "s03", "dcape"]
		#predictors_deep = ["k_index", "Umean01", "Umean800_600", "t_totals", "s06", "dcape"]

		#logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
		#logit_mod_shallow = logit.fit(df.loc[df["ml_el"] < 6000, predictors_shallow], \
		#			df.loc[df["ml_el"] < 6000, "is_conv_aws"])
		#p_shallow = logit_mod_shallow.predict_proba(df.loc[df["ml_el"] < 6000, \
		#			predictors_shallow])[:,1]
		#df.loc[df["ml_el"] < 6000, "logit"] = (p_shallow >= 0.9) * 1
#
		#logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
		#logit_mod_deep = logit.fit(df.loc[df["ml_el"] >= 6000, predictors_deep], \
		#			df.loc[df["ml_el"] >= 6000, "is_conv_aws"])
		#p_deep = logit_mod_deep.predict_proba(df.loc[df["ml_el"] >= 6000, predictors_deep])[:,1]
		#df.loc[df["ml_el"] >= 6000, "logit"] = (p_deep >= 0.6) * 1

		predictors = ["k_index", "Umean01", "Umean800_600", "t_totals", "s03", "dcape"]
		logit = LogisticRegression(class_weight="balanced", solver="liblinear",max_iter=1000)
		logit_mod = logit.fit(df[predictors], df["is_conv_aws"])
		p = logit_mod.predict_proba(df[predictors])[:,1]
		df.loc[:, "logit"] = (p >= 0.8) * 1

		df["cond"] = ( (df["ml_el"]>=6000) & (df["k_index"]>=30) & (df["Umean800_600"]>=5) & \
				(df["t_totals"]>=46) ) | \
			( (df["ml_el"]<6000) & (df["Umean800_600"]>=16) & (df["k_index"]>=20) & \
				(df["Umean01"]>=10) & (df["s03"]>=10) )

		#Plot
		month_df = pd.concat({"conv_aws":df[df["is_conv_aws"]==1]["month"].value_counts(), \
					"sta":df[df["is_sta"]==1]["month"].value_counts(), \
					"logit":df[df["logit"]==1]["month"].value_counts(), \
					"cond":df[df["cond"]==1]["month"].value_counts()}, \
				axis=1, sort=False)
		bars = (month_df / month_df.sum())[["conv_aws","sta","cond","logit"]].\
			reindex([7,8,9,10,11,12,1,2,3,4,5,6]).\
			plot(kind="bar", color=["none", "none", "k", "grey"], edgecolor="k")
		for i in np.arange(0,12):
			bars.patches[i].set_hatch("///")
		plt.legend()

		wd_df = pd.concat({"conv_aws":df[df["is_conv_aws"]==1]["wind_dir"].value_counts(), \
					"sta":df[df["is_sta"]==1]["wind_dir"].value_counts(), \
					"logit":df[df["logit"]==1]["wind_dir"].value_counts(), \
					"cond":df[df["cond"]==1]["wind_dir"].value_counts()}, \
				axis=1, sort=False)
		bars = (wd_df / wd_df.sum())[["conv_aws","sta","cond","logit"]].\
			reindex(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW","SW", "WSW", "W", \
				"WNW", "NW", "NNW"]).\
			plot(kind="bar", color=["none", "none", "k", "grey"], edgecolor="k")
		for i in np.arange(0,16):
			bars.patches[i].set_hatch("///")
		plt.legend()
