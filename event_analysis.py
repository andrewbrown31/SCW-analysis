import os
import matplotlib.pyplot as plt
#from plot_param import *
#from plot_clim import *
import pandas as pd
import numpy as np

def load_netcdf_points_mf(points,loc_id,domain,model,year_range,out_domain):
	#Same as load_netcdf_points, but use MFDataset to lead in all netcdf points at once

	#Domain tells the function which netcdf dataset to load. out_domain just saves the resulting dataframe
	# with "domain" information, corresponding to whichever points are given

	#Load in convective parameter netcdf files for a given model/year range/domain
	if model == "barra_r_fc":
		#Special treatment of BARRA-R forecast model, as the number of variables aren't consisent 
		# throughout the files in /g/data/eg3/ab4502/ExtremeWind/sa_small/barra_r_fc/
		f = load_ncdf(domain,model,year_range,var_list=["max_wg10"],exclude_vars=True)
	else:
		f = load_ncdf(domain,model,year_range)

	#Get lat/lon inds to use based on points input, taking in to account the lsm
	if model == "erai":
		from erai_read import get_lat_lon,reform_lsm
		lon_orig,lat_orig = get_lat_lon()
		lsm = reform_lsm(lon_orig,lat_orig)
		smooth = False		#TURN SMOOTHING OFF FOR ERA-I (ALREADY 0.75 DEG)
	elif model == "barra":
		from barra_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").\
			variables["lnd_mask"][:]
	elif model == "barra_ad":
		from barra_ad_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/lnd_mask-fc-slv-PT0H-BARRA_AD-v1.nc")\
			.variables["lnd_mask"][:]
	elif model == "barra_r_fc":
		from barra_r_fc_read import get_lat_lon
		lon_orig,lat_orig = get_lat_lon()
		lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").\
			variables["lnd_mask"][:]
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
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

	times = nc.num2date(f.variables["time"][:],f.variables["time"].units)
	days = (np.array([x.day for x in times]))
	unique_days = np.unique(days)
	if model == "barra_r_fc":
		var = ["max_wg10"]
	else:
		var = np.array([str(list(f.variables.items())[i][0]) for i in np.arange(0,\
			len(list(f.variables.items())))])
		var = var[~(var=="time") & ~(var=="lat") & ~(var=="lon")]
	values = np.empty((len(points)*len(times),len(var)))
	values_lat = np.tile(np.array(points)[:,1],[len(times),1]).flatten()
	values_lon = np.tile(np.array(points)[:,0],[len(times),1]).flatten()
	values_lon_used = np.tile(lon_used,[len(times),1]).flatten()
	values_lat_used = np.tile(lat_used,[len(times),1]).flatten()
	values_loc_id = np.tile(np.array(loc_id),[len(times),1]).flatten()
	values_year = np.tile(np.array([t.year for t in times]),[len(points),1]).T.flatten()
	values_month = np.tile(np.array([t.month for t in times]),[len(points),1]).T.flatten()
	values_day = np.tile(np.array([t.day for t in times]),[len(points),1]).T.flatten()
	values_hour = np.tile(np.array([t.hour for t in times]),[len(points),1]).T.flatten()
	values_date = np.tile(times,[len(points),1]).T.flatten()

	cnt = 0
	print("EXTRACTING POINT DATA FROM NETCDF FILES...")
	for v in np.arange(len(var)):
		print(var[v])
		temp = f.variables[var[v]][:]
		values[:,v] = temp[:,lat_ind.astype(int),lon_ind.astype(int)].flatten()

	df = pd.DataFrame(values,columns=var,index=values_date)
	df["lat"] = values_lat
	df["lon"] = values_lon
	df["lon_used"] = values_lon_used
	df["lat_used"] = values_lat_used
	df["loc_id"] = values_loc_id
	df["year"] = values_year
	df["month"] = values_month
	df["day"] = values_day
	df["hour"] = values_hour

	#SAVE AT ORIGINAL FREQUENCY
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+model+"_points_"+out_domain+"_"+str(year_range[0])+"_"+\
			str(year_range[1])+".pkl")

	#FOR BARRA-R and BARRA-AD, CONVERT TO 6-HOURLY
	if (model == "barra_r_fc") | (model == "barra_ad"):
		df6 = df[np.in1d(df.hour, np.array([0,6,12,18]))]
		df6.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+model+"_points_"+out_domain+"_"+\
			str(year_range[0])+"_"+str(year_range[1])+"_6hr.pkl")

	#RESAMPLE TO DAILY MAX AND SAVE
	print("REAMPLING TO DAILY MAX...")
	daily_df = pd.DataFrame()
	for loc in loc_id:
		print(loc)
		daily_df = pd.concat([daily_df,df[df["loc_id"]==loc].resample("1D").max()],axis=0)
	daily_df.reset_index().rename(columns={"index":"date"}).to_pickle("/g/data/eg3/ab4502/ExtremeWind"+\
		"/points/"+model+"_points_"+out_domain+"_"+str(year_range[0])+"_"+str(year_range[1])+"_daily_max.pkl")

	if (model == "barra_r_fc") | (model == "barra_ad"):
		print("REAMPLING TO DAILY MAX USING 6-HOURLY DATA...")
		daily_df6 = pd.DataFrame()
		for loc in loc_id:
			print(loc)
			daily_df6 = pd.concat([daily_df6,df6[df6["loc_id"]==loc].resample("1D").max()],axis=0)
		daily_df6.reset_index().rename(columns={"index":"date"}).to_pickle("/g/data/eg3/ab4502/ExtremeWind"+\
			"/points/"+model+"_points_"+out_domain+"_"+str(year_range[0])+"_"+\
			str(year_range[1])+"_daily_max_6hr.pkl")


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

def match_aws(aws,reanal,location,model,lightning=[0]):
	#Take ~half-hourly AWS data and add to 6-hourly reanalysis by resampling (max)
	#Also add lightning data to dataframe

	#Create df of aws closest to reanalysis times
	print("Matching ERA-Interim with AWS observations...")

	#Re-sample AWS to 6-hourly by taking maximum
	#NOTE need to take the max, or else extreme gusts are missed
	aws_erai = aws.resample("6H",on="date").max()
	#aws.index = aws.date
	#aws_erai = aws.resample("6H").nearest()

	aws_erai = aws_erai[(aws_erai.index <= reanal["date"].max())]
	aws_erai = aws_erai[(aws_erai.index >= reanal["date"].min())]
	reanal = reanal[(reanal["date"] <= aws_erai.index.max())]
	reanal = reanal[(reanal["date"] >= aws_erai.index.min())]
	reanal.index = reanal["date"]
	reanal = reanal.sort_index()
	reanal["wind_gust"] = aws_erai.wind_gust

	#Remove corrupted BARRA date from AWS data
	if model == "barra":
		aws_erai = aws_erai[~(aws_erai.index == dt.datetime(2014,11,22,6,0))]

	#Eliminate NaNs
	na_ind = ~(aws_erai.wind_gust.isna())
	aws_erai = aws_erai[na_ind]
	reanal = reanal[na_ind]
	
	#Add lightning data
	if len(lightning)<=1:
		lightning = read_lightning()
	lightning = lightning[lightning.loc_id==location]
	lightning = lightning[(lightning["date"] <= reanal.index.max())]
	lightning = lightning[(lightning["date"] >= reanal.index.min())]
	lightning.index = lightning["date"]
	if model == "barra":
		lightning = lightning[~(lightning.index == dt.datetime(2014,11,22,6,0))]
	lightning = lightning[na_ind]
	reanal["lightning"] = lightning.lightning
	return reanal

def load_jdh_points_barra_r_fc(daily_max=True,smooth=False):
	#FOR THE BARRA-R FC SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS FOUND 
	#IN THE JDH DATASET 2010-2015
	#Smooth can be "max" "mean" or False
	ls = np.sort(os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_r_fc/"))
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_r_fc/"+ls[i] \
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
		print(ls[i])
		df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"barra_r_fc",smooth=smooth,\
			daily_max=daily_max))
	if smooth in ["max","mean"]:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc_points_"+\
			smooth+"_2010.pkl"
	else:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc_points_2003_2016.pkl"
	df.to_pickle(outname)

def load_jdh_points_barra_ad(smooth=False):
	#FOR THE BARRA-AD SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS FOUND IN THE JDH
	# DATASET 2010-2015
	#Smooth can be "max" "mean" or False
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_ad/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_ad/"+ls[i] \
			for i in np.arange(0,len(ls))]
#	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
#			"Renmark","Clare HS","Adelaide AP","Whyalla",\
#			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
#			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
#			"Tarcoola","Edinburgh"]
#	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(136.8054,-31.15),\
#			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
#			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
#			(137.5206,-33.0539),(140.5212,-36.6539),\
#			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
#			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
#			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
#			(138.6222,-34.7111)]
	loc_id = ["Mount Gambier"]
	points = [(140.7739,-37.7473)]
	#Remove mount gambier if getting data from the sa_small domain and smoothing
	if smooth:
		points = np.array(points)[~(np.array(loc_id)=="Mount Gambier")]
		loc_id = np.array(loc_id)[~(np.array(loc_id)=="Mount Gambier")]
	
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		print(ls[i])
		df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"barra_ad",smooth=smooth))
	if smooth in ["max","mean"]:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad_points_"+\
			smooth+"_2010.pkl"
	else:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad_points_2010.pkl"
	df.to_pickle(outname)

def load_jdh_points_barra(smooth=False):
	#FOR THE BARRA SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS FOUND IN THE JDH
	# DATASET 2010-2015
	#Smooth can be "max" "mean" or False
	ls = np.sort(os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"))
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"+ls[i] \
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
	#Remove mount gambier if getting data from the sa_small domain and smoothing
	#points = np.array(points)[~(np.array(loc_id)=="Mount Gambier")]
	#loc_id = np.array(loc_id)[~(np.array(loc_id)=="Mount Gambier")]
	
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		print(ls[i])
		df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"barra",daily_max=True,smooth=smooth))
	if smooth in ["max","mean"]:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_points_"+\
			smooth+"_2010_2015.csv"
	else:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_points_daily_2003_2016.pkl"
	df.to_pickle(outname)

def load_jdh_points_erai(loc_id,points,fc,daily_max):
	#FOR THE ERA-Interim SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS 
	#FOUND IN THE JDH DATASET
	if fc:
		ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/erai_fc/")
		ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/erai_fc/"+ls[i] \
			for i in np.arange(0,len(ls))]
	else:
		ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/")
		ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"+ls[i] \
			for i in np.arange(0,len(ls))]
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		if fc:
		#if int(ls_full[i][56:60]) >= 2010:
			print(ls[i])
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
				smooth=False,daily_max=daily_max,erai_fc=fc))
		else:
		#if int(ls_full[i][50:54]) >= 2010:
			print(ls[i])
			print(str(i)+"/"+str(len(ls_full)))
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
				smooth=False,daily_max=daily_max))
	if fc:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/erai_fc_points_1979_2017"
	else:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2017"
	if daily_max:
		outname=outname+"_daily_max.pkl"
	else:
		outname=outname+".pkl"
	df.to_pickle(outname)
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

def match_jdh_erai():
	#Match non-synoptic wind gusts from JDH dataset with era-interim
	#Note JDH dataset provides only the "day" of the event. So, provide ERA-Interim data in the
	# form of daily max
	jdh = read_non_synoptic_wind_gusts()
	erai_df = load_erai_df(True,False)
	erai_df.index = erai_df.date

	jdh_erai = pd.DataFrame()
	for i in np.arange(0,jdh.shape[0]):
		date = str(jdh.dates[i].year)+"-"+str(jdh.dates[i].month)+"-"+str(jdh.dates[i].day)
		jdh_erai = jdh_erai.append(erai_df[erai_df.loc_id==jdh.station[i]][date].max(),\
			ignore_index=True)

	jdh = pd.concat([jdh.reset_index(drop=True),jdh_erai],axis=1)
	jdh = jdh[~(jdh.lat.isna())]
	return jdh

def get_wind_sa(model):
	#Load in the wind_sa.csv dataset, and for each event, extract convective parameters from 
	# reanalysis into a dataframe
	#For each observation, get the closest gridpoint, and closest time.

	#Load observations
	wind_sa = load_wind_sa()

	#Set up reanalysis 
	if model == "erai":
		path = "/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"
		f0 = nc.Dataset(path+"erai_19790101_19790131.nc")
		lat = f0.variables["lat"][:]
		lon = f0.variables["lon"][:]
		smooth = False
	elif model == "barra":
		path = "/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"
		f0 = nc.Dataset(path+"barra_20100101_20100131.nc")
		lat = f0.variables["lat"][:]
		lon = f0.variables["lon"][:]
		smooth = "max"

	#Initialise lists/arrays to store reanalysis data
	var = np.array([str(f0.variables.items()[i][0]) for i in np.arange(0,\
		len(f0.variables.items()))])
	var = var[~(var=="time") & ~(var=="lat") & ~(var=="lon")]
	values = np.empty((wind_sa.shape[0],len(var)))
	values_lat = []
	values_lon = []
	values_lon_used = []
	values_lat_used = []
	values_date_obs = []
	values_date_model = []
	values_year = [];values_month=[];values_day=[]

	cnt = 0
	print("Getting Observations...")
	for i in np.arange(0,wind_sa.shape[0]):
		fname = glob.glob(path+"erai_"+dt.datetime.strftime(wind_sa["date"][i],"%Y%m")+\
			"*.nc")
		f = nc.Dataset(fname[0])
		lat_ind = np.argmin(abs(wind_sa.Latitude[i]-lat))
		lon_ind = np.argmin(abs(wind_sa.Longitude[i]-lon))
		times = nc.num2date(f.variables["time"][:],f.variables["time"].units)
		time_ind = np.argmin(abs(times - wind_sa["date"][i]))
		
		for v in np.arange(0,len(var)):
			if smooth=="mean":
			#SMOOTH OVER ~1 degree
				lat_points = np.arange(lat_ind-4,lat_ind+5)
				lon_points = np.arange(lat_ind-4,lat_ind+5)
				values[cnt,v] = np.nanmean(f.variables[var[v]][time_ind,\
				int(lat_points[0]):int(lat_points[-1]),\
				int(lon_points[0]):int(lon_points[-1])])
			elif smooth=="max":
			#Max OVER ~1 degree 
				lat_points = np.arange(lat_ind-4,lat_ind+5)
				lon_points = np.arange(lat_ind-4,lat_ind+5)
				values[cnt,v] = np.nanmax(f.variables[var[v]][time_ind,\
				int(lat_points[0]):int(lat_points[-1]),\
				int(lon_points[0]):int(lon_points[-1])])
			elif smooth==False:
				values[cnt,v] = f.variables[var[v]][time_ind,lat_ind,lon_ind]

		values_lat.append(wind_sa["Latitude"][i])
		values_lon.append(wind_sa["Longitude"][i])
		values_lat_used.append(lat[lat_ind])
		values_lon_used.append(lon[lon_ind])
		values_date_obs.append(wind_sa["date"][i])
		values_year.append(wind_sa["date"][i].year)
		values_month.append(wind_sa["date"][i].month)
		values_day.append(wind_sa["date"][i].day)
		values_date_model.append(times[time_ind])
		cnt = cnt+1
		f.close()
	
	df = pd.DataFrame(values,columns=var)
	df["year"] = values_year
	df["month"] = values_month
	df["day"] = values_day
	df["lat"] = values_lat
	df["lon"] = values_lon
	df["lon_used"] = values_lon_used
	df["lat_used"] = values_lat_used
	df["obs_date"] = values_date_obs
	df["model_date"] = values_date_model

	return df

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
	load_netcdf_points_mf(points,loc_id,"sa_small","barra_ad",[2006,2016],"sa_small")

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
