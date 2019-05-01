import seaborn as sb
import os
from obs_read import *
import matplotlib.pyplot as plt
from plot_param import *
from plot_clim import *

def load_netcdf_points_mf(points,loc_id,domain,model,year_range):
	#Same as load_netcdf_points, but use MFDataset to lead in all netcdf points at once

	#Load in convective parameter netcdf files for a given model/year range/domain
	f = load_ncdf(domain,model,year_range,var_list=False)

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
	var = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
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

	#SAVE AT 6-HOURLY FREQUENCY
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2017.pkl")

	#RESAMPLE TO DAILY MAX AND SAVE
	print("REAMPLING TO DAILY MAX...")
	daily_df = pd.DataFrame()
	for loc in loc_id:
		print(loc)
		daily_df = pd.concat([daily_df,df[df["loc_id"]==loc].resample("1D").max()],axis=0)
	daily_df.reset_index().rename(columns={"index":"date"}).to_pickle("/g/data/eg3/ab4502/ExtremeWind"+\
		"/points/erai_points_1979_2017_daily_max.pkl")

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
					values[cnt,v] = param_out[v][t,lat_ind[int(point)],\
						lon_ind[int(point)]]
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
		aws_erai = aws_erai[~(aws_erai.index == dt.datetime(2014,11,22,06,0))]

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
		lightning = lightning[~(lightning.index == dt.datetime(2014,11,22,06,0))]
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

def analyse_jdh_events():
        #Read data and combine
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"\
                        +"all_daily_max_wind_gusts_sa_1979_2017.pkl").set_index(["date","stn_name"])
	jdh = read_non_synoptic_wind_gusts().set_index("station",append=True)
	erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
                +"erai_points_1979_2017_daily_max.pkl").set_index(["date","loc_id"])
	erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
                +"erai_fc_points_1979_2017_daily_max.pkl").set_index(["date","loc_id"])
	barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
                +"barra_r_fc_points_daily_2006_2016.pkl").set_index(["loc_id"],append=True)

	df = pd.concat([aws["wind_gust"],jdh["gust (m/s)"],erai,erai_fc["wg10"],\
                barra_r_fc["max_wg10"]],axis=1)

        #Create a binary column for JDH events in the original combined dataframe
	#Only include JDH events where the AWS data is above 20 m/s
	df["jdh"] = 0
	df.loc[((~df["gust (m/s)"].isna()) & (~df["wind_gust"].isna()) & (df["wind_gust"]>=20)),"jdh"] = 1

	#Create dataframe with JDH events, which are able to be cross-validated with AWS data (i.e.
	# where AWS data is present
	jdh_df = df.dropna(subset=["gust (m/s)","wind_gust"])

	#Create another dataframe without the JDH events
	non_jdh_df = df[((df["gust (m/s)"].isna()) | (df["wind_gust"].isna()))]
	
	return (df,jdh_df,non_jdh_df)

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
		vplot = [dt.datetime(1993,07,05),dt.datetime(1995,05,25)]
	elif (loc == "Port Augusta") & (var == "wind_gust"):
		vplot = [dt.datetime(1997,6,12),dt.datetime(2001,07,02),dt.datetime(2014,02,20)]
	elif (loc == "Edinburgh") & (var == "wind_gust"):
		vplot = [dt.datetime(1993,05,04)]
	elif (loc == "Parafield") & (var == "wind_gust"):
		vplot = [dt.datetime(1992,07,22),dt.datetime(2006,06,27)]
	else:
		year_range = [1979,2017]
		vplot = []

	df = df[np.in1d(df.month,months) & (df.stn_name==loc) & (df.year>=year_range[0]) & \
		(df.year<=year_range[1])].sort_values("date").reset_index()

	if method=="threshold":
		years = df.year.unique()
		fig,[ax1,ax3]=plt.subplots(figsize=[10,8],nrows=2,ncols=1)
		ax1.set_ylabel("Wind gust (m/s)",fontsize="large")
		df_monthly = df.set_index("date").resample("1M").max()
		y = np.array([dt.datetime(t.year,t.month,t.day) for t in df_monthly.index])
		ax1.plot(y,df_monthly[var])
		ax2 = ax1.twinx()
		cnt = [df[(df.stn_name==loc)&(df.year==y)].shape[0] for y in np.arange(1979,2018,1)]
		y = np.array([dt.datetime(y,6,1) for y in np.arange(1979,2018,1)])
		ax2.plot(y,cnt,color="k",marker="s",linestyle="none",markersize=8)
		ax2.set_ylabel("Obs. per year",fontsize="large")
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
		ax3.set_ylabel("Days",fontsize="large")
		ax3.set_xlabel("Year",fontsize="large")
		ax3.set_yscale('log')
		ax1.set_title(loc,fontsize="large")
		ax3.tick_params(labelsize="large")
		ax2.tick_params(labelsize="large")
		ax1.tick_params(labelsize="large")
	if method=="threshold_only":
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
					scatter_kws={"s":75,"linewidth":2})
			else:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax,marker="s",\
					scatter_kws={"s":75})
			cnt=cnt+1
		if dataset == "BARRA-AD":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		elif dataset == "BARRA-R":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		else:
			ax.set_xlim([events.years.min()-1,events.years.max()+1])
		ax.set_ylim([0.5,1000])
		ax.set_ylabel("Days",fontsize="large")
		ax.set_xlabel("Year",fontsize="large")
		ax.set_yscale('log')
		ax.tick_params(labelsize="large")
		ax.set_title(loc)
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
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/trends/"+\
		dataset+"_"+loc+"_"+var+"_"+method+"_"+str(months[0])+"_"+str(months[-1])+".png",\
		bbox_inches="tight")
	plt.close()


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

	if two_thirds:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var],df5[var],df6[var]],whis=[33,100],\
			labels=["5-15 m/s\n("+str(df1.shape[0])+")","15-25 m/s\n("+str(df2.shape[0])+")",\
			"25-30 m/s\n("+str(df3.shape[0])+")","30-35 m/s\n("+str(df4.shape[0])+")",\
			"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	else:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var]],
			#df5[var],df6[var]],
			whis="range",\
			labels=["5-15 m/s\n("+str(df1.shape[0])+")","15-25 m/s\n("+str(df2.shape[0])+")",\
			"25-30 m/s\n("+str(df3.shape[0])+")","30+ m/s\n("+str(df4.shape[0])+")"])
			#"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	plt.xticks(fontsize="x-large")
	plt.yticks(fontsize="x-large")

	if loc==False:
		plt.title("All Stations",fontsize="x-large")
	else:
		plt.title(loc,fontsize="x-large")
	t1,t2,t3,units,t4,log_plot,t6 = contour_properties(var)
	plt.ylabel(units,fontsize="x-large")

	if log_plot:
		plt.yscale("log")
		plt.ylim(ymin=0.1)

	if var == "mu_cape":
		plt.ylim([0,5000])
	elif var == "s06":
		plt.ylim([0,60])

	if loc==False:
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_AllStations_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".png",\
			bbox_inches="tight")
	else:
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_"+loc+"_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".png",\
			bbox_inches="tight")
	plt.close()

def plot_conv_seasonal_cycle(df,loc,var,trend=False):

	#For a reanalysis dataframe (df), create a seasonal mean plot forrameter "var" for each wind speed gust
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
		ax.set_ylabel(units,fontsize="x-large")
		ax.tick_params(labelcolor="b",axis="y",labelsize="x-large")
		ax2 = ax.twinx()
		ax2.plot(np.arange(1,13,1),[np.mean(df[df.month==m][var[1]]) for m in np.arange(1,13,1)],\
			color="red")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax2.tick_params(labelcolor="red",axis="y",labelsize="x-large")
		ax2.tick_params(labelcolor="k",axis="x",labelsize="x-large")
		ax.tick_params(labelcolor="k",axis="x",labelsize="x-large")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[1])
		ax2.set_ylabel(units,fontsize="x-large")
		plt.title(loc,fontsize="x-large")
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
			"_"+var[0]+"_"+var[1]+".png",bbox_inches="tight")
	else:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		if trend:
			plt.plot([np.mean(df[(df.month==m) & (df.year<=1998)][var[0]]) for m in np.arange(1,13,1)],\
						label = "1979-1998",color="b")
			plt.plot([np.mean(df[(df.month==m) & (df.year>=1998)][var[0]]) for m in np.arange(1,13,1)],\
						label = "1998-2017",color="b",linestyle="--")
			plt.title(loc)
			plt.legend()
			plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
					"_"+var[0]+"_trend.png",bbox_inches="tight")
		else:
			plt.plot([np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)])
			plt.title(loc)
			plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
					"_"+var[0]+".png",bbox_inches="tight")
	plt.close()

def hypothesis_test(a,b,B):

	#For two samples (a,b) perform a student t test and bootstrap hypothesis test that the mean is different

	if (np.all(np.isnan(a)) ) | (np.all(np.isnan(b)) ):
		return (np.nan)
	else:
		abs_diff = np.nanmean(b) - np.nanmean(a)

		total = np.concatenate((a,b))

		tot_mean = np.nanmean(total)
		a_shift = a - np.nanmean(a) + tot_mean
		b_shift = b - np.nanmean(b) + tot_mean

		a_samples = np.random.choice(a_shift,(B,a_shift.shape[0]))
		b_samples = np.random.choice(b_shift,(B,b_shift.shape[0]))

		a_sample_means = np.array(map(np.nanmean,a_samples))
		b_sample_means = np.array(map(np.nanmean,b_samples))

		sample_diff = b_sample_means - a_sample_means

		p_up = np.sum(sample_diff >= abs_diff) / float(B)
		p_low = np.sum(sample_diff <= abs_diff) / float(B)

		return (2*np.min([p_low,p_up]))

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
	df,jdh_df,non_jdh_df = analyse_jdh_events()
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
		"cape*s06","dcp","td850","td800","td950","dcape","mlm","dlm","mlm+dcape","dlm+dcape",\
		"dcape*cs6","dlm*dcape*cs6","mlm*dcape*cs6","mlm*dcape*cs3","wg10"]
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

def magnitude_trends(param,hr,param_names):

	#Attempt to identify trends in a range of magnitudes, following the method of Dowdy (2014)
	#Param is a string, hr is an array or percentiles which identify a daignostic threshold

	#Load datasets and combine
	df,t1,t2 = analyse_jdh_events()
	df = df.dropna(subset=["wind_gust"])
	
	markers=["s","o","^","+","v"]
	cols = ["b","r","g","c","y"]
	for i in np.arange(len(hr)):
		#Find the proportion of event days given all days, as a function of wind speed catergory
		speed_cats = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50]]
		speed_labs = ["0-5","5-10","10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50"]
		all_days = np.zeros(len(speed_cats))
		event_days = np.zeros(len(speed_cats))
		for s in np.arange(len(speed_cats)):
			all_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				(df["wind_gust"]<speed_cats[s][1])].shape[0]
			event_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				(df["wind_gust"]<speed_cats[s][1]) & \
				(df[param[i]]>=np.percentile(df[df.jdh==1][param[i]],hr[i]))].shape[0]

		#Plot the relationship
		if i==0:
			plt.plot(np.arange(len(speed_cats)),all_days,"kx",linestyle="none",markersize=12,\
				label = "All days")
		lab = "Days with "+param_names[i]+" = "+str(np.percentile(df[df.jdh==1][param[i]],hr[i]).round(3))
		plt.plot(np.arange(len(speed_cats)),event_days,color=cols[i],marker=markers[i],linestyle="none",\
			fillstyle="none",markersize=12,label = lab)
		print(lab+"\n"+str((event_days/all_days.astype(float)).round(3)))

	plt.yscale("log")
	plt.ylabel("Number of Days",fontsize="x-large")
	plt.xlabel("Wind Speed (m/s)\nAt 23 AWS locations from 1979-2017",fontsize="x-large")
	ax=plt.gca();ax.set_xticks(np.arange(len(speed_cats)));ax.set_xticklabels(speed_labs)
	ax.tick_params(labelsize="x-large")
	plt.xlim([-0.5,len(speed_cats)]);plt.ylim([0.5,10e4])
	plt.legend(numpoints=1,fontsize="x-large")
	plt.show()

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

if __name__ == "__main__":

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
	#load_netcdf_points_mf(points,loc_id,"sa_small","erai",[1979,2017])

	#aws = remove_incomplete_aws_years(pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
	#	"all_daily_max_wind_gusts_sa_1979_2017.pkl"),"Port Augusta")
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
		"all_daily_max_wind_gusts_sa_1979_2017.pkl")
	#erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"erai_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"erai_fc_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
	#	+"barra_r_fc_points_daily_2006_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
	#	+"barra_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"\
	#	+"barra_ad_points_daily_2006_2016.pkl").rename(columns={"loc_id":"stn_name"}).\
	#	reset_index()   
	#interannual_time_series([aws,erai_fc],["wind_gust","wg10"],["AWS","ERA-Interim"],\
	#		"Adelaide AP",[1989,2017],"am",[10,11,12,1,2,3])
	locs = ["Adelaide AP","Edinburgh","Mount Gambier","Parafield",\
                "Port Augusta","Woomera"]
	[var_trends(aws,"wind_gust",l,"threshold","AWS",threshold=[[15,25],[25,30],[30]]) \
		for l in locs]
	#[var_trends(erai_fc,"wg10",l,"threshold_only","ERA-Interim",threshold=[[15,25],[25,30],[30]]) \
	#	for l in locs]
	#[var_trends(barra_r_fc,"max_wg10",l,"threshold_only","BARRA-R",threshold=[[15,25],[25,30],[30]],year_range=[2003,2016]) \
	#	for l in locs]
	#[var_trends(barra_ad,"max_wg10",l,"threshold_only","BARRA-AD",threshold=[[15,25],[25,30],[30]],year_range=[2006,2016]) \
	#	for l in locs]
	#trend_table()
	#seasons = [np.arange(1,13,1),[11,12,1],[2,3,4],[5,6,7],[8,9,10]]
	#for loc in ["Woomera","Adelaide AP","Mount Gambier", "Port Augusta"]:
	   #plot_conv_seasonal_cycle(erai,loc,["ml_cape","s06"],trend=True)
	   #for var in ["ml_cape","s06","dcape","dlm"]:
	 #  for var in ["dlm*dcape*cs6"]:
	#     for s in seasons:
		#wind_gust_boxplot(erai,aws,var,loc=loc,two_thirds=False)
	#far_table()
	#magnitude_trends(["cond"],[33],\
	#			["COND"])
