import os
from obs_read import *
import matplotlib.pyplot as plt

def load_netcdf_points(fname,points,loc_id,model,smooth,ad_data=False,daily_max=False):
	#Load a single netcdf file created by calc_param.py, and create dataframe for a 
	# list of lat/lon points (names given by loc_id)
	#If model = barra and smooth = False, the closest point in BARRA to "point" is taken. 
	# Otherwise, smooth = "mean" takes the mean over ~0.75 degrees (same as ERA-Interim),
	# or smooth = "max" takes the max over the same area for all variables 
	

	#Load netcdf file containing convective parameters saved by calc_param.py
	f = nc.Dataset(fname)

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
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
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
		times = f.variables["time"][:]
		times = [dt.datetime(int(fname[-7:-3]),1,1,0,0,0) + dt.timedelta(hours=6*x) \
			for x in times]
	else:
		times = nc.num2date(f.variables["time"][:],f.variables["time"].units)
	var = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
	var = var[~(var=="time") & ~(var=="lat") & ~(var=="lon")]
	if daily_max:
		values = np.empty(((len(points)*len(times))/4,len(var)))
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
			for t in np.arange(len(times)):
			    if times[t].hour==0:
				for v in np.arange(0,len(var)):
					values[cnt,v] = np.nanmax(f.variables[var[v]][t:(t+4),\
						lat_ind[point],lon_ind[point]],axis=0)
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
	else:
		for point in np.arange(0,len(points)):
			print(lon_used[point],lat_used[point])
			for t in np.arange(len(times)):
				for v in np.arange(0,len(var)):
				    if smooth=="mean":
					#SMOOTH OVER ~1 degree
					lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
					lon_points = np.arange(lon_ind[point]-4,lon_ind[point]+5)
					values[cnt,v] = np.nanmean(f.variables[var[v]][t,\
						int(lat_points[0]):int(lat_points[-1]),\
						int(lon_points[0]):int(lon_points[-1])])
				    elif smooth=="max":
					#Max OVER ~1 degree 
					lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
					lon_points = np.arange(lon_ind[point]-4,lon_ind[point]+5)
					values[cnt,v] = np.nanmax(f.variables[var[v]][t,\
						int(lat_points[0]):int(lat_points[-1]),\
						int(lon_points[0]):int(lon_points[-1])])
				    elif smooth==False:
					values[cnt,v] = f.variables[var[v]][t,lat_ind[point],\
						lon_ind[point]]
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


def load_erai_df(is1979,fc,location=False):
	if fc:
		erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_fc_points_2010_2017.csv")
		erai_dt = []
		for i in np.arange(0,erai_df.shape[0]):
			erai_dt.append(dt.datetime(int(erai_df["year"][i]),int(erai_df["month"][i]),\
					int(erai_df["day"][i]),int(erai_df["hour"][i]),\
					int(erai_df["minute"][i])))
		erai_df["date"] = erai_dt

		if location != False:
			erai_df = erai_df[erai_df["loc_id"]==location]
		else:
			print("INFO: RETURNING ALL LOCATIONS AVAILABLE...")
	else:
	    if is1979:
		erai_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2015.pkl")
	    else:
		erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_2010_2017.csv")
		erai_dt = []
		for i in np.arange(0,erai_df.shape[0]):
			erai_dt.append(dt.datetime(int(erai_df["year"][i]),int(erai_df["month"][i]),\
					int(erai_df["day"][i]),int(erai_df["hour"][i]),\
					int(erai_df["minute"][i])))
		erai_df["date"] = erai_dt

		if location != False:
			erai_df = erai_df[erai_df["loc_id"]==location]
		else:
			print("INFO: RETURNING ALL LOCATIONS AVAILABLE...")
	return erai_df

def load_erai_daily_max_df():
	return (pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2017_daily_max.pkl"))

def load_ADversion_df(param,location=False):
	erai_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_ADdata_"+param+\
		"_2010_2015.pkl")
	if location != False:
		erai_df = erai_df[erai_df["loc_id"]==location]
	return erai_df

def load_barra_df(smooth="max",location=False):
	if smooth == False:
		barra_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_"+\
			"2010_2015.csv")
	else:
		barra_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_"+\
			smooth+"_2010_2015.csv")
	#Create datetime column in reanalysis dataframe
	barra_dt = []
	for i in np.arange(0,barra_df.shape[0]):
		barra_dt.append(dt.datetime(int(barra_df["year"][i]),int(barra_df["month"][i]),\
				int(barra_df["day"][i]),int(barra_df["hour"][i]),\
				int(barra_df["minute"][i])))
	barra_df["date"] = barra_dt
	if location != False:
		barra_df = barra_df[barra_df["loc_id"]==location]
	return barra_df

def load_obs_df(location=False):
	obs_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_obs_points_wrf_20100101_20151231.csv")
	#Create datetime column in reanalysis dataframe
	obs_dt = []
	for i in np.arange(0,obs_df.shape[0]):
		obs_dt.append(dt.datetime(int(obs_df["year"][i]),int(obs_df["month"][i]),\
				int(obs_df["day"][i]),int(obs_df["hour"][i]),\
				int(obs_df["minute"][i])))
	obs_df["date"] = obs_dt
	if location != False:
		obs_df = obs_df[obs_df["loc_id"]==location]
	return obs_df

def load_jdh_points_barra(smooth=False):
	#FOR THE BARRA SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS FOUND IN THE JDH
	# DATASET 2010-2015
	#Smooth can be "max" "mean" or False
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_large/barra/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_large/barra/"+ls[i] \
			for i in np.arange(0,len(ls))]
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]
	#Remove mount gambier if getting data from the sa_small domain and smoothing
	points = np.array(points)[~(np.array(loc_id)=="Mount Gambier")]
	loc_id = np.array(loc_id)[~(np.array(loc_id)=="Mount Gambier")]
	
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		print(ls[i])
		df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"barra",smooth=smooth))
	if smooth in ["max","mean"]:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_points_"+\
			smooth+"_2010_2015.csv"
	else:
		outname = "/g/data/eg3/ab4502/ExtremeWind/points/barra_points_2010_2015.csv"
	df.to_csv(outname)

def load_jdh_points_erai(fc,daily_max):
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
	#loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
	#		"Renmark","Clare HS","Adelaide AP"]
	#points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
	#		(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
	#		(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96)]
	#A LIST OF ALL STATIONS MENTIONED IN JDH DATASET:
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
	    if fc:
		#if int(ls_full[i][56:60]) >= 2010:
			print(ls[i])
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
				smooth=False,daily_max=daily_max))
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
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
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

def resample_erai_daily():
	#For comparison with JDH dataset (which only has "daily" observations), resample the ERA-I
	# dataframe to daily frequency by taking the daily max

	#Load erai df from 1979-2015
	erai_df = load_erai_df(fc=False,is1979=True)

	#Resample to daily max
	erai_df_resamp = pd.DataFrame()
	groups = erai_df.groupby("loc_id")
	for name, group in groups:
		print(name)
		erai_df_resamp = erai_df_resamp.append(\
			group.set_index("date").resample("1D").max())

if __name__ == "__main__":

	#aws_model,model_df = plot_scatter(model)
	#plot_scatter(model,False,False)
	#df = load_AD_data()
	#df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_mean_2010_2015.csv",\
	#	float_format="%.3f")
	#df = load_jdh_points_barra(smooth="max")
	#df = get_wind_sa("erai")

	load_jdh_points_erai(fc=False,daily_max=True)
