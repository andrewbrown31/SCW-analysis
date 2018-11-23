import os
from obs_read import *
import matplotlib.pyplot as plt

def load_netcdf_points(fname,points,loc_id,model,smooth,ad_data=False):
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
	values = np.empty((len(points)*len(times),len(var)))
	values_lat = []
	values_lon = []
	values_lon_used = []
	values_lat_used = []
	values_loc_id = []
	values_year = []; values_month = []; values_day = []; values_hour = []; values_minute = []
	values_date = []
	cnt = 0
	for point in np.arange(0,len(points)):
		print(lon_used[point],lat_used[point])
		for t in np.arange(len(times)):
			for v in np.arange(0,len(var)):
			    if smooth=="mean":
				#SMOOTH OVER ~1 degree
				lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
				lon_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
				values[cnt,v] = np.mean(f.variables[var[v]][t,\
					int(lat_points[0]):int(lat_points[-1]),\
					int(lon_points[0]):int(lon_points[-1])])
			    elif smooth=="max":
				#Max OVER ~1 degree 
				lat_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
				lon_points = np.arange(lat_ind[point]-4,lat_ind[point]+5)
				values[cnt,v] = np.max(f.variables[var[v]][t,\
					int(lat_points[0]):int(lat_points[-1]),\
					int(lon_points[0]):int(lon_points[-1])])
			    elif smooth==False:
				values[cnt,v] = f.variables[var[v]][t,lat_ind[point],lon_ind[point]]
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

def load_erai_df(location=False):
	erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_2010_2015.csv")
	#erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_erai_points_SHARPpy_20100101_20151231.csv")
	#Create datetime column in reanalysis dataframe
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

def load_ADversion_df(location=False):
	erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_ADdata_2010_2015.csv")
	erai_dt = []
	for i in np.arange(0,erai_df.shape[0]):
		erai_dt.append(dt.datetime(int(erai_df["year"][i]),int(erai_df["month"][i]),\
				int(erai_df["day"][i]),int(erai_df["hour"][i]),\
				int(erai_df["minute"][i])))
	erai_df["date"] = erai_dt
	if location != False:
		erai_df = erai_df[erai_df["loc_id"]==location]
	return erai_df

def load_barra_df(location=False):
	barra_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_max_2010_2015.csv")
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

def load_jdh_points_barra():
	#FOR THE BARRA SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS FOUND IN THE JDH
	# DATASET 2010-2015
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"+ls[i] \
			for i in np.arange(0,len(ls))]
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(-29.0347,134.7222),(-33.0539,137.5206),(-36.6539,140.5212),\
			(-34.4761,139.0056),(-33.7676,138.2182),(-37.7473,140.7739),\
			(-36.9813,140.7270),(-36.9655,139.7164),(-34.7977,138.6281),\
			(-35.3778,140.5378),(-34.5106,138.6763),(-30.7051,134.5786),\
			(-34.7111,138.6222)]
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		print(ls[i])
		df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"barra",smooth="mean"))
	return df

def load_jdh_points_erai():
	#FOR THE ERA-Interim SA_SMALL DATSET, DRIVE LOAD_NETCDF_POINTS FOR LOACATIONS 
	#FOUND IN THE JDH DATASET
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
		if int(ls_full[i][50:54]) >= 2010:
			print(ls[i])
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
			smooth=False))
	df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_2010_2015.csv")
		#float_format="%.3f")
	return df

def load_AD_data():
	#Load Andrew Dowdy's CAPE/S06 data
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/ad_data/mu_cape/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/ad_data/mu_cape/"+ls[i] \
			for i in np.arange(0,len(ls))]
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96)]
	df = pd.DataFrame()
	for i in np.arange(0,len(ls_full)):
		if int(ls_full[i][-7:-3]) >= 2010:
			print(ls[i])
			df = df.append(load_netcdf_points(ls_full[i],points,loc_id,"erai",\
				smooth=False,ad_data=True))
	df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_ADdata_mu_cape_2010_2015.csv",\
		float_format="%.3f")
	return df

if __name__ == "__main__":

	model = "barra"
	#aws_model,model_df = plot_scatter(model)
	#plot_scatter(model,False,False)
	#df = load_AD_data()
	#df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_mean_2010_2015.csv",\
	#	float_format="%.3f")
	df = load_jdh_points_erai()
