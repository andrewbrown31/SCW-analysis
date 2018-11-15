from obs_read import *
import matplotlib.pyplot as plt
from plot_param import contour_properties

def load_netcdf_points(fname,points,loc_id):
	#Load a single netcdf file created by calc_param.py, and create dataframe for a 
	# list of lat/lon points

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
	lsm_new = lsm[((lat_orig<=lat[0]) & (lat_orig>=lat[-1]))]
	lsm_new = lsm_new[:,((lon_orig>=lon[0]) & (lon_orig<=lon[-1]))]
	x,y = np.meshgrid(lon,lat)
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

def plot_scatter(model,convective,threshold=False):
	#Plot scatter plots of reanalysis/observed convective indices (see load_*dataset*_df()) 
	# against AWS max gust strength (see read_aws())
	#Set convective argument to True to restrict data to times when CAPE > 100
	#Set threshold to an integer to restrict gust data to values over the threshold. 
	# Otherwise, False

	#Read AWS data
	print("Loading AWS...");aws = read_aws()

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df()
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	else:
		raise NameError("Invalid model selection")
	
	#Create df of aws closest to reanalysis times
	print("Matching ERA-Interim with AWS observations...")
	#time_ind = []
	#for i in np.arange(0,erai_df.shape[0]):
	#	time_ind.append((aws["date"] - erai_df["date"][i]).abs().values.argmin())
	#time_ind = np.array(time_ind)

	#aws_erai = aws.loc[time_ind]
	#aws_erai.index = np.arange(0,aws_erai.shape[0])

	#Re-sample AWS to 6-hourly by taking maximum
	aws_erai = aws.resample("6H",on="date").max()
	aws_erai = aws_erai[(aws_erai.index <= erai_df["date"].max())]
	aws_erai = aws_erai[(aws_erai.index >= erai_df["date"].min())]
	erai_df = erai_df[(erai_df["date"] <= aws_erai.index.max())]
	erai_df = erai_df[(erai_df["date"] >= aws_erai.index.min())]
	erai_df.index = erai_df["date"]

	#Remove corrupted date
	if model == "barra":
		aws_erai = aws_erai[~(aws_erai.index == dt.datetime(2014,11,22,06,0))]

	#Eliminate NaNs
	na_ind = ~(aws_erai.wind_gust.isna())
	aws_erai = aws_erai[na_ind]
	erai_df = erai_df[na_ind]
	if convective:
		#Create CAPE >= 100 (Convective events)
		aws_erai = aws_erai[(erai_df.mu_cape>=100)]
		erai_df = erai_df[(erai_df.mu_cape>=100)]
	if threshold:
		thresh_inds = (aws_erai.wind_gust >= threshold)
		aws_erai = aws_erai[thresh_inds]
		erai_df = erai_df[thresh_inds]

	param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","s06","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","lr1000","lcl","cape*s06","lr1000",\
		"lcl"]
	
	
	for param in param_list:
			
		plt.figure()
		x = np.array(erai_df[param])
		y = np.array(aws_erai.wind_gust)
		if param == "lcl":
			nan_ind = np.isnan(x)
			x = x[~nan_ind]
			y = y[~nan_ind]
		m,b = np.polyfit(x,y,deg=1)
		r = np.corrcoef(x,y)[0,1]
		plt.hist2d(x,y,bins=50)
		plt.colorbar()
		plt.plot(x, b+m*x,"k")
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s")
		plt.title(str(r))
		outname = "/home/548/ab4502/working/ExtremeWind/figs/2dhist/"+model+"_"+param
		if convective:
			outname = outname + "_conv"	
		if threshold:
			outname = outname + "_" + str(threshold)	
		outname = outname + ".png"
		plt.savefig(outname,bbox_inches="tight")
		plt.close()
	
	return [aws_erai]

def load_erai_df():
	erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_erai_points_wrf_20100101_20151231.csv")
	#erai_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_erai_points_SHARPpy_20100101_20151231.csv")
	#Create datetime column in reanalysis dataframe
	erai_dt = []
	for i in np.arange(0,erai_df.shape[0]):
		erai_dt.append(dt.datetime(int(erai_df["year"][i]),int(erai_df["month"][i]),\
				int(erai_df["day"][i]),int(erai_df["hour"][i]),\
				int(erai_df["minute"][i])))
	erai_df["date"] = erai_dt
	return erai_df

def load_barra_df():
	barra_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_barra_points_wrf_20100101_20151231.csv")
	#Create datetime column in reanalysis dataframe
	barra_dt = []
	for i in np.arange(0,barra_df.shape[0]):
		barra_dt.append(dt.datetime(int(barra_df["year"][i]),int(barra_df["month"][i]),\
				int(barra_df["day"][i]),int(barra_df["hour"][i]),\
				int(barra_df["minute"][i])))
	barra_df["date"] = barra_dt
	return barra_df

def load_obs_df():
	obs_df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/adelaideAP/data_obs_points_wrf_20100101_20151231.csv")
	#Create datetime column in reanalysis dataframe
	obs_dt = []
	for i in np.arange(0,obs_df.shape[0]):
		obs_dt.append(dt.datetime(int(obs_df["year"][i]),int(obs_df["month"][i]),\
				int(obs_df["day"][i]),int(obs_df["hour"][i]),\
				int(obs_df["minute"][i])))
	obs_df["date"] = obs_dt
	return obs_df

def plot_distributions(model):
	#Read wind events, reanalysis point data for Adelaide AP
	df_wind = read_synoptic_wind_gusts(" Adelaide AP")
	if model == "barra":
		model_df = load_barra_df()
	elif model == "erai":
		model_df = load_erai_df()
	elif model == "obs":
		model_df = load_obs_df()
	else:
		raise NameError("Invalid model selection")

	#Create df of reanalysis closest to wind times
	time_ind = []
	for i in np.arange(0,df_wind.shape[0]):
		time_ind.append((df_wind["dates_utc"][i] - model_df.date).abs().values.argmin())
	time_ind = np.array(time_ind)

	wind_erai = model_df.loc[time_ind]

	#Plot distributions

	ms=1

	plt.figure()
	param = "mu_cape"
	plt.hist(model_df[param],bins=np.linspace(0,1000,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "ml_cape"
	plt.hist(model_df[param],bins=np.linspace(0,1000,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "mu_cin"
	plt.hist(model_df[param],bins=np.linspace(0,600,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "ml_cin"
	plt.hist(model_df[param],bins=np.linspace(0,600,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "scp"
	plt.hist(model_df[param],bins=np.linspace(0,1,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "stp"
	plt.hist(model_df[param],bins=np.linspace(0,1,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "mmp"
	plt.hist(model_df[param],bins=np.linspace(0,1,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "lr1000"
	plt.hist(model_df[param],bins=np.linspace(0,10,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

#	plt.figure()
#	param = "ship"
#	plt.hist(model_df[param],bins=np.linspace(0,1,100),log=True)
#	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
#	[cmap,levels,cb_lab] = contour_properties(param)
#	plt.title(param)
#	plt.xlabel(cb_lab)	

	plt.figure()
	param = "srh01"
	plt.hist(model_df[param],bins=np.linspace(0,300,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "srh03"
	plt.hist(model_df[param],bins=np.linspace(0,300,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "srh06"
	plt.hist(model_df[param],bins=np.linspace(0,300,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "s06"
	plt.hist(model_df[param],bins=np.linspace(0,75,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "crt"
	vals = model_df[~(model_df[param].isna())][param]
	plt.hist(vals,bins=np.linspace(0,180,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "relhum850-500"
	vals = model_df[~(model_df[param].isna())][param]
	plt.hist(vals,bins=np.linspace(0,100,100),log=False)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")

	plt.figure()
	param = "cape*s06"
	plt.hist(model_df[param],bins=np.linspace(0,100000,100),log=True)
	plt.plot(wind_erai[param],np.ones(wind_erai.shape[0])*10,"rx",markeredgewidth=ms)
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+model+"_"+param+\
			"_adelaideAP.png",bbox_inches="tight")


model = "erai"
#aws_model,model_df = plot_scatter(model)
plot_scatter(model,False,False)
#df = load_netcdf_points("/g/data/eg3/ab4502/ExtremeWind/aus/erai_wrf_20160928_20160930.nc",\
		#[(138.5204, -34.5924)],["Adelaide AP"])
