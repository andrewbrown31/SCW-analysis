from obs_read import *
import matplotlib.pyplot as plt
from plot_param import contour_properties

def plot_scatter(model):
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
	time_ind = []
	for i in np.arange(0,erai_df.shape[0]):
		time_ind.append((aws["date"] - erai_df["date"][i]).abs().values.argmin())
	time_ind = np.array(time_ind)

	aws_erai = aws.loc[time_ind]
	aws_erai.index = np.arange(0,aws_erai.shape[0])

	na_ind = ~(aws_erai.wind_gust.isna())
	aws_erai = aws_erai[na_ind]
	erai_df = erai_df[na_ind]
	aws_erai_nzcape = aws_erai[~(erai_df.mu_cape==0)]
	erai_df_nzcape = erai_df[~(erai_df.mu_cape==0)]

	param = ["ml_cape","ml_cin","mu_cin","mu_cape","s06","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","lr1000","lcl","cape*s06"]
	
	
	for param in param_list:
		plt.figure()
		x = np.array(erai_df[param])
		y = np.array(aws_erai.wind_gust)
		m,b = np.polyfit(x,y,deg=1)
		r = np.corrcoef(x,y)[0,1]
		plt.scatter(x,y)
		plt.plot(x, b+m*x)
		plt.title(str(r))
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/scatter/"+model+"_"+param+\
				".png",bbox_inches="tight")
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


model = "barra"
aws_model,model_df = plot_scatter(model)
