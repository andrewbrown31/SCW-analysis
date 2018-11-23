import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import netCDF4 as nc
import matplotlib.animation as animation
import os
from event_analysis import *
from scipy.interpolate import griddata

def plot_param(df,param,domain,time,model):

#Plot parameters from a dataframe with parameter and lat lon data 

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "i")
	no_lon = len(np.unique(df["lon"]))
	no_lat = len(np.unique(df["lat"]))
	x = np.reshape(df["lon"],(no_lon,no_lat))
	y = np.reshape(df["lat"],(no_lon,no_lat))

	for p in param:

		#Define contour levels and colourmaps
		if p == "mu_cape":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1000,11)
		if p == "s06":
			cmap = cm.Reds
			levels = np.linspace(0,30,7)
		if p == "mu_cin":
			cmap = cm.RdBu_r
			levels = np.linspace(-100,300,9)
		if (p == "hel03") | (p == "hel06"):
			cmap = cm.RdBu_r
			levels = np.linspace(-500,500,11)
		if p == "ship":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1,11)
		if p == "lhp":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1,0.1)
		if p == "hgz_depth":
			cmap = cm.YlGnBu
			levels = np.linspace(50,300,6)
		if p == "dcp":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1,11)
		if p == "mburst":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1,11)
		if p == "mmp":
			cmap = cm.YlGnBu
			levels = np.linspace(0,1,11)

		#Plot a parameter for a given time for the whole domain
		vals_grid = np.reshape(df[p],(no_lon,no_lat))
		m.drawcoastlines()
		m.drawmeridians(np.arange(domain[2],domain[3],2),labels=[True,False,False,True])
		m.drawparallels(np.arange(domain[0],domain[1],2),labels=[True,False,True,False])
		m.contourf(x,y,vals_grid,latlon=True,levels=levels,cmap=cmap,extend="both")
		plt.colorbar()
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+model+"_"+\
p+"_"+dt.datetime.strftime(time,"%Y%m%d_%H%M")+".png",bbox_inches="tight")
		plt.close()

def plot_netcdf(fname,param,outname,domain,time,model):

#Load a single netcdf file and plot time mean if time is a list of [start_time,end_time]
# or else plot for a single time if time is length=1

	f = nc.Dataset(fname)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)
	if len(time)==2:
		values = np.mean(f.variables[param][(times_dt>=time[0]) & (times_dt<=time[-1])],\
			axis=0)
	else:
		values = np.squeeze(f.variables[param][times_dt == time])

	[cmap,levels,cb_lab] = contour_properties(param)

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "l")
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	m.contourf(x,y,values,latlon=True,levels=levels,cmap=cmap,extend="both")
	cb = plt.colorbar()
	cb.set_label(cb_lab)
	plt.title(str(time[0]) + "-" + str(time[-1]))
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+".png",\
			bbox_inches="tight")
	plt.close()

def plot_netcdf_clim(param,domain,model):
	#Load all netcdf files in to a numpy array and plot the average
	ls = os.listdir("/g/data/eg3/ab4502/ExtremeWind/sa_small/"+model+"/")
	ls_full = ["/g/data/eg3/ab4502/ExtremeWind/sa_small/"+model+"/"+ls[i] \
			for i in np.arange(0,len(ls))]
	f = nc.MFDataset(ls_full)
	vals = f.variables[param][:]
	lat = f.variables["lat"][:]
	lon = f.variables["lon"][:]
	vals_avg = np.mean(vals,axis=0)
	x,y = np.meshgrid(lon,lat)

	[cmap,levels,cb_lab] = contour_properties(param)

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "l")
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	m.contourf(x,y,vals_avg,latlon=True,levels=levels/4,cmap=cmap,extend="both")
	cb = plt.colorbar()
	cb.set_label(cb_lab)
	outname = param + "_mean_2010-2015"
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+".png",\
			bbox_inches="tight")
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
	if param == "mu_cape":
		cmap = cm.YlGnBu
		levels = np.linspace(0,4000,11)
		cb_lab = "J/Kg"
	if param == "ml_cape":
		cmap = cm.YlGnBu
		levels = np.linspace(0,4000,11)
		cb_lab = "J/Kg"
	if param == "mu_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
	if param == "ml_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
	if param in ["s06","ssfc6","ssfc3"]:
		cmap = cm.Reds
		levels = np.linspace(0,30,7)
		cb_lab = "m/s"
	if param in ["ssfc1","ssfc850","ssfc500"]:
		cmap = cm.Reds
		levels = np.linspace(0,20,7)
		cb_lab = "m/s"
	if param in ["stp","ship","non_sc_stp"]:
		cmap = cm.Reds
		levels = np.linspace(0,0.5,11)
		cb_lab = ""
	if param in ["mmp","scp"]:
		cmap = cm.Reds
		levels = np.linspace(0,1,11)
		cb_lab = ""
	if param in ["srh01","srh03","srh06"]:
		cmap = cm.Reds
		levels = np.linspace(0,200,11)
		cb_lab = "m^2/s^2"
	if param == "crt":
		cmap = cm.YlGnBu
		levels = [60,120]
		cb_lab = "degrees"
	if param in ["relhum850-500","relhum1000-700"]:
		cmap = cm.YlGnBu
		levels = [0,100]
		cb_lab = "%"
	if param == "vo":
		cmap = cm.RdBu_r
		levels = np.linspace(-8e-5,8e-5,17)
		cb_lab = "s^-1"
	if param == "lr1000":
		cmap = cm.Blues
		levels = np.linspace(2,12,11)
		cb_lab = "deg/km"
	if param == "lcl":
		cmap = cm.YlGnBu_r
		levels = np.linspace(0,8000,9)
		cb_lab = "m"
	if param in ["cape*s06","cape*ssfc6"]:
		cmap = cm.YlGnBu
		levels = np.linspace(0,60000,9)
		cb_lab = ""
	return [cmap,levels,cb_lab]

def plot_distributions(model,location):
	#Read wind events, reanalysis point data for Port Augusta

	#Load the JDH dataset for 2010-2015 (NOTE DATES HAVE BEEN CHECKD FOR ONLY A FEW LOCATIONS)
	df_wind = read_non_synoptic_wind_gusts()
	df_wind = df_wind[df_wind.station == location]
	df_wind = df_wind[(df_wind.index>="20100101") & (df_wind.index<"20160101")]

	#Load reanalysis data
	if model == "barra":
		model_df = load_barra_df(location)
	elif model == "erai":
		model_df = load_erai_df(location)
	elif model == "obs":
		model_df = load_obs_df(location)
	else:
		raise NameError("Invalid model selection")

	#Read AWS data to check JDH data
	aws = read_aws(location)
	aws.index = aws.date

	#Extract highest reanalysis parameters for a day either side of JDH wind gust, 
	# to put on dist.
	#NOTE WE DON'T WANT MAX CRT (OR LCL?)
	wind_erai_min = pd.DataFrame()
	wind_erai_mean = pd.DataFrame()
	wind_erai_max = pd.DataFrame()
	wind_aws = pd.DataFrame()
	model_df.index = model_df.date
	for i in np.arange(0,df_wind.shape[0]):
		date1 = str(df_wind.dates[i]-dt.timedelta(hours=24))
		date2 = str(df_wind.dates[i]+dt.timedelta(hours=42))
		wind_erai_max = wind_erai_max.append(model_df[date1:date2].max(),ignore_index=True)
		wind_erai_min = wind_erai_min.append(model_df[date1:date2].min(),ignore_index=True)
		wind_erai_mean = wind_erai_mean.append(model_df[date1:date2].mean(),ignore_index=True)
		wind_aws = wind_aws.append(aws[str(df_wind.dates[i].year) + "-"+ str(df_wind.dates[i].month) + "-" +str(df_wind.dates[i].day)].max(),ignore_index=True)

	#Plot distributions
	param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	for param in param_list:
		if param in ["crt"]:
			plot_distributions_plotting(model_df,wind_erai_mean,df_wind,wind_aws,\
				param,location,model)	
		else:
			plot_distributions_plotting(model_df,wind_erai_max,df_wind,wind_aws,param,\
				location,model)	

def plot_distributions_plotting(model_df,wind_erai,df_wind,wind_aws,param,location,model):
	ms=1
	plt.figure()
	print(param)
	plt.hist(model_df[param],log=True)
	for i in np.arange(0,wind_erai.shape[0]):
		plt.text(wind_erai[param][i],100*(i+1),\
			"x "+ df_wind["gust (m/s)"].values.astype(str)[i]+" AWS: "+\
			wind_aws["wind_gust"].values.astype(str)[i],\
			color="r",horizontalalignment="left")
	[cmap,levels,cb_lab] = contour_properties(param)
	plt.title(param)
	plt.xlabel(cb_lab)	
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+location+\
			"/"+model+"_"+param+".png",bbox_inches="tight")


def match_aws(aws,reanal,location,model,lightning=[0]):
	#Take ~half-hourly AWS data and match to 6-hourly reanalysis. Add aws wind gusts to reanal
	# dataframe. Also add lightning data

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
	reanal["aws_wind_gust"] = aws_erai.wind_gust

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

def plot_scatter_full(location,model):
	#Same as plot_scatter, but showing seasonality, and plotting convective/non-convective
	#wind gusts on the same plot

	#Read AWS data
	print("Loading AWS...");aws = read_aws(location)

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df()
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	elif model == "ADversion":
		erai_df = load_ADversion_df()
	else:
		raise NameError("Invalid model selection")

	#Get reanalysis data at "location"
	erai_df = erai_df[erai_df.loc_id==location]
	
	#Match half-hourly aws data to 6-hourly reanalysis, add lightning and aws wind gusts to df
	erai_df = match_aws(aws,erai_df,location,model)

	#Extract convective/non-convective wind gusts for warm/cold season
	erai_df_conv = erai_df[(erai_df.lightning>=2) & (erai_df.mu_cin <= 100)]
	erai_df_non_conv = erai_df[(erai_df.lightning<2) | (erai_df.mu_cin > 100)]
	non_conv_warm_inds = np.array([erai_df_non_conv.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df_non_conv.shape[0])])
	conv_warm_inds = np.array([erai_df_conv.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df_conv.shape[0])])
	erai_df_non_conv_warm = erai_df_non_conv[non_conv_warm_inds]
	erai_df_non_conv_cold = erai_df_non_conv[~(non_conv_warm_inds)]
	erai_df_conv_warm = erai_df_conv[conv_warm_inds]
	erai_df_conv_cold = erai_df_conv[~(conv_warm_inds)]

	#Plot for all parameters
	param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	
	for param in param_list:
			
		x_conv_warm = np.array(erai_df_conv_warm[param])
		x_conv_cold = np.array(erai_df_conv_cold[param])
		x_non_conv_warm = np.array(erai_df_non_conv_warm[param])
		x_non_conv_cold = np.array(erai_df_non_conv_cold[param])
		y_conv_warm = np.array(erai_df_conv_warm.aws_wind_gust)
		y_conv_cold = np.array(erai_df_conv_cold.aws_wind_gust)
		y_non_conv_warm = np.array(erai_df_non_conv_warm.aws_wind_gust)
		y_non_conv_cold = np.array(erai_df_non_conv_cold.aws_wind_gust)
		#REMOVE NANs	
		x_conv_warm_nan_ind = np.isnan(x_conv_warm)
		x_conv_cold_nan_ind = np.isnan(x_conv_cold)
		x_non_conv_warm_nan_ind = np.isnan(x_non_conv_warm)
		x_non_conv_cold_nan_ind = np.isnan(x_non_conv_cold)
		x_conv_warm = x_conv_warm[~x_conv_warm_nan_ind]
		x_conv_cold = x_conv_cold[~x_conv_cold_nan_ind]
		x_non_conv_warm = x_non_conv_warm[~x_non_conv_warm_nan_ind]
		x_non_conv_cold = x_non_conv_cold[~x_non_conv_cold_nan_ind]
		y_conv_warm = y_conv_warm[~x_conv_warm_nan_ind]
		y_conv_cold = y_conv_cold[~x_conv_cold_nan_ind]
		y_non_conv_warm = y_non_conv_warm[~x_non_conv_warm_nan_ind]
		y_non_conv_cold = y_non_conv_cold[~x_non_conv_cold_nan_ind]

		outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/"+location+"/"+\
				model+"_"+param+"_seasonal"

		#PLOT STYLE 1 (two scatter plots on top of each other)
		plt.figure()
		s3 = plt.scatter(x_non_conv_warm,y_non_conv_warm,c="k",marker="^",s=14)
		s4 = plt.scatter(x_non_conv_cold,y_non_conv_cold,c="k",marker="s",s=14)
		s1 = plt.scatter(x_conv_warm,y_conv_warm,c="r",marker="^",s=30)
		s2 = plt.scatter(x_conv_cold,y_conv_cold,c="b",marker="s",s=30)
		plt.legend((s1,s2,s3,s4),("Convective Warm Season","Convective Cold Season",\
			"Non-Convective Warm Season","Non-Convective Cold Season"),\
			bbox_to_anchor=(0., -.4, 1., .102),fontsize=10,\
			loc="lower left")
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s)")
		plt.title(location)
		plt.savefig(outname+"_full.png",bbox_inches="tight")
		plt.close()
		plt.figure()
		s1 = plt.scatter(x_conv_warm,y_conv_warm,c="r",marker="^")
		s2 = plt.scatter(x_conv_cold,y_conv_cold,c="b",marker="s")
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s)")
		plt.legend((s1,s2),("Convective Warm Season","Convective Cold Season"),\
			bbox_to_anchor=(0., -.4, 1., .102),fontsize=10,\
			loc="lower left")
		plt.title(location)
		plt.savefig(outname+"_conv.png",bbox_inches="tight")
		plt.close()

		#PLOT STYLE 2 (2d-histogram with scatterplot on top)
		print(param)
		#if param in ["mu_cape","ml_cape","cape*ssfc6","cape*s06"]:
		#	vm = 200
		#else:
		#	vm = None
		#cmap,levels,clab=contour_properties(param)
		#plt.figure(figsize=(20,7))
		#plt.subplot(121)
		#plt.hist2d(x_non_conv_warm,y_non_conv_warm,cmap=cm.Reds,bins=20,vmin=0,vmax=vm)
		#cb1 = plt.colorbar();cb1.set_label("Number of Non-Convective Warm Season Gusts")
		#s1 = plt.scatter(x_conv_warm,y_conv_warm,c="k",marker="^",s=30,\
		#	label="Convective Gusts Warm Season")
		#plt.xlabel(param)
		#plt.ylabel("AWS Warm Season Wind Gust (m/s)")
		#plt.legend(bbox_to_anchor=(0., 1, 1., .102),fontsize=10,\
		#	loc="upper left")
		#plt.subplot(122)
		#plt.hist2d(x_non_conv_cold,y_non_conv_cold,cmap=cm.Blues,bins=20,vmin=0,vmax=vm)
		#cb1 = plt.colorbar();cb1.set_label("Number of Non-Convective Cool Season Gusts")
		#s1 = plt.scatter(x_conv_cold,y_conv_cold,c="k",marker="^",s=30,\
		#	label="Convective Gusts Cool Season")
		#plt.xlabel(param)
		#plt.ylabel("AWS Cool Season Wind Gust (m/s)")
		#plt.legend(bbox_to_anchor=(0., 1, 1., .102),fontsize=10,\
		#	loc="upper right")
		#plt.suptitle(location,size=20)
		#plt.savefig(outname+".png",bbox_inches="tight")
		#plt.close()


def plot_scatter(location,model,convective,threshold=False):
	#Plot scatter plots of reanalysis/observed convective indices (see load_*dataset*_df()) 
	# against AWS max gust strength (see read_aws())
	#Set convective argument to True to restrict data to times when CAPE > 100
	#Set threshold to an integer to restrict gust data to values over the threshold. 
	# Otherwise, False

	#Read AWS data
	print("Loading AWS...");aws = read_aws(location)

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df()
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	elif model == "ADversion":
		erai_df = load_ADversion_df()
	else:
		raise NameError("Invalid model selection")

	erai_df = erai_df[erai_df.loc_id==location]
	
	erai_df = match_aws(aws,erai_df,location,model)

	if convective:
		#Create convective events using lightning
		if model == "ADversion":
			erai_df = erai_df[(erai_df.lightning>=2)]
		else:
			erai_df = erai_df[(erai_df.lightning>=2) & (erai_df.mu_cin <= 100)]
	if threshold:
		thresh_inds = (erai_df.aws_wind_gust >= threshold)
		erai_df = erai_df[thresh_inds]

	#param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
	#		"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
	#		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
	#		"cape*s06","cape*ssfc6"]
	param_list = ["mu_cape","s06"]
	
	for param in param_list:
			
		plt.figure()
		x = np.array(erai_df[param])
		y = np.array(erai_df.aws_wind_gust)
		if param == "lcl":
			nan_ind = np.isnan(x)
			x = x[~nan_ind]
			y = y[~nan_ind]
		m,b = np.polyfit(x,y,deg=1)
		r = np.corrcoef(x,y)[0,1]
		#plt.hist2d(x,y,bins=50)
		#plt.colorbar()
		plt.scatter(x,y)
		plt.plot(x, b+m*x,"k")
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s")
		plt.title(str(r))
		outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/"+location+"/"+\
				model+"_"+param
		if convective:
			outname = outname + "_conv"	
		if threshold:
			outname = outname + "_" + str(threshold)	
		outname = outname + ".png"
		plt.savefig(outname,bbox_inches="tight")
		plt.close()

def plot_density_contour(location,param1,param2,model,convective,threshold=15):
	#Probability of wind gust over "threshold" for all data (convective = False) or just
	# for convective events (convective = True)
	
	#Read AWS data
	print("Loading AWS...");aws = read_aws(location)

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

	erai_df = erai_df[erai_df.loc_id==location]
	
	erai_df = match_aws(aws,erai_df,location,model)

	if convective:
		#Create convective events using lightning
		erai_df = erai_df[(erai_df.lightning>=2) & (erai_df.mu_cin <= 100) & \
				(erai_df.aws_wind_gust>=15)]


	x = erai_df[param1]
	y = erai_df[param2]
	plt.hist2d(x,y,bins=10,range=[[0,10],[0,50]],cmap=cm.Reds);plt.colorbar();
	plt.xlabel(param1)
	plt.ylabel(param2)
	out_name = location+"_"+model+"_"+param1+"_"+param2+"_"+str(threshold)
	if convective:
		out_name = out_name + "conv"
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/2dhist/"+out_name+".png")
	

def plot_density_contour_all(param1,param2,model,convective,threshold=15):
	#Same as for plot_density_contour but for multiple stations. I.e. stations which are 
	# present in both reanalysis dataframes and aws data
	
	#Read AWS data
	print("loading aws...");aws = load_aws_all()

	#Load lightning data (may be "smoothed", created by taking the max count +/- 100 km 
	# around each point
	lightning = load_lightning()

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


	#Group each dataset by station locations, and match datasets
	aws_groups = aws.groupby("stn_name")
	erai_groups = erai_df.groupby("loc_id")
	total_erai = pd.DataFrame()
	for name, group in aws_groups:
		if name in erai_groups.groups.keys():
			group
			print(name)
			total_erai = total_erai.append(match_aws(group,erai_groups.get_group(name),\
				name,model,lightning))

	#Create convective events using lightning
	conv_erai = total_erai[(total_erai.lightning>=2) & (total_erai.mu_cin <= 100) & \
			(total_erai.aws_wind_gust>=20)]
	non_conv_erai = total_erai[(total_erai.lightning<2) | (total_erai.mu_cin > 100) | \
			(total_erai.aws_wind_gust<20)]

	#Example plot for distributions
	plt.figure(figsize=[12,8])
	plt.hist(non_conv_erai.mu_cape,range=(0,3000),histtype="step",normed=True,label="Non-Convective Wind Gusts + Convective Wind Gusts < 20 m/s" ,log=True)
	plt.hist(conv_erai.mu_cape,range=(0,3000),histtype="step",normed=True,label="Convective Wind Gusts > 20 m/s",log=True)
	plt.legend(loc="upper left",bbox_to_anchor=[0,1.05,1,.05],mode="expand",fontsize=10)
	
	plt.show()

	#Plot density plot
	bins=10
	x_conv = conv_erai[param1]
	y_conv = conv_erai[param2]
	r = [[0,3000],[0,60]]
	conv_hist = plt.hist2d(x_conv,y_conv,range=r,bins=bins,cmap=cm.Reds);plt.colorbar();
	x_non_conv = total_erai[param1]
	y_non_conv = total_erai[param2]
	non_conv_hist = plt.hist2d(x_non_conv,y_non_conv,range=r,bins=bins,cmap=cm.Reds);plt.colorbar();
	plt.xlabel(param1)
	plt.ylabel(param2)

	prob = conv_hist[0] / non_conv_hist[0]
	rx = np.linspace((r[0][1] - r[0][0])/bins,r[0][1],bins)
	ry = np.linspace((r[1][1] - r[1][0])/bins,r[1][1],bins)
	rx,ry = np.meshgrid(rx,ry)
	plt.contourf(rx,ry,prob);plt.colorbar()
	
if __name__ == "__main__":


	path = '/g/data/eg3/ab4502/ExtremeWind/aus/'
	#f = "barra_wrf_20140622_20140623"
	f = "erai_wrf_20140622_20140624"
	model = "erai"
	fname = path+f+".nc"
	param = "cape*s06"
	region = "sa_small"

	time = [dt.datetime(2014,10,6,6,0,0),dt.datetime(2014,06,10,12,0,0)]
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

	#plot_netcdf("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/barra_20141001_20141030.nc"\
	#	,"mu_cape","mucape_20140610",domain,time,"barra")
	#plot_netcdf_animate(fname,param,outname,domain)
	#plot_netcdf_clim("cape*s06",domain,"erai")
	#plot_scatter_full("Port Augusta","erai")
	#plot_density_contour("Port Augusta","mu_cape","s06","erai",True,threshold=15)
	#plot_scatter("Adelaide AP","ADversion",True,threshold=False)
	plot_distributions("erai","Port Augusta")

