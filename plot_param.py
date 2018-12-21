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
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

def plot_param(df,param,domain,time,model):

#Plot parameters from a dataframe with parameter and lat lon data 

	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "i")
	no_lon = len(np.unique(df["lon"]))
	no_lat = len(np.unique(df["lat"]))
	x = np.reshape(df["lon"],(no_lon,no_lat))
	y = np.reshape(df["lat"],(no_lon,no_lat))

	for p in param:

		[cmap,levels,cb_lab,range,log_plot] = contour_properties(p)

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

def plot_netcdf(fname,outname,time,model):

#Load a single netcdf file and plot time max if time is a list of [start_time,end_time]
# or else plot for a single time if time is length=1

	f = nc.Dataset(fname)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)

	vars = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
	len(f.variables.items()))])
	vars = vars[~(vars=="time") & ~(vars=="lat") & ~(vars=="lon")]	
	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "i")
	for param in vars:
		if len(time)==2:
			values = np.nanmax(f.variables[param][(times_dt>=time[0]) & \
				(times_dt<=time[-1])],axis=0)
		else:
			values = np.squeeze(f.variables[param][times_dt == time])

		print(param)
		[cmap,levels,cb_lab,range,log_plot] = contour_properties(param)

		m.drawcoastlines()
		m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),5),\
				labels=[True,False,False,True])
		m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),5),\
				labels=[True,False,True,False])
		m.contourf(x,y,values,latlon=True,cmap=cmap,levels=levels,extend="both")
		cb = plt.colorbar()
		cb.set_label(cb_lab)
		plt.title(str(time[0]) + "-" + str(time[-1]))
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+\
			"_"+param+".png",bbox_inches="tight")
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
	if param in ["mu_cape","cape","cape700"]:
		cmap = cm.YlGnBu
		levels = np.linspace(0,2000,11)
		cb_lab = "J/Kg"
		range = [0,4000]
		log_plot = True
	if param == "ml_cape":
		cmap = cm.YlGnBu
		levels = np.linspace(0,1000,11)
		cb_lab = "J/Kg"
		range = [0,4000]
		log_plot = True
	if param == "mu_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
		range = [0,500]
		log_plot = True
	if param == "ml_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
		range = [0,500]
		log_plot = True
	if param in ["s06","ssfc6"]:
		cmap = cm.Reds
		levels = np.linspace(0,30,7)
		cb_lab = "m/s"
		range = [0,50]
		log_plot = False
	if param in ["ssfc3","wg10"]:
		cmap = cm.Reds
		levels = np.linspace(0,30,7)
		cb_lab = "m/s"
		range = [0,30]
		log_plot = False
	if param in ["ssfc850"]:
		cmap = cm.Reds
		levels = np.linspace(0,20,7)
		cb_lab = "m/s"
		range = [0,25]
		log_plot = False
	if param in ["ssfc500","ssfc1"]:
		cmap = cm.Reds
		levels = np.linspace(0,20,7)
		cb_lab = "m/s"
		range = [0,20]
		log_plot = False
	if param in ["stp","ship","non_sc_stp","scp"]:
		cmap = cm.Reds
		levels = np.linspace(0,0.5,11)
		cb_lab = ""
		range = [0,3]
		log_plot = True
	if param in ["mmp"]:
		cmap = cm.Reds
		levels = np.linspace(0,1,11)
		cb_lab = ""
		range = [0,1]
		log_plot = True
	if param in ["srh01","srh03","srh06"]:
		cmap = cm.Reds
		levels = np.linspace(0,200,11)
		cb_lab = "m^2/s^2"
		range = [0,400]
		log_plot = True
	if param == "crt":
		cmap = cm.YlGnBu
		levels = [60,120]
		cb_lab = "degrees"
		range = [0,180]
		log_plot = False
	if param in ["relhum850-500","relhum1000-700"]:
		cmap = cm.YlGnBu
		levels = np.linspace(0,100,11)
		cb_lab = "%"
		range = [0,100]
		log_plot = False
	if param == "vo":
		cmap = cm.RdBu_r
		levels = np.linspace(-8e-5,8e-5,17)
		cb_lab = "s^-1"
		range = [-8e-5,-8e-5]
		log_plot = False
	if param == "lr1000":
		cmap = cm.Blues
		levels = np.linspace(2,12,11)
		cb_lab = "deg/km"
		range = [0,12]
		log_plot = False
	if param == "lcl":
		cmap = cm.YlGnBu_r
		levels = np.linspace(0,8000,9)
		cb_lab = "m"
		range = [0,8000]
		log_plot = False
	if param in ["td800","td850","td950"]:
		cmap = cm.Reds
		levels = np.linspace(0,30,9)
		cb_lab = "deg"
		range = [0,40]
		log_plot = False
	if param in ["cape*s06","cape*ssfc6"]:
		cmap = cm.YlGnBu
		levels = np.linspace(0,100000,11)
		cb_lab = ""
		range = [0,60000]
		log_plot = True
	if param in ["cape*td850"]:
		cmap = cm.YlGnBu
		levels = None
		cb_lab = ""
		range = None
		log_plot = True
	if param in ["dp850-500","dp1000-700"]:
		cmap = cm.YlGnBu
		levels = np.linspace(-10,10,11)
		cb_lab = "degC"
		range = [-20,20]
		log_plot = False
	return [cmap,levels,cb_lab,range,log_plot]

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
		model_df = load_erai_df(False,location)
	elif model == "obs":
		model_df = load_obs_df(location)
	else:
		raise NameError("Invalid model selection")

	#Read AWS data to check JDH data
	aws = load_aws_all(resample=True)
	aws.index = aws.date
	aws = aws[aws.stn_name==location]

	#Extract highest reanalysis parameters for the day of the JDH wind gust, 
	# to put on dist.
	#NOTE WE DON'T WANT MAX CRT (OR LCL?)
	wind_erai_min = pd.DataFrame()
	wind_erai_mean = pd.DataFrame()
	wind_erai_max = pd.DataFrame()
	wind_aws = pd.DataFrame()
	model_df.index = model_df.date
	for i in np.arange(0,df_wind.shape[0]):
		date = str(df_wind.dates[i].year)+"-"+str(df_wind.dates[i].month)+"-"+str(df_wind.dates[i].day)
		wind_erai_max = wind_erai_max.append(model_df[date].max(),ignore_index=True)
		wind_erai_min = wind_erai_min.append(model_df[date].min(),ignore_index=True)
		wind_erai_mean = wind_erai_mean.append(model_df[date].mean(),ignore_index=True)
		wind_aws = wind_aws.append(aws[date].max(),ignore_index=True)

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

def plot_scatter_full(model,location=False):
	#Same as plot_scatter, but showing seasonality, and plotting convective/non-convective
	#wind gusts on the same plot

	#Read AWS data
	print("Loading AWS...");aws = load_aws_all(True)

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df(False,False)
	elif model == "erai_fc":
		erai_df = load_erai_df(False,True)
		erai_df = erai_df.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	elif model == "ADversion":
		s06 = load_ADversion_df("s06")
		mu_cape = load_ADversion_df("mu_cape")
		s06 = s06.set_index(["date","loc_id"])
		mu_cape = mu_cape.set_index(["date","loc_id"])
		erai_df = pd.concat([s06.sort_index(),mu_cape["MUCAPE_ADowdy_version"].sort_index()\
			],axis=1,)
		erai_df = erai_df.rename(columns={"S06_ADowdy_version":"s06",\
			"MUCAPE_ADowdy_version":"mu_cape"})
		erai_df["cape*s06"] = erai_df.mu_cape * erai_df.s06
	else:
		raise NameError("Invalid model selection")

	#Match half-hourly aws data to 6-hourly reanalysis, add lightning and aws wind gusts to df
	if location == False:
		lightning=load_lightning()
		aws = aws.set_index(["stn_name"],append=True)
		if model != "ADversion":
			erai_df = erai_df.set_index(["date","loc_id"])
		lightning = lightning.set_index(["date","loc_id"])
		erai_df = pd.concat([aws.wind_gust,erai_df,lightning.lightning],axis=1)
		erai_df = erai_df[~(erai_df.lat.isna()) & ~(erai_df.wind_gust.isna())]
	else:
		erai_df = erai_df[erai_df.loc_id==location]
		erai_df = match_aws(aws,erai_df,location,model)

	#HACK: ADD TD*CAPE
	if model == "erai":
		erai_df["cape*td850"] = erai_df["mu_cape"]*erai_df["td850"]

	if model == "erai_fc":
		erai_df = erai_df[~(erai_df["wg10"].isna())]
		erai_df = erai_df[~(erai_df["cape"].isna())]

	#Extract convective/non-convective wind gusts for warm/cold season
	erai_df_conv = erai_df[(erai_df.lightning>=2)]
	erai_df_non_conv = erai_df[(erai_df.lightning<2)]
	non_conv_warm_inds = np.array([erai_df_non_conv.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df_non_conv.shape[0])])
	conv_warm_inds = np.array([erai_df_conv.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df_conv.shape[0])])
	erai_df_non_conv_warm = erai_df_non_conv[non_conv_warm_inds]
	erai_df_non_conv_cold = erai_df_non_conv[~(non_conv_warm_inds)]
	erai_df_conv_warm = erai_df_conv[conv_warm_inds]
	erai_df_conv_cold = erai_df_conv[~(conv_warm_inds)]

	#Plot for all parameters
	if model == "erai":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6","td850","td800","td950"]
	elif model == "barra":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif model == "ADversion":
		param_list = ["mu_cape","s06"]
	elif model == "erai_fc":
		param_list = ["wg10","cape"]
	
	for param in param_list:
			
		x_conv_warm = np.array(erai_df_conv_warm[param])
		x_conv_cold = np.array(erai_df_conv_cold[param])
		x_non_conv_warm = np.array(erai_df_non_conv_warm[param])
		x_non_conv_cold = np.array(erai_df_non_conv_cold[param])
		y_conv_warm = np.array(erai_df_conv_warm.wind_gust)
		y_conv_cold = np.array(erai_df_conv_cold.wind_gust)
		y_non_conv_warm = np.array(erai_df_non_conv_warm.wind_gust)
		y_non_conv_cold = np.array(erai_df_non_conv_cold.wind_gust)

		#Get correlation
		m_conv_warm,b_conv_warm = np.polyfit(x_conv_warm,y_conv_warm,deg=1)
		r_conv_warm = np.corrcoef(x_conv_warm,y_conv_warm)[0,1]
		m_conv_cold,b_conv_cold = np.polyfit(x_conv_cold,y_conv_cold,deg=1)
		r_conv_cold = np.corrcoef(x_conv_cold,y_conv_cold)[0,1]
		m_non_conv_warm,b_non_conv_warm = np.polyfit(x_non_conv_warm,y_non_conv_warm,deg=1)
		r_non_conv_warm = np.corrcoef(x_non_conv_warm,y_non_conv_warm)[0,1]
		m_non_conv_cold,b_non_conv_cold = np.polyfit(x_non_conv_cold,y_non_conv_cold,deg=1)
		r_non_conv_cold = np.corrcoef(x_non_conv_cold,y_non_conv_cold)[0,1]
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

		if location == False:
		    outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/all_stations/"+\
				model+"_"+param+"_seasonal"
		    plot_title = "All Stations " + model
		else:
		    outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/"+location+"/"+\
				model+"_"+param+"_seasonal"
		    plot_title = location + " " + model

		#PLOT STYLE 1 (two scatter plots on top of each other)
		#plt.figure()
		#s3 = plt.scatter(x_non_conv_warm,y_non_conv_warm,c="k",marker="^",s=14)
		#s4 = plt.scatter(x_non_conv_cold,y_non_conv_cold,c="k",marker="s",s=14)
		#s1 = plt.scatter(x_conv_warm,y_conv_warm,c="r",marker="^",s=30)
		#s2 = plt.scatter(x_conv_cold,y_conv_cold,c="b",marker="s",s=30)
		#plt.legend((s1,s2,s3,s4),("Convective Warm Season","Convective Cold Season",\
		#	"Non-Convective Warm Season","Non-Convective Cold Season"),\
		#	bbox_to_anchor=(0., -.4, 1., .102),fontsize=10,\
		#	loc="lower left")
		#plt.xlabel(param)
		#plt.ylabel("AWS Wind Gust (m/s)")
		#plt.title(location)
		#plt.savefig(outname+"_full.png",bbox_inches="tight")
		#plt.close()
		#plt.figure()
		#s1 = plt.scatter(x_conv_warm,y_conv_warm,c="r",marker="^")
		#s2 = plt.scatter(x_conv_cold,y_conv_cold,c="b",marker="s")
		#plt.xlabel(param)
		#plt.ylabel("AWS Wind Gust (m/s)")
		#plt.legend((s1,s2),("Convective Warm Season","Convective Cold Season"),\
		#	bbox_to_anchor=(0., -.4, 1., .102),fontsize=10,\
		#	loc="lower left")
		#plt.title(location)
		#plt.savefig(outname+"_conv.png",bbox_inches="tight")
		#plt.close()

		#PLOT STYLE 2 (2d-histogram with scatterplot on top)
		print(param)
		if param in ["mu_cape","ml_cape","cape*ssfc6","cape*s06"]:
			vm = 200
		else:
			vm = None
		cmap,levels,clab,range,log_plot=contour_properties(param)
		fig = plt.figure(figsize=(20,10))

		plt.subplot(221)
		s1 = plt.hist2d(x_conv_warm,y_conv_warm,cmap=cm.Reds,bins=20,vmin=0,vmax=200)
		cb1 = plt.colorbar();cb1.set_label("Number of Convective Warm Season Gusts")
		plt.xlabel(param)
		plt.ylabel("AWS Warm Season Wind Gust (m/s)")
		plt.title("Convective Wind Gusts (Warm Season)")
		plt.plot(x_conv_warm, b_conv_warm+m_conv_warm*x_conv_warm,"k",\
			label="r = "+str(round(r_conv_warm,3)))
		plt.legend()

		plt.subplot(222)
		s1 = plt.hist2d(x_conv_cold,y_conv_cold,cmap=cm.Blues,bins=20,vmin=0,vmax=200)
		cb1 = plt.colorbar();cb1.set_label("Number of Convective Cold Season Gusts")
		plt.xlabel(param)
		plt.ylabel("AWS Cold Season Wind Gust (m/s)")
		plt.title("Convective Wind Gusts (Cold Season)")
		plt.plot(x_conv_cold, b_conv_cold+m_conv_cold*x_conv_cold,"k",\
			label="r = "+str(round(r_conv_cold,3)))
		plt.legend()

		plt.subplot(223)
		s1 = plt.hist2d(x_non_conv_warm,y_non_conv_warm,cmap=cm.Reds,bins=20,vmin=0,vmax=2000)
		cb1 = plt.colorbar();cb1.set_label("Number of Non-Convective Warm Season Gusts")
		plt.xlabel(param)
		plt.ylabel("AWS Warm Season Wind Gust (m/s)")
		plt.title("Non-Convective Wind Gusts (Warm Season)")
		plt.plot(x_conv_warm, b_conv_warm+m_conv_warm*x_conv_warm,"k",\
			label="r = "+str(round(r_non_conv_warm,3)))
		plt.legend()

		plt.subplot(224)
		s1 = plt.hist2d(x_non_conv_cold,y_non_conv_cold,cmap=cm.Blues,bins=20,vmin=0,vmax=2000)
		cb1 = plt.colorbar();cb1.set_label("Number of Non-Convective Cold Season Gusts")
		plt.xlabel(param)
		plt.ylabel("AWS Cold Season Wind Gust (m/s)")
		plt.title("Non-Convective Wind Gusts (Cold Season)")
		plt.plot(x_non_conv_cold, b_non_conv_cold+m_non_conv_cold*x_non_conv_cold,"k",\
			label="r = "+str(round(r_non_conv_cold,3)))
		plt.legend()


		plt.suptitle(plot_title,size=20)
		plt.savefig(outname+".png",bbox_inches="tight")
		plt.close()


def plot_scatter(model,location=False,threshold=False):
	#Plot scatter plots of reanalysis/observed convective indices (see load_*dataset*_df()) 
	# against AWS max gust strength (see read_aws()) for convective and all wind gusts
	#Set threshold to an integer to restrict gust data to values over the threshold. 
	# Otherwise, False

	#Read AWS data
	print("Loading AWS...");aws = load_aws_all(resample=True)

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai_fc":
		erai_df = load_erai_df(False,True)
		#Duplicates appear (uninvestigated). Probably something to do with first day
		# of following month being present in each monthl;y netcdf file
		erai_df = erai_df.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
	elif model == "erai_both":
		erai_df_fc = load_erai_df(False,True)
		erai_df_fc = erai_df_fc.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
		erai_df_an = load_erai_df(False,False)
		erai_df_an = erai_df_an.set_index(["date","loc_id"])
		erai_df_fc = erai_df_fc.set_index(["date","loc_id"])
		erai_df = pd.concat([erai_df_fc.wg10,erai_df_an],axis=1)
	elif model == "erai":
		erai_df = load_erai_df(False,False)
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	elif model == "ADversion":
		s06 = load_ADversion_df("s06")
		mu_cape = load_ADversion_df("mu_cape")
		s06 = s06.set_index(["date","loc_id"])
		mu_cape = mu_cape.set_index(["date","loc_id"])
		erai_df = pd.concat([s06.sort_index(),mu_cape["MUCAPE_ADowdy_version"].sort_index()\
			],axis=1,)
		erai_df = erai_df.rename(columns={"S06_ADowdy_version":"s06",\
			"MUCAPE_ADowdy_version":"mu_cape"})
		erai_df["cape*s06"] = erai_df.mu_cape * erai_df.s06
	else:
		raise NameError("Invalid model selection")

	if location == False:
		lightning=load_lightning()
		aws = aws.set_index(["stn_name"],append=True)
		if (model != "ADversion") & (model != "erai_both"):
			erai_df = erai_df.set_index(["date","loc_id"])
		lightning = lightning.set_index(["date","loc_id"])
		erai_df = pd.concat([aws.wind_gust,erai_df,lightning.lightning],axis=1)
		erai_df = erai_df[~(erai_df.lat.isna()) & ~(erai_df.wind_gust.isna())]
	else:
		erai_df = erai_df[erai_df.loc_id==location]
		erai_df = match_aws(aws,erai_df,location,model)

	#Create convective events using lightning
	erai_df_conv = erai_df[(erai_df.lightning>=2)]
	erai_df = erai_df[(erai_df.lightning<2)]

	if threshold:
		thresh_inds = (erai_df.aws_wind_gust >= threshold)
		erai_df = erai_df[thresh_inds]

	if model == "erai":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6","td850","td800","td950"]
	elif model == "barra":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif model == "ADversion":
		param_list = ["mu_cape","s06","cape*s06"]
	
	for param in param_list:
			
		plt.figure(figsize=[18,8])
		plt.subplot(121)
		x = np.array(erai_df[param])
		y = np.array(erai_df.wind_gust)
		nan_ind = np.isnan(x)
		x = x[~nan_ind]
		y = y[~nan_ind]
		m,b = np.polyfit(x,y,deg=1)
		r = np.corrcoef(x,y)[0,1]
		#plt.hist2d(x,y,bins=50)
		plt.scatter(x,y,color="k")
		#plt.colorbar()
		plt.plot(x, b+m*x,"r",label="r = "+str(round(r,3)))
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s)")
		plt.title("Non-Convective AWS Wind Gusts")
		plt.legend()

		plt.subplot(122)
		x = np.array(erai_df_conv[param])
		y = np.array(erai_df_conv.wind_gust)
		nan_ind = np.isnan(x)
		x = x[~nan_ind]
		y = y[~nan_ind]
		m,b = np.polyfit(x,y,deg=1)
		r = np.corrcoef(x,y)[0,1]
		#plt.hist2d(x,y,bins=50)
		plt.scatter(x,y,color="k")
		#plt.colorbar()
		plt.plot(x, b+m*x,"r",label="r = "+str(round(r,3)))
		plt.xlabel(param)
		plt.ylabel("AWS Wind Gust (m/s)")
		plt.legend()
		plt.title("Convective AWS Wind Gusts Only")
		if location == False:
			outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/all_stations/"\
				+model+"_"+param
		else:
			outname = "/home/548/ab4502/working/ExtremeWind/figs/scatter/"+location+"/"\
				+model+"_"+param
		if threshold:
			outname = outname + "_" + str(threshold)	
		outname = outname + "_subplot.png"
		plt.suptitle("2010-2015 " + model)
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
	
def plot_jdh_dist(seasonal):
	#Same as for plot_conv_wind_gust_dist but for 1979-2015 using jdh data as events 

	#NOTE need to be careful when matching JDH events (daily) to reanalysis dataframe (6hr)
	
	#Load erai df from 1979-2015
	erai_df = load_erai_daily_max_df()
	erai_df = erai_df.drop_duplicates(["date","loc_id"])

	#Combine datasets
	jdh = match_jdh_erai().set_index(["dates","station"])
	erai_df = erai_df.set_index(["date","loc_id"])
	jdh["jdh"] = 1
	total_erai = pd.concat([erai_df,jdh.jdh],axis=1)
	total_erai.loc[total_erai.jdh.isna(),"jdh"] = 0
		
	#Remove rows where there are no parameters
	total_erai = total_erai[~(total_erai.lat.isna())]

	#Define warm inds
	warm_inds = np.array([total_erai.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,total_erai.shape[0])])

	#Plot distributions
	#param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
	#		"stp","ship","mmp","relhum850-500","crt","lr1000",\
	#		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
	#		"cape*s06","cape*ssfc6"]
	param_list = ["dp850-500","dp1000-700"]
	warm_fa_rates = []
	cold_fa_rates = []
	warm_fa_thresholds = []
	cold_fa_thresholds = []

	for param in param_list:
		print(param)
		fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=[12,10])
		cmap1,levels1,cb_lab1,range,log_plot = contour_properties(param)
		x1_w = total_erai[(warm_inds) & (total_erai["jdh"]==0)][param]
		x2_w = total_erai[(warm_inds) & (total_erai["jdh"]==1)][param]
		x1_c = total_erai[(~warm_inds) & (total_erai["jdh"]==0)][param]
		x2_c = total_erai[(~warm_inds) & (total_erai["jdh"]==1)][param]
		warm_far,warm_fa_rate,warm_thresh = get_far66(total_erai[(warm_inds)],\
				"jdh",param)
		cold_far,cold_fa_rate,cold_thresh = get_far66(total_erai[(~warm_inds)],\
				"jdh",param)
		warm_fa_rates.append(warm_fa_rate)
		cold_fa_rates.append(cold_fa_rate)
		warm_fa_thresholds.append(warm_thresh)
		cold_fa_thresholds.append(cold_thresh)
		label1 = "Non-Convective Wind Gust N = "+\
			str(x1_w.shape[0]) + " (warm) "+\
			str(x1_c.shape[0]) + " (cold)"
		label2 = "Convective Wind Gusts > 30 m/s: N = "+\
			str(x2_w.shape[0]) + " (warm) "+\
			str(x2_c.shape[0]) + " (cold) \n False Alarm Rate = "+\
			str(round(warm_fa_rate,3)) + \
			" (warm) "+str(round(cold_fa_rate,3)) + " (cold) "
		ax1.hist((x1_w,x2_w),histtype="bar",normed=True,\
			log=log_plot,label=(label1,label2),\
			color = ("b","m"),range=range)
		ax1.set_title("October - March")
		ax1.set_ylabel("Normalised Counts")
		ax1.legend(loc="lower left",bbox_to_anchor=[0,-1.5,1,.05],\
			mode="expand",fontsize=10)
		ax1.plot([warm_thresh,warm_thresh],[0,ax1.get_ylim()[1]],\
			color="m",linestyle="--")

		ax2.set_title("April - September")
		ax2.hist((x1_c,x2_c),histtype="bar",normed=True,\
			log=log_plot,label=(label1,label2),\
			color = ("b","m"),range=range)
		ax2.set_ylabel("Normalised Counts")
		ax2.plot([cold_thresh,cold_thresh],[0,ax1.get_ylim()[1]],\
			color="m",linestyle="--")
		fig.subplots_adjust(bottom=0.2)
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/"+\
			"convective_all_stations/erai_"+param+"_jdh_1979-2017.png",\
			boox_inches="tight")
		plt.close()
		
	pd.DataFrame({"cold_fa":cold_fa_rates,"warm_fa":warm_fa_rates,\
		"cold_thresh":cold_fa_thresholds,"warm_thresh":warm_fa_thresholds},\
		index=param_list)

def plot_conv_wind_gust_dist(model,seasonal,smooth):
	#Same as for plot_density_contour but for multiple stations. I.e. stations which are 
	# present in both reanalysis dataframes and aws data
	
	#Read AWS data
	print("Loading aws...");aws = load_aws_all(resample=True)

	#Load lightning data (may be "smoothed", created by taking the max count +/- 100 km 
	# around each point
	lightning = load_lightning()

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df(fc=False,is1979=False)
	elif model == "erai_fc":
		erai_df = load_erai_df(False,True)
		erai_df = erai_df.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	elif model == "ADversion":
		s06 = load_ADversion_df("s06")
		mu_cape = load_ADversion_df("mu_cape")
		s06 = s06.set_index(["date","loc_id"])
		mu_cape = mu_cape.set_index(["date","loc_id"])
		erai_df = pd.concat([s06.sort_index(),mu_cape["MUCAPE_ADowdy_version"].sort_index()\
			],axis=1,)
		erai_df = erai_df.rename(columns={"S06_ADowdy_version":"s06",\
			"MUCAPE_ADowdy_version":"mu_cape"})
		erai_df["cape*s06"] = erai_df.mu_cape * erai_df.s06
	else:
		raise NameError("Invalid model selection")

	#Combine datasets
	aws = aws.set_index(["stn_name"],append=True)
	if model != "ADversion":
		erai_df = erai_df.set_index(["date","loc_id"])
	lightning = lightning.set_index(["date","loc_id"])
	if model == "erai":
		print("Loading JDH data...")
		jdh = match_jdh_erai()
		jdh_inds = [(jdh.year[i]>=2010) & (jdh.year[i]<=2015) \
				for i in np.arange(0,jdh.shape[0])]
		jdh = jdh[jdh_inds]
		jdh = jdh.set_index(["dates","station"])
	total_erai = pd.concat([aws.wind_gust,erai_df,lightning.lightning]\
		,axis=1)
		
	#Remove rows where there are no parameters or no wind gust data or no lightning
	total_erai = total_erai[~(total_erai.lat.isna()) & ~(total_erai.wind_gust.isna()) & \
		~(total_erai.lightning.isna())]

	#Create convective events using lightning
	#Note times are removed when reanalysis or wind gusts are NaNs
	total_erai["conv20"] = ((total_erai.lightning>=2) & (total_erai.wind_gust>=20))*1
	total_erai["conv25"] = ((total_erai.lightning>=2) & (total_erai.wind_gust>=25))*1
	total_erai["conv30"] = ((total_erai.lightning>=2) & (total_erai.wind_gust>=30))*1
	total_erai["conv_lessthan20"] = ((total_erai.lightning>=2) & (total_erai.wind_gust<20))*1
	total_erai["conv"] = ((total_erai.lightning>=2))*1
	total_erai["non_conv"] = ((total_erai.lightning<2))*1

	warm_inds = np.array([total_erai.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,total_erai.shape[0])])
	if model=="erai":
		jdh_warm_inds = np.array([jdh.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,jdh.shape[0])])


###################################################################################################
	#Plot distributions
	if model == "erai":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6","td850","td800","td950"]
	elif model == "barra":
		param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif model == "ADversion":
		param_list = ["mu_cape","s06","cape*s06"]
	elif model == "erai_fc":
		param_list = ["wg10"]
	if smooth:
		param_list = np.array(param_list)
		param_list = param_list[~("mu_cin"==param_list)]
		param_list = param_list[~("ml_cin"==param_list)]
		param_list = param_list[~("mmp"==param_list)]
		param_list = param_list[~("lcl"==param_list)]
		param_list = param_list[~("ssfc1"==param_list)]
		param_list = param_list[~("td850"==param_list)]
		param_list = param_list[~("td950"==param_list)]
		param_list = param_list[~("td800"==param_list)]

	for param in param_list:
		print(param)
		cmap1,levels1,cb_lab1,range,log_plot = contour_properties(param)

		if seasonal:
			fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=[12,10])
			x1_w = total_erai[(warm_inds) & (total_erai["non_conv"]==1)][param]
			x6_w = total_erai[(warm_inds) & (total_erai["conv_lessthan20"]==1)][param]
			x2_w = total_erai[(warm_inds) & (total_erai["conv20"]==1)][param]
			x3_w = total_erai[(warm_inds) & (total_erai["conv25"]==1)][param]
			x4_w = total_erai[(warm_inds) & (total_erai["conv30"]==1)][param]
			if model=="erai":
			    x5_w = jdh[(jdh_warm_inds)][param]
			x1_c = total_erai[~(warm_inds) & (total_erai["non_conv"]==1)][param]
			x2_c = total_erai[~(warm_inds) & (total_erai["conv20"]==1)][param]
			x3_c = total_erai[~(warm_inds) & (total_erai["conv25"]==1)][param]
			x4_c = total_erai[~(warm_inds) & (total_erai["conv30"]==1)][param]
			x6_c = total_erai[~(warm_inds) & (total_erai["conv_lessthan20"]==1)][param]

			conv20_far_w,conv20_fa_rate_w,conv20_thresh_w = get_far66(\
					total_erai[(warm_inds)],"conv20",param)
			conv30_far_w,conv30_fa_rate_w,conv30_thresh_w = get_far66(\
					total_erai[(warm_inds)],"conv30",param)
			conv25_far_w,conv25_fa_rate_w,conv25_thresh_w = get_far66(\
					total_erai[(warm_inds)],"conv25",param)
			conv20_far_c,conv20_fa_rate_c,conv20_thresh_c = get_far66(\
					total_erai[(~warm_inds)],"conv20",param)
			conv30_far_c,conv30_fa_rate_c,conv30_thresh_c = get_far66(\
					total_erai[(~warm_inds)],"conv30",param)
			conv25_far_c,conv25_fa_rate_c,conv25_thresh_c = get_far66(\
					total_erai[(~warm_inds)],"conv25",param)

			label1 = "Non-Convective N = "+\
				str(x1_w.shape[0]) + " (warm) "+\
				str(x1_c.shape[0]) + " (cold)"
			label2 = "Convective Wind Gusts > 20 m/s: N = "+\
				str(x2_w.shape[0]) + " (warm) "+\
				str(x2_c.shape[0]) + " (cold) \nFalse Alarm Rate = "+\
				str(round(conv20_fa_rate_w,3)) + \
				" (warm) "+str(round(conv20_fa_rate_c,3)) + " (cold) "
			label3 = "Convective Wind Gusts > 25 m/s: N = "+\
				str(x3_w.shape[0]) + " (warm) "+\
				str(x3_c.shape[0]) + " (cold) \nFalse Alarm Rate = "+\
				str(round(conv25_fa_rate_w,3)) + " (warm) "+\
				str(round(conv25_fa_rate_c,3)) + " (cold) "
			label4 = "Convective Wind Gusts > 30 m/s: N = "+\
				str(x4_w.shape[0]) + " (warm) "+\
				str(x4_c.shape[0]) + " (cold) \nFalse Alarm Rate = "+\
				str(round(conv30_fa_rate_w,3)) + " (warm) "+\
				str(round(conv30_fa_rate_c,3)) + " (cold) "
			label6 = "Convective Wind Gusts < 20 m/s: N = "+\
				str(x6_w.shape[0]) + " (warm) and "+\
				str(x6_c.shape[0]) + " (cold)"

			if smooth:
			    x1_w,x2_w,x3_w,x4_w,x6_w,x1_c,x2_c,x3_c,x6_c = dist_smooth(\
					[x1_w,x2_w,x3_w,x4_w,x6_w,x1_c,x2_c,x3_c,x6_c],range)
				
			if model=="erai":
			    x5_c = jdh[~(jdh_warm_inds)][param]
			    label5 = "JDH (daily max) non-synoptic events (2010-2015): N = "+\
				str(x5_w.shape[0]) + " (warm) and "+\
				str(x5_c.shape[0]) + " (cold)"
			    if smooth:
			    	x5_c = dist_smooth([x5_c],range)[0]
			    	x5_w = dist_smooth([x5_w],range)[0]

		else:
			fig,ax = plt.subplots(figsize=[12,8])
			x1 = total_erai[total_erai["non_conv"]==1][param]
			x2 = total_erai[total_erai["conv20"]==1][param]
			x3 = total_erai[total_erai["conv25"]==1][param]
			x4 = total_erai[total_erai["conv30"]==1][param]
			x6 = total_erai[total_erai["conv_lessthan20"]==1][param]
			if model == "erai":
			    x5 = jdh[param]
			    label5 = "JDH (daily max) non-synoptic events (2010-2015): N = 90"	   	
			    if smooth:
			    	x5 = dist_smooth([x5],range)

			conv20_far,conv20_fa_rate,conv20_thresh = get_far66(total_erai,\
					"conv20",param)
			conv25_far,conv25_fa_rate,conv25_thresh = get_far66(total_erai,\
					"conv25",param)
			conv30_far,conv30_fa_rate,conv30_thresh = get_far66(total_erai,\
					"conv30",param)

			label1 = "Non-Convective N = "+\
				str(x1.shape[0])
			label2 = "Convective Wind Gusts > 20 m/s: N = "+\
				str(x2.shape[0]) + " FAR = " + str(round(conv20_far))
			label3 = "Convective Wind Gusts > 25 m/s: N = "+\
				str(x3.shape[0]) + " FAR = " + str(round(conv25_far))
			label4 = "Convective Wind Gusts > 30 m/s: N = "+\
				str(x4.shape[0]) + " FAR = " + str(round(conv30_far))
			label6 = "Convective Wind Gusts < 20 m/s: N = "+\
				str(x6.shape[0])
			if smooth:
			    x1,x2,x3,x4,x6 = dist_smooth(\
					[x1,x2,x3,x4,x6],range)
		if model == "erai":
		    if seasonal:
			  if smooth:
			    	lines = [x1_w,x2_w,x3_w,x4_w,x5_w,x6_w]
			    	cols = ["b","g","r","c","m","k"]
			    	labels = [label1,label2,label3,label4,label5,label6]
			    	for i in np.arange(0,len(lines)):
			       		ax1.plot(np.linspace(range[0],range[1]),lines[i],\
						label=labels[i],color = cols[i])
			  else:
			  	ax1.hist((x1_w,x2_w,x3_w,x4_w,x5_w,x6_w),histtype="bar",normed=True,\
			    	log=log_plot,label=(label1,label2,label3,label4,label5,label6),\
			        	color = ("b","g","r","c","m","k"),range=range)
			  ax1.set_title("October - March")
			  ax1.set_ylabel("Normalised Counts")
			  ax1.legend(loc="lower left",bbox_to_anchor=[0,-2,1,.05],\
				mode="expand",fontsize=10)
			  thresholds = [conv20_thresh_w,conv25_thresh_w,conv30_thresh_w]
			  cols = ["g","r","c"]
			  for i in np.arange(0,len(thresholds)):
			  	ax1.plot([thresholds[i],thresholds[i]],[0,ax1.get_ylim()[1]],\
					color=cols[i],linestyle="--")
		    else:
			  if smooth:
			    	lines = [x1,x2,x3,x4,x5,x6]
			    	cols = ["b","g","r","c","m","k"]
			    	labels = [label1,label2,label3,label4,label5,label6]
			    	for i in np.arange(0,len(lines)):
			       		plt.plot(np.linspace(range[0],range[1]),lines[i],\
						label=labels[i],color = cols[i])
			  else:
			    	plt.hist((x1,x2,x3,x4,x5,x6),histtype="bar",normed=True,log=log_plot,\
					color = ("b","g","r","c","m","k"),\
					label=(label1,label2,label3,label4,label5,label6),\
					range=range)
			  plt.xlabel(param)
			  plt.ylabel("Normalised Counts")
			  thresholds = [conv20_thresh,conv25_thresh,conv30_thresh]
			  cols = ["g","r","c"]
			  for i in np.arange(0,len(thresholds)):
			  	plt.plot([thresholds[i],thresholds[i]],[0,plt.ylim()[1]],\
					color=cols[i],linestyle="--")
		else:
			if seasonal:
			  if smooth:
			    	lines = [x1_w,x2_w,x3_w,x4_w,x6_w]
			    	cols = ["b","g","r","c","k"]
			    	labels = [label1,label2,label3,label4,label6]
			    	for i in np.arange(0,len(lines)):
			       		ax1.plot(np.linspace(range[0],range[1]),lines[i],\
						label=labels[i],color = cols[i])
			  else:
			  	ax1.hist((x1_w,x2_w,x3_w,x4_w,x6_w),histtype="bar",normed=True,\
					log=log_plot,label=(label1,label2,label3,label4,label6),\
					color = ("b","g","r","c","k"),range=range)
			  ax1.set_title("October - March")
			  ax1.set_ylabel("Normalised Counts")
			  ax1.legend(loc="lower left",bbox_to_anchor=[0,-2,1,.05],\
				mode="expand",fontsize=10)
			  thresholds = [conv20_thresh_w,conv25_thresh_w,conv30_thresh_w]
			  cols = ["g","r","c"]
			  for i in np.arange(0,len(thresholds)):
			  	ax1.plot([thresholds[i],thresholds[i]],[0,ax1.get_ylim()[1]],\
					color=cols[i],linestyle="--")
			else:
			  if smooth:
			    	lines = [x1,x2,x3,x4,x6]
			    	cols = ["b","g","r","c","k"]
			    	labels = [label1,label2,label3,label4,label6]
			    	for i in np.arange(0,len(lines)):
			      		plt.plot(np.linspace(range[0],range[1]),lines[i],\
						label=labels[i],color = cols[i])
			  plt.xlabel(param)
			  plt.ylabel("Normalised Counts")

		if seasonal:
			#REMOVED >30m/s AS ONLY ONE EVENT IN COLD SEASON
		    if smooth:
			lines = [x1_c,x2_c,x3_c,x5_c,x6_c]
			cols = ["b","g","r","m","k"]
			labels = [label1,label2,label3,label5,label6]
			for i in np.arange(0,len(lines)):
			       ax2.plot(np.linspace(range[0],range[1]),lines[i],\
				label=labels[i],color = cols[i])
		    else:
			ax2.hist((x1_c,x2_c,x3_c,x6_c),histtype="bar",normed=True,\
				log=log_plot,label=(label1,label2,label3,label6),\
				color = ("b","g","r","k"),range=range)
		    ax2.set_title("April - September")
		    ax2.set_xlabel(param)
		    ax2.set_ylabel("Normalised Counts")
		    thresholds = [conv20_thresh_c,conv25_thresh_c]
		    cols = ["g","r","c"]
		    for i in np.arange(0,len(thresholds)):
			ax2.plot([thresholds[i],thresholds[i]],[0,ax1.get_ylim()[1]],\
				color=cols[i],linestyle="--")
		    plt.suptitle(model)
		    fig.subplots_adjust(bottom=0.25)
		    outname = "/home/548/ab4502/working/ExtremeWind/figs/distributions/"+\
				"convective_all_stations/"+model+"_"+param+"_seasonal_"+\
				str(erai_df.index.min()[0].year)+"_"+\
				str(erai_df.index.max()[0].year)
		else:
			fig.subplots_adjust(bottom=0.35)
			plt.legend(loc="lower left",bbox_to_anchor=[0,-0.5,1,.05],mode="expand",\
				fontsize=10)
			plt.title(model)
			outname = "/home/548/ab4502/working/ExtremeWind/figs/distributions/"+\
				"convective_all_stations/"+model+"_"+param+"_"+\
				str(erai_df.index.min()[0].year)+"_"+\
				str(erai_df.index.max()[0].year)
		if smooth:
			outname = outname+"_smooth.png"
		else:
			outname = outname+".png"

		plt.savefig(outname,bbox_inches="tight")
		plt.close()


def plot_wind_gust_prob(param1,param2,model,thresh):
	#Same as for plot_density_contour but for multiple stations. I.e. stations which are 
	# present in both reanalysis dataframes and aws data
	
	#Read AWS data
	print("Loading aws...");aws = load_aws_all(resample=True)

	#Load lightning data (may be "smoothed", created by taking the max count +/- 100 km 
	# around each point
	lightning = load_lightning()

	#Load Parameters
	print("Loading Parameters...")
	if model == "erai":
		erai_df = load_erai_df(False)
	elif model == "barra":
		erai_df = load_barra_df()
	elif model == "obs":
		erai_df = load_obs_df()
	else:
		raise NameError("Invalid model selection")

	aws = aws.set_index(["stn_name"],append=True)
	erai_df = erai_df.set_index(["date","loc_id"])
	lightning = lightning.set_index(["date","loc_id"])
	total_erai = pd.concat([aws,erai_df,lightning],axis=1)

	#Create convective events using lightning
	conv_erai_thresh = total_erai[(total_erai.lightning>=2) & (total_erai.mu_cin <= 100) & \
			(total_erai.wind_gust>=thresh)]
	conv_erai = total_erai[(total_erai.lightning>=2) & (total_erai.mu_cin <= 100) & \
			(total_erai.wind_gust<thresh)]
	non_conv_erai = total_erai[(total_erai.lightning<2) | (total_erai.mu_cin > 100) | \
			(total_erai.wind_gust<30)]

	#Plot density plot
	print(param1,param2)
	cmap1,levels1,cb_lab1,range1 = contour_properties(param1)
	cmap2,levels2,cb_lab2,range2 = contour_properties(param2)
	bins=10
	x_conv_thresh = conv_erai_thresh[param1]
	y_conv_thresh = conv_erai_thresh[param2]
	r = [range1,range2]
	conv_hist_thresh = plt.hist2d(x_conv_thresh,y_conv_thresh,range=r,bins=bins,cmap=cm.Reds,visible=False)
	x_conv = conv_erai[param1]
	y_conv = conv_erai[param2]
	conv_hist = plt.hist2d(x_conv,y_conv,range=r,bins=bins,cmap=cm.Reds,visible=False)
	plt.close()

	prob = conv_hist_thresh[0] / conv_hist[0]
	rx = np.linspace((r[0][1] - r[0][0])/bins,r[0][1],bins)
	ry = np.linspace((r[1][1] - r[1][0])/bins,r[1][1],bins)
	rx,ry = np.meshgrid(rx,ry)
	plt.contourf(rx,ry,prob,levels = [0,0.1,0.2,0.3,0.4,0.5],extend="max",cmap=cm.Reds)
	plt.colorbar()
	plt.title("Probability of convective gust > "+str(thresh)+" m/s given convective event")
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/probs/"+
		model+"_"+param1+"_"+param2+"_"+str(thresh)+".png")
	plt.show()

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

def dist_smooth(arrays,range):
	#Smooth each 1d array in arrays using gaussian kde
	
	smoothed_array = []
	for array in arrays:
		d = gaussian_kde(array)
		smoothed_array.append(d.evaluate(np.linspace(range[0],range[1])))
	return smoothed_array

def plot_param_scatter():
	#For ERA-Interim, plot scatterplots between all pairs of variables, highlighting "events"
	#Read AWS data
	print("Loading AWS...");aws = load_aws_all(resample=True)

	#Load Parameters
	print("Loading Parameters...")
	erai_df_fc = load_erai_df(False,True)
	erai_df_fc = erai_df_fc.drop_duplicates(subset=["lat","lon","year","month","day","hour"])
	erai_df_an = load_erai_df(False,False)
	erai_df_an = erai_df_an.set_index(["date","loc_id"])
	erai_df_fc = erai_df_fc.set_index(["date","loc_id"])
	erai_df = pd.concat([erai_df_fc.wg10,erai_df_an],axis=1)
	lightning=load_lightning()
	aws = aws.set_index(["stn_name"],append=True)
	lightning = lightning.set_index(["date","loc_id"])
	erai_df = pd.concat([aws.wind_gust,erai_df,lightning.lightning],axis=1)
	erai_df = erai_df[~(erai_df.lat.isna()) & ~(erai_df.wind_gust.isna())]

	#Get seasonal index
	warm_inds = np.array([erai_df.month[i] in np.array([10,11,12,1,2,3]) \
				for i in np.arange(0,erai_df.shape[0])])

	#Create convective events using lightning
	erai_df_cold = erai_df[(~warm_inds)]
	erai_df_conv20_cold = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>20) & (~warm_inds)]
	erai_df_conv25_cold = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>25) & (~warm_inds)]
	erai_df_conv30_cold = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>30) & (~warm_inds)]
	erai_df_warm = erai_df[(warm_inds)]
	erai_df_conv20_warm = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>20) & (warm_inds)]
	erai_df_conv25_warm = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>25) & (warm_inds)]
	erai_df_conv30_warm = erai_df[(erai_df.lightning>=2)&(erai_df.wind_gust>30) & (warm_inds)]

	#Parameters
	param_list = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
		"cape*s06","cape*ssfc6","td850","td800","td950","wg10"]
	
	ln = LogNorm()
	
	for i in np.arange(0,len(param_list)):
		print(param_list[i])
		for j in np.arange(i+1,len(param_list)):
			fig=plt.figure(figsize=[18,8])
			ax = plt.subplot(121)
			ax.set_xlabel(param_list[i])
			ax.set_ylabel(param_list[j])
			s1 = plt.hist2d(erai_df_cold[~(erai_df_cold[param_list[i]].isna()) & \
				~(erai_df_cold[param_list[j]].isna())][param_list[i]],\
				erai_df_cold[~(erai_df_cold[param_list[i]].isna()) & \
				~(erai_df_cold[param_list[j]].isna())][param_list[j]],\
				cmap=cm.Greys,bins=20,norm=ln)
			label1="All Gusts"
			s2 = plt.scatter(erai_df_conv20_cold[param_list[i]],erai_df_conv20_cold[param_list[j]],color="r",marker="x",s=30)
			label2="Convective Gusts > 20 m/s"
			s3 = plt.scatter(erai_df_conv25_cold[param_list[i]],erai_df_conv25_cold[param_list[j]],color="g",marker="x",s=30)
			label3="Convective Gusts > 25 m/s"
			s4 = plt.scatter(erai_df_conv30_cold[param_list[i]],erai_df_conv30_cold[param_list[j]],color="c",marker="x",s=30)
			label4="Convective Gusts > 30 m/s"
			plt.subplots_adjust(bottom=0.2)
			plt.title("April - Septermber")
			plt.colorbar(s1[3])

			ax = plt.subplot(122)
			plt.hist2d(erai_df_warm[~(erai_df_warm[param_list[i]].isna()) & \
				~(erai_df_warm[param_list[j]].isna())][param_list[i]],\
				erai_df_warm[~(erai_df_warm[param_list[i]].isna()) & \
				~(erai_df_warm[param_list[j]].isna())][param_list[j]],\
				cmap=cm.Greys,bins=20,norm=ln)
			plt.scatter(erai_df_conv20_warm[param_list[i]],erai_df_conv20_warm[param_list[j]],color="r",marker="x",s=30)
			plt.scatter(erai_df_conv25_warm[param_list[i]],erai_df_conv25_warm[param_list[j]],color="g",marker="x",s=30)
			plt.scatter(erai_df_conv30_warm[param_list[i]],erai_df_conv30_warm[param_list[j]],color="c",marker="x",s=30)
			plt.subplots_adjust(bottom=0.2)
			ax.set_xlabel(param_list[i])
			ax.set_ylabel(param_list[j])
			plt.title("October - March")
			plt.legend([s2,s3,s4],[label2,label3,label4],loc="lower left",bbox_to_anchor=[-0.6,-0.25,1,.05],mode="expand",\
				fontsize=10)
			plt.colorbar(s1[3])
		
			plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/scatter/erai_"+\
				param_list[i]+"_"+param_list[j]+"_hist2d_events.png",
				boox_inches="tight")
			plt.close()

def plot_temporal_wind_distribution():
	#Plot the diurnal and seasonal distributions of AWS-defined wind gust events
	#For 2010-2015 produce convective and all-gust distirbutions
	#For 1979-2015 produce a split distribution for all gusts (1979-1997 and 1998-2015)

	#Load resampled (6 hourly) aws data
	aws_6hr = load_aws_all(resample=True)
	aws_6hr = aws_6hr.set_index(["stn_name"],append=True)
	lightning = load_lightning(smoothing=True)
	lightning = lightning.set_index(["date","loc_id"])
	aws_6hr = pd.concat([aws_6hr,lightning.lightning],axis=1).dropna().reset_index()
	aws_6hr["date_lt"] = aws_6hr.level_0 + dt.timedelta(hours = -10.5)
	aws_6hr["hour"] = [x.hour for x in aws_6hr.date_lt]
	aws_6hr["month"] = [x.month for x in aws_6hr.date_lt]
	aws_6hr_warm_inds = np.in1d(aws_6hr.month,np.array([10,11,12,1,2,3]))
	
	#Load unsampled (half-horuly) aws data
	#aws_30min = load_aws_all(resample=False).reset_index()
	aws_30min = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
			"all_wind_gusts_sa_1985_2017.pkl")
	aws_30min["date_lt"] = aws_30min.date + dt.timedelta(hours = -10.5)
	aws_30min = aws_30min.set_index(["date","stn_name"],append=False)
	aws_30min["hour"] = [x.hour for x in aws_30min.date_lt]
	aws_30min["month"] = [x.month for x in aws_30min.date_lt]
	aws_30min_warm_inds = np.in1d(aws_30min.month,np.array([10,11,12,1,2,3]))

	#Get diurnal distribution for 30 min data
	aws_30min_warm_10_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min_warm_inds) & \
		(aws_30min.wind_gust>=10) & (aws_30min.wind_gust<20)])
	aws_30min_warm_20_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min_warm_inds) & \
		(aws_30min.wind_gust>=20)])
	aws_30min_cold_10_cnt,hours = get_diurnal_dist(aws_30min[(~aws_30min_warm_inds) & \
		(aws_30min.wind_gust>=10) & (aws_30min.wind_gust<20)])
	aws_30min_cold_20_cnt,hours = get_diurnal_dist(aws_30min[(~aws_30min_warm_inds) & \
		(aws_30min.wind_gust>=20)])

	#Get diurnal distribution for 6 hourly data
	aws_6hr_warm_10_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20)])
	aws_6hr_warm_20_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20)])
	aws_6hr_cold_10_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20)])
	aws_6hr_cold_20_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20)])

	#Get diurnal distribution for 6 hourly convective wind gusts
	aws_6hr_warm_10_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning >=2)])
	aws_6hr_warm_20_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning >=2)])
	aws_6hr_cold_10_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning >=2)])
	aws_6hr_cold_20_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning >=2)])

	#Get diurnal distribution for 6 hourly non-convective wind gusts
	aws_6hr_warm_10_non_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning <2)])
	aws_6hr_warm_20_non_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning <2)])
	aws_6hr_cold_10_non_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning <2)])
	aws_6hr_cold_20_non_conv_cnt,hours_6hr = get_diurnal_dist(aws_6hr[(~aws_6hr_warm_inds) & \
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning <2)])


	#Get monthly distribution for 30 min data
	aws_30min_10_mth_cnt,months = get_monthly_dist(aws_30min[\
		(aws_30min.wind_gust>=10) & (aws_30min.wind_gust<20)])
	aws_30min_20_mth_cnt,months = get_monthly_dist(aws_30min[\
		(aws_30min.wind_gust>=20)])

	#Get monthly distribution for 6 hourly data
	aws_6hr_10_mth_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20)])
	aws_6hr_20_mth_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=20)])

	#Get monthly distribution for 6 hourly convective wind gusts
	aws_6hr_10_mth_conv_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning >=2)])
	aws_6hr_20_mth_conv_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning >=2)])

	#Get monthly distribution for 6 hourly non-convective wind gusts
	aws_6hr_10_mth_non_conv_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=10) & (aws_6hr.wind_gust<20) & (aws_6hr.lightning <2)])
	aws_6hr_20_mth_non_conv_cnt,months_6hr = get_monthly_dist(aws_6hr[\
		(aws_6hr.wind_gust>=20) & (aws_6hr.lightning <2)])


	#Plot diurnal distributions
	fig=plt.figure(figsize=[12,10])
	plt.subplot(2,2,1)
	plt.plot(hours,aws_30min_warm_10_cnt,"r",label="Warm season wind gust 10-20 m/s")
	plt.plot(hours,aws_30min_warm_20_cnt,"r--",label="Warm season wind gust >20 m/s")
	plt.plot(hours,aws_30min_cold_10_cnt,"b",label="Cold season wind gust 10-20 m/s")
	plt.plot(hours,aws_30min_cold_20_cnt,"b--",label="Cold season wind gust >20 m/s")
	plt.title("AWS 30-minute wind gusts")
	fig.subplots_adjust(bottom=0.25)
	plt.legend(loc="lower left",bbox_to_anchor=[0.7,-1.7,1,.05],mode="expand",\
		fontsize=10)
	plt.ylabel("Normalised counts")
	plt.subplot(2,2,2)
	plt.plot(hours_6hr,aws_6hr_warm_10_cnt,"r")
	plt.plot(hours_6hr,aws_6hr_warm_20_cnt,"r--")
	plt.plot(hours_6hr,aws_6hr_cold_10_cnt,"b")
	plt.plot(hours_6hr,aws_6hr_cold_20_cnt,"b--")
	plt.title("AWS 6-hourly wind gusts")
	plt.subplot(2,2,3)
	plt.plot(hours_6hr,aws_6hr_warm_10_conv_cnt,"r")
	plt.plot(hours_6hr,aws_6hr_warm_20_conv_cnt,"r--")
	plt.plot(hours_6hr,aws_6hr_cold_10_conv_cnt,"b")
	plt.plot(hours_6hr,aws_6hr_cold_20_conv_cnt,"b--")
	plt.title("AWS 6-hourly convective wind gusts")
	plt.xlabel("Hours (local time)")
	plt.ylabel("Normalised counts")
	plt.subplot(2,2,4)
	plt.plot(hours_6hr,aws_6hr_warm_10_non_conv_cnt,"r")
	plt.plot(hours_6hr,aws_6hr_warm_20_non_conv_cnt,"r--")
	plt.plot(hours_6hr,aws_6hr_cold_10_non_conv_cnt,"b")
	plt.plot(hours_6hr,aws_6hr_cold_20_non_conv_cnt,"b--")
	plt.title("AWS 6-hourly non-convective wind gusts")
	plt.xlabel("Hours (local time)")
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
		"diurnal_all_2010_2015.png",bbox_inches="tight")

	#Plot monthly distributions
	fig=plt.figure(figsize=[12,10])
	plt.subplot(2,2,1)
	plt.plot(months,aws_30min_10_mth_cnt,"k",label="Wind gust 10-20 m/s")
	plt.plot(months,aws_30min_20_mth_cnt,"k--",label="Wind gust >20 m/s")
	plt.title("AWS 30-minute wind gusts")
	fig.subplots_adjust(bottom=0.25)
	plt.legend(loc="lower left",bbox_to_anchor=[0.7,-1.7,1,.05],mode="expand",\
		fontsize=10)
	plt.ylabel("Normalised counts")
	plt.subplot(2,2,2)
	plt.plot(months_6hr,aws_6hr_10_mth_cnt,"k")
	plt.plot(months_6hr,aws_6hr_20_mth_cnt,"k--")
	plt.title("AWS 6-hourly wind gusts")
	plt.subplot(2,2,3)
	plt.plot(months_6hr,aws_6hr_10_mth_conv_cnt,"k")
	plt.plot(months_6hr,aws_6hr_20_mth_conv_cnt,"k--")
	plt.title("AWS 6-hourly convective wind gusts")
	plt.ylabel("Normalised counts")
	plt.xlabel("Month")
	plt.subplot(2,2,4)
	plt.plot(months_6hr,aws_6hr_10_mth_non_conv_cnt,"k")
	plt.plot(months_6hr,aws_6hr_20_mth_non_conv_cnt,"k--")
	plt.xlabel("Month")
	plt.title("AWS 6-hourly non-convective wind gusts")
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
		"monthly_all_2010_2015.png",bbox_inches="tight")

def plot_daily_data_monthly_dist():
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
		"all_daily_max_wind_gusts_sa_1979_2017.pkl")
	aws = aws[np.in1d(aws.stn_name,["Adelaide AP","Woomera","Mount Gambier","Parafield",\
		"Edinburgh","Port Augusta Power Station","Port Augusta"])]
	aws_mth_cnt10,aws_mth_mean10,mths10 = get_monthly_dist(aws[(aws.wind_gust>=10)&\
		(aws.wind_gust<20)])
	n10 = aws[(aws.wind_gust>=10)&(aws.wind_gust<20)].shape[0]
	aws_mth_cnt20,aws_mth_mean20,mths20 = get_monthly_dist(aws[(aws.wind_gust>=20)])
	n20 = aws[(aws.wind_gust>=20)].shape[0]
	aws_mth_cnt25,aws_mth_mean25,mths25 = get_monthly_dist(aws[(aws.wind_gust>=25)])
	n25 = aws[(aws.wind_gust>=25)].shape[0]
	aws_mth_cnt30,aws_mth_mean30,mths30 = get_monthly_dist(aws[(aws.wind_gust>=30)])
	n30 = aws[(aws.wind_gust>=30)].shape[0]
	plt.figure()
	plt.plot(mths10,aws_mth_cnt10,"k",label="Wind gust 10-20 m/s: N = "+str(n10))
	plt.plot(mths20,aws_mth_cnt20,"k--",label="Wind gust > 20 m/s: N = "+str(n20))
	plt.plot(mths25,aws_mth_cnt25,"k:",label="Wind gust > 25 m/s: N = "+str(n25))
	plt.plot(mths30,aws_mth_cnt30,"k",label="Wind gust > 30 m/s: N = "+str(n30),marker="^")
	plt.xlabel("Month")
	plt.title("AWS daily maximum wind gusts")
	plt.legend(loc="upper left")
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
		"monthly_7stn_1979_2017.png",bbox_inches="tight")

	plt.figure()
	aws_mth_cnt,aws_mth_mean,mths = get_monthly_dist(aws)
	plt.plot(mths,aws_mth_mean,"k",marker="^")
	plt.xlabel("Month")
	plt.ylabel("Wind gust (m/s)")
	plt.title("Montly mean AWS daily maximum wind gusts")
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
		"monthly_mean_7stn_1979_2017.png",bbox_inches="tight")

	stns = aws.stn_name.unique()
	for i in np.arange(0,len(stns)):
		plt.figure()
		aws_mth_cnt10,aws_mth_mean10,mths = get_monthly_dist(aws[(aws.wind_gust>=10)&\
			(aws.wind_gust<20)&(aws.stn_name==stns[i])])
		n10 = aws[(aws.wind_gust>=10)&(aws.wind_gust<20)&\
			(aws.stn_name==stns[i])].shape[0]
		aws_mth_cnt20,aws_mth_mean20,mths = get_monthly_dist(aws[(aws.wind_gust>=20)&\
			(aws.stn_name==stns[i])])
		n20 = aws[(aws.wind_gust>=20)&(aws.stn_name==stns[i])].shape[0]
		plt.plot(mths,aws_mth_cnt10,"k",label="Wind gust 10-20 m/s: N = "+str(n10))
		plt.plot(mths,aws_mth_cnt20,"k--",label="Wind gust > 20 m/s: N = "+str(n20))
		plt.title(stns[i])
		plt.xlabel("Month")
		plt.legend(loc="upper left")
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"\
			+"monthly_"+stns[i]+"_1979_2017.png")

def get_diurnal_dist(aws):
	hours = np.sort(aws.hour.unique())
	hour_counts = np.empty(len(hours))
	for h in np.arange(0,len(hours)):
		hour_counts[h] = ((aws.hour==hours[h]).sum())/np.float(aws.shape[0])
	return hour_counts,hours

def get_monthly_dist(aws):
	months = np.sort(aws.month.unique())
	month_counts = np.empty(len(months))
	month_mean = np.empty(len(months))
	for m in np.arange(0,len(months)):
		month_counts[m] = ((aws.month==months[m]).sum())/np.float(aws.shape[0])
		month_mean[m] = aws[aws.month==months[m]].wind_gust.mean()
	return month_counts,month_mean,months

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
	#plot_scatter_full("ADversion",location=False)
	#plot_density_contour("Port Augusta","mu_cape","s06","erai",True,threshold=15)
	#plot_scatter("ADversion")
	#plot_distributions("erai","Port Augusta")
	#plot_conv_wind_gust_dist("erai_fc",True,False)
	#plot_wind_gust_prob("mu_cape","s06","erai",20)
	#plot_jdh_dist(seasonal=True)
	plot_daily_data_monthly_dist()

	#plot_netcdf("/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/erai_20160901_20160930.nc"\
		#,"system_black",[dt.datetime(2016,9,27,12,0,0),dt.datetime(2016,9,29,0,0,0)],\
		#"erai")
