import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import netCDF4 as nc
import matplotlib.animation as animation

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

#Load a single netcdf file and plot time mean it time = "mean" or else plot for a single time

	f = nc.Dataset(fname)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)
	if time == "mean":
		values = np.mean(f.variables[param][:],axis=0)
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
		levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
	if param == "ml_cape":
		cmap = cm.YlGnBu
		levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
	if param == "mu_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
	if param == "ml_cin":
		cmap = cm.YlGnBu
		levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
	if param == "s06":
		cmap = cm.Reds
		levels = np.linspace(0,30,7)
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
	if param == "relhum850-500":
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
	if param == "cape*s06":
		cmap = cm.YlGnBu
		levels = np.linspace(0,60000,9)
		cb_lab = ""
	return [cmap,levels,cb_lab]

if __name__ == "__main__":


	path = '/g/data/eg3/ab4502/ExtremeWind/aus/'
	#f = "barra_wrf_20140622_20140623"
	f = "erai_wrf_20140622_20140624"
	model = "erai"
	fname = path+f+".nc"
	param = "srh06"
	region = "sa_small"

	time = [dt.datetime(2014,06,23,0,0,0)]	#Either one datetime object/list or "mean"
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

	plot_netcdf(fname,param,outname,domain,time,model)
	#plot_netcdf_animate(fname,param,outname,domain)
