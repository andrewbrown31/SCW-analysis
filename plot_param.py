import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import netCDF4 as nc


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

def plot_netcdf(fname,param,outname):

#Load a single netcdf file and plot the time mean

	f = nc.Dataset(fname)
	values = np.mean(f.variables[param][:],axis=0)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]

	if param == "mu_cape":
		cmap = cm.YlGnBu
		levels = np.linspace(0,1000,11)
		cb_lab = "J/Kg"
	if param == "s06":
		cmap = cm.Reds
		levels = np.linspace(0,30,7)
		cb_lab = "m/s"

	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "l")
	m.drawcoastlines()
	m.drawmeridians(np.arange(lon.min().round(),lon.max().round(),5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(lat.min().round(),lat.max().round(),5),\
			labels=[True,False,True,False])
	m.contourf(x,y,values,latlon=True,levels=levels,cmap=cmap,extend="both")
	cb = plt.colorbar()
	cb.set_label(cb_lab)
	plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/mean/"+outname,bbox_inches="tight")
	plt.close()

if __name__ == "__main__":


	fname = '/g/data/eg3/ab4502/ExtremeWind/aus/barra_20121101_20121201.nc'
	param = "mu_cape"
	outname = "barra_wrf3d_mu_cape_20121101_20121201.png"
	plot_netcdf(fname,param,outname)
