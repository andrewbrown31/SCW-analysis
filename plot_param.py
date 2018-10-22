import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm as cm

#Plot parameters from a dataframe with parameter and lat lon data 

def plot_param(df,param,domain,time,model):

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="merc", resolution = "i")
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

#Try plotting profile
#plt.plot(prof.tmpc, prof.hght, 'r-')
#plt.plot(prof.dwpc, prof.hght, 'g-')
##plt.barbs(40*np.ones(len(prof.hght)), prof.hght, prof.u, prof.v)
#plt.xlabel("Temperature [C]")
#plt.ylabel("Height [m above MSL]")
#plt.grid()
#plt.show()
