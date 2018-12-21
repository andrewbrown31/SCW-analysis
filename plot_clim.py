import netCDF4 as nc
from plot_param import *

def clim(domain,model,year_range):
	
	print("INFO: Loading data for "+model+" on domain "+domain+" between "+str(year_range[0])+\
		" and "+str(year_range[1]))
	f = nc.MFDataset("/g/data/eg3/ab4502/ExtremeWind/"+domain+"/"+model+"/"+model+"_*.nc")
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	vars = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
	vars = vars[~(vars=="time") & ~(vars=="lat") & ~(vars=="lon")]
	time = nc.num2date(f.variables["time"][:],f.variables["time"].units)
	years = np.array([t.year for t in time])
	months = np.array([t.month for t in time])
	y1 = str(years.min())
	y2 = str(years.max())
	time_inds = (years>=year_range[0]) & (years<=year_range[1])
	warm_inds = np.array([m in np.array([10,11,12,1,2,3]) for m in months])
	warm_time_inds = (time_inds) & (warm_inds)
	cold_time_inds = (time_inds) & (~warm_inds)

	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "l")

	print("INFO: Plotting...")
	for var in vars:
		print(var)
		data = data = f.variables[var][:]
		mean = np.nanmean(data[time_inds,:,:],axis=0)
		mean_warm = np.nanmean(data[(time_inds) & (warm_inds),:,:],axis=0)
		mean_cold = np.nanmean(data[(time_inds) & (~warm_inds),:,:],axis=0)
		plot_clim(f,m,lat,lon,mean,var,model,domain,year_range,seasonal="full")
		plot_clim(f,m,lat,lon,mean_warm,var,model,domain,year_range,seasonal="warm")
		plot_clim(f,m,lat,lon,mean_cold,var,model,domain,year_range,seasonal="cold")
		

def plot_clim(f,m,lat,lon,mean,var,model,domain,year_range,seasonal=""):

	plt.figure()
	cmap,levels,cb_lab,range,log_plot = contour_properties(var)	
	m.drawcoastlines()
	m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),5),\
			labels=[True,False,True,False])
	x,y = np.meshgrid(lon,lat)
	m.contourf(x,y,mean,latlon=True,cmap=cmap,extend="both")
	cb=plt.colorbar()
	cb.set_label(f.variables[var].units)
	if seasonal == "warm":
		plt.title("October - March")
	elif seasonal == "cold":
		plt.title("April - September")
	elif seasonal == "full":
		plt.title("All Seasons")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/clim/"+model+"_"+var+\
	#	"_"+seasonal+"_"+str(year_range[0])+"_"+str(year_range[1])+".png",\
	#	boox_inches="tight")
	plt.show()

if __name__ == "__main__":
	clim("sa_small","erai",[1979,2017])
	clim("sa_small","erai_fc",[1979,2017])
	clim("sa_small","erai",[2010,2015])
	clim("sa_small","erai_fc",[2010,2015])
	clim("sa_small","barra",[2010,2015])
