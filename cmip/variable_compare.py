from matplotlib import ticker
import matplotlib as mpl
from cmip_scenario import get_mean, get_seasonal_sig, plot_mean_spatial_dist
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_mean(models, hist_y1, hist_y2, outname, qm=True):

	#Taken from cmip_analysis. However, this function accepts the dataframe created by save_mean(), which 
	# consists of 2d variables representing the mean for each month (and the total mean). This mean has been 
	# generated from 6-hourly data, quantile matched to ERA5.

	log = {"lr36":False,"mhgt":False,"ml_el":False,"qmean01":False,"srhe_left":False,\
		    "Umean06":False}
	titles = {"lr36":"LR36","mhgt":"MHGT","ml_el":"ML-EL","qmean01":"Qmean01","srhe_left":"SRHE",\
		    "Umean06":"Umean06"}
	rnge = {"lr36":[5.8,8],"mhgt":[1600,4800],"ml_el":[1500,7500],"qmean01": [5.5,13],\
		    "srhe_left":[0.1,15],"Umean06":[5,12.5]}
	units = {"lr36":"$DegC.km^{-1}$", "mhgt":"$m$", "ml_el":"$m$", "qmean01":"$g.kg^{-1}$", \
		    "srhe_left":"$m^2$.$s^{-2}$", "Umean06":"$m.s^{-1}$"}

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[6,10])
	tick_locator = ticker.MaxNLocator(nbins=4)
	n = 6
	r = 12
	cm = plt.get_cmap("YlOrBr")

	cnt=1
	for p in ["lr36", "mhgt", "ml_el", "qmean01", "srhe_left", "Umean06"]:
			if qm == True:
				data = get_mean(models, p, hist_y1, hist_y2, hist_y1, hist_y2, experiment="historical")
				cmip = np.median(np.stack([data[i][p].values for i in np.arange(1,13)]), axis=0)
				era5 = data[0][p].values
			else:
				data = [xr.open_dataarray("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+p+"_1979_2005_ensemble_mean.nc")]
				cmip = data[0].values
				era5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_"+p+"_historical_1979_2005.nc")[p].values
				
			lon = data[0].lon.values
			lat = data[0].lat.values
			x,y = np.meshgrid(lon,lat)

			plt.subplot(6,2,cnt)
			if p=="lr36":
				plt.title("ERA5")
			m.drawcoastlines()
			if log[p]:
				c = m.contourf(x, y, era5, norm=mpl.colors.LogNorm(), cmap=cm)
				cb = plt.colorbar()
			else:
				c = m.contourf(x, y, era5, cmap=cm, levels=n)
				cb = plt.colorbar(ticks=tick_locator, aspect=r)
			plt.ylabel(titles[p])
			cnt=cnt+1

			plt.subplot(6,2,cnt)
			if p=="lr36":
				plt.title("CMIP5")
			m.drawcoastlines()
			if log[p]:
				c = m.contourf(x, y, cmip,\
					norm=mpl.colors.LogNorm(), cmap=cm, levels=n)
				cb = plt.colorbar()
			else:
				c = m.contourf(x, y, cmip, cmap=cm)
				cb = plt.colorbar(ticks=tick_locator, aspect=r)
			cb.set_label(units[p])

			cnt=cnt+1

	plt.subplots_adjust(hspace=0.3)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png", bbox_inches="tight")

if __name__ == "__main__":

	models = [ ["ERA5",""] ,\
			["ACCESS1-3","r1i1p1",5,""] ,\
			["ACCESS1-0","r1i1p1",5,""] , \
			["BNU-ESM","r1i1p1",5,""] , \
			["CNRM-CM5","r1i1p1",5,""] ,\
			["GFDL-CM3","r1i1p1",5,""] , \
			["GFDL-ESM2G","r1i1p1",5,""] , \
			["GFDL-ESM2M","r1i1p1",5,""] , \
			["IPSL-CM5A-LR","r1i1p1",5,""] ,\
			["IPSL-CM5A-MR","r1i1p1",5,""] , \
			["MIROC5","r1i1p1",5,""] ,\
			["MRI-CGCM3","r1i1p1",5,""], \
			["bcc-csm1-1","r1i1p1",5,""], \
                        ]
	hist_y1 = 1979
	hist_y2 = 2005
    

	plot_mean(models, hist_y1, hist_y2,\
		"mean_variable_compare", qm=False)

