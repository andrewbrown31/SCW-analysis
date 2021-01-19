from cmip_analysis import get_era5_lsm
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

	log = {"t_totals":False,"eff_sherb":False,"dcp":False}
	titles = {"t_totals":"T-Totals","eff_sherb":"SHERBE","dcp":"DCP"}
	rnge = {"t_totals":[30,45], "eff_sherb":[0,0.4], "dcp":[0,.4]}
	units = {"t_totals":"","eff_sherb":"","dcp":""}

	m = Basemap(llcrnrlon=112, llcrnrlat=-44.5, urcrnrlon=156.25, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[6,8])
	tick_locator = ticker.MaxNLocator(nbins=4)
	n = 10
	r = 12
	cm = plt.get_cmap("YlOrBr")
	cnt=1
	for p in ["t_totals","eff_sherb","dcp"]:
			if qm == True:
				data = get_mean(models, p, hist_y1, hist_y2, hist_y1, hist_y2, experiment="historical")
				cmip = np.median(np.stack([data[i][p].values for i in np.arange(1,13)]), axis=0)
				era5 = data[0][p].values
			else:
				data = [xr.open_dataarray("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+p+"_1979_2005_ensemble_mean_cmip5.nc")]
				#data2 = [xr.open_dataarray("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+p+"_1979_2005_ensemble_mean_cmip6.nc")]
				cmip = data[0].values
				#cmip6 = data2[0].values
				#barpa = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/BARPA__mean_"+p+"_historical_1979_2005.nc")
				#era5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_"+p+"_historical_1979_2005.nc")[p].values
				era5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/era5_"+p+"_6hr_mean.nc")[p]
				era5 = era5.sel({"time":(era5["time.year"]<=hist_y2) & (era5["time.year"]>=hist_y1)}).mean("time").values
				#era5 = np.where(get_era5_lsm()==1, era5, np.nan)
				
			lon = data[0].lon.values
			lat = data[0].lat.values
			x,y = np.meshgrid(lon,lat)

			plt.subplot(3,2,cnt)
			if cnt==1:
				plt.title("ERA5")
			m.drawcoastlines()
			if log[p]:
				c = m.contourf(x, y, era5, norm=mpl.colors.LogNorm(n), cmap=cm)
				cb = plt.colorbar()
			else:
				if rnge[p][0] == None:
					c = m.contourf(x, y, era5, cmap=cm, levels=n, extend="max")
				else:
					c = m.contourf(x, y, era5, cmap=cm, levels=np.linspace(rnge[p][0], rnge[p][1], n), extend="max")
				cb = plt.colorbar(ticks=tick_locator, aspect=r)
			plt.ylabel(titles[p])
			cb.ax.tick_params(labelsize=12)
			cnt=cnt+1

			plt.subplot(3,2,cnt)
			if cnt==2:
				plt.title("CMIP5")
			m.drawcoastlines()
			if log[p]:
				c = m.contourf(x, y, cmip,\
					norm=mpl.colors.LogNorm(), cmap=cm, levels=n)
				cb = plt.colorbar()
			else:
				if rnge[p][0] == None:
					c = m.contourf(x, y, cmip, cmap=cm, levels=n, extend="max")
				else:
					c = m.contourf(x, y, cmip, cmap=cm, levels=np.linspace(rnge[p][0], rnge[p][1], n), extend="max")
				cb = plt.colorbar(ticks=tick_locator, aspect=r)
			cb.ax.tick_params(labelsize=12)
			cnt=cnt+1

			#plt.subplot(8,3,cnt)
			#if cnt==3:
			#	plt.title("CMIP6")
			#m.drawcoastlines()
			#if log[p]:
			#	c = m.contourf(x, y, cmip6,\
			#		norm=mpl.colors.LogNorm(), cmap=cm, levels=n)
			#	cb = plt.colorbar()
			#else:
			#	if rnge2[p][0] == None:
			#		c = m.contourf(x, y, cmip6, cmap=cm, levels=n, extend="max")
			#	else:
			#		c = m.contourf(x, y, cmip6, cmap=cm, levels=np.linspace(rnge2[p][0], rnge2[p][1], n), extend="max")
			#	cb = plt.colorbar(ticks=tick_locator, aspect=r)
			cb.set_label(units[p])
			#cnt=cnt+1

			#plt.subplot(8,4,cnt)
			#xb, yb = np.meshgrid(barpa.lon.values, barpa.lat.values)
			#if p=="lr36":
				#plt.title("BARPA")
			#m.drawcoastlines()
			#if log[p]:
			#	c = m.contourf(xb, yb, barpa[p].values,\
			#		norm=mpl.colors.LogNorm(), cmap=cm, levels=n)
				#cb = plt.colorbar()
			#else:
			#	if (p == "srhe_left"):
			#		c = m.contourf(xb, yb, barpa[p].values, cmap=cm, levels=np.linspace(0,10,10))
			#	else:
			#		c = m.contourf(xb, yb, barpa[p].values, cmap=cm)
				#cb = plt.colorbar(ticks=tick_locator, aspect=r)
			#cb.set_label(units[p])
			#cb.ax.tick_params(labelsize=12)
			#cnt=cnt+1

	plt.subplots_adjust(hspace=0.2, wspace=0.2)
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
		"mean_indices_compare", qm=False)

