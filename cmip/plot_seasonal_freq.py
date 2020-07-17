from era5_read import get_mask as get_era5_lsm
from cmip_scenario import get_seasonal_freq, get_seasonal_sig, plot_mean_spatial_dist
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_spatial_freq(data, p, models, lon, lat, outname, vmin=None, vmax=None):

	#Taken from cmip_analysis. However, this function accepts the dataframe created by save_mean(), which 
	# consists of 2d variables representing the mean for each month (and the total mean). This mean has been 
	# generated from 6-hourly data, quantile matched to ERA5.

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[6,8])

	lsm = get_era5_lsm(data[0].lon.values, data[0].lat.values)

	mod_names = [m[0] for m in models]
	cmip5_mods = np.where(np.in1d(mod_names, ["ERA5","ACCESS-CM2","ACCESS-ESM1-5"], invert=True))
	cmip6_mods = np.where(np.in1d(mod_names, ["ACCESS-CM2","ACCESS-ESM1-5"]))
	era5_mods = np.where(np.in1d(mod_names, ["ERA5"]))

	cnt=1
	x,y = np.meshgrid(lon,lat)
	for season in ["DJF","MAM","JJA","SON"]:
			cmip = np.median(np.stack([data[i][season].values for i in cmip5_mods[0]]), axis=0)
			cmip6 = np.median(np.stack([data[i][season].values for i in cmip6_mods[0]]), axis=0)
			era5 = data[era5_mods[0][0]][season].values

			plt.subplot(4,3,cnt)
			if season=="DJF":
				plt.title("ERA5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = np.linspace(0,vmax,11),\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			plt.ylabel(season)
			cnt=cnt+1

			plt.subplot(4,3,cnt)
			if season=="DJF":
				plt.title("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = np.linspace(0,vmax,11),\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			cnt=cnt+1

			plt.subplot(4,3,cnt)
			if season=="DJF":
				plt.title("CMIP6")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip6, np.nan), levels = np.linspace(0,vmax,11),\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			cnt=cnt+1

	cax = plt.axes([0.2,0.25,0.6,0.02])
	c=plt.colorbar(p, cax=cax, orientation="horizontal", extend="max" )
	c.set_label("Seasonal frequency")
	plt.subplots_adjust(top=0.95, bottom=0.3, wspace=0.1)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png", bbox_inches="tight")

if __name__ == "__main__":

	models = [ ["ERA5",""] ,\
			["ACCESS1-3","r1i1p1",5,""] ,\
			["ACCESS1-0","r1i1p1",5,""] , \
			["ACCESS-ESM1-5","r1i1p1f1",6,""] , \
			["ACCESS-CM2","r1i1p1f1",6,""] , \
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
	p = "logit_aws"
	hist_y1 = 1979
	hist_y2 = 2005
    

	hist = get_seasonal_freq(models, p, hist_y1, hist_y2, hist_y1, hist_y2,\
		experiment="historical")
	plot_mean_spatial_freq(hist, p, models,\
		hist[0].lon.values, hist[0].lat.values,\
		p+"_seasonal_freq", vmin=0, vmax=50)

