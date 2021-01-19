from era5_read import get_mask as get_era5_lsm
from cmip_scenario import get_seasonal_sig, plot_mean_spatial_dist
import xarray as xr
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

def get_seasonal_freq(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if p == "logit_aws":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_0.83"+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def plot_mean_spatial_freq_v2(data, p, models, lon, lat, outname, vmin=None, vmax=None):

	#Same as below but for multiple parameters, and only CMIP5

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	fig=plt.figure(figsize=[8,12])
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]
	alph = alph + ["aa","ab","ac","ad","ae","af"]
	alph = [alph[i]+")" for i in np.arange(len(alph))]

	lsm = get_era5_lsm(data[0][0].lon.values, data[0][0].lat.values)

	mod_names = [m[0] for m in models]
	cmip5_mods = np.where(np.in1d(mod_names, ["ERA5","ACCESS-CM2","ACCESS-ESM1-5"], invert=True))
	era5_mods = np.where(np.in1d(mod_names, ["ERA5"]))

	#pos = [1,5,9,13,17,21,2,6,10,14,18,22,3,7,11,15,19,23,4,8,12,16,20,24]
	pos = [1,5,9,13,17,21,25,29,2,6,10,14,18,22,26,30,3,7,11,15,19,23,27,31,4,8,12,16,20,24,28,32]

	cnt=1
	x,y = np.meshgrid(lon,lat)
	levels = [0,1,5,10,15,20,30,40,60,80]
	for season in ["DJF","MAM","JJA","SON"]:
			cmip = np.median(np.stack([data[0][i][season].values for i in cmip5_mods[0]]), axis=0)
			era5 = data[0][era5_mods[0][0]][season].values

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("ERA5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.title(season)
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			cmip = np.median(np.stack([data[1][i][season].values for i in cmip5_mods[0]]), axis=0)
			era5 = data[1][era5_mods[0][0]][season].values

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("ERA5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			cmip = np.median(np.stack([data[2][i][season].values for i in cmip5_mods[0]]), axis=0)
			era5 = data[2][era5_mods[0][0]][season].values

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("ERA5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			cmip = np.median(np.stack([data[3][i][season].values for i in cmip5_mods[0]]), axis=0)
			era5 = data[3][era5_mods[0][0]][season].values

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("ERA5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

			plt.subplot(8,4,pos[cnt-1])
			if season=="DJF":
				plt.ylabel("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,len(levels))), extend="max")
			plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
			cnt=cnt+1

	cax = plt.axes([0.25,0.25,0.50,0.02])
	c=plt.colorbar(p, cax=cax, orientation="horizontal", extend="max" )
	c.set_label("Seasonal frequency (days)")
	plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.1,left=0.1)
	fig.text(0.05,0.7,"T-Totals",ha="center",size=16,va="center",rotation=90)
	fig.text(0.05,0.55,"SHERBE",ha="center",size=16, va="center",rotation=90)
	fig.text(0.05,0.4,"DCP",ha="center",size=16, va="center",rotation=90)        
	fig.text(0.05,0.85,"BDSD",ha="center",size=16, va="center",rotation=90)        
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png", bbox_inches="tight")

def plot_mean_spatial_freq(data, p, models, lon, lat, outname, vmin=None, vmax=None):

	#Taken from cmip_analysis. However, this function accepts the dataframe created by save_mean(), which 
	# consists of 2d variables representing the mean for each month (and the total mean). This mean has been 
	# generated from 6-hourly data, quantile matched to ERA5.

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[6,8])

	lsm = get_era5_lsm(data[0][0].lon.values, data[0][0].lat.values)

	mod_names = [m[0] for m in models]
	cmip5_mods = np.where(np.in1d(mod_names, ["ERA5","ACCESS-CM2","ACCESS-ESM1-5"], invert=True))
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
			p=m.contourf(x, y, np.where(lsm==1, era5, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			plt.ylabel(season)
			cnt=cnt+1

			plt.subplot(4,3,cnt)
			if season=="DJF":
				plt.title("CMIP5")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			cnt=cnt+1

			plt.subplot(4,3,cnt)
			if season=="DJF":
				plt.title("CMIP6")
			m.drawcoastlines()
			p=m.contourf(x, y, np.where(lsm==1, cmip6, np.nan), levels = levels,\
                                colors=plt.get_cmap("Reds")(np.linspace(0,1,11)), extend="max")
			cnt=cnt+1

	cax = plt.axes([0.2,0.25,0.6,0.02])
	c=plt.colorbar(p, cax=cax, orientation="horizontal", extend="max" )
	c.set_label("Seasonal frequency")
	plt.subplots_adjust(top=0.95, bottom=0.3, wspace=0.1)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+outname+".png", bbox_inches="tight")

if __name__ == "__main__":

	p = "cs6"
	p_list = ["logit_aws","t_totals","eff_sherb","dcp"]
	hist_y1 = 1979
	hist_y2 = 2005
	multiple_vars = True
    
	if multiple_vars:
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
		hist = []
		for p in p_list:
			hist.append(get_seasonal_freq(models, p, hist_y1, hist_y2, hist_y1, hist_y2,\
			    experiment="historical"))
		plot_mean_spatial_freq_v2(hist, p, models,\
			hist[0][0].lon.values, hist[0][0].lat.values,\
			"seasonal_freq_multiple_p", vmin=0, vmax=90)

	else:
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
		hist = get_seasonal_freq(models, p, hist_y1, hist_y2, hist_y1, hist_y2,\
			experiment="historical")
		plot_mean_spatial_freq(hist, p, models,\
			hist[0].lon.values, hist[0].lat.values,\
			p+"_seasonal_freq", vmin=0, vmax=50)

