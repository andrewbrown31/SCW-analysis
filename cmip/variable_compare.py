from cmip_analysis import get_era5_lsm
from matplotlib import ticker
import matplotlib as mpl
from cmip_scenario import get_mean, get_seasonal_sig, plot_mean_spatial_dist
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_mean(models, hist_y1, hist_y2, outname, variables, qm=True):

	#Taken from cmip_analysis. However, this function accepts the dataframe created by save_mean(), which 
	# consists of 2d variables representing the mean for each month (and the total mean). This mean has been 
	# generated from 6-hourly data, quantile matched to ERA5.

	log = {"lr36":False,"mhgt":False,"ml_el":False,"qmean01":False,"srhe_left":False,\
		    "Umean06":False, "dcape":False, "mu_cape":False, "ml_cape":False, "s06":False, "srh01_left":False,\
		    "dcp":False,"scp_fixed":False,"mucape*s06":False,"ebwd":False,"lr03":False,"lr700_500":False,"ta850":False,\
		    "ta500":False, "dp850":False, "Umean800_600":False, "lr13":False, "rhmin13":False, "q_melting":False, "eff_lcl":False,\
		    "eff_sherb":False,"t_totals":False}
	titles = {"lr36":"LR36","mhgt":"MHGT","ml_el":"ML-EL","qmean01":"Qmean01","srhe_left":"SRHE",\
		    "Umean06":"Umean06", "dcape":"DCAPE", "mu_cape":"MU-CAPE", "ml_cape":"ML-CAPE", "s06":"S06", "srh01_left":"SRH01",\
		    "dcp":"DCP","scp_fixed":"SCP","mucape*s06":"MUCS6","ebwd":"EBWD", "lr03":"LR03","lr700_500":"LR75","dp850":"DP850",\
		    "ta850":"T850","ta500":"T500", "Umean800_600":"Umean800-600", "lr13":"LR13", "rhmin13":"RHMin13", "q_melting":"Q-Melting",\
		    "eff_lcl":"Eff-LCL","eff_sherb":"SHERBE","t_totals":"T-Totals"}
	rnge = {"lr03":[4,8],"lr700_500":[5,8],"mhgt":[None,None],"ml_el":[None,None],"qmean01":[None,None],"srhe_left":[0,15],\
		    "Umean06":[3,15], "dcape":[100,1000], "mu_cape":[0,1500], "ml_cape":[0,1200], \
		    "s06":[6,20],"srh01_left":[None,None],"ebwd":[0,10],"dp850":[-6,10], "ta850":[0,25], "ta500":[-20,-5],\
		    "dcp":[0,0.4],"scp_fixed":[0,0.5],"mucape*s06":[0,50000], "Umean800_600":[3,15], "lr13":[3,8], "rhmin13":[0,80], "q_melting":[1,4],\
		    "eff_lcl":[0,1000],"eff_sherb":[0,0.4],"t_totals":[30,50]}
	rnge2 = {"lr36":[None,None],"mhgt":[None,None],"ml_el":[None,None],"qmean01":[None,None],"srhe_left":[None,None],\
		    "Umean06":[None,None], "dcape":[None,None], "mu_cape":[None,None], "ml_cape":[0,1200], \
            "s06":[None,None],"srh01_left":[None,None],\
		    "dcp":[0,0.32],"scp_fixed":[0,0.05],"mucape*s06":[0,15000], "Umean800_600":[None,None], "lr13":[None,None], "rhmin13":[None,None], "q_melting":[None,None], "eff_lcl":[None,None]}
	units = {"lr03":"deg km$^{-1}$","mu_cape":"J kg$^{-1}$",
		    "ebwd":"m s$^{-1}$","Umean06":"m s$^{-1}$","s06":"m s$^{-1}$",\
		    "dp850":"deg C","ta500":"deg C","ta850":"deg C","srhe_left":"m$^{-2}$ s$^{-2}$",\
		    "lr700_500":"deg km$^{-1}$", "dcape":"J kg$^{-1}$", "Umean800_600":"m s$^{-1}$", "lr13":"deg km$^{-1}$", "rhmin13":"%", "q_melting":"g kg$^{-1}$", "eff_lcl":"m",\
		    "eff_sherb":"","t_totals":"","dcp":""}

	m = Basemap(llcrnrlon=112, llcrnrlat=-44.5, urcrnrlon=156.25, \
		urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[6,14])
	tick_locator = ticker.MaxNLocator(nbins=4)
	n = 10
	r = 12
	cm = plt.get_cmap("YlOrBr")
	cnt=1
	for p in variables:
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

			plt.subplot(len(variables),2,cnt)
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

			plt.subplot(len(variables),2,cnt)
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
		"mean_index_variable_compare",["eff_sherb","t_totals","dcp"], qm=False)
	plot_mean(models, hist_y1, hist_y2,\
		"mean_logit_variable_compare",["Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"], qm=False)
	plot_mean(models, hist_y1, hist_y2,\
		"mean_variable_compare",["mu_cape","dcape", "ebwd", "lr03", "lr700_500", "dp850", "ta850", "ta500", "Umean06", "s06"], qm=False)

