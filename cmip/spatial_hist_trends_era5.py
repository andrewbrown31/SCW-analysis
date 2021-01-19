from mpl_toolkits.basemap import Basemap
from scipy.stats import ttest_ind
from dask.diagnostics import ProgressBar
from mpl_toolkits.axes_grid1 import inset_locator
import seaborn as sns
import pandas as pd
import numpy as np
#import geopandas
#from rasterio import features
#from affine import Affine
import matplotlib.pyplot as plt
import xarray as xr
#from percent_mean_change import transform_from_latlon, rasterize
import netCDF4 as nc

#As in diurnal_compare.py but for different diagnostics

def get_lsm():
        #Load the ERA-Interim land-sea mask (land = 1)
        lsm_file = nc.Dataset("/g/data/ub4/era5/netcdf/static_era5.nc")
        lsm = np.squeeze(lsm_file.variables["lsm"][:])
        lsm_lon = np.squeeze(lsm_file.variables["longitude"][:])
        lsm_lat = np.squeeze(lsm_file.variables["latitude"][:])
        lsm_file.close()
        return [lsm,lsm_lon,lsm_lat]

def get_mask(lon,lat,thresh=0.5):
        #Return lsm for a given domain (with lats=lat and lons=lon)
        lsm,nat_lon,nat_lat = get_lsm()
        lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
        lat_ind = np.where((nat_lat >= lat[-1]) & (nat_lat <= lat[0]))[0]
        lsm_domain = lsm[(lat_ind[0]):(lat_ind[-1]+1),(lon_ind[0]):(lon_ind[-1]+1)]
        lsm_domain = np.where(lsm_domain > thresh, 1, 0)

        return lsm_domain

def get_era5_lsm(regrid=False):

        #load the era5 lsm, subset to Australia, and coarsen as in regrid_era5(). Consider the coarse
        # lsm to be land if greater than 50% of the points are land


        f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_19790101_19790131.nc")
        lsm = get_mask(f.lon.values,f.lat.values)
        if regrid:
                lsm_coarse = xr.DataArray(lsm, dims=("lat","lon")).coarsen({"lat":6,"lon":6},\
                            boundary="trim", side="left").sum()
                return xr.where(lsm_coarse>=18, 1, 0).values
        else:
                return lsm

def load_logit(model, ensemble, p, rcp=False):

	if rcp:
		try:
			model_6hr = xr.open_mfdataset(["/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_"+p+"_historical_1979_2005.nc",\
			    "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_"+p+"_rcp85_2006_2018.nc"])[p]
		except:
			model_6hr = xr.open_mfdataset(["/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_historical_"+p+"_qm_lsm_1979_2005.nc",\
			    "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_rcp85_"+p+"_qm_lsm_2006_2018.nc"])[p]
	else:
		try:
			model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_"+p+"_historical_1979_2005.nc")[p]
		except:
			model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    model+"_"+ensemble+"_historical_"+p+"_qm_lsm_1979_2005.nc")[p]

	return model_6hr

def load_resampled_era5(p, thresh, mean=False):
	if p == "dcp":
		var = "dcp"
	elif p == "scp":
		var = "scp_fixed"
	elif p == "cs6":
		var = "mucape*s06"
	elif p == "mu_cape":
		var = "mu_cape"
	else:
		var = p
	if mean:
		da = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/era5_"+var+"_6hr_mean.nc")[var]
	else:
		da = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/era5_"+var+"_6hr_"+str(thresh[p])+"*daily.nc", concat_dim="time", combine="by_coords")[var]
	return da

if __name__ == "__main__":

	lsm = get_era5_lsm()
	lsm = np.where(lsm==0, 1, 1)
	path = "/g/data/eg3/ab4502/ExtremeWind/trends/"
	compute = False
	cnt=1
	alph = ["","a","b","c"]
	thresh = {"logit_aws":0.83, "dcp":0.15, "scp":0.04, "cs6":30768, "mu_cape":230, "eff_sherb":0.47, "t_totals":48.1, "logit":"is_conv_aws"}

	seasons = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
	y1=1979; y2=1998; y3=1999; y4=2018

	freq = True
	plot_final = True

	for p in ["t_totals","eff_sherb","dcp","logit"]:

		if freq:
			era5 = load_resampled_era5(p, thresh).persist()
		else:
			era5 = load_resampled_era5(p, thresh, mean=True).persist()

		for s in ["DJF","MAM","JJA","SON"]:

			#Resample monthly data to annual occurence time series, for two time periods defined by y1-y4.
			t1 = (era5.sel({"time":\
			    (np.in1d(era5["time.month"], seasons[s])) &\
			    (era5["time.year"] >= y1) &\
			    (era5["time.year"] <= y2) })).resample({"time":"1Y"}).sum("time")
			t2 = (era5.sel({"time":\
			    (np.in1d(era5["time.month"], seasons[s])) &\
			    (era5["time.year"] >= y3) &\
			    (era5["time.year"] <= y4) })).resample({"time":"1Y"}).sum("time")
			if freq:
				era5_change = np.where(lsm==1, (t2.sum("time") / ((y2+1) - y1)).values - (t1.sum("time") / ((y4+1) - y3)).values, 0)
			else:
				era5_change = np.where(lsm==1, (t2.mean("time")).values - (t1.mean("time")).values, 0)
			temp, sig = ttest_ind(t1.values,t2.values,axis=0)
			era5_sig = np.where(lsm==1, sig, 1)
			if freq:
				out_name = "/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_era5_"+s+"_"+p+".nc"
			else:
				out_name = "/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+s+"_"+p+".nc"
			xr.Dataset(data_vars={\
                                        "era5_trend":(("lat","lon"), era5_change),\
                                        "era5_sig":(("lat","lon"), era5_sig)},\
                                        coords={"lat":era5.lat.values, "lon":era5.lon.values}).to_netcdf(out_name)

			

	#Now plot
	if plot_final:
		#m = Basemap(llcrnrlon=130, llcrnrlat=-45, urcrnrlon=155, \
			#urcrnrlat=-20,projection="cyl",resolution="i")
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
			urcrnrlat=-10,projection="cyl")
		fig=plt.figure(figsize=[8,8])
		#fig=plt.figure(figsize=[10,12])
		a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
		pos = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
		#pos = [1,5,9,2,6,10,3,7,11,4,8,12]
		cnt=1
		vmin = -9; vmax = 9
		levs=10
		cmip_scale = 1
		for season in ["DJF","MAM","JJA","SON"]:
				plt.subplot(4,4,pos[cnt-1])
				plt.title(season)
				f4 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_era5_"+season+"_logit.nc")
				x,y = np.meshgrid(f4.lon.values,f4.lat.values)
				if season=="DJF":
					plt.ylabel("BDSD")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f4["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,levs),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f4["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(4,4,pos[cnt-1])
				f1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_era5_"+season+"_t_totals.nc")
				if season=="DJF":
					plt.ylabel("T-Totals")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f1["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,levs),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f1["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(4,4,pos[cnt-1])
				f2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_era5_"+season+"_eff_sherb.nc")
				if season=="DJF":
					plt.ylabel("SHERBE")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f2["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,levs),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f2["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(4,4,pos[cnt-1])
				f3 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_era5_"+season+"_dcp.nc")
				if season=="DJF":
					plt.ylabel("DCP")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f3["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,levs),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f3["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1


		cax = plt.axes([0.2,0.25,0.6,0.02])
		c=plt.colorbar(p, cax=cax, orientation="horizontal", extend="max" )
		c.set_label("Seasonal frequency (days)")
		plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.1,left=0.1, hspace=0.3)

		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_trend_era5.png",bbox_inches="tight")
