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
	path = "/g/data/eg3/ab4502/ExtremeWind/trends/"
	compute = False
	cnt=1
	alph = ["","a","b","c"]
	thresh = {"logit_aws":0.72, "dcp":0.128, "scp":0.04, "cs6":30768, "mu_cape":230}

	seasons = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
	y1=1979; y2=1998; y3=1999; y4=2018

	freq = True
	plot_final = True

	if compute:
		for p in ["dcp","cs6","mu_cape"]:

			ProgressBar().register()
			if freq:
				era5 = load_resampled_era5(p, thresh).persist()
				gfdl_cm3 = load_logit("GFDL-CM3","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				gfdl2g = load_logit("GFDL-ESM2G","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				gfdl2m = load_logit("GFDL-ESM2M","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				ipsll = load_logit("IPSL-CM5A-LR","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				ipslm = load_logit("IPSL-CM5A-MR","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				miroc = load_logit("MIROC5","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
				mri = load_logit("MRI-CGCM3","r1i1p1",p,rcp=True).persist().resample({"time":"1D"}).max("time")
			else:
				era5 = load_resampled_era5(p, thresh, mean=True).persist()
				gfdl_cm3 = load_logit("GFDL-CM3","r1i1p1",p,rcp=True).persist()
				gfdl2g = load_logit("GFDL-ESM2G","r1i1p1",p,rcp=True).persist()
				gfdl2m = load_logit("GFDL-ESM2M","r1i1p1",p,rcp=True).persist()
				ipsll = load_logit("IPSL-CM5A-LR","r1i1p1",p,rcp=True).persist()
				ipslm = load_logit("IPSL-CM5A-MR","r1i1p1",p,rcp=True).persist()
				miroc = load_logit("MIROC5","r1i1p1",p,rcp=True).persist()
				mri = load_logit("MRI-CGCM3","r1i1p1",p,rcp=True).persist()

			x = [gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri]
			n = ["GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M","IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MRI-CGCM3"]

			for s in ["DJF","MAM","JJA","SON"]:
				if freq:
					#Get the seasonal frequency for period 1
					t1 = [ ((mod.sel({"time":\
					    (np.in1d(mod["time.month"], seasons[s])) &\
					    (mod["time.year"] >= y1) &\
					    (mod["time.year"] <= y2) })  >= thresh[p]).sum("time") / ((y2+1) - y1)).values for mod in x]
					#Get the seasonal frequency for period 2
					t2 = [ ((mod.sel({"time":\
					    (np.in1d(mod["time.month"], seasons[s])) &\
					    (mod["time.year"] >= y3) &\
					    (mod["time.year"] <= y4) })  >= thresh[p]).sum("time") / ((y4+1) - y3)).values for mod in x]
				else:
					#Get the seasonal frequency for period 1
					t1 = [ ((mod.sel({"time":\
					    (np.in1d(mod["time.month"], seasons[s])) &\
					    (mod["time.year"] >= y1) &\
					    (mod["time.year"] <= y2) })).mean("time")).values for mod in x]
					#Get the seasonal frequency for period 2
					t2 = [ ((mod.sel({"time":\
					    (np.in1d(mod["time.month"], seasons[s])) &\
					    (mod["time.year"] >= y3) &\
					    (mod["time.year"] <= y4) })).mean("time")).values for mod in x]
				#Save the median change, as well as the significance
				change = np.median(np.stack([t2[j] - t1[j] for j in np.arange(len(x))]), axis=0)
				for j in np.arange(len(x)):
					plt.figure()
					xr.Dataset(data_vars={"cmip_trend":\
					    (("lat","lon"), t2[j] - t1[j])}, \
					    coords={"lat":x[0].lat.values, "lon":x[0].lon.values})["cmip_trend"].plot()
					plt.title(n[j])
					if freq:
						plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_trend_rcp85_"+p+"_"+n[j]+"_"+s+".png")
					else:
						plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_mean_trend_rcp85_"+p+"_"+n[j]+"_"+s+".png")
					plt.close()
					
				mean1 = np.mean(np.stack(t1), axis=0)
				spatial_diff_pos = (np.stack([t2[j] - t1[j] for j in np.arange(len(x))]) > 0).sum(axis=0) 
				spatial_diff_neg = (np.stack([t2[j] -  t1[j] for j in np.arange(len(x))]) <= 0).sum(axis=0) 
				if freq:
					cmip_sig = np.where( ( ( (change > 0) & (spatial_diff_pos >= 6) ) |\
					 ( (change <= 0) & (spatial_diff_neg >= 6) ) ) & (mean1 >= 1), 1, 0) 
				else:
					cmip_sig = np.where( ( ( (change > 0) & (spatial_diff_pos >= 6) ) |\
					 ( (change <= 0) & (spatial_diff_neg >= 6) ) ), 1, 0) 

				#Now do ERA5 separately, get significence via ttest or bootstrap
				t1 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y1) &\
				    (era5["time.year"] <= y2) }))
				t2 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y3) &\
				    (era5["time.year"] <= y4) }))
				if freq:
					era5_change = np.where(lsm==1, (t2.sum("time") / ((y2+1) - y1)).values - (t1.sum("time") / ((y4+1) - y3)).values, 0)
				else:
					era5_change = np.where(lsm==1, (t2.mean("time")).values - (t1.mean("time")).values, 0)
				temp, sig = ttest_ind(t1.values,t2.values,equal_var=False,axis=0)
				era5_sig = np.where(lsm==1, sig, 1)
	    
				if freq:
					out_name = "/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_rcp85_"+s+"_"+p+".nc"
				else:
					out_name = "/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_rcp85_"+s+"_"+p+".nc"
				xr.Dataset(data_vars={"cmip_trend":\
					(("lat","lon"), change), "era5_trend":(("lat","lon"), era5_change),\
					"cmip_sig":(("lat","lon"), cmip_sig),"era5_sig":(("lat","lon"), era5_sig)},\
					coords={"lat":x[0].lat.values, "lon":x[0].lon.values}).to_netcdf(out_name)

			del x, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, era5

	#Now plot
	if plot_final:
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
			urcrnrlat=-10,projection="cyl")
		fig=plt.figure(figsize=[8,12])
		a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
		pos = [1,5,9,13,17,21,2,6,10,14,18,22,3,7,11,15,19,23,4,8,12,16,20,24]
		cnt=1
		vmin = -10; vmax = 10
		cmip_scale = 1
		for season in ["DJF","MAM","JJA","SON"]:
				plt.subplot(6,4,pos[cnt-1])
				f1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_rcp85_"+season+"_dcp.nc")
				x,y = np.meshgrid(f1.lon.values,f1.lat.values)
				if season=="DJF":
					plt.ylabel("ERA5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f1["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(xr.where(f1["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.title(season)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(6,4,pos[cnt-1])
				if season=="DJF":
					plt.ylabel("CMIP5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f1["cmip_trend"]*cmip_scale, np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(f1["cmip_sig"], colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(6,4,pos[cnt-1])
				f2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_rcp85_"+season+"_cs6.nc")
				if season=="DJF":
					plt.ylabel("ERA5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f2["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(xr.where(f2["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(6,4,pos[cnt-1])
				if season=="DJF":
					plt.ylabel("CMIP5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f2["cmip_trend"]*cmip_scale, np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(f2["cmip_sig"], colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(6,4,pos[cnt-1])
				f3 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_trend_rcp85_"+season+"_mu_cape.nc")
				if season=="DJF":
					plt.ylabel("ERA5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f3["era5_trend"], np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(xr.where(f3["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

				plt.subplot(6,4,pos[cnt-1])
				if season=="DJF":
					plt.ylabel("CMIP5")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f3["cmip_trend"]*cmip_scale, np.nan), levels = np.linspace(vmin,vmax,11),\
					colors=plt.get_cmap("RdBu_r")(np.linspace(0,1,11)), extend="both")
				xr.plot.contourf(f3["cmip_sig"], colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.annotate(alph[pos[cnt-1]-1], xy=(0.05, 0.05), xycoords='axes fraction') 
				cnt=cnt+1

		cax = plt.axes([0.33,0.25,0.33,0.02])
		c=plt.colorbar(p, cax=cax, orientation="horizontal", extend="max" )
		c.set_label("Seasonal frequency (days)")
		plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.1,left=0.1)
		fig.text(0.05,0.8,"DCP",ha="center",size=16,va="center",rotation=90)
		fig.text(0.05,0.6,"MUCS6",ha="center",size=16, va="center",rotation=90)
		fig.text(0.05,0.4,"MUCAPE",ha="center",size=16, va="center",rotation=90)        

		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_trend_rcp85.png",bbox_inches="tight")
