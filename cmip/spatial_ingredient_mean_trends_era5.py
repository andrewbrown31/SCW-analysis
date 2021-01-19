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
	cnt=1

	seasons = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
	y1=1979; y2=1998; y3=1999; y4=2018

	freq = False
	plot_final = True

	variables = ["mu_cape","dcape","ebwd","lr03","lr700_500","dp850","ta850","ta500","Umean06","s06","Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"]

	for p in variables:

		if freq:
			era5 = load_resampled_era5(p, thresh).persist()
		else:
			era5 = load_resampled_era5(p, None, mean=True).persist()

		for s in ["DJF","MAM","JJA","SON"]:

			if freq:
				t1 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y1) &\
				    (era5["time.year"] <= y2) })).resample({"time":"1Y"}).sum("time")
				t2 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y3) &\
				    (era5["time.year"] <= y4) })).resample({"time":"1Y"}).sum("time")
				era5_change = (t2.sum("time") / ((y2+1) - y1)).values - (t1.sum("time") / ((y4+1) - y3)).values
			else:
				t1 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y1) &\
				    (era5["time.year"] <= y2) })).resample({"time":"1Y"}).mean("time")
				t2 = (era5.sel({"time":\
				    (np.in1d(era5["time.month"], seasons[s])) &\
				    (era5["time.year"] >= y3) &\
				    (era5["time.year"] <= y4) })).resample({"time":"1Y"}).mean("time")
				era5_change = (t2.mean("time")).values - (t1.mean("time")).values
			temp, sig = ttest_ind(t1.values,t2.values,axis=0)
			era5_sig = sig
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
		m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
			urcrnrlat=-10,projection="cyl")
		vmin = {"mu_cape":-200,"dcape":-100, "ebwd":-1, "lr03":-0.5, "lr700_500":-0.5, "dp850":-2, "ta850":-1.5, "ta500":-1.5, "Umean06":-2, "s06":-2,\
			"Umean800_600":-2,"lr13":-1,"rhmin13":-10,"srhe_left":-5,"q_melting":-0.8,"eff_lcl":-250}
		units = {"lr03":"deg km$^{-1}$","mu_cape":"J kg$^{-1}$",
                    "ebwd":"m s$^{-1}$","Umean06":"m s$^{-1}$","s06":"m s$^{-1}$",\
                    "dp850":"deg C","ta500":"deg C","ta850":"deg C","srhe_left":"m$^{-2}$ s$^{-2}$",\
                    "lr700_500":"deg km$^{-1}$", "dcape":"J kg$^{-1}$", "Umean800_600":"m s$^{-1}$",\
		    "lr13":"deg km$^{-1}$", "rhmin13":"%", "q_melting":"g kg$^{-1}$", "eff_lcl":"m"}
		fig=plt.figure(figsize=[8,14])
		pos = np.concatenate([np.arange(1,41,4), np.arange(2,41,4), np.arange(3,41,4), np.arange(4,41,4)])
		cnt=1
		cmip_scale = 1
		for season in ["DJF","MAM","JJA","SON"]:
				ax=plt.subplot(10,4,pos[cnt-1])
				plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1,left=0.1)
				f1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_mu_cape.nc")
				x,y = np.meshgrid(f1.lon.values,f1.lat.values)
				if season=="DJF":
					plt.ylabel("MUCAPE")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f1["era5_trend"], np.nan), levels = np.linspace(vmin["mu_cape"],-vmin["mu_cape"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f1["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.title(season)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["mu_cape"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_dcape.nc")
				if season=="DJF":
					plt.ylabel("DCAPE")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f2["era5_trend"], np.nan), levels = np.linspace(vmin["dcape"],-vmin["dcape"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f2["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["dcape"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f3 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_ebwd.nc")
				if season=="DJF":
					plt.ylabel("EBWD")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f3["era5_trend"], np.nan), levels = np.linspace(vmin["ebwd"],-vmin["ebwd"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f3["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["ebwd"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f4 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_lr03.nc")
				if season=="DJF":
					plt.ylabel("LR03")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f4["era5_trend"], np.nan), levels = np.linspace(vmin["lr03"],-vmin["lr03"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f4["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["lr03"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_lr700_500.nc")
				if season=="DJF":
					plt.ylabel("LR75")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f5["era5_trend"], np.nan), levels = np.linspace(vmin["lr700_500"],-vmin["lr700_500"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f5["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["lr700_500"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f6 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_dp850.nc")
				if season=="DJF":
					plt.ylabel("DP850")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f6["era5_trend"], np.nan), levels = np.linspace(vmin["dp850"],-vmin["dp850"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f6["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["dp850"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f7 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_ta850.nc")
				if season=="DJF":
					plt.ylabel("T850")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f7["era5_trend"], np.nan), levels = np.linspace(vmin["ta850"],-vmin["ta850"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f7["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["ta850"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f8 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_ta500.nc")
				if season=="DJF":
					plt.ylabel("T500")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f8["era5_trend"], np.nan), levels = np.linspace(vmin["ta500"],-vmin["ta500"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f8["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["ta500"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f9 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_Umean06.nc")
				if season=="DJF":
					plt.ylabel("Umean06")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f9["era5_trend"], np.nan), levels = np.linspace(vmin["Umean06"],-vmin["Umean06"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f9["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["Umean06"])
				cnt=cnt+1

				ax=plt.subplot(10,4,pos[cnt-1])
				f10 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_s06.nc")
				if season=="DJF":
					plt.ylabel("S06")
				m.drawcoastlines()
				p=m.contourf(x, y, np.where(lsm==1, f10["era5_trend"], np.nan), levels = np.linspace(vmin["s06"],-vmin["s06"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f10["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["s06"])
				cnt=cnt+1



		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_ingredients_mean_trend_era5.png",bbox_inches="tight")

		#NOW, FOR LOGISTIC MODEL INGREDIENTS
		fig=plt.figure(figsize=[8,10])
		pos = np.concatenate([np.arange(1,25,4), np.arange(2,25,4), np.arange(3,25,4), np.arange(4,25,4)])
		cnt=1
		cmip_scale = 1
		for season in ["DJF","MAM","JJA","SON"]:
				plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1,left=0.1)
				ax=plt.subplot(6,4,pos[cnt-1])
				f1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_Umean800_600.nc")
				x,y = np.meshgrid(f1.lon.values,f1.lat.values)
				if season=="DJF":
					plt.ylabel("Umean800-600")
				m.drawcoastlines()
				p=m.contourf(x, y, f1["era5_trend"], levels = np.linspace(vmin["Umean800_600"],-vmin["Umean800_600"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f1["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				plt.title(season)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["Umean800_600"])
				cnt=cnt+1

				ax=plt.subplot(6,4,pos[cnt-1])
				f2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_lr13.nc")
				if season=="DJF":
					plt.ylabel("LR13")
				m.drawcoastlines()
				p=m.contourf(x, y, f2["era5_trend"], levels = np.linspace(vmin["lr13"],-vmin["lr13"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f2["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["lr13"])
				cnt=cnt+1

				ax=plt.subplot(6,4,pos[cnt-1])
				f3 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_rhmin13.nc")
				if season=="DJF":
					plt.ylabel("RHMin13")
				m.drawcoastlines()
				p=m.contourf(x, y, f3["era5_trend"], levels = np.linspace(vmin["rhmin13"],-vmin["rhmin13"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f3["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["rhmin13"])
				cnt=cnt+1

				ax=plt.subplot(6,4,pos[cnt-1])
				f4 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_srhe_left.nc")
				if season=="DJF":
					plt.ylabel("SRHE")
				m.drawcoastlines()
				p=m.contourf(x, y, f4["era5_trend"], levels = np.linspace(vmin["srhe_left"],-vmin["srhe_left"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f4["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["srhe_left"])
				cnt=cnt+1

				ax=plt.subplot(6,4,pos[cnt-1])
				f5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_q_melting.nc")
				if season=="DJF":
					plt.ylabel("Q-Melting")
				m.drawcoastlines()
				p=m.contourf(x, y, f5["era5_trend"], levels = np.linspace(vmin["q_melting"],-vmin["q_melting"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f5["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["q_melting"])
				cnt=cnt+1

				ax=plt.subplot(6,4,pos[cnt-1])
				f6 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/trends/spatial_hist_mean_trend_era5_"+season+"_eff_lcl.nc")
				if season=="DJF":
					plt.ylabel("Eff-LCL")
				m.drawcoastlines()
				p=m.contourf(x, y, f6["era5_trend"], levels = np.linspace(vmin["eff_lcl"],-vmin["eff_lcl"],11),\
					cmap=plt.get_cmap("RdBu_r"), extend="both")
				xr.plot.contourf(xr.where(f6["era5_sig"]<=0.1, 1, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
				if season=="SON":
					pos1 = ax.get_position() # get the original position 
					cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
					c=plt.colorbar(p,ax=ax,cax=cax,orientation="vertical")
					c.set_label(units["eff_lcl"])
				cnt=cnt+1



		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/spatial_logit_ingredients_mean_trend_era5.png",bbox_inches="tight")
