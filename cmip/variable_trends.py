from percent_mean_change import transform_from_latlon, rasterize
import geopandas
from rasterio import features
from affine import Affine
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from scw_compare import get_era5_lsm
from dask.diagnostics import ProgressBar
import xarray as xr
import glob
import numpy as np

def drop_duplicates(da):

        #Drop time duplicates

        a, ind = np.unique(da.time.values, return_index=True)
        return(da[ind])

def load_monthly_mean_var(model, ensemble, variable, period="p1"):

	if period=="p2":
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_rcp85_"+variable+"_qm_lsm_2006_2018.nc", combine='by_coords')
	elif period=="p3":
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_historical_"+variable+"_qm_lsm_2006_2014.nc", combine='by_coords')
	elif period=="p1":
		model_6hr = xr.open_mfdataset(\
		    ["/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_historical_"+variable+"_qm_lsm_1970_1978.nc", 
		    "/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model+"_"+ensemble+"_historical_"+variable+"_qm_lsm_1979_2005.nc"], combine='by_coords')
	model_monthly = drop_duplicates(model_6hr[variable]).resample({"time":"1M"}).mean("time").persist()
	
	return model_monthly

def load_era5(var):

	era5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/era5_"+var+"_6hr_mean.nc")

	return era5

if __name__ == "__main__":
	lsm = get_era5_lsm()
	temp = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_lr36_historical_1979_2005.nc")
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]
	nrm_da = rasterize(shapes, {"lon":temp.lon,"lat":temp.lat})
	aus_da = rasterize([f2.loc[f2.name=="Australia"].geometry.values[0]], {"lon":temp.lon,"lat":temp.lat})
	titles = {"lr36":"LR36","mhgt":"MHGT","ml_el":"ML-EL","qmean01":"Qmean01","srhe_left":"SRHE",\
		    "Umean06":"Umean06"}
	units = {"lr36":"$DegC.km^{-1}$", "mhgt":"$m$", "ml_el":"$m$", "qmean01":"$g.kg^{-1}$", \
		    "srhe_left":"$m^2$.$s^{-2}$", "Umean06":"$m.s^{-1}$"}
	nrm_titles = ["Northern Australia", "b) Rangelands", "c) Eastern Australia", "d) Southern Australia"]
	    
	for v in ["lr36","mhgt","ml_el","qmean01","srhe_left","Umean06"]: 
		print(v)
		fname1 = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+"_nrm0.csv"
		fname2 = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+"_nrm1.csv"
		fname3 = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+"_nrm2.csv"
		fname4 = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+"_nrm3.csv"

		if not (os.path.exists(fname1)) or not (os.path.exists(fname2)) or not (os.path.exists(fname3)) or not (os.path.exists(fname4)):

			#Load 1970-2005 historical data and mask ocean with NaN
			access0 = xr.where(lsm==1,load_monthly_mean_var("ACCESS1-0","r1i1p1",v), np.nan)
			access3 = xr.where(lsm==1,load_monthly_mean_var("ACCESS1-3","r1i1p1",v), np.nan)
			bnu = xr.where(lsm==1,load_monthly_mean_var("BNU-ESM","r1i1p1",v), np.nan)
			cnrm = xr.where(lsm==1,load_monthly_mean_var("CNRM-CM5","r1i1p1",v), np.nan)
			gfdl_cm3 = xr.where(lsm==1,load_monthly_mean_var("GFDL-CM3","r1i1p1",v), np.nan)
			gfdl2g = xr.where(lsm==1,load_monthly_mean_var("GFDL-ESM2G","r1i1p1",v), np.nan)
			gfdl2m = xr.where(lsm==1,load_monthly_mean_var("GFDL-ESM2M","r1i1p1",v), np.nan)
			ipsll = xr.where(lsm==1,load_monthly_mean_var("IPSL-CM5A-LR","r1i1p1",v), np.nan)
			ipslm = xr.where(lsm==1,load_monthly_mean_var("IPSL-CM5A-MR","r1i1p1",v), np.nan)
			miroc = xr.where(lsm==1,load_monthly_mean_var("MIROC5","r1i1p1",v), np.nan)
			mri = xr.where(lsm==1,load_monthly_mean_var("MRI-CGCM3","r1i1p1",v), np.nan)
			bcc = xr.where(lsm==1,load_monthly_mean_var("bcc-csm1-1","r1i1p1",v), np.nan)
			era5 = xr.where(lsm==1, load_era5(v), np.nan)
			access_esm = xr.where(lsm==1,load_monthly_mean_var("ACCESS-CM2","r1i1p1f1",v), np.nan)
			access_cm2 = xr.where(lsm==1,load_monthly_mean_var("ACCESS-ESM1-5","r1i1p1f1",v), np.nan)

			#Load 2006-2018 rcp85 data and mask ocean with NaN
			gfdl_cm32 = xr.where(lsm==1,load_monthly_mean_var("GFDL-CM3","r1i1p1",v, period="p2"), np.nan)
			gfdl2g2 = xr.where(lsm==1,load_monthly_mean_var("GFDL-ESM2G","r1i1p1",v, period="p2"), np.nan)
			gfdl2m2 = xr.where(lsm==1,load_monthly_mean_var("GFDL-ESM2M","r1i1p1",v, period="p2"), np.nan)
			ipsll2 = xr.where(lsm==1,load_monthly_mean_var("IPSL-CM5A-LR","r1i1p1",v, period="p2"), np.nan)
			ipslm2 = xr.where(lsm==1,load_monthly_mean_var("IPSL-CM5A-MR","r1i1p1",v, period="p2"), np.nan)
			miroc2 = xr.where(lsm==1,load_monthly_mean_var("MIROC5","r1i1p1",v, period="p2"), np.nan)
			mri2 = xr.where(lsm==1,load_monthly_mean_var("MRI-CGCM3","r1i1p1",v, period="p2"), np.nan)
			access_esm2 = xr.where(lsm==1,load_monthly_mean_var("ACCESS-ESM1-5","r1i1p1f1",v, period="p3"), np.nan)
			access_cm22 = xr.where(lsm==1,load_monthly_mean_var("ACCESS-CM2","r1i1p1f1",v, period="p3"), np.nan)
            
		fig=plt.figure(figsize=[14,8])
		cnt=1
        
		for n in [0,1,2,3]:


			fname = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+"_nrm"+str(n)+".csv"

			if not os.path.exists(fname):

				#Combine annual mean time series into a DF and save
				df = pd.DataFrame({\
				    "ACCESS1-0":xr.where(nrm_da.isin([n]), access0, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "ACCESS1-3":xr.where(nrm_da.isin([n]), access3, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "BNU-ESM":xr.where(nrm_da.isin([n]), bnu, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "CNRM":xr.where(nrm_da.isin([n]), cnrm, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "GFDL-CM3":xr.where(nrm_da.isin([n]), gfdl_cm3, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "GFDL-ESM2G":xr.where(nrm_da.isin([n]), gfdl2g, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "GFDL-ESM2M":xr.where(nrm_da.isin([n]), gfdl2m, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "IPSL-CM5A-LR":xr.where(nrm_da.isin([n]), ipsll, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "IPSL-CM5A-MR":xr.where(nrm_da.isin([n]), ipslm, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "MIROC5":xr.where(nrm_da.isin([n]), miroc, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "MRI-CGCM3":xr.where(nrm_da.isin([n]), mri, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "bcc-csm1-1":xr.where(nrm_da.isin([n]), bcc, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
					},\
				    index = np.arange(1970,2006))
				df2 = pd.DataFrame({\
				    "GFDL-CM3":xr.where(nrm_da.isin([n]), gfdl_cm32, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "GFDL-ESM2G":xr.where(nrm_da.isin([n]), gfdl2g2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "GFDL-ESM2M":xr.where(nrm_da.isin([n]), gfdl2m2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "IPSL-CM5A-LR":xr.where(nrm_da.isin([n]), ipsll2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "IPSL-CM5A-MR":xr.where(nrm_da.isin([n]), ipslm2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "MIROC5":xr.where(nrm_da.isin([n]), miroc2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "MRI-CGCM3":xr.where(nrm_da.isin([n]), mri2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values},\
				    index = np.arange(2006,2019))
				df3 = pd.DataFrame({\
				    "ACCESS-ESM1-5":xr.where(nrm_da.isin([n]), access_esm, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "ACCESS-CM2":xr.where(nrm_da.isin([n]), access_cm2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values},\
				    index = np.arange(1970,2006))
				df4 = pd.DataFrame({\
				    "ACCESS-ESM1-5":xr.where(nrm_da.isin([n]), access_esm2, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values,\
				    "ACCESS-CM2":xr.where(nrm_da.isin([n]), access_cm22, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon")).values},\
				    index = np.arange(2006,2015))
				era5_df = pd.DataFrame({"ERA5":xr.where(nrm_da.isin([n]), era5, np.nan).resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values}, index=np.arange(1979,2019))
				df = (pd.concat([ pd.concat([ pd.concat([df,df2]), pd.concat([df3,df4]) ], axis=1) , era5_df], axis=1))
				df.to_csv(fname)
			else:
				df = pd.read_csv(fname)    
				df = df.rename(columns={"Unnamed: 0":"index"}).set_index('index')


			cmip5_df = df.loc[:, ~np.in1d(df.columns, ["ERA5", "ACCESS-ESM1-5", "ACCESS-CM2"])]
			era5_df = df.loc[:, np.in1d(df.columns, "ERA5")]
			access_esm_df = df.loc[:, np.in1d(df.columns, "ACCESS-ESM1-5")]
			access_cm2_df = df.loc[:, np.in1d(df.columns, "ACCESS-CM2")]
			cmip_rp = pd.DataFrame()
			for name in cmip5_df.columns:
				cmip_rp = pd.concat([cmip_rp, cmip5_df.loc[:, name]], axis=0)

			plt.subplot(4,1,cnt)
			if cnt==1:            
				plt.title("a) " + titles[v] + " " + nrm_titles[n])
			else:
				plt.title(nrm_titles[n])
			ax=plt.gca()
			[ cmip5_df.loc[:, n].plot(color="grey", alpha=0.25, legend=False) for n in cmip5_df.columns ]
			cmip5_df.median(axis=1).plot(color="k", label="CMIP5", marker="o", ax=ax, legend=False)
			access_esm_df.plot(color="tab:green", label="ACCESS-ESM1-5", ax=ax, marker="o", legend=False)
			access_cm2_df.plot(color="tab:blue", label="ACCESS-CM2", ax=ax, marker="o", legend=False)
			era5_df.plot(color="tab:red", label="ERA5", ax=ax, marker="o", legend=False)
			plt.xlabel("")
			ax.yaxis.set_ticks_position("both")
			ax.grid(b=True, which="major", axis="both")
			if cnt<4:            
				plt.xticks(ticks=np.arange(1970, 2020, 5), labels="")
			else:
				plt.xticks(ticks=np.arange(1970, 2020, 5))
			plt.xlim([1969.5, 2018.5])                
			cnt=cnt+1
		fig.text(0.05, 0.5, units[v], va="center", ha="left",rotation=90)
		plt.subplots_adjust(hspace=0.25)
		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/trends_"+v+".png", bbox_inches="tight")
		plt.close()
