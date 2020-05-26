import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from scw_compare import get_era5_lsm
from dask.diagnostics import ProgressBar
import xarray as xr
import glob
import numpy as np

def load_monthly_mean_var(model, ensemble, variable, p2=False):

	if p2:
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_rcp85_"+variable+"_qm_lsm_2006_2018.nc")
	else:
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_historical_"+variable+"_qm_lsm_1979_2005.nc")
	model_monthly = model_6hr.resample({"time":"1M"}).mean("time").persist()
	
	return model_monthly

def load_era5(var):

	era5 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/threshold_data/era5_"+var+"_6hr_mean.nc")
	era5 = era5.isel({"time":era5["time.year"]<=2005})

	return era5

if __name__ == "__main__":
	lsm = get_era5_lsm()
	names = ["ACCESS1-0","ACCESS1-3","BNU-ESM","CNRM","GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M"\
		,"IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MRI-CGCM3","bcc-csm1-1","ERA5"]
	titles = {"lr36":"LR36","mhgt":"MHGT","ml_el":"ML-EL","qmean01":"Qmean01","srhe_left":"SRHE",\
		    "Umean06":"Umean06"}
	units = {"lr36":"$DegC.km^{-1}$", "mhgt":"$m$", "ml_el":"$m$", "qmean01":"$g.kg^{-1}$", \
		    "srhe_left":"$m^2$.$s^{-2}$", "Umean06":"$m.s^{-1}$"}
	plt.figure(figsize=[14,8])
	cnt=1
	#for v in ["lr36","mhgt","ml_el","qmean01","srhe_left","Umean06"]: 
	for v in ["srhe_left"]: 

		fname = "/g/data/eg3/ab4502/ExtremeWind/trends/cmip5_"+v+".csv"
		if not os.path.exists(fname):
			#Load data and mask ocean with NaN
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
			access_esm = xr.where(lsm==1,load_monthly_mean_var("ACCESS-ESM1-5","r1i1p1f1",v), np.nan)

			#Data diagnostics via plotting monthly time series
			plt.figure(figsize=[12,8])
			x = [access0, access3, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5,\
				access_esm]
			for i in np.arange(len(x)):
				x[i].mean(("lat","lon"))[v].plot(label=names[i])
			plt.legend(ncol=3, fontsize="small")
			plt.savefig("/g/data/eg3/ab4502/figs/CMIP/monthly_ts_"+v+".png", bbox_inches="tight")
			plt.close()

			#Combine annual mean time series into a DF and save
			df = pd.DataFrame({\
			    "ACCESS1-0":access0.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "ACCESS1-3":access3.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "BNU-ESM":bnu.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "CNRM":cnrm.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "GFDL-CM3":gfdl_cm3.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "GFDL-ESM2G":gfdl2g.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "GFDL-ESM2M":gfdl2m.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "IPSL-CM5A-LR":ipsll.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "IPSL-CM5A-MR":ipslm.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "MIROC5":miroc.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "MRI-CGCM3":mri.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "bcc-csm1-1":bcc.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "ACCESS-ESM1-5":access_esm.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values,\
			    "ERA5":era5.resample({"time":"1Y"}).mean("time").mean(("lat","lon"))[v].values},\
			    index = np.arange(1979,2006))
			df.to_csv(fname)
		else:
			df = pd.read_csv(fname)    
			df = df.rename(columns={"Unnamed: 0":"index"}).set_index('index')


		cmip5 = df.loc[:, ~np.in1d(names, ["ERA5", "ACCESS-ESM1-5"])]
		era5 = df.loc[:, np.in1d(names, "ERA5")]
		access = df.loc[:, np.in1d(names, "ACCESS-ESM1-5")]
		cmip_rp = pd.DataFrame()
		for n in cmip5.columns:
			cmip_rp = pd.concat([cmip_rp, cmip5.loc[:, n]], axis=0)

		plt.subplot(2,3,cnt)
		plt.title(titles[v])
		ax=plt.gca()
		[ cmip5.loc[:, n].plot(color="grey", alpha=0.25, legend=False) for n in cmip5.columns ]
		cmip5.median(axis=1).plot(color="k", label="CMIP5", marker="o", ax=ax, legend=False)
		access.plot(color="tab:red", label="ACCESS-ESM1-5", ax=ax, marker="o", legend=False)
		era5.plot(color="tab:red", label="ERA5", ax=ax, marker="o", legend=False)
		sns.regplot(x="index", y=v, data=cmip_rp.reset_index().rename(columns={0:v}), x_estimator=np.median,\
			line_kws={"linestyle":"--","color":"tab:purple"},scatter_kws={"color":"k"})
		sns.regplot(x="index", y="ERA5", data=era5.reset_index(),\
			color="tab:orange", line_kws={"linestyle":"--","color":"tab:orange"})
		plt.ylabel(units[v])
		plt.xlabel("")
		ax.yaxis.set_ticks_position("both")
		ax.grid(b=True, which="major", axis="both")
		plt.xlim([1978.5,2005.5])
		cnt=cnt+1
	plt.subplots_adjust(wspace=0.3)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/trends_all_variables.png", bbox_inches="tight")
