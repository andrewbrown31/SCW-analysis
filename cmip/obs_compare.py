from mpl_toolkits.axes_grid1 import inset_locator
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt
import xarray as xr
from percent_mean_change import transform_from_latlon, rasterize
import netCDF4 as nc

#This script provides functions to compare Quantile-matched CMIP5 logit model with ERA5
#Because the data has been quantile-matched monthly and for each grid point, the only 
#   way of independantly assessing GCM reliability is through interannual variability. Namely,
#   by comparing an annual occurrence time series and trends with ERA5.

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

def create_df(ts, model):
	return pd.DataFrame({"logit":ts.values, "model":model, "time":ts.time, "time_ind":np.arange(ts.shape[0])})

def load_resampled(model, ensemble):

	model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_logit_aws_historical_1979_2005.nc")
	model_daily = model_6hr.resample({"time":"1D"}).max("time") 
	model_daily = (model_daily >= 0.72) * 1
	model_monthly = model_daily.resample({"time":"1M"}).sum("time").persist()
	
	return model_monthly

def polyfit(df):

	x = np.arange(df.time.unique().shape[0])
	m,y0 = np.polyfit(x, df.groupby("time").median()["logit"].values.squeeze(), deg=1)
	return [df.time.unique(), m*x + y0]

if __name__ == "__main__":
	access0 = load_resampled("ACCESS1-0","r1i1p1")
	access3 = load_resampled("ACCESS1-3","r1i1p1")
	bnu = load_resampled("BNU-ESM","r1i1p1")
	cnrm = load_resampled("CNRM-CM5","r1i1p1")
	gfdl_cm3 = load_resampled("GFDL-CM3","r1i1p1")
	gfdl2g = load_resampled("GFDL-ESM2G","r1i1p1")
	gfdl2m = load_resampled("GFDL-ESM2M","r1i1p1")
	ipsll = load_resampled("IPSL-CM5A-LR","r1i1p1")
	ipslm = load_resampled("IPSL-CM5A-MR","r1i1p1")
	miroc = load_resampled("MIROC5","r1i1p1")
	mri = load_resampled("MRI-CGCM3","r1i1p1")
	bcc = load_resampled("bcc-csm1-1","r1i1p1")
	era5 = load_resampled("ERA5","")
	lsm = get_era5_lsm()

	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]
	era5["nrm"] = rasterize(shapes, {"lon":era5.lon,"lat":era5.lat})

	plt.figure(figsize=[12,10])
	seaborn=True
	titles = ["a) Northern Australia", "b) Rangelands", "c) Southern Australia", "d) Eastern Australia"]
	cnt=1
	for nrm in [ [0], [1], [2], [3] ]:

		df = pd.concat([\
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    access0.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "ACCESS1-0"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    access3.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "ACCESS1-3"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    bnu.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "BNU-ESM"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    cnrm.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "CNRM-CM5"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    gfdl_cm3.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "GFDL-CM3"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    gfdl2g.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "GFDL-ESM2G"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    gfdl2m.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "GFDL-ESM2M"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    ipsll.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "IPSL-CM5A-LR"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    ipslm.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "IPSL-CM5A-MR"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    miroc.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "MIROC5"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    bcc.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "bcc-csm1-1"),
		    create_df(xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    mri.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")), "MRI-CGCM3")])
		era5_df = create_df(\
			    xr.where(era5["nrm"].isin(nrm) & lsm==1,\
			    era5.resample({"time":"1Y"}).sum("time")["logit_aws"],\
			    np.nan).mean(("lat","lon")),"ERA5")
		x,y=polyfit(df)
		erax,eray=polyfit(era5_df)
		group = df.groupby("model")

		ax=plt.subplot(2,2,cnt)
		if seaborn is True:
			for n, g in group:
				g.plot(x="time_ind",y="logit",ax=ax,color="grey", alpha=0.25, legend=False)
			sns.regplot(x="time_ind",y="logit",data=df,x_estimator=np.median,color="k",\
				line_kws={"linestyle":"--"})
			sns.regplot(x="time_ind",y="logit",data=era5_df,color="tab:red",line_kws={"linestyle":"--"})
			era5_df.plot(x="time_ind",y="logit",ax=ax,color="tab:red", legend=False)
			df.groupby("time_ind").median()["logit"].plot(color="k", legend=False, ax=ax)
		else:
			ax.plot(x,y,"k--")
			ax.plot(erax,eray,color="tab:red",linestyle="--")
			for n, g in group:
				g.plot(x="time",y="logit",ax=ax,color="grey", alpha=0.25, legend=False)
			era5_df.plot(x="time",y="logit",ax=ax,color="tab:red", legend=False)
			df.groupby("time").median()["logit"].plot(color="k", legend=False, ax=ax)
		plt.xlabel("")
		plt.ylabel("Mean annual environments")
		ax.set_xticks([1,5,9,13,17,21,25])
		ax.set_xticklabels(["1980","1984","1988","1992","1996","2000","2004"])
		ax.yaxis.set_ticks_position("both")
		plt.title(titles[nrm[0]])
		plt.xlim([0,26.5])
		cnt=cnt+1

	#TODO Label Tasmania as being in region C
	inset = inset_locator.inset_axes(ax, width="30%", height="33%",loc=1)
	[xr.plot.contour(xr.where(era5["nrm"]==n, 1.5, 0.5), colors=["k"], levels=[1], ax=inset, add_labels=False) for n in [0,1,2,3] ]
	inset.yaxis.set_ticks([]) 
	inset.xaxis.set_ticks([])
	plt.text(142,-37,"c",fontdict={"size":12}); plt.text(130, -28, "b",fontdict={"size":12}); plt.text(131,-16,"a",fontdict={"size":12}); plt.text(149,-30,"d",fontdict={"size":12});
	plt.text(116,-34,"c",fontdict={"size":12})  

	plt.subplots_adjust(hspace=0.25)    
	plt.show()
