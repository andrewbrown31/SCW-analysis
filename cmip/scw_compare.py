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

def std_boot(x):
	x=np.array(x)
	s_samp = [np.median(x[np.random.randint(len(x)-1,size=len(x))]) for i in np.arange(10000)]
	return [np.percentile(s_samp, 2.5), np.percentile(s_samp, 97.5)]
    
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

def load_resampled_logit(model, ensemble, p2=False, cmip6=False):

	if cmip6:
		model_6hr = xr.open_mfdataset(["/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
		    model+"_"+ensemble+"_logit_aws_historical_1970_1978.nc",\
			"/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+model+"_"+ensemble+\
		    "_logit_aws_historical_1979_2005.nc",\
			"/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+model+"_"+ensemble+\
		    "_logit_aws_historical_2006_2014.nc"], combine="by_coords")
	else:
		if p2:
			try:
				model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
				    model+"_"+ensemble+"_logit_aws_historical_2006_2018.nc")
			except:
				model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
				    model+"_"+ensemble+"_logit_aws_rcp85_2006_2018.nc")
		else:
			try:
				model_6hr = xr.open_mfdataset(["/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
				    model+"_"+ensemble+"_logit_aws_historical_1979_2005.nc",\
					"/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+model+"_"+ensemble+\
					"_logit_aws_historical_1970_1978.nc"], combine="by_coords")
			except:
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

	load = True
	path = "/g/data/eg3/ab4502/ExtremeWind/trends/"

	temp = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_lr36_historical_1979_2005.nc")
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]
	nrm_da = rasterize(shapes, {"lon":temp.lon,"lat":temp.lat})
	aus_da = rasterize([f2.loc[f2.name=="Australia"].geometry.values[0]], {"lon":temp.lon,"lat":temp.lat})

	lsm = get_era5_lsm()

	#If load is false, then instead of loading in the annual dataframes, load in the entire model datasets, and
	#	create the dataframes

	if not load:

		#Load 1970-2005 historical experiment data, from CMIP5
		access0 = load_resampled_logit("ACCESS1-0","r1i1p1")
		access3 = load_resampled_logit("ACCESS1-3","r1i1p1")
		bnu = load_resampled_logit("BNU-ESM","r1i1p1")
		cnrm = load_resampled_logit("CNRM-CM5","r1i1p1")
		gfdl_cm3 = load_resampled_logit("GFDL-CM3","r1i1p1")
		gfdl2g = load_resampled_logit("GFDL-ESM2G","r1i1p1")
		gfdl2m = load_resampled_logit("GFDL-ESM2M","r1i1p1")
		ipsll = load_resampled_logit("IPSL-CM5A-LR","r1i1p1")
		ipslm = load_resampled_logit("IPSL-CM5A-MR","r1i1p1")
		miroc = load_resampled_logit("MIROC5","r1i1p1")
		mri = load_resampled_logit("MRI-CGCM3","r1i1p1")
		bcc = load_resampled_logit("bcc-csm1-1","r1i1p1")

		#Load the ERA5 dataset, 1979-2018
		era5 = load_resampled_logit("ERA5","")
		era52 = load_resampled_logit("ERA5","",p2=True)

		#Load 2006-2018 RCP85 data from the reduced CMIP5 ensemble
		gfdl_cm3_2 = load_resampled_logit("GFDL-CM3","r1i1p1",p2=True)
		gfdl2g_2 = load_resampled_logit("GFDL-ESM2G","r1i1p1",p2=True)
		gfdl2m_2 = load_resampled_logit("GFDL-ESM2M","r1i1p1",p2=True)
		ipsll_2 = load_resampled_logit("IPSL-CM5A-LR","r1i1p1",p2=True)
		ipslm_2 = load_resampled_logit("IPSL-CM5A-MR","r1i1p1",p2=True)
		miroc_2 = load_resampled_logit("MIROC5","r1i1p1",p2=True)
		mri_2 = load_resampled_logit("MRI-CGCM3","r1i1p1",p2=True)

		#Load historical CMIP6 data from 1970-2014
		access_esm = load_resampled_logit("ACCESS-ESM1-5","r1i1p1f1", cmip6=True)
		access_cm2 = load_resampled_logit("ACCESS-CM2","r1i1p1f1", cmip6=True)

		data = []

		for nrm in [ [0], [1], [2], [3] ]:

			#Create historical CMIP5 dataframe
			df = pd.concat([\
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    access0.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "ACCESS1-0"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    access3.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "ACCESS1-3"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    bnu.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "BNU-ESM"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    cnrm.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "CNRM-CM5"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl_cm3.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-CM3"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl2g.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-ESM2G"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl2m.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-ESM2M"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    ipsll.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "IPSL-CM5A-LR"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    ipslm.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "IPSL-CM5A-MR"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    miroc.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "MIROC5"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    bcc.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "bcc-csm1-1"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    mri.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "MRI-CGCM3")])
			#Create RCP85 CMIP5 dataframe
			df2 = pd.concat([\
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl_cm3_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-CM3"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl2g_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-ESM2G"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    gfdl2m_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "GFDL-ESM2M"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    ipsll_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "IPSL-CM5A-LR"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    ipslm_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "IPSL-CM5A-MR"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    miroc_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "MIROC5"),\
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    mri_2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "MRI-CGCM3")])
			#Create historical CMIP6 dataframe
			df3 = pd.concat([\
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    access_esm.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "ACCESS-ESM1-5"),
			    create_df(xr.where(nrm_da.isin(nrm) & lsm==1,\
				    access_cm2.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")), "ACCESS-CM2")])
			#Create ERA5 dataframe
			era5_df = create_df(\
				    xr.where(nrm_da.isin(nrm) & lsm==1,\
				    era5.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")),"ERA5")
			era5_df2 = create_df(\
				    xr.where(nrm_da.isin(nrm) & lsm==1,\
				    era52.resample({"time":"1Y"}).sum("time")["logit_aws"],\
				    np.nan).mean(("lat","lon")),"ERA5")
			era5_df3 = pd.concat([era5_df, era5_df2], axis=0).reset_index()
			era5_df3["time_ind"] = era5_df3.index
		
			#Save dataframes for each NRM region to disk
			df.to_csv(path+"cmip5_logit_1970_2005_nrm"+str(nrm[0])+".csv", index=False)
			df2.to_csv(path+"cmip5_logit_2006_2018_nrm"+str(nrm[0])+".csv", index=False)
			df3.to_csv(path+"cmip6_logit_1970_2014_nrm"+str(nrm[0])+".csv", index=False)
			era5_df.to_csv(path+"era5_logit_1979_2005_nrm"+str(nrm[0])+".csv", index=False)
			era5_df2.to_csv(path+"era5_logit_2006_2018_nrm"+str(nrm[0])+".csv", index=False)
			era5_df3.to_csv(path+"era5_logit_1979_2018_nrm"+str(nrm[0])+".csv", index=False)

			data.append([df, df2, df3, era5_df, era5_df2, era5_df3])

	else:
		#Or else, just load the df
		data = []
		for nrm in [ [0], [1], [2], [3] ]:
			df = pd.read_csv(path+"cmip5_logit_1970_2005_nrm"+str(nrm[0])+".csv")
			df2 = pd.read_csv(path+"cmip5_logit_2006_2018_nrm"+str(nrm[0])+".csv")
			df3 = pd.read_csv(path+"cmip6_logit_1970_2014_nrm"+str(nrm[0])+".csv")
			era5_df = pd.read_csv(path+"era5_logit_1979_2005_nrm"+str(nrm[0])+".csv")
			era5_df2 = pd.read_csv(path+"era5_logit_2006_2018_nrm"+str(nrm[0])+".csv")
			era5_df3 = pd.read_csv(path+"era5_logit_1979_2018_nrm"+str(nrm[0])+".csv")

			data.append([df, df2, df3, era5_df, era5_df2, era5_df3])

	#Plot the data
	fig=plt.figure(figsize=[12,10])
	seaborn=False
	titles = ["a) Northern Australia", "b) Rangelands", "c) Eastern Australia", "d) Southern Australia"]
	cnt=1
	for i in np.arange(4):

		df = data[i][0]
		df2 = data[i][1]
		df3 = data[i][2]
		df2["time_ind"] = df2["time_ind"] + df["time_ind"].max() + 1
		era5_df = data[i][3]
		era5_df2 = data[i][4]
		era5_df3 = data[i][5]
		era5_df3["time_ind"] = era5_df3["time_ind"] + 9
		x,y=polyfit(df)
		erax,eray=polyfit(era5_df)
		erax2,eray2=polyfit(era5_df3)
		group = df.groupby("model")
		group2 = df2.groupby("model")
		ax=plt.subplot(4,1,cnt)
		if seaborn is True:
			for n, g in group:
				g.plot(x="time_ind",y="logit",ax=ax,color="grey", alpha=0.25, legend=False)
			sns.regplot(x="time_ind",y="logit",data=df,x_estimator=np.median,\
				line_kws={"linestyle":"--", "color":"tab:purple"}, scatter_kws={"color":"k"})
			sns.regplot(x="time_ind",y="logit",data=era5_df3,color="tab:orange",line_kws={"linestyle":"--"})
			sns.regplot(x="time_ind",y="logit",data=era5_df,color="tab:red",line_kws={"linestyle":"--"})
			era5_df.plot(x="time_ind",y="logit",ax=ax,color="tab:red", legend=False)
			era5_df3.loc[era5_df3.time_ind>=26].plot(x="time_ind",y="logit",ax=ax,color="tab:orange", legend=False)
			df.groupby("time_ind").median()["logit"].plot(color="k", legend=False, ax=ax)
		else:
			std_cmip5 = []            
			for mod in df.model.unique():
				pd.concat([df[df.model==mod], df2[df2.model==mod]]).plot(x="time_ind",y="logit",ax=ax,color="grey", alpha=0.4, legend=False)
				std_cmip5.append(pd.concat([df[df.model==mod], df2[df2.model==mod]])["logit"].std()) 
			cmip_low, cmip_up = std_boot(std_cmip5)
			a=era5_df3.plot(x="time_ind",y="logit",ax=ax,color="tab:red", legend=False, marker="o")
			b=pd.concat([df.groupby("time_ind").median()["logit"], df2.groupby("time_ind").median()["logit"]]).plot(color="k", legend=False, ax=ax, marker="o")
			c=df3[df3["model"]=="ACCESS-ESM1-5"].plot(x="time_ind",y="logit",color="tab:green", legend=False, ax=ax, marker="o")
			d=df3[df3["model"]=="ACCESS-CM2"].plot(x="time_ind",y="logit",color="tab:blue", legend=False, ax=ax, marker="o")
		plt.xlabel("")
		ax.set_xticks([0,8,16,24,32,40,48])
		ax.set_xticks(np.arange(0,50,2), minor=True)
		if i == 3:
			ax.set_xticklabels(["1970","1978","1986","1994","2002","2010","2018"])
			plt.legend((a.lines[-4],b.lines[-3],c.lines[-2],d.lines[-1]),("ERA5","CMIP5","ACCESS-ESM1-5","ACCESS-CM2"),ncol=4,bbox_to_anchor=(.5,-.5), loc=8)
		else:
			ax.set_xticklabels("")
		ax.set_yticks([20,40,60,80,100]); plt.ylim([15,110])
		ax.yaxis.set_ticks_position("both")
		ax.grid(b=True, which="both", axis="both")
		plt.title(titles[i])
		plt.xlim([0,49.5])

		if i == 0:
			plt.text(0.5,20,str(round(np.median(std_cmip5),1))+" ("+str(round(cmip_low,1))+", "+str(round(cmip_up,1))+")", fontdict={"weight":"bold"})
			plt.text(9,20,str(round(era5_df3["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:red"})
			plt.text(12,20,str(round(df3[df3["model"]=="ACCESS-ESM1-5"]["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:green"})
			plt.text(15,20,str(round(df3[df3["model"]=="ACCESS-CM2"]["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:blue"})
			inset = inset_locator.inset_axes(ax, width="12%", height="40%",loc=4)
		else:
			plt.text(0.5,90,str(round(np.median(std_cmip5),1))+" ("+str(round(cmip_low,1))+", "+str(round(cmip_up,1))+")", fontdict={"weight":"bold"})
			plt.text(9,90,str(round(era5_df3["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:red"})
			plt.text(12,90,str(round(df3[df3["model"]=="ACCESS-ESM1-5"]["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:green"})
			plt.text(15,90,str(round(df3[df3["model"]=="ACCESS-CM2"]["logit"].std(),1)), fontdict={"weight":"bold", "color":"tab:blue"})
			inset = inset_locator.inset_axes(ax, width="12%", height="40%",loc=1)
		xr.plot.contour(xr.where(aus_da==1, 1.5, 0.5), colors=["k"], levels=[1], ax=inset, add_labels=False)
		xr.plot.contourf(xr.where(nrm_da==i, 1, 0), colors=["none","gray"], levels=[0.5,1], ax=inset, add_labels=False, add_colorbar=False)
		inset.yaxis.set_ticks([]) 
		inset.xaxis.set_ticks([])
		cnt=cnt+1
	fig.text(0.05, 0.4, "Mean annual environments (days)", rotation=90)
	plt.subplots_adjust(hspace=0.3, top=0.95)    
	plt.savefig("out.png",bbox_inches="tight")
