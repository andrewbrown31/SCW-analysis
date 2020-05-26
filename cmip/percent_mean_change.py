from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
from rasterio import features
from affine import Affine
import xarray as xr

#For each ingredient of the logit_aws equation, plot the percentage change across a region for each month.
#On a different y-axis, plot the percentage change in environment occurrence frequency.
#Include options to specify the region with a lat/lon bounding box (defaults to whole of Australia).
#Do this for each model.

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))

def get_mean(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_mean_"+p+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_mean_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def get_seasonal_freq(models, p, y1, y2, era5_y1, era5_y2, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def plot_boxplot(v, models, data_list, nrm_da, spatial_diff, spatial_diff_sig, mean_envs, min_envs=0, plot_map=True):

	#min_envs is the minimum number of environments at a spatial point per season, required to draw statistical significance at that point.

	#Plot
	fig=plt.figure(figsize=[8,10])
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	titles = ["a) Northern Australia", "b) Rangelands", "c) Eastern Australia", \
		    "d) Southern Australia"]
	ranges = [-5.5,5.5]
	for i in np.arange(len(data_list)):
		ax=plt.subplot2grid((5,4), (i, 0), colspan=4, rowspan=1)
		boxes = []
		for month in return_months():
			temp = data_list[i].loc[month].values
			boxes.append({"med":np.percentile(temp, 50, interpolation="nearest"),\
			    "q1":np.percentile(temp, 16.67, interpolation="nearest"),\
			    "q3":np.percentile(temp, 83.33, interpolation="nearest"),\
			    "whislo":temp.min(),"whishi":temp.max()})
		ax.bxp(boxes, showfliers=False)
		#ax.grid(b=True, which="both", axis="both")
		plt.ylim(ranges)
		plt.axhline(0,ls="--", color="k")
		if i not in [3]:
			ax.set_xticklabels("")
		else:
			ax.set_xticks(np.arange(1,13))
			ax.set_xticklabels(return_months())

		plt.title(titles[i])
		if plot_map:
			if i==0:
				inset = inset_locator.inset_axes(ax, width="10%", height="30%",loc=8)
			else:
				inset = inset_locator.inset_axes(ax, width="10%", height="30%",loc=3)
			xr.plot.contour((nrm_da["nrm"].isin([0,1,2,3]).values*1), levels=[0.5,1.5], colors="k", ax=inset)
			xr.plot.contourf((nrm_da["nrm"].isin([i]).values*1), levels=[0,0.5], colors=["none","grey"],extend="max", ax=inset)
			plt.tick_params(axis='both',which='both', bottom=False, top=False, labelbottom=False, left=False,\
				labelleft=False)

	cnt=0
	seasons = ["DJF","MAM","JJA","SON"]
	for spatial_diff, spatial_diff_sig in zip(spatial_diffs, spatial_diff_sigs):
		ax=plt.subplot2grid((5,4), (4,cnt), colspan=1, rowspan=1)
		xr.plot.contour(xr.where(nrm_da["aus"]==1, 1, 0), levels=[0.5,1.5], colors="k", ax=ax, add_labels=False)
		[xr.plot.contour(xr.where((nrm_da["nrm"]==i) & (~(nrm_da["aus"].isnull())), 1, 0), levels=[0.5,1.5], colors="k", ax=ax, add_labels=False) for i in [0,1,2,3]]
		spatial_diff = xr.DataArray(data=spatial_diff, coords=(nrm_da.lat, nrm_da.lon))
		spatial_diff_sig = xr.DataArray(data=spatial_diff_sig, coords=(nrm_da.lat, nrm_da.lon))
		c=xr.plot.contourf(xr.where((~(nrm_da["aus"].isnull())), spatial_diff, np.nan), cmap=plt.get_cmap("RdBu_r"), vmin=-12, vmax=12, extend="both", ax=ax, add_colorbar=False, add_labels=False)
		xr.plot.contourf(xr.where((~(nrm_da["aus"].isnull()) & (spatial_diff_sig==1) & (mean_envs[cnt]>=min_envs)), spatial_diff_sig, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
		ax.text(115,-42,seasons[cnt])
		ax.set_xticklabels("")
		ax.set_yticklabels("")
		plt.tick_params(axis='both',which='both', bottom=False, top=False, labelbottom=False, left=False,\
			labelleft=False)
		cnt=cnt+1

	fig.text(0.08, 0.22, "e)")

	cax = plt.axes([0.2, 0.075, 0.6, 0.015])
	cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
	cb.set_label("Mean environmental frequency change per season (days)")
	
	fig.text(0.05, 0.3, "Mean environmental frequency change per month (days)", rotation=90)
	plt.subplots_adjust(hspace=0.4, top=0.95)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/percent_mean_change_"+v+".png")

def plot_lines(models, lr36_diff, mhgt_diff, ml_el_diff, qmean01_diff, srhe_left_diff, \
	Umean06_diff, logit_aws_diff):

	#Plot
	fig=plt.figure(figsize=[12,7])
	for m in np.arange(len(models)):
		ax = plt.subplot(3,4,m+1)
		ax2 = ax.twinx()
		l1=rolling(lr36_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l2=rolling(mhgt_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l3=rolling(ml_el_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l4=rolling(qmean01_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l5=rolling(srhe_left_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l6=rolling(Umean06_diff.loc[:, models[m][0]]).plot(ax=ax, legend=False)
		l7=rolling(logit_aws_diff.loc[:, models[m][0]]).plot(ax=ax2, legend=False, color="k")
		plt.axhline(0,linestyle=":",color="k")
		ax.set_ylim([-130, 130])
		ax2.set_ylim([-100, 100])
		plt.title(models[m][0])
		if m not in [0, 4, 8]:
			ax.set_yticklabels("")
		else:
			ax.set_yticks([-100,-50,0,50,100])
		if m not in [3, 7, 11]:
			ax2.set_yticklabels("")
		if m not in [8, 9, 10, 11]:
			ax2.set_xticklabels("")
		else:
			ax.set_xticks(np.arange(12))
			ax.set_xticklabels(return_months())
			ax.tick_params(rotation=45)
		if m == 11:
			fig.legend((l1.lines[0], l2.lines[1], l3.lines[2], l4.lines[3], l5.lines[4],\
				l6.lines[5], l7.lines[0]),\
				("Lapse rate 3-6 km", "Melting height", "Equilibrium level height", \
				"Water vapor mixing ratio 0-1 km", "Effective storm relative helicity",\
				"Mean wind speed 0-6 km", "SCW environments"),\
				loc=8, ncol=4)
		if m == 4:
			ax.set_ylabel("Percentage change in the mean\nbetween historical and RCP8.5")
		if m == 7:
			ax2.set_ylabel("Percentage change in daily occurrence\nfrequency between historical and RCP8.5")
	plt.subplots_adjust(bottom=0.15, top=0.95)

def rolling(x):
	temp = pd.concat([pd.DataFrame(data=[x.loc["Dec"]]), x], axis=0) 
	temp = pd.concat([temp, pd.DataFrame(data=[x.loc["Jan"]])], axis=0) 
	return temp.rolling(3).mean(center=True).iloc[1:-1] 

def return_months():
	return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def get_diff(hist, scenario, models, shapes, nrm, rel=True):
	
	months = return_months()
	mnames = [m[0] for m in models]
	df = pd.DataFrame(columns=mnames, index=months)
	hist[0]["nrm"] = rasterize(shapes, hist[0].coords)
	scenario[0]["nrm"] = rasterize(shapes, scenario[0].coords)
	for i in np.arange(len(hist)):
		for m in np.arange(len(months)):
			hist_reg = xr.where(hist[0]["nrm"].isin(nrm), hist[i][months[m]], np.nan).values
			scenario_reg = xr.where(scenario[0]["nrm"].isin(nrm), scenario[i][months[m]], np.nan).values
			if rel:
				temp = ((np.nanmean(scenario_reg) - np.nanmean(hist_reg)) /\
					np.nanmean(hist_reg))
				df.loc[months[m], mnames[i]] = temp * 100
			else:
				df.loc[months[m], mnames[i]] = (np.nanmean(scenario_reg) -\
					np.nanmean(hist_reg))
	return df

def get_spatial_diff(hist, scenario, models, season):

	spatial_diff = np.median(np.stack([scenario[i][season].values - hist[i][season].values for i in np.arange(len(models))]),axis=0) 
	spatial_diff_pos = (np.stack([scenario[i][season].values - hist[i][season].values for i in np.arange(len(models))]) > 0).sum(axis=0)
	spatial_diff_neg = (np.stack([scenario[i][season].values - hist[i][season].values for i in np.arange(len(models))]) <= 0).sum(axis=0)
	spatial_diff_sig = np.where( ( (spatial_diff > 0) & (spatial_diff_pos >= 10) ) | ( (spatial_diff <= 0) & (spatial_diff_neg >= 10) ), 1, 0)

	mme_mean_hist = np.mean(np.stack([hist[i][season].values for i in np.arange(len(models))]),axis=0)
	mme_mean_scenario = np.mean(np.stack([scenario[i][season].values for i in np.arange(len(models))]),axis=0)
	mean_envs = np.mean(np.stack([mme_mean_hist, mme_mean_scenario]), axis=0)
	
	return [spatial_diff, spatial_diff_sig, mean_envs]

if __name__ == "__main__":

	#Settings
	models = [["ACCESS1-3","r1i1p1",5,""] ,\
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
	hist_y1 = 1979; hist_y2 = 2005
	scenario_y1 = 2081; scenario_y2 = 2100
	lon1 = 112; lon2 = 156
	lat1 = -45; lat2 = -10
	experiment = "rcp85"
	v = "logit_aws"
	nrm = [0,1,2,3]	#0: Northern Australia;  1: Rangelands ; 2: Eastern Australia; 3: Suthern Australia

	#Get NRM shapes with geopandas
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]

	#Load mean netcdf files
	if v == "logit_aws":
		hist = get_seasonal_freq(models, v, hist_y1, hist_y2, None, None, "historical")
		scenario = get_seasonal_freq(models, v, scenario_y1, scenario_y2, None, None, experiment)
	else:
		hist = get_mean(models, v, hist_y1, hist_y2, None, None, "historical")
		scenario = get_mean(models, v, scenario_y1, scenario_y2, None, None, experiment)

	#Get pandas dataframes which summarise percentage changes for each month/model
	diff0 = get_diff(hist, scenario, models, shapes, [0], rel=False)
	diff1 = get_diff(hist, scenario, models, shapes, [1], rel=False)
	diff2 = get_diff(hist, scenario, models, shapes, [2], rel=False)
	diff3 = get_diff(hist, scenario, models, shapes, [3], rel=False)
	diff = [diff0, diff1, diff2, diff3]

	#The MME median spatial diff
	spatial_diff_djf, spatial_diff_sig_djf, mean_envs_djf = get_spatial_diff(hist, scenario, models, "DJF")
	spatial_diff_mam, spatial_diff_sig_mam, mean_envs_mam = get_spatial_diff(hist, scenario, models, "MAM")
	spatial_diff_jja, spatial_diff_sig_jja, mean_envs_jja = get_spatial_diff(hist, scenario, models, "JJA")
	spatial_diff_son, spatial_diff_sig_son, mean_envs_son = get_spatial_diff(hist, scenario, models, "SON")
	spatial_diffs = [spatial_diff_djf, spatial_diff_mam, spatial_diff_jja, spatial_diff_son]
	spatial_diff_sigs = [spatial_diff_sig_djf, spatial_diff_sig_mam, spatial_diff_sig_jja, spatial_diff_sig_son]
	mean_envs = [mean_envs_djf, mean_envs_mam, mean_envs_jja, mean_envs_son]

	#Raster info
	temp = hist[0]
	temp["nrm"] = rasterize(shapes, {"lon":temp.lon,"lat":temp.lat})
	temp["aus"] = rasterize([f2.loc[f2.name=="Australia"].geometry.values[0]], {"lon":temp.lon,"lat":temp.lat})

	plot_boxplot(v, models, diff, temp, spatial_diffs, spatial_diff_sigs, mean_envs, min_envs=1, plot_map=False)

