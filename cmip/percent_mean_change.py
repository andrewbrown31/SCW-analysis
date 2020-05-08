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

def plot_boxplot(models, lr36_diff, mhgt_diff, ml_el_diff, qmean01_diff, srhe_left_diff, \
	Umean06_diff, logit_aws_diff, nrm_da, nrm, plot_map=True):

	#Plot
	fig=plt.figure(figsize=[12,7])
	data_list = [lr36_diff, mhgt_diff, ml_el_diff, qmean01_diff, srhe_left_diff,\
		Umean06_diff, logit_aws_diff]
	loc = [ [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0] ]
	if plot_map:
		rs = [1,1,1,1,1,1,2]
	else:
		rs = [1,1,1,1,1,1,3]
	titles = ["a) Lapse rate 3-6 km", "b) Melting height", "c) Equilibrium level height", \
			"d) Water vapor mixing\nratio 0-1 km", "e) Effective storm\nrelative helicity",\
			"f) Mean wind speed 0-6 km", "g) SCW environments"]
	ylabels = ["$DegC.km^{-1}$", "$m$", "$m$", "$g.kg^{-1}$", "$m^2$.$s^{-2}$", "$m.s^{-1}$",\
		    "Environments per\nmonth"]
	ranges = [ [-1.25,1.25], [-1400,1400], [-2000,2000], [-4,4], [-10,10], [-3.5,3.5], [-5.5,5.5] ]
	for i in np.arange(len(data_list)):
		ax=plt.subplot2grid([3,3], loc[i], colspan=rs[i])
		boxes = []
		for month in return_months():
			temp = data_list[i].loc[month].values
			boxes.append({"med":np.percentile(temp, 50, interpolation="nearest"),\
			    "q1":np.percentile(temp, 16.67, interpolation="nearest"),\
			    "q3":np.percentile(temp, 83.33, interpolation="nearest"),\
			    "whislo":temp.min(),"whishi":temp.max()})
		ax.bxp(boxes, showfliers=False)
		plt.ylim(ranges[i])
		plt.axhline(0,ls="--", color="k")
		if i not in [3,4,5,6]:
			ax.set_xticklabels("")
		else:
			ax.set_xticks(np.arange(1,13))
			ax.set_xticklabels(return_months())
			if i in [3,4,5]:
				ax.tick_params("x",rotation=60)
		ax.tick_params("y",rotation=45)

		plt.title(titles[i])
		plt.ylabel(ylabels[i])
	if plot_map:
		ax=plt.subplot2grid([3,3], [2,2])
		plt.contourf(np.flipud(nrm_da.isin([0,1,2,3]).values*1), levels=[0.5,1.5], colors="grey")
		plt.contourf(np.flipud(nrm_da.isin(nrm).values*1), levels=[0.5,1.5], colors="grey", hatches=["/////"])
		plt.tick_params(axis='both',which='both', bottom=False, top=False, labelbottom=False, left=False,\
			labelleft=False)
	
	plt.subplots_adjust(hspace=0.65, wspace=0.4)

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
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/percentage_mean_change.png")

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
	nrm = [2,3]	#0: Northern Australia;  1: Rangelands ; 2: Southern Australia; 3: Eastern Australia

	#Get NRM shapes with geopandas
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]

	#Load mean netcdf files
	lr36_hist = get_mean(models, "lr36", hist_y1, hist_y2, None, None, "historical")
	lr36_scenario = get_mean(models, "lr36", scenario_y1, scenario_y2, None, None, experiment)
	mhgt_hist = get_mean(models, "mhgt", hist_y1, hist_y2, None, None, "historical")
	mhgt_scenario = get_mean(models, "mhgt", scenario_y1, scenario_y2, None, None, experiment)
	ml_el_hist = get_mean(models, "ml_el", hist_y1, hist_y2, None, None, "historical")
	ml_el_scenario = get_mean(models, "ml_el", scenario_y1, scenario_y2, None, None, experiment)
	qmean01_hist = get_mean(models, "qmean01", hist_y1, hist_y2, None, None, "historical")
	qmean01_scenario = get_mean(models, "qmean01", scenario_y1, scenario_y2, None, None, experiment)
	srhe_left_hist = get_mean(models, "srhe_left", hist_y1, hist_y2, None, None, "historical")
	srhe_left_scenario = get_mean(models, "srhe_left", scenario_y1, scenario_y2, None, None, experiment)
	Umean06_hist = get_mean(models, "Umean06", hist_y1, hist_y2, None, None, "historical")
	Umean06_scenario = get_mean(models, "Umean06", scenario_y1, scenario_y2, None, None, experiment)
	logit_aws_hist = get_seasonal_freq(models, "logit_aws", hist_y1, hist_y2, None, None, "historical")
	logit_aws_scenario = get_seasonal_freq(models, "logit_aws", scenario_y1, scenario_y2, None, None, experiment)

	#Get pandas dataframes which summarise percentage changes for each month/model
	lr36_diff = get_diff(lr36_hist, lr36_scenario, models, shapes, nrm, rel=False)
	mhgt_diff = get_diff(mhgt_hist, mhgt_scenario, models, shapes, nrm, rel=False)
	ml_el_diff = get_diff(ml_el_hist, ml_el_scenario, models, shapes, nrm, rel=False)
	qmean01_diff = get_diff(qmean01_hist, qmean01_scenario, models, shapes, nrm, rel=False)
	srhe_left_diff = get_diff(srhe_left_hist, srhe_left_scenario, models, shapes, nrm, rel=False)
	Umean06_diff = get_diff(Umean06_hist, Umean06_scenario, models, shapes, nrm, rel=False)
	logit_aws_diff = get_diff(logit_aws_hist, logit_aws_scenario, models, shapes, nrm, rel=False)

	plot_boxplot(models, lr36_diff, mhgt_diff, ml_el_diff, qmean01_diff, srhe_left_diff, \
	    Umean06_diff, logit_aws_diff, rasterize(shapes, lr36_hist[0])["nrm"], nrm, plot_map=True)

	plt.savefig("nrm_projections.png", bbox_inches="tight")
