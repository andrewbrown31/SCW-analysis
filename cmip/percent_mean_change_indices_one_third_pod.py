import matplotlib
import matplotlib.lines as mlines
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

def get_seasonal_freq(models, p, y1, y2, era5_y1, era5_y2, threshold, experiment=""):

	mean_out = []
	for i in np.arange(len(models)):
		if models[i][0]=="ERA5":
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+str(threshold)+"_historical_"+\
			    str(era5_y1)+"_"+str(era5_y2)+".nc"))
		else:
			mean_out.append(xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
			    models[i][0]+"_"+models[i][1]+"_seasonal_freq_"+p+"_"+str(threshold)+"_"+experiment+"_"+str(y1)+"_"+str(y2)+".nc"))
	return mean_out

def plot_boxplot(v_list, models, data_list, cmip6, barpa, nrm_da, spatial_diff, spatial_diff_sig, mean_envs, ens_df, min_envs=0, plot_map=True):

	#min_envs is the minimum number of environments at a spatial point per season, required to draw statistical significance at that point.

	#Plot
	matplotlib.rcParams.update({'font.size': 12})
	fig=plt.figure(figsize=[8,10])
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	titles = ["a) Northern Australia", "b) Rangelands", "c) Eastern Australia", \
		    "d) Southern Australia"]
	ranges = [-8,6]
	ranges2 = [-48,36]
	#For each NRM region...
	for i in np.arange(len(data_list[0])):
		ax=plt.subplot2grid((4,4), (i, 0), colspan=4, rowspan=1)
		boxes = []
		#For each month...
		for month in return_months():
			#For each variable...
			for v in np.arange(len(v_list)+1):
				if v == len(v_list):
					temp = np.ones(12)*-999
				else:
					temp = data_list[v][i].loc[month].values
				boxes.append({"med":np.percentile(temp, 50, interpolation="nearest"),\
				    "q1":np.percentile(temp, 16.67, interpolation="nearest"),\
				    "q3":np.percentile(temp, 83.33, interpolation="nearest"),\
				    "whislo":temp.min(),"whishi":temp.max()})
		boxes.append({"med":np.percentile(ens_df[i], 50, interpolation="nearest"),\
			"q1":np.percentile(ens_df[i], 16.67, interpolation="nearest"),\
			"q3":np.percentile(ens_df[i], 83.33, interpolation="nearest"),\
			"whislo":ens_df[i].values.min(),"whishi":ens_df[i].values.max()})
		for v in np.arange(len(v_list)+1):
			if v < len(v_list):
				ax.plot(np.arange(0,12)*4 + (v+1), np.percentile(data_list[v][i], 50, interpolation="nearest", axis=1),lw=2, ls=":")
		box=ax.bxp(boxes[0:len(boxes)-1], showfliers=False)
		bcnt = 0
		for b in box["boxes"]:
			if bcnt%4 == 0:
				b.set_color("tab:blue")
			elif bcnt%4 == 1:
				b.set_color("tab:orange")
			elif bcnt%4 == 2:
				b.set_color("tab:green")
			bcnt=bcnt+1
		[b.set_color("k") for b in box["medians"]]

		ax2 = ax.twinx()
		box2=ax2.bxp([boxes[-1]], showfliers=False, positions=[50.5], widths=1)
		box2["boxes"][0].set_color("k")
		box2["medians"][0].set_color("k")
		#ax.grid(b=True, which="both", axis="both")
		ax.set_ylim(ranges)
		ax2.set_ylim(ranges2)
		plt.axhline(0,ls="--", color="k")
		if i not in [3]:
			ax.set_xticks(np.arange(0,13)*4 + 2.5)
			ax.set_xticklabels("")
		else:
			ax.set_xticks(np.arange(0,13)*4 + 2.5)
			ax.set_xticklabels(return_months() + ["Annual\n(all diagnostics)"], rotation=20)
		ax.set_xlim([0,52])
		#cmip6plot = []
		#for m in np.arange(12):
		#	for v in np.arange(len(v_list)):
		#		cmip6plot.append(cmip6[v][i].iloc[m,0])
		#plt.plot((np.arange(len(cmip6plot))+1)[0:-1:3],cmip6plot[0:-1:3], marker="o", color="tab:blue", linestyle="none")
		#plt.plot((np.arange(len(cmip6plot))+1)[1:-1:3],cmip6plot[1:-1:3], marker="o", color="tab:orange", linestyle="none")
		#plt.plot((np.arange(len(cmip6plot))+1)[2:-1:3],cmip6plot[2:-1:3], marker="o", color="tab:green", linestyle="none")
		#plt.plot((np.arange(len(cmip6plot))+1)[-1],cmip6plot[-1], marker="o", color="tab:green", linestyle="none")
		#if i == 2:
		#    plt.plot(np.arange(1,13),barpa[i].values[:,0], marker="o", color="tab:red", linestyle="none")
		if i == 0:
			l1 = mlines.Line2D([], [], color='tab:blue', label='T-Totals')
			l2 = mlines.Line2D([], [], color='tab:orange', label='SHERBE')
			l3 = mlines.Line2D([], [], color='tab:green', label='DCP')
			l4 = mlines.Line2D([], [], color='k', label='All')
			ax.legend(handles=[l1,l2,l3,l4], ncol=4, loc='lower right',fontsize=12)

		plt.title(titles[i])
		if plot_map:
			if i==0:
				inset = inset_locator.inset_axes(ax, width="10%", height="30%",loc=3)
			else:
				inset = inset_locator.inset_axes(ax, width="10%", height="30%",loc=3)
			xr.plot.contour((nrm_da["nrm"].isin([0,1,2,3])), levels=[0.5,1.5], colors="k", ax=inset, add_labels=False)
			xr.plot.contourf((nrm_da["nrm"].isin([i])), levels=[0,0.5], colors=["none","grey"],extend="max", ax=inset,\
			    add_labels=False, add_colorbar=False)
			plt.tick_params(axis='both',which='both', bottom=False, top=False, labelbottom=False, left=False,\
				labelleft=False)

	
	fig.text(0.025, 0.5, "Change in mean values\n(days with SCW environments per month)", rotation=90, va="center" , multialignment='center')
	fig.text(0.95, 0.5, "Change in mean values\n(days with SCW environments per year)", rotation=90, va="center", multialignment='center')
	plt.subplots_adjust(hspace=0.4, top=0.95, right=0.85, left=0.15)

	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/percent_mean_change_indices_one_third_pod.png")

def plot_spatial_diffs(v_list, models, nrm_da, spatial_diffs, spatial_diff_sigs, mean_envs, min_envs=1):
	seasons = ["DJF","MAM","JJA","SON"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	fig=plt.figure(figsize=[10,8])
	p_cnt=0
	for v in np.arange(len(v_list)):
		cnt=0
		for spatial_diff, spatial_diff_sig in zip(spatial_diffs[v], spatial_diff_sigs[v]):
			ax=plt.subplot(len(v_list),4,p_cnt+1)
			xr.plot.contour(xr.where(nrm_da["aus"]==1, 1, 0), levels=[0.5,1.5], colors="k", ax=ax, add_labels=False)
			#[xr.plot.contour(xr.where((nrm_da["nrm"]==i) & (~(nrm_da["aus"].isnull())), 1, 0), levels=[0.5,1.5], colors="k", ax=ax, add_labels=False) for i in [0,1,2,3]]
			spatial_diff = xr.DataArray(data=spatial_diff, coords=(nrm_da.lat, nrm_da.lon))
			spatial_diff_sig = xr.DataArray(data=spatial_diff_sig, coords=(nrm_da.lat, nrm_da.lon))
			c=xr.plot.contourf(xr.where((~(nrm_da["aus"].isnull())), spatial_diff, np.nan), cmap=plt.get_cmap("RdBu_r"), vmin=-12, vmax=12, extend="both", ax=ax, add_colorbar=False, add_labels=False)
			xr.plot.contourf(xr.where((~(nrm_da["aus"].isnull()) & (spatial_diff_sig==1) & (mean_envs[v][cnt]>=min_envs)), spatial_diff_sig, 0), colors="none", hatches=[None,"////"], levels=[0.5,1.5], add_colorbar=False, add_labels=False)
			ax.text(115,-42,alph[p_cnt])
			ax.set_xticklabels("")
			ax.set_yticklabels("")
			plt.tick_params(axis='both',which='both', bottom=False, top=False, labelbottom=False, left=False,\
				labelleft=False)
			if cnt == 0:
				if v_list[v] == "mu_cape":
					plt.ylabel("MUCAPE")
				elif v_list[v] == "t_totals":
					plt.ylabel("T-Totals")
				elif v_list[v] == "eff_sherb":
					plt.ylabel("SHERBE")
				elif v_list[v] == "cs6":
					plt.ylabel("MUCS6")
				else:
					plt.ylabel(v_list[v].upper())
			if v == 0:
				plt.title(seasons[cnt])
			cnt=cnt+1
			p_cnt=p_cnt+1

	cax = plt.axes([0.2, 0.075, 0.6, 0.015])
	cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
	cb.set_label("Mean environmental frequency change per season (days)")
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/percent_mean_change_indices_map_one_third_pod.png")

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

def get_diff(hist, scenario, models, shapes, nrm, rel=True, annual=False, var=None):
	
	if not annual:
		months = return_months()
		mnames = [m[0] for m in models]
	else:
		months = [var]
		mnames = [m[0] for m in models]
	df = pd.DataFrame(columns=mnames, index=months)
	hist[0]["nrm"] = rasterize(shapes, hist[0].coords)
	scenario[0]["nrm"] = rasterize(shapes, scenario[0].coords)
	for i in np.arange(len(hist)):
		for m in np.arange(len(months)):
			#Take the mean seasonal frequency across the nrm region, and then take the historical-scenario difference
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

	mean_envs = np.median(np.stack([scenario[i][season].values for i in np.arange(len(models))]),axis=0)
	
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
	v_list = ["t_totals","eff_sherb","dcp"]
	nrm = [0,1,2,3]	#0: Northern Australia;  1: Rangelands ; 2: Eastern Australia; 3: Suthern Australia

	#Get NRM shapes with geopandas
	f2 = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
	f = geopandas.read_file("/home/548/ab4502/NRM_super_clusters/NRM_super_clusters.shp")
	shapes = [(shape, n) for n, shape in enumerate(f.geometry)]

	#Load mean netcdf files
	hist = []; scenario = []; hist_cmip6 = []; scenario_cmip6 = []
	diff = []; diff_cmip6 = []
	spatial_diffs = []; spatial_diff_sigs = []; mean_envs = []
	cnt = 0

	#Set up dataframe for total change (percentage change for all months)
	change0_rel = pd.DataFrame()
	change1_rel = pd.DataFrame()
	change2_rel = pd.DataFrame()
	change3_rel = pd.DataFrame()
	change0_abs = pd.DataFrame()
	change1_abs = pd.DataFrame()
	change2_abs = pd.DataFrame()
	change3_abs = pd.DataFrame()
	thresh = {"dcp":0.91, "t_totals":50.9, "eff_sherb":0.90}
	for v in v_list:
		hist.append(get_seasonal_freq(models, v, hist_y1, hist_y2, None, None, thresh[v], "historical"))
		scenario.append(get_seasonal_freq(models, v, scenario_y1, scenario_y2, None, None, thresh[v], experiment))
		#hist_cmip6.append(get_seasonal_freq([["ACCESS-CM2","r1i1p1f1",6,""]], v, hist_y1, hist_y2, None, None, "historical"))
		#scenario_cmip6.append(get_seasonal_freq([["ACCESS-CM2","r1i1p1f1",6,""]], v, scenario_y1, scenario_y2, None, None, "ssp585"))
		#hist_barp = get_seasonal_freq([["BARPA",""]], v, hist_y1, hist_y2, None, None, "historical")
		#scenario_barp = get_seasonal_freq([["BARPA",""]], v, 2080, 2099, None, None, "rcp85")

		#Get pandas dataframes which summarise percentage changes for each month/model
		diff0 = get_diff(hist[cnt], scenario[cnt], models, shapes, [0], rel=False)
		diff1 = get_diff(hist[cnt], scenario[cnt], models, shapes, [1], rel=False)
		diff2 = get_diff(hist[cnt], scenario[cnt], models, shapes, [2], rel=False)
		diff3 = get_diff(hist[cnt], scenario[cnt], models, shapes, [3], rel=False)
		diff.append([diff0, diff1, diff2, diff3])

		#Get pandas dataframes which summarise percentage changes for each month/model
		#diff0 = get_diff(hist_cmip6[cnt], scenario_cmip6[cnt], [["ACCESS-CM2","r1i1p1f1",6,""]], shapes, [0], rel=False)
		#diff1 = get_diff(hist_cmip6[cnt], scenario_cmip6[cnt], [["ACCESS-CM2","r1i1p1f1",6,""]], shapes, [1], rel=False)
		#diff2 = get_diff(hist_cmip6[cnt], scenario_cmip6[cnt], [["ACCESS-CM2","r1i1p1f1",6,""]], shapes, [2], rel=False)
		#diff3 = get_diff(hist_cmip6[cnt], scenario_cmip6[cnt], [["ACCESS-CM2","r1i1p1f1",6,""]], shapes, [3], rel=False)
		#diff_cmip6.append([diff0, diff1, diff2, diff3])

		#Same but for BARPA
		#diff_barp = [None, None, get_diff(hist_barp, scenario_barp, [["BARPA",""]], shapes, [2], rel=False), None]

		#The MME median spatial diff
		spatial_diff_djf, spatial_diff_sig_djf, mean_envs_djf = get_spatial_diff(hist[cnt], scenario[cnt], models, "DJF")
		spatial_diff_mam, spatial_diff_sig_mam, mean_envs_mam = get_spatial_diff(hist[cnt], scenario[cnt], models, "MAM")
		spatial_diff_jja, spatial_diff_sig_jja, mean_envs_jja = get_spatial_diff(hist[cnt], scenario[cnt], models, "JJA")
		spatial_diff_son, spatial_diff_sig_son, mean_envs_son = get_spatial_diff(hist[cnt], scenario[cnt], models, "SON")
		spatial_diffs.append([spatial_diff_djf, spatial_diff_mam, spatial_diff_jja, spatial_diff_son])
		spatial_diff_sigs.append([spatial_diff_sig_djf, spatial_diff_sig_mam, spatial_diff_sig_jja, spatial_diff_sig_son])
		mean_envs.append([mean_envs_djf, mean_envs_mam, mean_envs_jja, mean_envs_son])

		change0_rel = pd.concat([change0_rel, get_diff(hist[cnt], scenario[cnt], models, shapes, [0], rel=True, annual=True, var=v_list[cnt])], axis=0)
		change1_rel = pd.concat([change1_rel, get_diff(hist[cnt], scenario[cnt], models, shapes, [1], rel=True, annual=True, var=v_list[cnt])], axis=0)
		change2_rel = pd.concat([change2_rel, get_diff(hist[cnt], scenario[cnt], models, shapes, [2], rel=True, annual=True, var=v_list[cnt])], axis=0)
		change3_rel = pd.concat([change3_rel, get_diff(hist[cnt], scenario[cnt], models, shapes, [3], rel=True, annual=True, var=v_list[cnt])], axis=0)
		change0_abs = pd.concat([change0_abs, get_diff(hist[cnt], scenario[cnt], models, shapes, [0], rel=False, annual=True, var=v_list[cnt])], axis=0)
		change1_abs = pd.concat([change1_abs, get_diff(hist[cnt], scenario[cnt], models, shapes, [1], rel=False, annual=True, var=v_list[cnt])], axis=0)
		change2_abs = pd.concat([change2_abs, get_diff(hist[cnt], scenario[cnt], models, shapes, [2], rel=False, annual=True, var=v_list[cnt])], axis=0)
		change3_abs = pd.concat([change3_abs, get_diff(hist[cnt], scenario[cnt], models, shapes, [3], rel=False, annual=True, var=v_list[cnt])], axis=0)
	    
		cnt=cnt+1

	#Raster info
	temp = hist[0][0]
	temp["nrm"] = rasterize(shapes, {"lon":temp.lon,"lat":temp.lat})
	temp["aus"] = rasterize([f2.loc[f2.name=="Australia"].geometry.values[0]], {"lon":temp.lon,"lat":temp.lat})

	#Save change dataframes
	change_rel = pd.DataFrame({\
		"NA":[np.quantile(np.array(change0_rel), 0.1), np.quantile(np.array(change0_rel), 0.5),np.quantile(np.array(change0_rel), 0.9)],\
		"RA":[np.quantile(np.array(change1_rel), 0.1), np.quantile(np.array(change1_rel), 0.5),np.quantile(np.array(change1_rel), 0.9)],\
		"EA":[np.quantile(np.array(change2_rel), 0.1), np.quantile(np.array(change2_rel), 0.5),np.quantile(np.array(change2_rel), 0.9)],\
		"SA":[np.quantile(np.array(change3_rel), 0.1), np.quantile(np.array(change3_rel), 0.5),np.quantile(np.array(change3_rel), 0.9)]})
	change_rel.to_csv("/g/data/eg3/ab4502/figs/CMIP/percent_nrm_changes_one_third_pod.csv")
	change_abs = pd.DataFrame({\
		"NA":[np.quantile(np.array(change0_abs), 0.1), np.quantile(np.array(change0_abs), 0.5),np.quantile(np.array(change0_abs), 0.9)],\
		"RA":[np.quantile(np.array(change1_abs), 0.1), np.quantile(np.array(change1_abs), 0.5),np.quantile(np.array(change1_abs), 0.9)],\
		"EA":[np.quantile(np.array(change2_abs), 0.1), np.quantile(np.array(change2_abs), 0.5),np.quantile(np.array(change2_abs), 0.9)],\
		"SA":[np.quantile(np.array(change3_abs), 0.1), np.quantile(np.array(change3_abs), 0.5),np.quantile(np.array(change3_abs), 0.9)]})
	change_abs.to_csv("/g/data/eg3/ab4502/figs/CMIP/abs_nrm_changes_one_third_pod.csv")

	#Plot
	plot_boxplot(v_list, models, diff, None, None, temp, spatial_diffs, spatial_diff_sigs, mean_envs,\
		[change0_abs, change1_abs, change2_abs, change3_abs], min_envs=1, plot_map=True)
	plot_spatial_diffs(v_list, models, temp, spatial_diffs, spatial_diff_sigs, mean_envs, min_envs=1)


