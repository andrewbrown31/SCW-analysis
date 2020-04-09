import numpy as np
from cmip_scenario import get_mean, get_daily_freq
import pandas as pd
import matplotlib.pyplot as plt

#For each ingredient of the logit_aws equation, plot the percentage change across a region for each month.
#On a different y-axis, plot the percentage change in environment occurrence frequency.
#Include options to specify the region with a lat/lon bounding box (defaults to whole of Australia).
#Do this for each model.

def rolling(x):
	temp = pd.concat([pd.DataFrame(data=[x.loc["Dec"]]), x], axis=0) 
	temp = pd.concat([temp, pd.DataFrame(data=[x.loc["Jan"]])], axis=0) 
	return temp.rolling(3).mean(center=True).iloc[1:-1] 

def return_months():
	return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def get_diff(hist, scenario, models, lat1, lat2, lon1, lon2):
	
	months = return_months()
	mnames = [m[0] for m in models]
	df = pd.DataFrame(columns=mnames, index=months)
	for i in np.arange(len(hist)):
		for m in np.arange(len(months)):
			hist_reg = hist[i][months[m]].sel({"lat":slice(lat2,lat1),\
				    "lon":slice(lon1,lon2)}).values
			scenario_reg = scenario[i][months[m]].sel({"lat":slice(lat2,lat1), \
				    "lon":slice(lon1,lon2)}).values
			temp = ((np.nanmean(scenario_reg) - np.nanmean(hist_reg)) / np.nanmean(hist_reg))
			df.loc[months[m], mnames[i]] = temp * 100
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
	logit_aws_hist = get_daily_freq(models, "logit_aws", hist_y1, hist_y2, None, None, "historical")
	logit_aws_scenario = get_daily_freq(models, "logit_aws", scenario_y1, scenario_y2, None, None, experiment)

	#Get pandas dataframes which summarise percentage changes for each month/model
	lr36_diff = get_diff(lr36_hist, lr36_scenario, models, lat1, lat2, lon1, lon2)
	mhgt_diff = get_diff(mhgt_hist, mhgt_scenario, models, lat1, lat2, lon1, lon2)
	ml_el_diff = get_diff(ml_el_hist, ml_el_scenario, models, lat1, lat2, lon1, lon2)
	qmean01_diff = get_diff(qmean01_hist, qmean01_scenario, models, lat1, lat2, lon1, lon2)
	srhe_left_diff = get_diff(srhe_left_hist, srhe_left_scenario, models, lat1, lat2, lon1, lon2)
	Umean06_diff = get_diff(Umean06_hist, Umean06_scenario, models, lat1, lat2, lon1, lon2)
	logit_aws_diff = get_diff(logit_aws_hist, logit_aws_scenario, models, lat1, lat2, lon1, lon2)

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



