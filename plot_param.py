from barra_read import date_seq
import numpy as np
import datetime as dt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import netCDF4 as nc
import matplotlib.animation as animation
import os
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from obs_read import load_lightning, analyse_events
import pandas as pd
from event_analysis import bootstrap_slope
import seaborn as sb

def three_month_average(a):

	#Takes an array of length 12 (months) and returns 3-month rolling avg	
	
	assert (len(a) == 12), "ARRAY MUST BE LENGTH 12"

	rolling = np.zeros(len(a))

	for i in np.arange(12):
		if i == 0:
			rolling[i] = (a[-1] + a[i] + a[i+1]) / 3.
		elif i == 11:
			rolling[i] = (a[i-1] + a[i] + a[0]) / 3.
		else:
			rolling[i] = (a[i-1] + a[i] + a[i+1]) / 3.
	return rolling

def sta_versus_aws():

	#Plot spatially smoothed maps of STA wind reports and AWS + lightning

	from scipy.ndimage.filters import gaussian_filter as filter

	df = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/convective_wind_gust_aus_2005_2015.pkl")

	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl");\
	 
	plt.subplot(131);\
	plt.title("STA REPORTS NEAR AWS STATIONS\nRELATIVE FREQUENCY");\
	d,y,x=np.histogram2d(df[~(df.sta_wind.isna()) & (df.tc_affected==0)].lat,df[~(df.sta_wind.isna()) & (df.tc_affected==0)].lon,bins=20,range=([-45,-10],[110,160]));\
	x = np.array([(x[i-1] + x[i]) / 2. for i in np.arange(1,len(x))])
	y = [(y[i-1] + y[i]) / 2. for i in np.arange(1,len(y))]
	x,y=np.meshgrid(x,y);\
	m.pcolor(x,y,filter(d,1) / filter(d,1).sum(), vmin=0, vmax=0.025);\
	plt.colorbar()
	m.drawcoastlines()
	 
	plt.subplot(132);\
	plt.title("AWS (OVER 25 m/s) + LIGHTNING\nRELATIVE FREQUENCY");\
	d_aws,t1,t2=np.histogram2d(df[(df.lightning>=2) & (df.wind_gust>=25) & (df.tc_affected==0)].lat,df[(df.lightning>=2) & (df.wind_gust>=25) &(df.tc_affected==0)].lon,bins=20,range=([-45,-10],[110,160]));\
	m.pcolor(x,y,filter(d_aws,1) / filter(d_aws,1).sum(), vmin=0, vmax=0.025);\
	plt.colorbar()
	m.drawcoastlines();\
	plt.subplot(133);\
	plt.title("AWS - STA\nDIFFERENCE IN RELATIVE FREQUENCY")
	m.pcolor(x,y,(filter(d_aws,1) / filter(d_aws,1).sum()) - (filter(d,1) / filter(d,1).sum()), vmin=-0.01, vmax=0.01, cmap=plt.get_cmap("RdBu_r"));\
	m.drawcoastlines();
	plt.colorbar()
	 
	plt.show()
 
def temporal_dist_plots():
	#New function to plot diurnal distribution of extreme convective gusts, using the "time of max gust" 
	# daily data in combination with 6 hourly lightning data

	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_6hrly_aus_1979_2017.pkl")
	lightning = load_lightning(daily=False,smoothing=True).reset_index()
	lightning["lightning_hour"] = [lightning.date[i].hour for i in np.arange(lightning.shape[0])]

	aws = aws.set_index(["an_gust_time_utc","stn_name"])
	lightning = lightning.set_index(["date","loc_id"])

	df = pd.concat([aws,lightning.lightning],axis=1).dropna(subset=["wind_gust"])
	df_lightning = df.dropna(subset=["lightning"])

	l = 30
	ymax=50
	plt.figure()
	plt.subplot(221);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat > -l) & (df_lightning.tc_affected==0)].month.hist();plt.ylim([0,ymax]);\
		plt.title("Convective gusts greater than 25 m/s \nfor stations north of "+str(l)+"$^{o}$S") 
	plt.subplot(222);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat < -l) & (df_lightning.tc_affected==0)].month.hist();plt.ylim([0,ymax]);\
		plt.title("Convective gusts greater than 25 m/s \nfor stations south of "+str(l)+"$^{o}$S") 
	plt.subplot(223);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat > -l) & (df_lightning.tc_affected==0)].hour.hist();plt.ylim([0,ymax]);\
	plt.subplot(224);df_lightning[(df_lightning.lightning>=2) & (df_lightning.wind_gust>=25) & \
		(df_lightning.lat < -l) & (df_lightning.tc_affected==0)].hour.hist();plt.ylim([0,ymax]);\

def probability_plot(param1,param2):

	#Plot a 2d probability plot for two parameters, as in Taszarek (2017) Fig 10 and Brooks 2013 (Fig 5)
	#Currently just for ERA-Interim on aus domain

	df = analyse_events(domain="aus",event_type="aws",model="erai")

	#Convert CAPE to WMAX
	if param1 in ["ml_cape","mu_cape","dcape"]:
		df[param1] = np.power( df[param1] * 2, 0.5)
	if param2 in ["ml_cape","mu_cape","dcape"]:
		df[param2] = np.power( df[param2] * 2, 0.5)

	#Define "convective" or "thunderstorm" events by when daily maximum lightning over a ~1 degree area around each 
	# AWS wind station is greater than or equal to 2. Define convective wind events as above but when a gust over 25 m/s is 
	# recorded
	conv_df = df[(df["lightning"]>=2)].dropna(subset=["lightning","wind_gust",param1])
	conv_wind_df = df[(df["lightning"]>=2) & (df["wind_gust"]>=25)].dropna(subset=["lightning","wind_gust",param1])

	#Bin data and get centre points
	xbins = np.linspace(df[param1].min(),df[param1].max(),20)
	ybins = np.linspace(df[param2].min(),df[param2].max(),20)
	h_wind, xe, ye = np.histogram2d(conv_wind_df[param1], conv_wind_df[param2], bins = [xbins,ybins])
	h_conv, xe, ye = np.histogram2d(conv_df[param1], conv_df[param2], bins = [xbins, ybins])
	h_all, xe, ye = np.histogram2d(df[param1], df[param2], bins = [xbins, ybins])
	x = [(xe[i] + xe[i]) /2. for i in np.arange(0,len(xe)-1)] 
	y = [(ye[i] + ye[i]) /2. for i in np.arange(0,len(ye)-1)] 
	xg,yg = np.meshgrid(x,y)

	#Smooth binned data converted to probability, using a gaussian filter
	from scipy.ndimage.filters import gaussian_filter
	#sigx = int((conv_df[param1].max() - conv_df[param1].min()) / 20.)
	#sigy = int((conv_df[param2].max() - conv_df[param2].min()) / 20.)
	sigx = 1
	sigy = 1
	h_wind_smooth = gaussian_filter(h_wind.T, (sigy, sigx))
	h_conv_smooth = gaussian_filter(h_conv.T, (sigx, sigy))
	h_all_smooth = gaussian_filter(h_all.T, (sigx, sigy))

	#Plot
	plt.subplot(221);
	plt.contourf(xg,yg,h_conv_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar();
	plt.title("Convective events")
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r",s=4)
	plt.ylabel(param2)

	plt.subplot(222);
	plt.contourf(xg,yg,h_wind_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar();plt.title("Convective wind events (>25m/s)");
	plt.xlabel(param1);
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r",s=4)

	plt.subplot(223);
	plt.contourf(xg,yg,(h_wind_smooth / h_conv_smooth),\
		levels=np.linspace(0,1,11),extend="max",cmap=plt.get_cmap("Greys"));
	plt.colorbar()
	plt.title("\nProbability of convective wind event given convective event");plt.xlabel(param1)

	plt.subplot(224);
	plt.contourf(xg,yg,h_all_smooth,cmap=plt.get_cmap("Greys"));
	plt.colorbar()
	plt.scatter(conv_wind_df[param1],conv_wind_df[param2],color="r", s=4)
	plt.title("All (convective and non convective) events")
	
	plt.show()
	

def var_trends(df,var,loc,method,dataset,year_range=[1979,2017],months=np.arange(1,13,1),percentile=0,threshold=0):

	#Load long-term daily max data, and look for trends in "var"
	#Df should have a reset index upon parsing
	#Df should have column name for station name as "stn_name"

	#method is either:
	#	- percebtile: Plot both the monthly mean time series, and model a linear best fit of
	#		annual frequency exceeding 99th percentile
	#	- amean: Fit a linear best fit model to the annual mean
	#	- amax: Fit a linear best fit model to the annual max
	#	- threshold: Same as percentile, except plot monthly max time series, and linear 
	#		best fit/counts of being within a range of "thresholds". 
	#		"thresholds" is a list of lists, where each list is either one integer 
	#		(to be exceeded) or two	increasing integers (to be between)
	#	- threshold_only: Same as threshold, except don't plot monthly max time series

	df = df.dropna(subset=[var])

	if (loc == "Adelaide AP") & (var == "wind_gust"):
		vplot = [dt.datetime(1988,10,21),dt.datetime(2002,12,17)]
	elif (loc == "Woomera") & (var == "wind_gust"):
		vplot = [dt.datetime(1991,5,16)]
	elif (loc == "Mount Gambier") & (var == "wind_gust"):
		#Note that trees and bushes have been growing on S side of anemometer for a 20 yr
		# period
		vplot = [dt.datetime(1993,7,5),dt.datetime(1995,5,25)]
	elif (loc == "Port Augusta") & (var == "wind_gust"):
		vplot = [dt.datetime(1997,6,12),dt.datetime(2001,7,2),dt.datetime(2014,2,20)]
	elif (loc == "Edinburgh") & (var == "wind_gust"):
		vplot = [dt.datetime(1993,5,4)]
	elif (loc == "Parafield") & (var == "wind_gust"):
		vplot = [dt.datetime(1992,7,22),dt.datetime(2006,6,27)]
	else:
		year_range = [1979,2017]
		vplot = []

	df = df[np.in1d(df.month,months) & (df.stn_name==loc) & (df.year>=year_range[0]) & \
		(df.year<=year_range[1])].sort_values("date").reset_index()

	if method=="threshold":
		years = df.year.unique()
		fig,[ax1,ax3]=plt.subplots(figsize=[10,8],nrows=2,ncols=1)
		ax1.set_ylabel("Wind gust (m.s$^{-1}$)",fontsize="xx-large")
		df_monthly = df.set_index("date").resample("1M").max()
		y = np.array([dt.datetime(t.year,t.month,t.day) for t in df_monthly.index])
		ax1.plot(y,df_monthly[var])
		ax2 = ax1.twinx()
		cnt = [df[(df.stn_name==loc)&(df.year==y)].shape[0] for y in np.arange(1979,2018,1)]
		y = np.array([dt.datetime(y,6,1) for y in np.arange(1979,2018,1)])
		ax2.plot(y,cnt,color="k",marker="s",linestyle="none",markersize=8)
		ax2.set_ylabel("Obs. per year",fontsize="xx-large")
		[plt.axvline(v,color="k",linestyle="--") for v in vplot]
		plt.xlabel("")
		cnt=0
		for thresh in threshold:
			if (len(thresh) == 1):
				event_years = [df[(df[var]>=thresh[0]) & (df.year == y)].shape[0] \
					for y in years]
				lab1 = "Over "+str(thresh[0])+" m/s: "
			elif (len(thresh) == 2):
				event_years = [df[(df[var]>=thresh[0]) & (df[var]<thresh[1]) & \
					(df.year == y)].shape[0] for y in years]
				lab1 = "Between "+str(thresh[0])+" - "+str(thresh[1])+" m/s: "

			events = pd.DataFrame({"years":years,"event_years":event_years})
			df_monthly = df.set_index("date").resample("1M").mean()

			x = events.years
			y = events.event_years
			m,std = bootstrap_slope(x.values,y.values,1000)

			lab = lab1+" "+str(round(m,3))+" +/- "+\
				str(round(std*2,3))+" d y^-1"
			if cnt == 2:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,color="k",\
					n_boot=10000,order=1,label=lab,ax=ax3,marker="x",\
					scatter_kws={"s":150,"linewidth":1.5})
			else:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax3,marker="s",\
					scatter_kws={"s":75})
			cnt=cnt+1
		ax1.set_xlim([dt.datetime(years.min(),1,1),dt.datetime(years.max(),12,31)])
		ax3.set_xlim([events.years.min()-1,events.years.max()+1])
		ax3.set_ylim([0.5,1000])
		ax3.set_ylabel("Days",fontsize=fs)
		ax3.set_xlabel("Year",fontsize=fs)
		ax3.set_yscale('log')
		ax1.set_title(loc,fontsize=fs)
		ax3.tick_params(labelsize=fs)
		ax2.tick_params(labelsize=fs)
		ax1.tick_params(labelsize=fs)
	if method=="threshold_only":
		fs=35
		if dataset=="ERA-Interim":
			ssize=150
			xticklabs = np.arange(1980, 2020, 5)
		elif dataset=="BARRA-R":
			ssize=500
			xticklabs = np.array([2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016])
		elif dataset=="BARRA-AD":
			ssize=500
			xticklabs = np.array([2006, 2008, 2010, 2012, 2014, 2016])
		else:
			raise ValueError("Incorrect Dataset")
		years = df.year.unique()
		fig,ax=plt.subplots(figsize=[12,5])
		cnt = [df[(df.stn_name==loc)&(df.year==y)].shape[0] for y in np.arange(1979,2018,1)]
		y = [dt.datetime(y,6,1) for y in np.arange(1979,2018,1)]
		cnt=0
		for thresh in threshold:
			if (len(thresh) == 1):
				event_years = [df[(df[var]>=thresh[0]) & (df.year == y)].shape[0] \
					for y in years]
				lab1 = "Over "+str(thresh[0])+" m/s: "
			elif (len(thresh) == 2):
				event_years = [df[(df[var]>=thresh[0]) & (df[var]<thresh[1]) & \
					(df.year == y)].shape[0] for y in years]
				lab1 = "Between "+str(thresh[0])+" - "+str(thresh[1])+" m/s: "

			events = pd.DataFrame({"years":years,"event_years":event_years})
			df_monthly = df.set_index("date").resample("1M").mean()

			x = events.years
			y = events.event_years
			m,std = bootstrap_slope(x.values,y.values,1000)

			lab = lab1+" "+str(round(m,3))+" +/- "+\
				str(round(std*2,3))+" d y^-1"
			if cnt==2:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax,marker="x",color="k",\
					scatter_kws={"s":ssize,"linewidth":2})
			else:
				sb.regplot("years","event_years",events,ci=95,fit_reg=False,\
					n_boot=10000,order=1,label=lab,ax=ax,marker="s",\
					scatter_kws={"s":ssize})
			cnt=cnt+1
		if dataset == "BARRA-AD":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		elif dataset == "BARRA-R":
			ax.set_xlim([events.years.min()-1,events.years.max()-0.5])
		else:
			ax.set_xlim([events.years.min()-1,events.years.max()+1])
		ax.set_ylim([0.5,1000])
		ax.set_ylabel("Days",fontsize=fs)
		ax.set_yscale('log')
		ax.tick_params(labelsize=fs)
		ax.set_title(loc,fontsize=fs)
		ax.set_xlabel("")
		for label in ax.get_xticklabels():
        		label.set_rotation(90) 
		plt.grid()
		plt.xticks(xticklabs, xticklabs.astype(str))
	if method=="percentile":
		years = df.year.unique()
		thresh = np.percentile(df[var],percentile)
		event_years = [df[(df[var]>=thresh) & (df.year == y)].shape[0] for y in\
			 years]

		events = pd.DataFrame({"years":years,"event_years":event_years})
		df_monthly = df.set_index("date").resample("1M").mean()

		x = events.years
		y = events.event_years
		m,std = bootstrap_slope(x.values,y.values,1000)

		plt.figure(figsize=[12,8]);plt.subplot(211)
		df_monthly[~df_monthly[var].isna()][var].plot()
		plt.ylabel("Monthly mean wind gust (m/s)")
		plt.subplot(212)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" d y^-1"
		sb.regplot("years","event_years",events,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.xlim([events.years.min(),events.years.max()])
		plt.ylabel("Days exceeding "+str(percentile)+" percentile")
		plt.suptitle(loc)
		plt.legend(fontsize=10)
			
	elif method=="amean":
		plt.figure(figsize=[12,8])
		df_yearly = df.set_index("date").resample("1Y").mean()
		y = df_yearly[var]
		x = df_yearly.year
		m,std = bootstrap_slope(x,y,1000)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" units y^-1"
		sb.regplot("year",var,df_yearly,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.legend(fontsize=10)
	elif method=="amax":
		plt.figure(figsize=[12,8])
		df_yearly = df.set_index("date").resample("1Y").max()
		y = df_yearly[var]
		x = df_yearly.year
		m,std = bootstrap_slope(x,y,1000)
		lab = str(round(m,3))+" +/- "+str(round(std*2,3))+" units y^-1"
		sb.regplot("year",var,df_yearly,ci=95,fit_reg=True,\
				n_boot=10000,order=1,label=lab)
		plt.legend(fontsize=10)
	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+\
		dataset+"_"+loc+"_"+var+"_"+method+"_"+str(months[0])+"_"+str(months[-1])+".tiff",\
		bbox_inches="tight")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/trends/"+\
	#	dataset+"_"+loc+"_"+var+"_"+method+"_"+str(months[0])+"_"+str(months[-1])+".png",\
	#	bbox_inches="tight")
	plt.close()

def interannual_time_series(df_list,var,names,loc,year_range,method,months=np.arange(1,13,1),\
		percentile=0):

	#Takes a list of dataframes with reset index, and plots an interannual time series for
	# each dataset at a location

	#Depending on the method, the time series is either:
	#	- Monthly mean
	#	- Days exceeding 99th percentile annually

	#"months" defines the months used for this analysis

	plt.figure()

	for i in np.arange(0,len(df_list)):
		temp = df_list[i][(df_list[i].year>=year_range[0]) & \
				(df_list[i].year<=year_range[1]) & \
				(df_list[i].stn_name==loc) & (np.in1d(df_list[i].month,months))]\
				.sort_values("date").set_index("date")
		if method=="am":
			temp = temp.resample("1Y").mean()
			temp[var[i]].plot(label=names[i])
		elif method=="percentile":
			p99 = np.percentile(temp[var[i]],percentile)
			years = temp.year.unique()
			event_years = [temp[(temp[var[i]]>=p99) & (temp.year == y)].shape[0] \
				for y in years]
			plt.plot(years,event_years,label=names[i]+" "+str(round(p99,3)),marker="s")
	plt.legend()
	plt.show()
		
def wind_gust_boxplot(df,aws,var,loc=False,season=np.arange(1,13,1),two_thirds=True):

	#For a reanalysis dataframe (df), create a boxplot for parameter "var" for each wind speed gust
	# category

	if loc==False:
		df = df.reset_index().set_index(["date","stn_name"])
		aws = aws.reset_index().set_index(["date","stn_name"])
	else:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		aws = aws[(aws.stn_name == loc)].reset_index().set_index("date")

	df = pd.concat([df,aws.wind_gust],axis=1)

	#if "mlcape*s06" == var:
	#	df["mlcape*s06"] = df["ml_cape"] * np.power(df["s06"],1.67)

	df1 = df[(df.wind_gust >= 5) & (df.wind_gust<15) & (np.in1d(df.month,np.array(season)))]
	df2 = df[(df.wind_gust >= 15) & (df.wind_gust<25) & (np.in1d(df.month,np.array(season)))]
	df3 = df[(df.wind_gust >= 25) & (df.wind_gust<30) & (np.in1d(df.month,np.array(season)))]
	df4 = df[(df.wind_gust >= 30) & (np.in1d(df.month,np.array(season)))]
	#df5 = df[(df.wind_gust >= 35) & (df.wind_gust<40) & (np.in1d(df.month,np.array(season)))]
	#df6 = df[(df.wind_gust >= 40) & (np.in1d(df.month,np.array(season)))]

	plt.figure(figsize=[8,5])

	if two_thirds:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var],df5[var],df6[var]],whis=[33,100],\
			labels=["5-15 m/s\n("+str(df1.shape[0])+")","15-25 m/s\n("+str(df2.shape[0])+")",\
			"25-30 m/s\n("+str(df3.shape[0])+")","30-35 m/s\n("+str(df4.shape[0])+")",\
			"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	else:
		plt.boxplot([df1[var],df2[var],df3[var],df4[var]],
			#df5[var],df6[var]],
			whis="range",\
			labels=["5-15 m.s$^{-1}$\n("+str(df1.shape[0])+")","15-25 m.s$^{-1}$\n("+str(df2.shape[0])+")",\
			"25-30 m.s$^{-1}$\n("+str(df3.shape[0])+")","30+ m.s$^{-1}$\n("+str(df4.shape[0])+")"])
			#"35-40 m/s\n("+str(df5.shape[0])+")","40+ m/s\n("+str(df6.shape[0])+")"])
	plt.xticks(fontsize="xx-large")
	plt.yticks(fontsize="xx-large")

	if loc==False:
		plt.title("All Stations",fontsize="x-large")
	else:
		plt.title(loc,fontsize="xx-large")
	t1,t2,t3,units,t4,log_plot,t6 = contour_properties(var)
	plt.ylabel(units,fontsize="xx-large")

	if log_plot:
		plt.yscale("symlog")
		plt.ylim(ymin=0)

	if var == "mu_cape":
		plt.ylim([0,5000])
	elif var == "s06":
		plt.ylim([0,65])

	if loc==False:
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_AllStations_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".png",\
			bbox_inches="tight")
	else:
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/boxplot_"+loc+"_"+var+"_"+\
			str(season[0])+"_"+str(season[-1])+".tiff",\
			bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/boxplot_"+loc+"_"+var+"_"+\
		#	str(season[0])+"_"+str(season[-1])+".png",\
		#	bbox_inches="tight")
	plt.close()

def plot_conv_seasonal_cycle(df,loc,var,trend=False,mf_days=False):

	#For a reanalysis dataframe (df), create a seasonal mean plot for parameter "var" for each wind speed gust
	# category
	
	if len(var) == 2:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		#barra = barra[(barra.stn_name == loc)].reset_index().set_index("date")
		fig,ax = plt.subplots()
		ax.plot(np.arange(1,13,1),[np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)],\
			color="b")
		#ax.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[0]]) for m in np.arange(1,13,1)],\
		#	color="b",linestyle="--")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[0])
		ax.set_ylabel(units,fontsize="xx-large")
		ax.tick_params(labelcolor="b",axis="y",labelsize="xx-large")
		ax2 = ax.twinx()
		ax2.plot(np.arange(1,13,1),[np.mean(df[df.month==m][var[1]]) for m in np.arange(1,13,1)],\
			color="red")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax2.tick_params(labelcolor="red",axis="y",labelsize="xx-large")
		ax2.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.set_xticks(np.arange(1,13))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[1])
		ax2.set_ylabel(units,fontsize="xx-large")
		plt.title(loc,fontsize="xx-large")
		ax.set_xlabel("Months",fontsize="xx-large")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
			"_"+var[0]+"_"+var[1]+".tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
		#	"_"+var[0]+"_"+var[1]+".png",bbox_inches="tight")
	if len(var) == 3:
		df = df[(df.loc_id == loc)].reset_index().set_index("date")
		#barra = barra[(barra.stn_name == loc)].reset_index().set_index("date")
		fig,ax = plt.subplots(figsize=[8,6])

		x = [np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)]
		ax.plot(np.arange(1,13,1), three_month_average(x),\
			color="b")
		#ax.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[0]]) for m in np.arange(1,13,1)],\
		#	color="b",linestyle="--")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[0])
		ax.set_ylabel(units,fontsize="xx-large",color="b")
		ax.tick_params(labelcolor="b",axis="y",labelsize="xx-large")

		ax2 = ax.twinx()
		x = [np.mean(df[df.month==m][var[1]]) for m in np.arange(1,13,1)]
		ax2.plot(np.arange(1,13,1), three_month_average(x),\
			color="red")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax2.tick_params(labelcolor="red",axis="y",labelsize="xx-large")
		ax2.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[1])
		#ax2.set_ylabel(units,fontsize="xx-large")

		ax3 = ax.twinx()
		x = [np.mean(df[df.month==m][var[2]]) for m in np.arange(1,13,1)]
		ax3.plot(np.arange(1,13,1), three_month_average(x),\
			color="green")
		#ax2.plot(np.arange(1,13,1),[np.mean(barra[barra.month==m][var[1]]) for m in np.arange(1,13,1)],\
		#	color="red",linestyle="--")
		ax3.tick_params(labelcolor="green",axis="y",labelsize="xx-large")
		ax3.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		t1,t2,t3,units,t4,t5,t6 = contour_properties(var[2])
		ax3.set_ylabel(units,fontsize="xx-large")

		ax.tick_params(labelcolor="k",axis="x",labelsize="xx-large")
		ax.set_xticks(np.arange(1,13))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		plt.title(loc,fontsize="xx-large")
		ax.set_xlabel("Months",fontsize="xx-large")
		ax3.tick_params(pad=30)
		fig.subplots_adjust(right=0.8)
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
			"_"+var[0]+"_"+var[1]+".tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
		#	"_"+var[0]+"_"+var[1]+".png",bbox_inches="tight")
	else:
		df = df[(df.stn_name == loc)].reset_index().set_index("date")
		if trend:
			if mf_days:
				plt.plot([np.mean(df[(df.month==m) & (df.year<=1998) & (df.mf==1)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1979-1998",color="b")
				plt.plot([np.mean(df[(df.month==m) & (df.year>=1998) & (df.mf==1)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1998-2017",color="b",linestyle="--")
				plt.title(loc)
				plt.legend(fonsize="xx-large")
				ax=plt.gca()
				ax.tick_params(labelsize="xx-large")
				plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
						"_"+var[0]+"_trend_mf_days.png",bbox_inches="tight")
			else:
				plt.plot([np.mean(df[(df.month==m) & (df.year<=1998)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1979-1998",color="b")
				plt.plot([np.mean(df[(df.month==m) & (df.year>=1998)][var[0]]) \
						for m in np.arange(1,13,1)],\
						label = "1998-2017",color="b",linestyle="--")
				plt.title(loc,fontsize="xx-large")
				plt.legend(fontsize="xx-large")
				ax=plt.gca()
				ax.tick_params(labelsize="xx-large")
				plt.ylabel("J.kg$^{-1}$",fontsize="xx-large")
				ax.set_xticks(np.arange(0,12))
				ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
				plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/monthly_mean_"+loc+\
						"_"+var[0]+"_trend.tiff",bbox_inches="tight")
				#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
				#		"_"+var[0]+"_trend.png",bbox_inches="tight")
		else:
			plt.plot([np.mean(df[df.month==m][var[0]]) for m in np.arange(1,13,1)])
			plt.title(loc)
			plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/monthly_mean_"+loc+\
					"_"+var[0]+".png",bbox_inches="tight")
	plt.close()

def magnitude_trends(param,hr,param_names):

	#Attempt to identify trends in a range of magnitudes, following the method of Dowdy (2014)
	#Param is a string, hr is an array or percentiles which identify a daignostic threshold

	#Load datasets and combine
	df = analyse_events("jdh","sa_small")
	df = df.dropna(subset=["wind_gust"])
	
	markers=["s","o","^","+","v"]
	cols = ["b","r","g","c","y"]
	plt.figure(figsize=[12,5])
	for i in np.arange(len(hr)):
		#Find the proportion of event days given all days, as a function of wind speed catergory
		speed_cats = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50]]
		speed_labs = ["0-5","5-10","10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50"]
		all_days = np.zeros(len(speed_cats))
		event_days = np.zeros(len(speed_cats))
		for s in np.arange(len(speed_cats)):
			all_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				(df["wind_gust"]<speed_cats[s][1])].shape[0]
			if hr[i] == 1:
				event_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				    (df["wind_gust"]<speed_cats[s][1]) & \
				    (df[param[i]]==hr[i])].shape[0]
				lab = "Days with "+param_names[i]+" = 1"
			else:
				event_days[s] = df[(df["wind_gust"]>=speed_cats[s][0]) & \
				    (df["wind_gust"]<speed_cats[s][1]) & \
				    (df[param[i]]>=np.percentile(df[df.jdh==1][param[i]],hr[i]))].shape[0]
				lab = "Days with "+param_names[i]+" = "+\
					str(np.percentile(df[df.jdh==1][param[i]],hr[i]).round(3))

		#Plot the relationship
		if i==0:
			plt.plot(np.arange(len(speed_cats)),all_days,"kx",linestyle="none",markersize=12,\
				label = "All days")
		plt.plot(np.arange(len(speed_cats)),event_days,color=cols[i],marker=markers[i],linestyle="none",\
			fillstyle="none",markersize=12,label = lab)
		print(lab+"\n"+str((event_days/all_days.astype(float)).round(3)))

	plt.yscale("log")
	plt.ylabel("Number of Days",fontsize="xx-large")
	plt.xlabel("Wind Speed (m.s$^{-1}$)",fontsize="xx-large")
	ax=plt.gca();ax.set_xticks(np.arange(len(speed_cats)));ax.set_xticklabels(speed_labs)
	ax.tick_params(labelsize="xx-large")
	plt.xlim([-0.5,len(speed_cats)]);plt.ylim([0.5,10e4])
	plt.legend(numpoints=1,fontsize="xx-large")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/magnitude_trends.png",bbox_inches="tight")
	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/magnitude_trends.tiff",bbox_inches="tight")

def plot_aus_station_wind_gusts():

	from adjustText import adjust_text
	
	df = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_aus_1979_2017.pkl")
	plt.figure()
	locs = df.stn_name.unique()
	max_gusts = np.array([df[df.stn_name==l]["wind_gust"].max() for l in locs])
	no_of_gusts = np.array([df[(df.stn_name==l)&(df.wind_gust>=30)].shape[0] for l in locs])
	no_of_points = np.array([df[(df.stn_name==l)].shape[0] for l in locs])
	plt.plot(max_gusts,no_of_gusts/no_of_points.astype(float),"bo")
	texts = [plt.text(max_gusts[i],no_of_gusts[i]/float(no_of_points[i]),locs[i]) for i in np.arange(len(locs))]
	adjust_text(texts,arrowprops={"arrowstyle":"->","color":"red"})
	plt.xlabel("Max wind gust (m/s)")
	plt.ylabel("No. of gusts above 30 m/s")
	plt.xlim([28,60])
	plt.show()

def plot_diurnal_wind_distribution():
	#Plot the diurnal distributions of AWS-defined wind gust events

	#Load unsampled (half-horuly) aws data
	aws_30min = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
			"all_wind_gusts_sa_1985_2017.pkl").reset_index().sort_values(["date","stn_name"])
	#Change to local time
	aws_30min["date_lt"] = aws_30min.date + dt.timedelta(hours = -10.5)
	aws_30min["hour"] = [x.hour for x in aws_30min.date_lt]
	aws_30min["month"] = [x.month for x in aws_30min.date_lt]
	#Attempt to eliminate erroneous values
	inds = aws_30min[aws_30min.wind_gust>=40].index.values
	err_cnt=0
	for i in inds:
		prev = aws_30min.loc[i-1].wind_gust
		next = aws_30min.loc[i+1].wind_gust
		if (prev < 20) & (next < 20):
			aws_30min.wind_gust.iat[i] = np.nan
			err_cnt = err_cnt+1

	aws_30min_warm_inds = np.in1d(aws_30min.month,np.array([10,11,12,1,2,3]))

	#Get diurnal distribution for 30 min data
	for loc in ["Adelaide AP","Port Augusta","Woomera","Mount Gambier"]:
		aws_30min_5_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=5) & (aws_30min.wind_gust<15)])
		aws_30min_15_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=15) & (aws_30min.wind_gust<25)])
		aws_30min_25_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=25) & (aws_30min.wind_gust<30)])
		aws_30min_30_cnt,hours = get_diurnal_dist(aws_30min[(aws_30min.stn_name==loc) & \
			(aws_30min.wind_gust>=30)])

	#Get diurnal distribution for 6 hourly data
		plt.figure()
		plt.plot(hours,aws_30min_15_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_25_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_5_cnt,marker="s",linestyle="none",markersize=11)
		plt.plot(hours,aws_30min_30_cnt,marker="x",linestyle="none",color="k",\
				markeredgewidth=1.5,markersize=17.5)
		plt.ylabel("Count",fontsize="large")
		plt.xlabel("Hour (Local Time)",fontsize="large")
		plt.yscale("log")
		plt.title(loc)
		plt.xlim([0,24]);plt.ylim([0.5,plt.ylim()[1]])
		ax = plt.gca()
		ax.tick_params(labelsize=20)
		plt.grid()
		plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"+\
			"diurnal_"+loc+".png",bbox_inches="tight")
		plt.close()


def plot_daily_data_monthly_dist(aws,stns,outname):

	#Plot seasonal distribution of wind gusts over certain thresholds for each AWS station
	s = 11
	no_years = float(2018-1979)
	for i in np.arange(0,len(stns)):
		fig = plt.figure(figsize=[8,6])
		aws_mth_cnt5,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[5,15])
		aws_mth_cnt15,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[15,25])
		aws_mth_cnt25,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[25,30])
		aws_mth_cnt30,mths = get_monthly_dist(aws[(aws.stn_name==stns[i])],\
			threshold=[30])
		plt.plot(mths,aws_mth_cnt15/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt25/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt5/no_years,linestyle="none",marker="s",markersize=s)
		plt.plot(mths,aws_mth_cnt30/no_years,linestyle="none",marker="x",markersize=17.5,color="k",\
			markeredgewidth=1.5)
		plt.title(stns[i],fontsize="xx-large")
		plt.xlabel("Month",fontsize="xx-large");plt.ylabel("Gusts per month",fontsize="xx-large")
		#plt.legend(loc="upper left")
		ax=plt.gca();ax.set_yscale('log')
		ax.set_xticks(np.arange(1,13,1))
		ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
		ax.set_xlim([0.01,12.5])
		ax.set_ylim([0.01,ax.get_ylim()[1]])
		ax.tick_params(labelsize="xx-large")
		ax.grid()
		fig.subplots_adjust(bottom=0.2)
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"\
			+outname+"_"+"monthly_"+stns[i]+"_1979_2017.tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/temporal_distributions/"\
		#	+outname+"_"+"monthly_"+stns[i]+"_1979_2017.png")
		plt.close()

def get_diurnal_dist(aws):
	hours = np.arange(0,24,1)
	hour_counts = np.empty(len(hours))
	for h in np.arange(0,len(hours)):
		hour_counts[h] = (aws.hour==hours[h]).sum()
	return hour_counts,hours

def get_monthly_dist(aws,threshold=0,mean=False):
	#Get the distribution of months in a dataframe. If threshold is specified, restrict the 
	# dataframe to where the field "wind_gust" is above that threshold
	#If mean = "var", then also get the monthly mean for "var"
	months = np.sort(aws.month.unique())
	month_counts = np.empty(len(months))
	month_mean = np.empty(len(months))
	for m in np.arange(0,len(months)):
		if len(threshold)==1:
			month_count = ((aws[aws.wind_gust>=threshold[0]].month==months[m]).sum())
		elif len(threshold)==2:
			month_count = ((aws[(aws.wind_gust>=threshold[0]) & \
				(aws.wind_gust<threshold[1])].month==months[m]).sum())
		total_month = ((aws.month==months[m]).sum())
		#month_counts[m] = month_count / float(total_month)
		#month_counts[m] = month_count / float(aws[aws.wind_gust>=threshold].shape[0])
		month_counts[m] = month_count
		if mean != False:
			month_mean[m] = aws[aws.month==months[m]][mean].mean()
	if mean != False:
		return month_counts,month_mean,months
	else:
		return month_counts,months

def plot_stations():
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
		"all_daily_max_wind_gusts_sa_1979_2017.pkl")
	stns = aws.stn_name.unique()
	lons = aws.lon.unique()
	lats = aws.lat.unique()
	#remove Port Augusta power station coordinates, as it has merged with Port Augusta in aws df
	lats = lats[~(lats==-32.528)]
	lons = lons[~(lons==137.79)]
	
	start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	m = Basemap(llcrnrlon=start_lon, llcrnrlat=start_lat, urcrnrlon=end_lon, urcrnrlat=end_lat,\
		projection="cyl",resolution="i")
	m.drawcoastlines()

	for i in np.arange(0,len(stns)):
		x,y = m(lons[i],lats[i])
		plt.annotate(stns[i],xy=(x,y),color="k",size="small")
		plt.plot(x,y,"ro")
	plt.show()

def plot_netcdf(domain,fname,outname,time,model,vars=False):

#Load a single netcdf file and plot time max if time is a list of [start_time,end_time]
# or else plot for a single time if time is length=1

	f = nc.Dataset(fname)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)

	if vars == False:
		vars = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
		vars = vars[~(vars=="time") & ~(vars=="lat") & ~(vars=="lon")]	
	
	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "i")
	for param in vars:
		if len(time)==2:
			values = np.nanmax(f.variables[param][(times_dt>=time[0]) & \
				(times_dt<=time[-1])],axis=0)
			plt.title(str(time[0]) + "-" + str(time[-1]))
		else:
			if param == "mlcape*s06":
				values = np.squeeze(f.variables["ml_cape"][times_dt == time] * \
					np.power(f.variables["s06"][times_dt == time],1.67))
			elif param == "cond":
				mlcape = np.squeeze(f.variables["ml_cape"][times_dt == time])
				dcape = np.squeeze(f.variables["dcape"][times_dt == time])
				s06 = np.squeeze(f.variables["s06"][times_dt == time])
				mlm = np.squeeze(f.variables["mlm"][times_dt == time])
				sf = ((s06>=30) & (dcape<500) & (mlm>=26) ) 
				wf = ((mlcape>120) & (dcape>350) & (mlm<26) ) 
				values = sf | wf 
				values = values * 1.0; sf = sf * 1.0; wf = wf * 1.0
				values_disc = np.zeros(values.shape)
				#Separate conditions 1 and 2 (strong forcing and weak forcing)
				#Make in to an array with 1's for SF, 2's for WF and 3's for both types
				values_disc[sf==1] = 1
				values_disc[wf==1] = 2
			else:
				values = np.squeeze(f.variables[param][times_dt == time])
			plt.title(str(time[0]))

		print(param)
		[cmap,mean_levels,levels,cb_lab,range,log_plot,threshold] = contour_properties(param)

		plt.figure()

		m.drawcoastlines()
		if (domain[0]==-38) & (domain[1]==-26):
			m.drawmeridians([134,137,140],\
					labels=[True,False,False,True],fontsize="xx-large")
			m.drawparallels([-36,-34,-32,-30,-28],\
					labels=[True,False,True,False],fontsize="xx-large")
		else:
			m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
					labels=[True,False,False,True])
			m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
					labels=[True,False,True,False])
		if param == "cond":
			m.pcolor(x,y,values,latlon=True,cmap=plt.get_cmap("Reds",2))
		else:
			m.contourf(x,y,values,latlon=True,cmap=cmap,levels=levels,extend="max")
		cb = plt.colorbar()
		cb.set_label(cb_lab,fontsize="xx-large")
		cb.ax.tick_params(labelsize="xx-large")
		if param == "cond":
			cb.set_ticks([0,1])
		m.contour(x,y,values,latlon=True,colors="grey",levels=threshold)
		if (outname == "system_black_2016092806"):
			tx = (138.5159,138.3505,138.0996)
			ty = (-33.7804,-32.8809,-32.6487)
			m.plot(tx,ty,color="k",linestyle="none",marker="^",markersize=10)
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+\
		#	"_"+param+".png",bbox_inches="tight")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+model+"_"+outname+\
			"_"+param+".tiff",bbox_inches="tight")
		plt.close()
		#IF COND, ALSO DRAW COND TYPES
		if param == "cond":
			plt.figure()
			m.drawcoastlines()
			if (domain[0]==-38) & (domain[1]==-26):
				m.drawmeridians([134,137,140],\
						labels=[True,False,False,True],fontsize="xx-large")
				m.drawparallels([-36,-34,-32,-30,-28],\
						labels=[True,False,True,False],fontsize="xx-large")
			else:
				m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
						labels=[True,False,False,True])
				m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
						labels=[True,False,True,False])
			m.pcolor(x,y,values_disc,latlon=True,cmap=plt.get_cmap("Accent_r",3),vmin=0,vmax=2)
			cb = plt.colorbar()
			cb.set_label("Forcing type",fontsize="xx-large")
			cb.set_ticks([0,1,2,3])
			cb.set_ticklabels(["None","SF","MF","Both"])
			cb.ax.tick_params(labelsize="xx-large")
			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+model+"_"+outname+\
				"_"+"forcing_types"+".tiff",bbox_inches="tight")
			#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/"+model+"/"+outname+\
			#	"_"+"forcing_types"+".png",bbox_inches="tight")
			plt.close()

def plot_netcdf_animate(fname,param,outname,domain):

#Load a single netcdf file and plot the time animation

	global values, times_str, x, y, levels, cmap, cb_lab, m

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=15, bitrate=1800)

	f = nc.Dataset(fname)
	values = f.variables[param][:]
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	x,y = np.meshgrid(lon,lat)
	times = f.variables["time"][:]
	times_dt = nc.num2date(times,f.variables["time"].units)
	times_str = [dt.datetime.strftime(t,"%Y-%m-%d %H:%M") for t in times_dt]

	[cmap,levels,cb_lab] = contour_properties(param)

	m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
				urcrnrlat = domain[1], projection="cyl", resolution = "l")
	
	fig = plt.figure()
	m.contourf(x,y,np.zeros(x.shape),latlon=True,levels=levels,cmap=cmap,extend="both")
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	cb = plt.colorbar()
	cb.set_label(cb_lab)
	anim = animation.FuncAnimation(fig, animate, frames=values.shape[0],interval=500)
	anim.save("/home/548/ab4502/working/ExtremeWind/figs/mean/"+outname+".mp4",\
			writer=writer)
	plt.show()

def animate(i):
	z = values[i]
	m.drawcoastlines()
	m.drawmeridians(np.arange(domain[2],domain[3],5),\
			labels=[True,False,False,True])
	m.drawparallels(np.arange(domain[0],domain[1],5),\
			labels=[True,False,True,False])
	im = m.contourf(x,y,values[i],latlon=True,levels=levels,cmap=cmap,extend="both")
	plt.title(times_str[i] + " UTC")
	return im
	
def contour_properties(param):
	threshold = [1]
	if param in ["mu_cape","cape","cape700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,2000,11)
		extreme_levels = np.linspace(0,2000,11)
		cb_lab = "J.kg$^{-1}$"
		range = [0,4000]
		log_plot = True
	if param in ["dlm*dcape*cs6","mlm*dcape*cs6"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,4,11)
		extreme_levels = np.linspace(0,40,11)
		cb_lab = ""
		range = [0,40]
		log_plot = True
		threshold = [0.669]
	elif param == "ml_cape":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,60,11)
		extreme_levels = np.linspace(0,2000,11)
		cb_lab = "J.kg$^{-1}$"
		threshold = [127]
		range = [0,4000]
		log_plot = True
	elif param == "mu_cin":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,500,11)
		cb_lab = "J/Kg"
		range = [0,500]
		log_plot = True
	elif param == "ml_cin":
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,400,11)
		cb_lab = "J/Kg"
		range = [0,600]
		log_plot = False
	elif param in ["s06","ssfc6"]:
		cmap = cm.Reds
		mean_levels = np.linspace(14,18,17)
		extreme_levels = np.linspace(0,50,11)
		cb_lab = "m.s$^{-1}$"
		threshold = [23.83]
		range = [0,60]
		log_plot = False
	elif param in ["ssfc3"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,30,7)
		extreme_levels = np.linspace(0,40,11)
		cb_lab = "m/s"
		range = [0,30]
		log_plot = False
	elif param in ["ssfc850"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,20,7)
		extreme_levels = np.linspace(0,30,11)
		cb_lab = "m/s"
		range = [0,25]
		log_plot = False
	elif param in ["ssfc500","ssfc1"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,20,7)
		extreme_levels = np.linspace(0,30,11)
		cb_lab = "m/s"
		range = [0,20]
		log_plot = False
	elif param in ["dcp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,4,11)
		cb_lab = "DCP"
		range = [0,3]
		threshold = [0.028,1]
		log_plot = True
	elif param in ["scp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,4,11)
		cb_lab = ""
		range = [0,3]
		threshold = [0.025,1]
		log_plot = True
	elif param in ["stp","ship","non_sc_stp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.5,11)
		extreme_levels = np.linspace(0,2.5,11)
		cb_lab = ""
		threshold = [0.027,1]
		range = [0,3]
		log_plot = True
	elif param in ["mmp"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,1,11)
		extreme_levels = np.linspace(0,1,11)
		cb_lab = ""
		range = [0,1]
		log_plot = True
	elif param in ["srh01","srh03","srh06"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(0,600,11)
		cb_lab = "m^2/s^2"
		range = [0,400]
		log_plot = True
	elif param == "crt":
		cmap = cm.YlGnBu
		mean_levels = [60,120]
		extreme_levels = [60,120]
		cb_lab = "degrees"
		range = [0,180]
		log_plot = False
	elif param in ["relhum850-500","relhum1000-700","hur850","hur700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,100,11)
		extreme_levels = np.linspace(0,100,11)
		cb_lab = "%"
		range = [0,100]
		log_plot = False
	elif param == "vo":
		cmap = cm.RdBu_r
		mean_levels = np.linspace(-8e-5,8e-5,17)
		extreme_levels = np.linspace(-8e-5,8e-5,17)
		cb_lab = "s^-1"
		range = [-8e-5,-8e-5]
		log_plot = False
	elif param == "lr1000":
		cmap = cm.Blues
		mean_levels = np.linspace(2,12,11)
		extreme_levels = np.linspace(2,12,11)
		cb_lab = "deg/km"
		range = [0,12]
		log_plot = False
	elif param == "lcl":
		cmap = cm.YlGnBu_r
		mean_levels = np.linspace(0,8000,9)
		extreme_levels = np.linspace(0,8000,9)
		cb_lab = "m"
		range = [0,8000]
		log_plot = False
	elif param in ["td800","td850","td950"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,30,9)
		extreme_levels = np.linspace(0,30,9)
		cb_lab = "deg"
		range = [0,40]
		log_plot = False
	elif param in ["cape*s06","cape*ssfc6","mlcape*s06"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,500000,11)
		extreme_levels = np.linspace(0,500000,11)
		cb_lab = ""
		range = [0,100000]
		threshold = [10344.542,68000]
		log_plot = True
	elif param in ["cape*td850"]:
		cmap = cm.YlGnBu
		mean_levels = None
		extreme_levels = None
		cb_lab = ""
		range = None
		log_plot = True
	elif param in ["dp850-500","dp1000-700","dp850","dp700"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(-10,10,11)
		extreme_levels = np.linspace(-10,10,11)
		cb_lab = "degC"
		range = [-20,20]
		log_plot = False
	elif param in ["dlm","mlm"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(5,15,11)
		extreme_levels = np.linspace(0,50,11)
		threshold = [1]
		cb_lab = "m/s"
		range = [0,50]
		log_plot = False
	elif param in ["dcape"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,400,11)
		extreme_levels = np.linspace(0,2000,11)
		threshold = [500]
		cb_lab = "J/kg"
		range = [0,2000]
		log_plot = False
	elif param in ["mfper"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,0.1,11)
		extreme_levels = np.linspace(0,1,2)
		threshold = [1]
		cb_lab = "% of WF"
		range = [0,1]
		log_plot = False
	elif param in ["cond","sf","mf"]:
		cmap = cm.Reds
		mean_levels = np.linspace(0,0.1,11)
		extreme_levels = np.linspace(0,1,2)
		threshold = [1]
		cb_lab = "CEWP"
		range = [0,1]
		log_plot = False
	elif param in ["max_wg10","wg10"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [12.817,21.5]
		cb_lab = "no. of days"
		range = [0,30]
		log_plot = False
	elif param in ["tas","ta850","ta700","tos"]:
		cmap = cm.Reds
		mean_levels = np.linspace(285,320,5)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [1]
		cb_lab = "K"
		range = [-20,20]
		log_plot = False
	elif param in ["ta2d"]:
		cmap = cm.YlGnBu
		mean_levels = np.linspace(0,200,11)
		extreme_levels = np.linspace(-10,10,11)
		threshold = [12.817,21.5]
		cb_lab = "no. of days"
		range = [-20,20]
		log_plot = False
	
	return [cmap,mean_levels,extreme_levels,cb_lab,range,log_plot,threshold]

if __name__ == "__main__":

	path = '/g/data/eg3/ab4502/ExtremeWind/aus/'
	f = "erai_wrf_20140622_20140624"
	model = "erai"
	fname = path+f+".nc"
	param = "cape*s06"
	region = "sa_small"

	time = [dt.datetime(2014,10,6,6,0,0),dt.datetime(2014,6,10,12,0,0)]
	outname = param+"_"+dt.datetime.strftime(time[0],"%Y%m%d_%H%M")+"_"+\
			dt.datetime.strftime(time[-1],"%Y%m%d_%H%M")

	#NOTE PUT IN FUNCTION
	if region == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	else:
	    raise NameError("Region must be one of ""aus"", ""sa_small"" or ""sa_large""")
	domain = [start_lat,end_lat,start_lon,end_lon]

#	PLOT SEASONAL DISTRIBUTIONS
	#plot_daily_data_monthly_dist(pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
	#	"erai_fc_points_1979_2017_daily_max.pkl").reset_index().\
	#	rename(columns={"wg10":"wind_gust","loc_id":"stn_name"}),\
	#	["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="ERA-Interim")
	#barra_r_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
	#	"barra_r_fc_points_daily_2003_2016.pkl").reset_index()
	#barra_r_df["month"] = [t.month for t in barra_r_df.date]
	#plot_daily_data_monthly_dist(barra_r_df.rename(columns={"max_wg10":"wind_gust","loc_id":"stn_name"}),\
	#	["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="BARRA-R")
	#plot_daily_data_monthly_dist(remove_incomplete_aws_years(\
	#		pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"+\
	#		"all_daily_max_wind_gusts_sa_1979_2017.pkl"),"Port Augusta").reset_index()\
	#		,["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="AWS")
	#barra_ad_df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
	#	"barra_ad_points_daily_2006_2016.pkl").reset_index()
	#barra_ad_df["month"] = [t.month for t in barra_ad_df.date]
	#plot_daily_data_monthly_dist(barra_ad_df.rename(columns={"max_wg10":"wind_gust","loc_id":"stn_name"}),\
		#["Adelaide AP","Woomera","Mount Gambier","Port Augusta"],outname="BARRA-AD")

	#CASE STUDIES
	#for time in date_seq([dt.datetime(1979,11,14,0),dt.datetime(1979,11,14,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"+\
	#		"erai_19791101_19791130.nc"\
	#		,"event_1979_"+time.strftime("%Y%m%d%H"),\
	#		[time],"erai",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/"+\
	#		"erai_20160901_20160930.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"erai",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/barra/"+\
	#		"barra_20160901_20160930.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"barra",vars=["scp"])
	#for time in date_seq([dt.datetime(2016,9,28,0),dt.datetime(2016,9,28,12)],"hours",6):
	#	plot_netcdf(domain,"/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_ad/"+\
	#		"barra_ad_20160928_20160929.nc"\
	#		,"system_black_"+time.strftime("%Y%m%d%H"),\
	#		[time],"barra_ad",vars=["scp"])

#	PLOT OBSERVED DIURNAL DISTRIBUTION
	#plot_diurnal_wind_distribution()

	#probability_plot("ml_cape","mlm")
	
	#INTERANNUAL TIME SERIES
	erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_fc_points_1979_2017_daily_max.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/barra_r_fc_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	barra_r_fc["month"] = [t.month for t in barra_r_fc.date]
	barra_r_fc["year"] = [t.year for t in barra_r_fc.date]
	barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_daily_2003_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_daily_2006_2016.pkl").rename(columns={"loc_id":"stn_name"}).reset_index()   
	barra_ad["month"] = [t.month for t in barra_ad.date]
	barra_ad["year"] = [t.year for t in barra_ad.date]
	locs = ["Adelaide AP","Woomera","Mount Gambier","Port Augusta"]
	#locs = ["Adelaide AP"]
	#[var_trends(aws,"wind_gust",l,"threshold","AWS",threshold=[[15,25],[25,30],[30]]) \
	#	for l in locs]
	[var_trends(erai_fc,"wg10",l,"threshold_only","ERA-Interim",threshold=[[15,25],[25,30],[30]]) \
		for l in locs]
	[var_trends(barra_r_fc,"max_wg10",l,"threshold_only","BARRA-R",threshold=[[15,25],[25,30],[30]],year_range=[2003,2016]) \
		for l in locs]
	[var_trends(barra_ad,"max_wg10",l,"threshold_only","BARRA-AD",threshold=[[15,25],[25,30],[30]],year_range=[2006,2016]) \
		for l in locs]

	#erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sa_small_1979_2017_daily_max.pkl")
	#locs = ["Woomera", "Port Augusta", "Adelaide AP", "Mount Gambier"]
	#[plot_conv_seasonal_cycle(erai, l, ["ml_cape", "s06", "cape*s06"] ) for l in locs]
