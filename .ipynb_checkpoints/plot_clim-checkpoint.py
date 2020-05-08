import xarray
import matplotlib
#matplotlib.use("Agg")
import netCDF4 as nc
from plot_param import *
from erai_read import get_lat_lon, drop_erai_fc_duplicates
from obs_read import read_clim_ind
from scipy.stats import spearmanr as rho
import numpy as np
from event_analysis import hypothesis_test
import xarray as xr
from barra_read import get_mask as get_barra_mask
from era5_read import get_mask as get_era5_mask
from scipy.stats import pearsonr as pearson

def clim(domain,model,year_range,var_list,seasons=[[1,2,3,4,5,6,7,8,9,10,11,12]],levels=False,\
		plot_trends=False):
	
	f = load_ncdf(domain,model,year_range,var_list)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	time = nc.num2date(f.variables["time"][:],f.variables["time"].units)
	months = np.array([t.month for t in time])

	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "l")

	print("INFO: Plotting...")
	for var in var_list:

		print(var)
		if plot_trends:
			dmean,daily_dates = netcdf_resample(f,time,var,model,"mean")
			daily_months = np.array([t.month for t in daily_dates])
			daily_years = np.array([t.year for t in daily_dates])

		for season in seasons:

			print(season)

			#IF DOING CORRELATIONS WITH CLIMATE INDICES, MAY HAVE TO RESAMPLE TO DAILY MEAN.
			#CURRENTLY, DO FOR 6-HOURLY DATA

			data = f.variables[var][np.in1d(months,np.array(season))]
			print("NAN REPORT: "+str((data==np.nan).sum()))
			mean = np.nanmean(data,axis=0)
			plot_clim(f,m,lat,lon,mean,var,model,domain,year_range,season=season,levels=levels)

			if plot_trends:
				t1 = np.array(daily_dates) <= dt.datetime(1998,12,31)
				t2 = np.array(daily_dates) >= dt.datetime(1998,1,1)
				dmean_season1 = dmean[np.in1d(daily_months,np.array(season)) & (t1)]
				dmean_season2 = dmean[np.in1d(daily_months,np.array(season)) & (t2)]
				a = np.mean(dmean_season1,axis=0)
				b = np.mean(dmean_season2,axis=0)
				trend = ((b - a) / a) * 100
		
def load_xarray(var,year_range,start_lon,start_lat,end_lon,end_lat,plevel=False):

	#Function to load ERA-Interim directly from the ub4 directory. Have to use xarray to load all files
	#at once

	#Note, if loading pressure level data (e.g. ta850), the whole era-interim dataset takes too long to load
	# with xarray.open_mfdataset. So, we loop over each file, extract the pressure level we want, and append
	# to a numpy array. To save time, this array is saved to the eg3 /g/data directory as a netcdf file, which
	# can be loaded if it has already been processed for the right variable/pressure level/time period.

	fnames = list()
	if var in ["tas","ta2d"]:
		for y in np.arange(year_range[0],year_range[1]+1):
			fnames.extend(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/"+var+"/"+var+\
			"*"+str(y)+"*.nc"))
	elif var == "tos":
		for y in np.arange(year_range[0],year_range[1]+1):
			fnames.extend(glob.glob("/g/data/ub4/erai/netcdf/6hr/ocean/oper_an_sfc/v01/"+var+"/"+var+\
			"*"+str(y)+"*.nc"))
	else:
		for y in np.arange(year_range[0],year_range[1]+1):
			fnames.extend(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/"+var+"/"+var+\
			"*"+str(y)+"*.nc"))
	fnames.sort()
	
	#Get lat/lon inds
	lon,lat = get_lat_lon()
	lon_ind = np.where((lon >= start_lon) & (lon <= end_lon))[0]
	lat_ind = np.where((lat >= start_lat) & (lat <= end_lat))[0]
	lon = lon[lon_ind]
	lat = lat[lat_ind]

	#Load data
	if var in ["tas","ta2d","tos"]:
		f = xarray.open_mfdataset(fnames,decode_times=False)
		values = f.data_vars[var][:,lat_ind,lon_ind].values
	else:
		fname = "/g/data/eg3/ab4502/ExtremeWind/sa_small/erai_"+var+str(plevel)+"_"+str(year_range[0])+"_"+\
		str(year_range[1])+".nc"
		if not os.path.isfile(fname):
			print("This variable has not yet been extracted for this pressure level. Processing...\n")
			values = np.empty((0,len(lat_ind),len(lon_ind)))
			for i in np.arange(len(fnames)):
				print(i)
				f = xarray.open_dataset(fnames[i],decode_times=False)
				if i == 0:
					lev = f.coords["lev"].values
				temp = f.data_vars[var][:,\
				np.where(np.array(abs(lev-plevel))==np.array(abs(lev-plevel)).min())[0][0],\
				lat_ind,lon_ind].values
				values = np.concatenate((values,temp),axis=0)
				f.close()
			out_file = nc.Dataset(fname,"w",format="NETCDF4_CLASSIC")
			time_dim = out_file.createDimension("time",None)
			lat_dim = out_file.createDimension("lat",len(lat_ind))
			lon_dim = out_file.createDimension("lon",len(lon_ind))
			out_var = out_file.createVariable(var+str(plevel),lat.dtype,("time","lat","lon"))
			out_var[:] = values
			out_file.close()
		else:
			values = nc.Dataset(fname).variables[var+str(plevel)][:]
		
	#Format times
	ref = dt.datetime(1900,1,1,0,0,0)
	if var in ["tas","ta2d"]:
		times = f.coords["time"].data
		times = nc.num2date(times,nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/"+var+\
			"/"+var+"_6hrs_ERAI_historical_an-sfc_19790101_19790131.nc").variables["time"].units)
	elif var == "tos":
		times = f.coords["time"].data
		times = nc.num2date(times,nc.Dataset("/g/data/ub4/erai/netcdf/6hr/ocean/oper_an_sfc/v01/"+var+\
			"/"+var+"_6hrs_ERAI_historical_an-sfc_19790101_19790131.nc").variables["time"].units)
	else:
		times = nc.num2date(xarray.open_mfdataset(fnames,decode_times=False).coords["time"].data,\
			nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/"+var+\
			"/"+var+"_6hrs_ERAI_historical_an-pl_19790101_19790131.nc").variables["time"].units)

	return [values, times, lat, lon]

def load_ncdf(domain,model,year_range,var_list=False,exclude_vars=False):

	#Load netcdf files of convective parameters

	print("INFO: Loading data for "+model+" on domain "+domain+" between "+str(year_range[0])+\
		" and "+str(year_range[1]))

	#Create filenames in year range
	fnames = list()
	for y in np.arange(year_range[0],year_range[1]+1):
		fnames.extend(glob.glob("/g/data/eg3/ab4502/ExtremeWind/"+domain+"/"+model+"/"+model+\
			"_"+str(y)+"*.nc"))
	fnames.sort()
	#Only get dataset for var_list (unless False). For BARRA-R, inconsistent name, for wg, need to change 
	file_vars = np.array(list(nc.Dataset(fnames[0]).variables.keys()))
	if var_list != False:
		if exclude_vars:
			print("EXCLUDING ALL VARIABLES FROM LOAD_NCDF EXCEPT THOSE PARSED BY VAR_LIST...")
			if (model == "barra_r_fc") | (model == "erai_fc") | (model == "barra_ad"):
				exclude_vars = file_vars[~(np.in1d(file_vars,var_list)) & ~(file_vars=="lon") & \
				~(file_vars=="lat") & ~(file_vars=="time")]
			else:
				exclude_vars = file_vars[~(np.in1d(file_vars,var_list)) & ~(file_vars=="lon") & \
				~(file_vars=="lat") & ~(file_vars=="time") & ~(file_vars=="ml_cape") & \
				~(file_vars=="s06")]
			return(nc.MFDataset(fnames,exclude=exclude_vars))
		else:
	    		return(nc.MFDataset(fnames))
	else:
		return(nc.MFDataset(fnames))

def daily_resample(f,time,var,model,method,ftype="netcdf"):

	#Resample a file from sub-daily to a daily max array
	#File can either be "netcdf" or "xarray"

	if ftype == "netcdf":
		if var == "mlcape*s06":
			a = (f.variables["ml_cape"][:]) * np.power(f.variables["s06"][:],1.67)
		else:
			a =  f.variables[var][:]
		if model == "erai_fc":
			a,time = drop_erai_fc_duplicates(a,time)
	else:
		#OR ELSE, IS A ERAI ANALYSIS, ALREADY EXTRACTED
		a = f

	hours = np.array([t.hour for t in time])
	#If using BARRA-R forecast data, then only take 6 hourly data
	if model == "barra_r_fc":
		time = time[np.in1d(hours,np.array([0,6,12,18]))]
		a = a[np.in1d(hours,np.array([0,6,12,18]))]
		hours = hours[np.in1d(hours,np.array([0,6,12,18]))]
	years = np.array([t.year for t in time])
	months = np.array([t.month for t in time])
	days = np.array([t.day for t in time])
	ymd,idx = np.unique([dt.datetime.strftime(t,"%Y%m%d") for t in time],return_index=True)
	daily_dates = [dt.datetime.strptime(t,"%Y%m%d") for t in ymd]
	daily_dates = daily_dates[0:-1]

	#dmax = np.empty((len(daily_dates),f.variables["lat"].shape[0],f.variables["lon"].shape[0]))
	#for i in np.arange(0,len(daily_dates)):
	#	year_ind = daily_dates[i].year == years
	#	month_ind = daily_dates[i].month == months
	#	day_ind = daily_dates[i].day == days
	#	if var == "mlcape*s06":
	#		temp_arr = (f.variables["ml_cape"][(year_ind)&(month_ind)&(day_ind)]) * \
	#			np.power(f.variables["s06"][(year_ind)&(month_ind)&(day_ind)],1.67)
	#	else:
	#		temp_arr = f.variables[var][(year_ind)&(month_ind)&(day_ind)]
	#	if ("wg" in var) or ("gust" in var):
	#		temp_arr = wind_gust_filter(temp_arr)
	#	dmax[i] = np.nanmax(temp_arr,axis=0)
	a = np.split(a,idx[1:-1],axis=0)
	if method=="max":
		dmax = np.stack([np.nanmax(i,axis=0) for i in a])
	elif method=="mean":
		dmax = np.stack([np.nanmean(i,axis=0) for i in a])
	else:
		raise ValueError("METHOD MUST BE ""max"" or ""mean""")

	if model=="erai_fc":
		return(dmax.data,daily_dates)
	else:
		return(dmax,daily_dates)
	

def daily_max_clim(domain,model,year_range,threshold=False,var_list=False,seasons=[[1,2,3,4,5,6,7,8,9,10,11,12]],\
			plot_trends=False,levels=False,trend_levels=np.linspace(-50,50,11),plot_corr=False,\
			n_boot=1000,trend_on_cond_days=False,log_cscale=False):
	#Plot daily max climatology
	#Re-sample to daily maximum first
	#If var = False, plot all vars. Else, pass a list of var names

	#Set up domain
	if domain == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif domain == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif domain == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	m = Basemap(llcrnrlon = start_lon, llcrnrlat = start_lat, urcrnrlon = end_lon, \
				urcrnrlat = end_lat, projection="cyl", resolution = "l")

	#If we are looking at 2 m air temp or dew point, then get the data straight from the ERA-Interim
	# directory, rather than from the calculated convective parameter directory
	if ("tas" in var_list) | ("ta2d" in var_list) | ("ta850" in var_list) | ("dp850" in var_list)\
		 | ("ta700" in var_list) | ("dp700" in var_list) | ("tos" in var_list):
		pass
	else:
		if model == "barra_r_fc":
			f = load_ncdf(domain,model,year_range,var_list,exclude_vars=True)
		else:
			f = load_ncdf(domain,model,year_range,np.concatenate((var_list,["cond","mf","sf"])))
		lon = f.variables["lon"][:]
		lat = f.variables["lat"][:]
		if var_list == False:
			var_list = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
				len(f.variables.items()))])
			var_list = var_list[~(var_list=="time") & ~(var_list=="lat") & ~(var_list=="lon")]
		time = nc.num2date(f.variables["time"][:],f.variables["time"].units)

	#Resample to daily max
	print("Resampling gridded data to daily max...")
	for v in np.arange(0,len(var_list)):
		print(var_list[v])

		if var_list[v] in ["tas","ta2d","tos"]:
			dmax, daily_dates, lat, lon = load_xarray(var_list[v],year_range,start_lon,start_lat,\
				end_lon,end_lat)
			dmax,daily_dates = daily_resample(dmax,daily_dates,None,"erai","max",ftype="xarray")
		elif var_list[v] in ["ta850"]:
			dmax, daily_dates, lat, lon = load_xarray("ta",year_range,start_lon,start_lat,\
				end_lon,end_lat,85000)
			dmax,daily_dates = daily_resample(dmax,daily_dates,None,"erai","max",ftype="xarray")
		elif var_list[v] in ["ta700"]:
			dmax, daily_dates, lat, lon = load_xarray("ta",year_range,start_lon,start_lat,\
				end_lon,end_lat,70000)
			dmax,daily_dates = daily_resample(dmax,daily_dates,None,"erai","max",ftype="xarray")
		elif var_list[v] in ["dp850"]:
			hur_dmax, daily_dates, lat, lon = load_xarray("hur",year_range,start_lon,start_lat,\
				end_lon,end_lat,85000)
			ta_dmax, daily_dates, lat, lon = load_xarray("ta",year_range,start_lon,start_lat,\
				end_lon,end_lat,85000)
			dmax,daily_dates = daily_resample(get_dp(ta_dmax,hur_dmax),daily_dates,None,\
				"erai","max",ftype="xarray")
		elif var_list[v] in ["dp700"]:
			hur_dmax, daily_dates, lat, lon = load_xarray("hur",year_range,start_lon,start_lat,\
				end_lon,end_lat,70000)
			ta_dmax, daily_dates, lat, lon = load_xarray("ta",year_range,start_lon,start_lat,\
				end_lon,end_lat,70000)
			dmax,daily_dates = daily_resample(get_dp(ta_dmax,hur_dmax),daily_dates,None,\
				"erai","max",ftype="xarray")
		else:
			dmax,daily_dates = daily_resample(f,time,var_list[v],model,"max")
		daily_months = np.array([t.month for t in daily_dates])
		daily_years = np.array([t.year for t in daily_dates])

		print("Plotting for each season...")
		for season in seasons:
			print(season)

			dmax_season = dmax[np.in1d(daily_months,np.array(season))]

			#PLOT MEAN
			#If a threshold is given, then plot the number of days which exceed this threshold
			if threshold[v] != False:
				exceeds = dmax_season > threshold[v]
				exceeds_sum = (np.sum(exceeds,axis=0)) / float(((year_range[1]-year_range[0])+1))
				try:
				    plot_clim(f,m,lat,lon,exceeds_sum,var_list[v],model,domain,year_range,season,\
					threshold=threshold[v],levels=levels[v],log_cscale=log_cscale)
				except:
				    plot_clim(f,m,lat,lon,exceeds_sum,var_list[v],model,domain,year_range,season,\
					threshold=threshold[v],levels=levels,log_cscale=log_cscale)
			#If no threshold is given...
			else:
				#For surface temp and dew point, plot mean 
				if var_list[v] in ["tas","ta2d","tos"]:
					try:
				    		plot_clim(None,m,lat,lon,dmax_season.mean(axis=0),var_list[v],model,domain,\
						year_range,season,levels=levels[v],log_cscale=log_cscale)
					except:
						plot_clim(None,m,lat,lon,dmax_season.mean(axis=0),var_list[v],model,domain,\
						year_range,season,threshold=threshold[v],levels=levels,log_cscale=log_cscale)
			   #For pressure-level temperature and humidity, plot mean
				elif var_list[v] in ["ta850","ta700","dp850","dp700"]:
					try:
						plot_clim(None,m,lat,lon,dmax_season.mean(axis=0),var_list[v],model,domain,\
						year_range,season,levels=levels[v],log_cscale=log_cscale)
					except:
						plot_clim(None,m,lat,lon,dmax_season.mean(axis=0),var_list[v],model,domain,\
						year_range,season,threshold=threshold[v],levels=levels,log_cscale=log_cscale)
				#Or else, try and plot the mean
				else:
					exceeds_sum = np.sum(dmax_season,axis=0) / float(((year_range[1]-year_range[0])+1))
					try:
						plot_clim(f,m,lat,lon,exceeds_sum,var_list[v],model,domain,year_range,season,\
						threshold=threshold[v],levels=levels[v],log_cscale=log_cscale)
					except:
						plot_clim(f,m,lat,lon,exceeds_sum,var_list[v],model,domain,year_range,season,\
						threshold=threshold[v],levels=levels,log_cscale=log_cscale)

			#PLOT TRENDS
			if plot_trends:

			   #For temperature and humidity
				if var_list[v] in ["tas","ta2d","ta850","ta700","dp850","dp700","tos"]:

					trend,sig = plot_trend_function(var_list[v],threshold[v],\
					dmax,daily_dates,daily_months,\
					season,dt.datetime(1998,12,31),dt.datetime(1998,1,1),\
					n_boot,lon,lat,m,trend_levels[v],domain,year_range,model)

				#If interested, repeat on cond days.
				if trend_on_cond_days:

					trend,sig = plot_trend_function(var_list[v]+"_cond_days",\
						threshold[v],dmax,\
						daily_dates,daily_months,\
						season,dt.datetime(1998,12,31),dt.datetime(1998,1,1),\
						n_boot,lon,lat,m,trend_levels[v],domain,year_range,model,\
						"cond")

					trend,sig = plot_trend_function(var_list[v]+"_mf_days",\
						threshold[v],dmax,\
						daily_dates,daily_months,\
						season,dt.datetime(1998,12,31),dt.datetime(1998,1,1),\
						n_boot,lon,lat,m,trend_levels[v],domain,year_range,model,\
						"mf")

					trend,sig = plot_trend_function(var_list[v]+"_sf_days",\
						threshold[v],dmax,\
						daily_dates,daily_months,\
						season,dt.datetime(1998,12,31),dt.datetime(1998,1,1),\
						n_boot,lon,lat,m,trend_levels[v],domain,year_range,model,\
						"sf")

				else:
	
					trend,sig = plot_trend_function(var_list[v],threshold[v],\
					dmax,daily_dates,daily_months,\
					season,dt.datetime(1998,12,31),dt.datetime(1998,1,1),\
					n_boot,lon,lat,m,trend_levels[v],domain,year_range,model)

		#PLOT CLIM IND CORRELATIONS
		#BACK OUTSIDE OF THE SEASONS LOOP
		if plot_corr:
			nino34 = read_clim_ind("nino34")
			dmi = read_clim_ind("dmi")
			sam = read_clim_ind("sam")
	
			plot_corr_func(m,lon,lat,model,nino34,dmi,sam,dmax,daily_years,daily_months,var_list[v],\
				threshold[v],year_range,n_boot,domain)	

def plot_trend_function(var,threshold,dmax,daily_dates,daily_months,season,time1,time2,n_boot,lon,lat,m,\
		trend_levels,domain,year_range,model,trend_on_cond_days=False):

	#Define the first and second half of the period 1979-2017
	t1 = np.array(daily_dates) <= time1
	t2 = np.array(daily_dates) >= time2

	#Split in to two periods
	if threshold != False:
		a = (dmax[np.in1d(daily_months,np.array(season)) & (t1)] >= threshold) * 1
		b = (dmax[np.in1d(daily_months,np.array(season)) & (t2)] >= threshold) * 1
	else:
		a = dmax[np.in1d(daily_months,np.array(season)) & (t1)]
		b = dmax[np.in1d(daily_months,np.array(season)) & (t2)]

	#Get the trends for variable var for the region bound by [137,-35,139,-33]. Output a time series of 
	# events in this region (events1 and events2)
	x,y=np.meshgrid(lon,lat)
	events1, events2 = spatial_trends(var, a, b, [137,-35,139,-33], x, y, year_range, n_boot)

	if trend_on_cond_days == "cond":
		print("Getting trends for COND days...")
		f_temp = load_ncdf(domain,model,year_range,["cond"])
		time = nc.num2date(f_temp.variables["time"][:],f_temp.variables["time"].units)
		cond,daily_dates = daily_resample(f_temp,time,"cond",model,"max")
		f_temp.close()
		a = np.ma.masked_where(cond[np.in1d(daily_months,np.array(season)) \
			& (t1)]==0,a)
		b = np.ma.masked_where(cond[np.in1d(daily_months,np.array(season)) \
			& (t2)]==0,b)
		trend = (b.mean(axis=0) - a.mean(axis=0))
	elif trend_on_cond_days == "mf":
		print("Getting trends for MF days...")

		f_temp = load_ncdf(domain,model,year_range,["mf"])
		time = nc.num2date(f_temp.variables["time"][:],f_temp.variables["time"].units)
		mf,daily_dates = daily_resample(f_temp,time,"mf",model,"max")

		mf_events1, mf_events2 = spatial_trends(var,\
			mf[np.in1d(daily_months,np.array(season)) & (t1)],\
			mf[np.in1d(daily_months,np.array(season)) & (t2)],\
			[137,-35,139,-33], x, y, year_range, n_boot)

		f_temp.close()
		a[mf_events1==0] = np.nan
		b[mf_events2==0] = np.nan
		a = np.ma.masked_where(np.isnan(a),a)
		b = np.ma.masked_where(np.isnan(b),b)
		trend = (b.mean(axis=0) - a.mean(axis=0))
	elif trend_on_cond_days == "sf":
		print("Getting trends for SF days...")
		f_temp = load_ncdf(domain,model,year_range,["sf"])
		time = nc.num2date(f_temp.variables["time"][:],f_temp.variables["time"].units)
		sf,daily_dates = daily_resample(f_temp,time,"sf",model,"max")
		f_temp.close()
		a = np.ma.masked_where(sf[np.in1d(daily_months,np.array(season)) \
			& (t1)]==0,a)
		b = np.ma.masked_where(sf[np.in1d(daily_months,np.array(season)) \
			& (t2)]==0,b)
		trend = (b.mean(axis=0) - a.mean(axis=0))
	else:
		#Get trend as a % of the initial frequency
		trend = (np.sum(b,axis=0) - np.sum(a,axis=0)) / np.nansum(a,axis=0).astype(float) * 100

	#For grid points where there is no event in either periods, set trend to nan
	trend[(np.nansum(a,axis=0)==0)|(np.nansum(b,axis=0)==0)] = np.nan

	#Significance test using bootstrap (n_boot)
	if n_boot > 0:
		print("Performing bootstrap resampling "+str(n_boot)+" times...")
		sig = hypothesis_test(a,b,n_boot)
		#Mask signiciance where there is no event in either period
		sig[(np.nansum(a,axis=0)==0)|(np.nansum(b,axis=0)==0)] = np.nan
		#Mask significance where there is no data
		sig[np.isnan(np.nansum(a,axis=0))|np.isnan(np.nansum(b,axis=0))] = np.nan

	#Plot trend
	plt.figure()
	m.drawcoastlines()
	try:
		m.contourf(x,y,trend,cmap=cm.RdBu_r,levels=trend_levels,extend="both")
	except:
		m.contourf(x,y,trend,cmap=cm.RdBu_r,extend="both")
	if domain=="sa_small":
		m.drawmeridians([134,137,140],\
				labels=[True,False,False,True],fontsize="xx-large")
		m.drawparallels([-36,-34,-32,-30,-28],\
				labels=[True,False,True,False],fontsize="xx-large")
	else:
		m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
				labels=[True,False,False,True])
		m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
				labels=[True,False,True,False])
	if n_boot > 0:
		plt.plot(x[np.where(sig<=0.05)],y[np.where(sig<=0.05)],"ko")
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize="xx-large")
	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/contour_"+\
		model+"_"+var+"_"+str(season[0])+"_"+str(season[-1])+"_"+str(threshold)+"_"+\
		str(year_range[0])+"_"+str(year_range[1])+".tiff",bbox_inches="tight")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/trends/contour_"+\
	#	model+"_"+var+"_"+str(season[0])+"_"+str(season[-1])+"_"+str(threshold)+"_"+\
	#	str(year_range[0])+"_"+str(year_range[1])+".png",bbox_inches="tight")


	return [trend,sig]

def plot_corr_func(m,lon,lat,model,nino34,dmi,sam,dmax,daily_years,daily_months,var_name,threshold,year_range,n_boot,domain):
		corr_seasons = [[2,3,4],[5,6,7],[8,9,10],[11,12,1],np.arange(1,13,1)]
		names = nino34.columns
		years = nino34.index
		monthly_ts = np.empty((len(names),len(years),lat.shape[0],lon.shape[0]))
		nino34_p = np.empty((len(names),lat.shape[0],lon.shape[0]))
		sam_p = np.empty((len(names),lat.shape[0],lon.shape[0]))
		dmi_p = np.empty((len(names),lat.shape[0],lon.shape[0]))
		years_p = np.empty((len(names),lat.shape[0],lon.shape[0]))
		nino34_corr = np.empty((len(names),lat.shape[0],lon.shape[0]))
		sam_corr = np.empty((len(names),lat.shape[0],lon.shape[0]))
		dmi_corr = np.empty((len(names),lat.shape[0],lon.shape[0]))
		years_corr = np.empty((len(names),lat.shape[0],lon.shape[0]))
		for s in np.arange(0,len(names)):
			for y in np.arange(0,len(years)):
			#If the variable/parameter being considered is using a threshold, then find a monthly time
			#series of threshold exceedence
				if threshold != False:
					if s == 3:	#IF NDJ
						cond = ((daily_years==years[y]) & (daily_months==corr_seasons[s][0])) \
						| ((daily_years==years[y]) & (daily_months==corr_seasons[s][1])) \
						| ((daily_years==years[y]+1) & (daily_months==corr_seasons[s][2]))
						monthly_ts[s,y] = np.sum(dmax[cond] >= threshold,axis=0)
					else:
						cond = (daily_years==years[y]) & (np.in1d(daily_months,corr_seasons[s]))
						monthly_ts[s,y] = np.sum(dmax[cond] >= threshold,axis=0)
			#If variable is "cond", then we do not need to find threshold exceedence, as it is already
			# in the form of binary/boolean condition
				else:
					if s == 3:	#IF NDJ
						cond = ((daily_years==years[y]) & (daily_months==corr_seasons[s][0])) \
						| ((daily_years==years[y]) & (daily_months==corr_seasons[s][1])) \
						| ((daily_years==years[y]+1) & (daily_months==corr_seasons[s][2]))
						monthly_ts[s,y] = np.sum(dmax[cond],axis=0)
					else:
						cond = (daily_years==years[y]) & (np.in1d(daily_months,corr_seasons[s]))
						monthly_ts[s,y] = np.sum(dmax[cond],axis=0)

		    #Calculate correlations
			for i in np.arange(0,len(lat)):
				for j in np.arange(0,len(lon)):
					c,p = rho(nino34[names[s]],monthly_ts[s,:,i,j])
					nino34_corr[s,i,j] = c
					nino34_p[s,i,j] = p

					c,p = rho(dmi[names[s]],monthly_ts[s,:,i,j])
					dmi_corr[s,i,j] = c
					dmi_p[s,i,j] = p

					c,p = rho(sam[names[s]],monthly_ts[s,:,i,j])
					sam_corr[s,i,j] = c
					sam_p[s,i,j] = p

					c,p = rho(np.arange(0,nino34.shape[0]),monthly_ts[s,:,i,j])
					years_corr[s,i,j] = c
					years_p[s,i,j] = p
			#PLOT
			#NINO 3.4
			plt.figure()
			m.drawcoastlines()
			if domain=="sa_small":
				m.drawmeridians([134,137,140],\
		    	    		labels=[True,False,False,True],fontsize="xx-large")
				m.drawparallels([-36,-34,-32,-30,-28],\
		    	    		labels=[True,False,True,False],fontsize="xx-large")
			else:
				m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
		    	    		labels=[True,False,False,True])
				m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
		    	    		labels=[True,False,True,False])
			x,y=np.meshgrid(lon,lat)
			m.contourf(x,y,nino34_corr[s],cmap=cm.RdBu_r,levels=np.linspace(-0.5,0.5,11)\
		    	    ,extend="both")
			plt.plot(x[np.where(nino34_p[s]<=0.05)],y[np.where(nino34_p[s]<=0.05)],"ko")
			cb = plt.colorbar()
			cb.ax.tick_params(labelsize="xx-large")

			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/nino34_"+\
			model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
			+"_"+str(year_range[0])+"_"+str(year_range[1])+".tiff",bbox_inches="tight")
		    #plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/corr/nino34_"+\
		#	model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
		#	+"_"+str(year_range[0])+"_"+str(year_range[1])+".png",bbox_inches="tight")
			plt.close()
			#DMI
			plt.figure()
			m.drawcoastlines()
			if domain=="sa_small":
				m.drawmeridians([134,137,140],\
					labels=[True,False,False,True],fontsize="xx-large")
				m.drawparallels([-36,-34,-32,-30,-28],\
					labels=[True,False,True,False],fontsize="xx-large")
			else:
				m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
					labels=[True,False,False,True])
				m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
					labels=[True,False,True,False])
			x,y=np.meshgrid(lon,lat)
			m.contourf(x,y,dmi_corr[s],cmap=cm.RdBu_r,levels=np.linspace(-0.5,0.5,11)\
			,extend="both")
			plt.plot(x[np.where(dmi_p[s]<=0.05)],y[np.where(dmi_p[s]<=0.05)],"ko")
			plt.colorbar()
			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/dmi_"+\
			model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
			+"_"+str(year_range[0])+"_"+str(year_range[1])+".tiff",bbox_inches="tight")
			#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/corr/dmi_"+\
		#	model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
		#	+"_"+str(year_range[0])+"_"+str(year_range[1])+".png",bbox_inches="tight")
			plt.close()
		    #SAM 3.4
			plt.figure()
			m.drawcoastlines()
			if domain=="sa_small":
				m.drawmeridians([134,137,140],\
					labels=[True,False,False,True],fontsize="xx-large")
				m.drawparallels([-36,-34,-32,-30,-28],\
					labels=[True,False,True,False],fontsize="xx-large")
			else:
				m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
					labels=[True,False,False,True])
				m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
					labels=[True,False,True,False])
			x,y=np.meshgrid(lon,lat)
			m.contourf(x,y,sam_corr[s],cmap=cm.RdBu_r,levels=np.linspace(-0.5,0.5,11)\
			,extend="both")
			plt.plot(x[np.where(sam_p[s]<=0.05)],y[np.where(sam_p[s]<=0.05)],"ko")
			plt.colorbar()
			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/sam_"+\
			model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
			+"_"+str(year_range[0])+"_"+str(year_range[1])+".tiff",bbox_inches="tight")
		    #plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/corr/sam_"+\
		#	model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
		#	+"_"+str(year_range[0])+"_"+str(year_range[1])+".png",bbox_inches="tight")
			plt.close()
		    #YEARS (trend)
			plt.figure()
			m.drawcoastlines()
			m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),3),\
				labels=[True,False,False,True])
			m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),3),\
				labels=[True,False,True,False])
			x,y=np.meshgrid(lon,lat)
			m.contourf(x,y,years_corr[s],cmap=cm.RdBu_r,levels=np.linspace(-0.5,0.5,11)\
			,extend="both")
			plt.plot(x[np.where(years_p[s]<=0.05)],y[np.where(years_p[s]<=0.05)],"ko")
			plt.colorbar()
			plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/corr/years_"+\
			model+"_"+var_name+"_"+names[s]+"_"+str(threshold)\
			+"_"+str(year_range[0])+"_"+str(year_range[1])+".png",bbox_inches="tight")
			plt.close()



def diurnal_clim(domain,model,year_range,var,threshold=0):

	#Import a netcdf dataset and plot the mean time for which a threshold is exceeded

	print("INFO: Loading data for "+model+" on domain "+domain+" between "+str(year_range[0])+\
		" and "+str(year_range[1]))
	#Create filenames in year range
	fnames = list()
	for y in np.arange(year_range[0],year_range[1]+1):
		fnames.extend(glob.glob("/g/data/eg3/ab4502/ExtremeWind/"+domain+"/"+model+"/"+model+\
			"_"+str(y)+"*.nc"))
	fnames.sort()
	f = nc.MFDataset(fnames)
	lon = f.variables["lon"][:]
	lat = f.variables["lat"][:]
	vars = np.array([str(f.variables.items()[i][0]) for i in np.arange(0,\
		len(f.variables.items()))])
	vars = vars[~(vars=="time") & ~(vars=="lat") & ~(vars=="lon")]
	time = nc.num2date(f.variables["time"][:],f.variables["time"].units)
	years = np.array([t.year for t in time])
	months = np.array([t.month for t in time])
	days = np.array([t.day for t in time])
	hours = np.array([t.hour for t in time])
	y1 = str(years.min())
	y2 = str(years.max())
	warm_inds = np.array([m in np.array([10,11,12,1,2,3]) for m in months])
	cool_inds = ~warm_inds
	hours_warm = np.array([t.hour for t in time[warm_inds]])
	hours_cool = np.array([t.hour for t in time[cool_inds]])

	data = f.variables[var][:]

	hours_clim = np.empty((lat.shape[0],lon.shape[0]))
	hours_warm_clim = np.empty((lat.shape[0],lon.shape[0]))
	hours_cool_clim = np.empty((lat.shape[0],lon.shape[0]))
	for i in np.arange(lat.shape[0]):
		print(i)
		for j in np.arange(lon.shape[0]):
			hours_clim[i,j] = np.mean(hours[np.where(data[:,i,j]>=threshold)])
			hours_warm_clim[i,j] = np.mean(hours_warm[np.where(data[warm_inds,i,j]>=threshold)])
			hours_cool_clim[i,j] = np.mean(hours_cool[np.where(data[cool_inds,i,j]>=threshold)])

	plot_diurnal_clim(hours_clim,lon,lat,model,var,threshold,"all")
	plot_diurnal_clim(hours_warm_clim,lon,lat,model,var,threshold,"warm")
	plot_diurnal_clim(hours_cool_clim,lon,lat,model,var,threshold,"cold")

def plot_diurnal_clim(hours_clim,lon,lat,model,var,threshold,outname):
	m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
				urcrnrlat = lat.max(), projection="cyl", resolution = "l")
	lon,lat = np.meshgrid(lon,lat)
	fig = plt.figure()
	m.drawcoastlines()
	m.contourf(lon,lat,hours_clim,latlon=True,levels = np.arange(0,27,3))
	plt.colorbar()
	fig.savefig("/short/eg3/ab4502/figs/ExtremeWind/clim/"+model+\
		"_"+var+"_hours_"+str(threshold)+"_"+outname+".png")
	plt.close()

def wind_gust_filter(array):

	#Remove points over 60 m/s. Also remove points over 40 m/s, if there is no other gust over
	# 30 m/s for the day

	array[array>=60] = np.nan
	gust_spikes_ind = np.where(array>40)
	for j in np.arange(0,len(gust_spikes_ind[0])):
		if np.sort(array[:,gust_spikes_ind[1][j],gust_spikes_ind[2][j]])[-2]<30:
			array[gust_spikes_ind[0][j],gust_spikes_ind[1][j],\
			gust_spikes_ind[2][j]] = np.nan
	return(array)

def plot_clim(f,m,lat,lon,mean,var,model,domain,year_range,season,levels=False,threshold="",log_cscale=False):

	plt.figure()
	#if levels==False:
	#	cmap,levels,extreme_levels,cb_lab,range,log_plot,thresholds = contour_properties(var)	
	#else:
	cmap,temp,extreme_levels,cb_lab,range,log_plot,thresholds = contour_properties(var)	
	m.drawcoastlines()
	if domain == "sa_small":
		m.drawmeridians([134,137,140],\
				labels=[True,False,False,True],fontsize="xx-large")
		m.drawparallels([-36,-34,-32,-30,-28],\
				labels=[True,False,True,False],fontsize="xx-large")
	else:
		m.drawmeridians(np.arange(np.floor(lon.min()),np.floor(lon.max()),10),\
				labels=[True,False,False,True])
		m.drawparallels(np.arange(np.floor(lat.min()),np.floor(lat.max()),10),\
				labels=[True,False,True,False])
	x,y = np.meshgrid(lon,lat)
	
	try:
		if log_cscale:
			norm = matplotlib.colors.LogNorm()
			m.contourf(x,y,mean,latlon=True,cmap=cmap,\
				norm = norm,levels=np.concatenate(([0.1],levels[-1]*np.logspace(-1,1,10))))
		else:
			norm = matplotlib.colors.BoundaryNorm(levels,cmap.N)
			m.contourf(x,y,mean,latlon=True,levels=levels,cmap=cmap,extend="max",\
				norm = norm)
	except:
		m.contourf(x,y,mean,latlon=True,cmap=cmap,extend="both")
	cb=plt.colorbar(format="%.1f")
	if log_cscale:
		cb.set_ticks(np.concatenate(([0.1],levels[-1]*np.logspace(-1,1,10))))
	cb.ax.tick_params(labelsize="xx-large")
	#if model == "barra_ad":
	#	terrain = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/topog-an-slv-PT0H-BARRA_AD-v1.nc").\
	#			variables["topog"][:]
	#	terrain_lon,terrain_lat = get_barra_ad_lat_lon()
	#	terrain_lon,terrain_lat = np.meshgrid(terrain_lon,terrain_lat)
	#if model == "barra_r_fc":
	terrain = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/topog-an-slv-PT0H-BARRA_R-v1.nc").\
			variables["topog"][:]
	terrain_lon,terrain_lat = get_barra_r_lat_lon()
	terrain_lon,terrain_lat = np.meshgrid(terrain_lon,terrain_lat)
	#elif model == "erai":
		#terrain = np.squeeze(nc.Dataset("/short/eg3/ab4502/erai_sfc_geopt.nc").\
		#		variables["z"][:] / 9.8)
		#terrain_lon,terrain_lat = get_lat_lon()
		#terrain_lon,terrain_lat = np.meshgrid(terrain_lon,terrain_lat)

	if domain == "sa_small":
		m.contour(terrain_lon,terrain_lat,terrain,latlon=True,levels=[250,500,750],colors="grey")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/"+model+"_"+var+\
			"_"+str(season[0])+"_"+str(season[-1])+"_"+str(threshold)+"_"+str(year_range[0])+"_"+\
			str(year_range[-1])+".tiff",bbox_inches="tight")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/clim/"+model+"_"+var+\
		#	"_"+str(season[0])+"_"+str(season[-1])+"_"+str(threshold)+"_"+str(year_range[0])+"_"+\
		#	str(year_range[-1])+".png",bbox_inches="tight")
	else:
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/clim/"+model+"_"+var+\
			"_"+domain+"_"+str(season[0])+"_"+str(season[-1])+"_"+str(threshold)+"_"+\
			str(year_range[0])+"_"+str(year_range[-1])+".png",bbox_inches="tight")
		
	plt.close()

def add_towns(m):

	#Add towns to Basemap contour
	names = ["A","W","PA","MG"]
	#,"Clare","Coober Pedy AP","Renmark","Munkora"]
	lons = [138.5196,136.8054,137.7169,140.7739]
	#,138.5933,134.7222,140.6766,140.3273]
	lats = [-34.9524,-31.1558,-32.5073,-37.7473]
	#,-33.8226,-29.0347,-34.1983,-36.1058]
	for i in np.arange(0,len(names)):
		x,y = m(lons[i],lats[i])
		plt.annotate(names[i],xy=(x,y+.12),color="k",size="xx-large",ha="center")
		plt.plot(x,y,"ko")
	
def spatial_trends(var,arr1, arr2, domain, lon, lat, year_range, n_boot):

	#Take a rectangular area given by domain ( = [start_lon,start_lat,end_lon,end_lat])
	#Calculate the trend in environment days from 1979-2017, by the difference in the 
	#	first (arr1) and second half (arr2) of the period
	#Convert trend to days/decade
	#Save trend as csv

	outside_bounds = (lon < domain[0]) | (lat < domain[1]) | (lon > domain[2]) | (lat > domain[3])

	#Take the change by asking for each timestep, is there an event in the region
	events1 = np.array([np.ma.masked_where(outside_bounds,arr1[t]).max() for t in np.arange(arr1.shape[0])])
	events2 = np.array([np.ma.masked_where(outside_bounds,arr2[t]).max() for t in np.arange(arr2.shape[0])])
	trend = (np.sum(events2) - np.sum(events1)) / float(((year_range[1] - year_range[0] + 1) / 2. / 10.))
	days1 = np.sum(events1)
	days2 = np.sum(events2)
	if n_boot > 0:
		print("Performing bootstrap resampling "+str(n_boot)+" times...")
		events1_samples = [events1[np.random.randint(0,high=events1.shape[0],size=events1.shape[0])] for temp in np.arange(0,n_boot)]
		events2_samples = [events2[np.random.randint(0,high=events2.shape[0],size=events2.shape[0])] for temp in np.arange(0,n_boot)]
		trend_samples = np.array([np.sum(events2_samples[n]) - np.sum(events1_samples[n]) for n in np.arange(n_boot)])
		trend_samples = trend_samples / float(((year_range[1] - year_range[0] + 1) / 2. / 10.))
		
	df = pd.DataFrame({"Trend":trend,"Upper trend (95%)":np.percentile(trend_samples,97.5),"Lower trend (95%)":np.percentile(trend_samples,2.5),\
				"Days1":days1,"Days2":days2},index=["Value"])
	df.to_csv("/short/eg3/ab4502/figs/ExtremeWind/trends/erai_spatial_events_"+var+".csv")

	return [events1, events2]


#THE FOLLOWING FUNCTIONS HAVE BEEN ADDED DURING THE ESCI PROJECT, AND WHAT I AM CURRENTLY USING

def read_aus_conv_wind_clim(fname, daily = True):

	#This will read the output from create_threshold_variable() in event_analysis.py, which are 3d netcdf
	# files containing the number of days exceeding a parameter threshold.

	path = "/g/data/eg3/ab4502/ExtremeWind/aus/"
	f = xr.open_dataset(path+fname)
	
	return f[list(f.variables.keys())[-1]]

def plot_aus_conv_wind_trends():

	#Plotting function/driver for the above function

	fnames = ["era5_logit_is_conv_aws_daily.nc","era5_logit_is_sta_daily.nc",\
			"era5_dcp_0.03_daily.nc","era5_t_totals_48.16_daily.nc"]
	vmax = [[-50,50],[-50,50],[-50,50],[-50,50]]
	titles = ["Logistic eq.\n (measured)","Logistic eq. \n(STA)","DCP","T-totals"]
	
	datasets = [read_aus_conv_wind_clim(fnames[i], daily = True) for i in np.arange(len(fnames))]
	n_boot = 1000
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	#m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	m = Basemap(llcrnrlon=132, llcrnrlat=-38, urcrnrlon=142, urcrnrlat=-26,projection="cyl",resolution="h")
	plt.figure(figsize=[10,8])
	cnt=1
	for s in [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]: 
		print(s)
		for i in np.arange(len(datasets)):
			data = datasets[i]
			data = data.sel(lon = ((data.lon>=132) & (data.lon<=142)),\
				lat = ((data.lat>=-38) & (data.lat<=-26)))
			plt.subplot(4,len(fnames),cnt)
			m.drawcoastlines()
			temp = data[np.in1d(data["time.month"],s)]
			temp_first = temp[temp["time.year"] < 1999]
			temp_second = temp[temp["time.year"] >= 1999]
			trend = ((temp_second.sum("time") - temp_first.sum("time")) / \
				temp_first.sum("time")) * 100
			lat = temp.lat.data
			lon = temp.lon.data
			x,y = np.meshgrid(lon,lat)
			lsm = get_era5_mask(lon,lat)
			trend = trend.where(lsm>=0.5, np.nan)
			c = trend.plot(cmap = plt.get_cmap("RdBu_r"), \
				vmin=vmax[i][0], vmax=vmax[i][1], add_colorbar = False,\
				add_labels=False)
			sig = hypothesis_test(temp_first.to_masked_array(), temp_second.to_masked_array(), n_boot)
			sig = np.where(lsm>=0.5, sig, 1)
			m.contourf(x,y,np.where(sig<=0.05, 1, 0), levels=[.5,1.5], \
				colors=["none", "grey"], hatches=["///////"], alpha=0)
			matplotlib.rcParams['hatch.linewidth'] = 0.5
			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if cnt <= len(fnames):
				plt.title(titles[i])
			if i == 0:
				if s == [12,1,2]:
					plt.ylabel("DJF       ",rotation=0)
				if s == [3,4,5]:
					plt.ylabel("MAM       ",rotation=0)
				if s == [6,7,8]:
					plt.ylabel("JJA       ",rotation=0)
				if s == [9,10,11]:
					plt.ylabel("SON       ",rotation=0)
			if (s == [9,10,11]) & (i == (len(datasets)-1) ):
				cax = plt.axes([0.2, 0.075, 0.6, 0.01])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal", \
					extend="both")
				cb.set_label("Trend (%)")
			cnt = cnt+1
	plt.subplots_adjust(wspace=0.2)
		
def plot_aus_syn_wind_trends():

	#Plot the trends in ERA5 10FG (10 m parametrised wind gust)

	fnames = ["era5_wg10_20.nc", "era5_wg10_25.nc","era5_wg10_30.nc"]
	datasets = [read_aus_conv_wind_clim(fnames[i], daily = True) for i in np.arange(len(fnames))]

	annmax_da1 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_wg10_annmax_1979_1998.nc")
	annmax_da2 = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_wg10_annmax_1999_2018.nc")
	
	vmax = [[-100,100],[-100,100],[-100,100],[-10,10]]
	n_boot = 1000
	titles = ["20 m/s","25 m/s","30 m/s","Annual maximum"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	s_months = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	seasons = ["DJF","MAM","JJA","SON"]
	cnt=1
	for s in np.arange(len(s_months)): 
		print(seasons[s])
		for i in np.arange(len(datasets)+1):
			plt.subplot(4,len(fnames)+1,cnt)
			m.drawcoastlines()
			if i <= 2:
				data = datasets[i]
				temp = data[np.in1d(data["time.month"],s_months[s])]
				temp_first = temp[temp["time.year"] < 1999]
				temp_second = temp[temp["time.year"] >= 1999]
			else:
				temp_first = annmax_da1[seasons[s]]
				temp_second = annmax_da2[seasons[s]]
			trend = ((temp_second.sum("time") - temp_first.sum("time")) / temp_first.sum("time")) * 100
			lat = temp.lat.data
			lon = temp.lon.data
			x,y = np.meshgrid(lon,lat)
			sig = hypothesis_test(temp_first.to_masked_array(), temp_second.to_masked_array(), n_boot)
			trend = trend.where(sig <= 0.05)
			c = trend.plot(cmap = plt.get_cmap("RdBu_r"), \
				vmin=vmax[i][0], vmax=vmax[i][1], add_colorbar = False,\
				add_labels=False)
			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if cnt <= (len(fnames)+1):
				plt.title(titles[i])
			if i == 0:
				plt.ylabel(seasons[s]+"       ",rotation=0)
				plt.ylabel(seasons[s]+"       ",rotation=0)
				plt.ylabel(seasons[s]+"       ",rotation=0)
				plt.ylabel(seasons[s]+"       ",rotation=0)
			if s == 3:
				ax = plt.gca()
				ax_pos = ax.get_position(False).extents
				cax = plt.axes([ ax_pos[0], ax_pos[1]-0.05, ax_pos[2]-ax_pos[0], 0.025])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
				if i == 3:
					cb.set_label("Trend (m/s)")
				else:
					cb.set_label("Trend (%)")
			cnt = cnt+1
	plt.subplots_adjust(wspace=0.2)
		
def plot_aus_syn_annmax_wind_trends():

	#Plot the trends in ERA5 10FG (10 m parametrised wind gust)

	vmax = [[-100,100],[-100,100],[-100,100]]
	n_boot = 1000
	titles = ["20 m/s","25 m/s","30 m/s"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	cnt=1
	for s in ["DJF","MAM","JJA","SON"]: 
		print(s)
		for i in np.arange(len(datasets)):
			data = datasets[i]
			plt.subplot(4,len(fnames),cnt)
			m.drawcoastlines()
			temp = data[np.in1d(data["time.month"],s)]
			temp_first = temp[temp["time.year"] < 1999]
			temp_second = temp[temp["time.year"] >= 1999]
			trend = ((temp_second.sum("time") - temp_first.sum("time")) / temp_first.sum("time")) * 100
			lat = temp.lat.data
			lon = temp.lon.data
			x,y = np.meshgrid(lon,lat)
			sig = hypothesis_test(temp_first.to_masked_array(), temp_second.to_masked_array(), n_boot)
			trend = trend.where(sig <= 0.05)
			c = trend.plot(cmap = plt.get_cmap("RdBu_r"), \
				vmin=vmax[i][0], vmax=vmax[i][1], add_colorbar = False,\
				add_labels=False)
			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if cnt <= len(fnames):
				plt.title(titles[i])
			if i == 0:
				if s == [12,1,2]:
					plt.ylabel("DJF       ",rotation=0)
				if s == [3,4,5]:
					plt.ylabel("MAM       ",rotation=0)
				if s == [6,7,8]:
					plt.ylabel("JJA       ",rotation=0)
				if s == [9,10,11]:
					plt.ylabel("SON       ",rotation=0)
			if s == [9,10,11]:
				ax = plt.gca()
				ax_pos = ax.get_position(False).extents
				cax = plt.axes([ ax_pos[0], ax_pos[1]-0.05, ax_pos[2]-ax_pos[0], 0.025])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
				cb.set_label("Trend (%)")
			cnt = cnt+1
	plt.subplots_adjust(wspace=0.2)
		
def plot_aus_conv_wind_corr():

	#Plotting function/driver for the above function

	nino34 = read_clim_ind("nino34")
	dmi = read_clim_ind("dmi")
	sam = read_clim_ind("sam")
	fnames = ["era5_logit_is_conv_aws_daily.nc","era5_logit_is_sta_daily.nc",\
			"era5_dcp_0.03_daily.nc","era5_t_totals_48.16_daily.nc"]
	datasets = [read_aus_conv_wind_clim(fnames[i], daily = True) for i in np.arange(len(fnames))]
	vmax = [[-1,1],[-1,1],[-1,1],[-1,1]]
	n_boot = 1000
	titles = ["Logistic eq. (measured)","Logistic eq. (reported)","DCP","T-totals"]
	seasons = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	s_names = ["DJF","MAM","JJA","SON"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	clim_inds = {"NINO3.4":nino34,"SAM":sam,"DMI":dmi}
	for clim_ind in ["DMI","NINO3.4","SAM"]:
		print("PLOTTING "+clim_ind)
		plt.figure(figsize=[10,8])
		cnt=1
		for s in np.arange(len(seasons)): 
			print(s_names[s])
			for i in np.arange(len(datasets)):
				data = datasets[i]
				plt.subplot(4,len(fnames),cnt)
				m.drawcoastlines()
				temp = data[np.in1d(data["time.month"],seasons[s])]
				lat = temp.lat.data
				lon = temp.lon.data
				if seasons[s] != [12,1,2]:
					if clim_ind == "DMI":
						temp_seasonal = temp.sel({"time":slice(dt.datetime(1979,1,1), \
							dt.datetime(2018,2,1))}).\
							coarsen(time=3, center=True).sum()
					else:
						temp_seasonal = temp.coarsen(time=3, center=True).sum()
					temp_seasonal_reshaped = np.reshape(temp_seasonal.values, \
							(temp_seasonal.time.shape[0],-1))
					spr_out = [pearson(temp_seasonal_reshaped[:,j], \
						clim_inds[clim_ind][s_names[s]]) \
						for j in np.arange(temp_seasonal_reshaped.shape[1])]
				else:
					temp_seasonal = temp.sel({"time":slice(dt.datetime(1979,12,1), \
						dt.datetime(2018,2,1))}).\
						coarsen(time=3, center=True).sum()
					temp_seasonal_reshaped = np.reshape(temp_seasonal.values, \
							(temp_seasonal.time.shape[0],-1))
					spr_out = [pearson(temp_seasonal_reshaped[:,j], \
						clim_inds[clim_ind][s_names[s]].loc[1979:2017]) \
						for j in np.arange(temp_seasonal_reshaped.shape[1])]
				r = np.array([spr_out[j][0] for j in np.arange(len(spr_out))])
				p = np.array([spr_out[j][1] for j in np.arange(len(spr_out))])
				r = r.reshape((lat.shape[0], lon.shape[0]))	
				p = p.reshape((lat.shape[0], lon.shape[0]))	

				x,y = np.meshgrid(lon,lat)
				lsm = get_era5_mask(lon,lat)
				r = np.where(lsm>=0.5, r, np.nan)
				p = np.where(lsm>=0.5, p, np.nan)
				c = m.pcolor(x, y, r, cmap = plt.get_cmap("RdBu_r"), \
					vmin=vmax[i][0], vmax=vmax[i][1])
				m.contourf(x,y,np.where(p<=0.05, 1, 0), levels=[.5,1.5], \
					colors=["none", "grey"], hatches=["///////"], alpha=0.05)
				matplotlib.rcParams['hatch.linewidth'] = 0.5
				plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
				if i == 0:
					if seasons[s] == [12,1,2]:
						plt.ylabel("DJF       ",rotation=0)
					if seasons[s] == [3,4,5]:
						plt.ylabel("MAM       ",rotation=0)
					if seasons[s] == [6,7,8]:
						plt.ylabel("JJA       ",rotation=0)
					if seasons[s] == [9,10,11]:
						plt.ylabel("SON       ",rotation=0)
				if s == 0:
					plt.title(titles[i])
				if (s == 3) & (i == (len(datasets)-1) ):
					cax = plt.axes([0.2, 0.075, 0.6, 0.01])
					cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
					cb.set_label("Pearson's r")
				cnt = cnt+1
		plt.subplots_adjust(wspace=0.2)
		plt.savefig("/g/data/eg3/ab4502/figs/ExtremeWind/corr/"+clim_ind+"_era5.tiff",\
			bbox_inches="tight")
		
def plot_aus_mjo_corr():

	#Plot the difference in the number of convective wind environment days between MJO active and in

	#Define the number of bootstraps
	n_boot = 100
	#Define file and variable names for the four diagnostics of interest. 
	# For each diagnostic, there are two daily netcdf files, one for active MJO days only, 
	# the other for inactive MJO days
	f = ["logit_is_conv_aws","logit_is_sta","dcp_0.03","t_totals_48.16"]
	var = ["logit","logit","dcp","t_totals"]
	seasons = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	s_names = ["DJF","MAM","JJA","SON"]
	#Plotting settings
	titles = ["Logistic eq. (measured)","Logistic eq. (STA)","DCP","T-totals"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" \
		for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	cnt=1
	for s in np.arange(len(seasons)):
		#For each diagnostic
		for i in np.arange(len(var)):
			print(f[i])
			#Draw the filled contour plot, which is the difference in the mean number of days where the diagnostic is above
			# a set threshold between MJO active days and inactive days
			plt.subplot(4,len(f),cnt)
			m.drawcoastlines()
			active_da = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_"+\
				f[i]+"_mjo_active.nc")[var[i]]
			inactive_da = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_"+\
				f[i]+"_mjo_inactive.nc")[var[i]]
			active_da = active_da[np.in1d(active_da["time.month"],seasons[s])]
			inactive_da = inactive_da[np.in1d(inactive_da["time.month"],seasons[s])]
			active_mean = active_da.mean("time")
			inactive_mean = inactive_da.mean("time")
			trend = ((active_mean - inactive_mean) / active_mean * 100)
			lsm = get_era5_mask(active_da.lon.values,active_da.lat.values)
			trend = trend.where(lsm>=0.5, np.nan)
			c = trend.plot(cmap = plt.get_cmap("RdBu_r"), \
				add_colorbar = False, add_labels=False,\
				vmax = 50, vmin=-50)

			#Perform bootstrapping
			#stat = []
			#combined_da = np.vstack([active_da.values, inactive_da.values])
			#for n in np.arange(0,n_boot):
			#	#Merge the active and inactive files into one, and randomly sample for active and inactive days
			#	temp_active_mean = combined_da[np.random.randint(0, active_da.mjo_days-1, \
			#			    active_da.mjo_days)].mean(axis=0)
			#	temp_inactive_mean = combined_da[np.random.randint(0, inactive_da.mjo_days-1, \
			#			    inactive_da.mjo_days)].mean(axis=0)
				#Calculate the trend for the randomly sampled files
			#	stat.append(\
			#	    ((temp_active_mean - temp_inactive_mean) / temp_active_mean * 100))
			#stat = np.stack(stat)
			#p_up = np.sum(trend.values >= stat,axis=0) / n_boot
			#p_low = np.sum(trend.values <= stat,axis=0) / n_boot
			#sig = (2*np.min(np.stack((p_low,p_up)),axis=0))
			from scipy.stats import ttest_ind
			temp, sig = ttest_ind(active_da,inactive_da,equal_var=False,axis=0)

			sig = np.where(lsm>=0.5, sig, 1)
			plt.rcParams['hatch.linewidth'] = 0.5
			x,y = np.meshgrid(active_da.lon.values,active_da.lat.values)
			m.contourf(x,y,np.where(sig<=.5/100., 1, 0), levels=[.5,1.5], \
				colors=["none", "grey"], hatches=["///////"], alpha=0)

			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if cnt <= len(var):
				plt.title(titles[i])
			if cnt == 3:
				ax = plt.gca()
				ax_pos = ax.get_position(False).extents
				cax = plt.axes([ 0.3, 0.08, 0.4, 0.025])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
				cb.set_label("Relative difference (%)")
			cnt = cnt+1
	plt.savefig("/g/data/eg3/ab4502/figs/ExtremeWind/corr/MJO_era5_"+\
	    f[i]+".png")
				
def plot_aus_daily_mjo_by_phase():
	from scipy.stats import ttest_ind

	f = "/g/data/eg3/ab4502/ExtremeWind/aus/mjo_full_phases_era5_mlcape*s06.nc"
	data = xr.open_dataset(f)
	lsm = get_era5_mask(data.lon.values,data.lat.values)
	x,y = np.meshgrid(data.lon, data.lat)
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
    
	seasons = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	s_names = ["DJF","MAM","JJA","SON"]
	cnt=0
	for s in np.arange(len(seasons)):
		#Get the total envs. per mjo day for the season
		total = np.concatenate([\
		    data["phase1"][np.in1d(data["phase1_times.month"],seasons[s])].values, \
		    data["phase2"][np.in1d(data["phase2_times.month"],seasons[s])].values, \
		    data["phase3"][np.in1d(data["phase3_times.month"],seasons[s])].values, \
		    data["phase4"][np.in1d(data["phase4_times.month"],seasons[s])].values, \
		    data["phase5"][np.in1d(data["phase5_times.month"],seasons[s])].values, \
		    data["phase6"][np.in1d(data["phase6_times.month"],seasons[s])].values, \
		    data["phase7"][np.in1d(data["phase7_times.month"],seasons[s])].values, \
		    data["phase8"][np.in1d(data["phase8_times.month"],seasons[s])].values, \
			], axis=0)
		for phase in [ [2,3], [4,5], [6,7], [8,1] ]:
			print(phase)
			plt.subplot(4,4,cnt+1)
			
			if cnt <= 3:
				plt.title("Phase "+str(phase[0])+"+" +str(phase[1]))
			if cnt in [0, 4, 8, 12]:
				plt.ylabel(s_names[s])

			temp = np.concatenate([\
			    data["phase"+str(phase[0])][\
			    np.in1d(data["phase"+str(phase[0])+"_times.month"],seasons[s])].values, \
			    data["phase"+str(phase[1])][\
			    np.in1d(data["phase"+str(phase[1])+"_times.month"],seasons[s])].values],\
				axis=0)

			temp = np.where(lsm>=0.5, temp, np.nan)
    
			c=m.pcolormesh(x,y,\
			    (temp.mean(axis=0) - total.mean(axis=0)) / total.mean(axis=0) * 100,\
				vmin=-50, vmax=50,cmap=plt.get_cmap("RdBu_r") )

			none, sig = ttest_ind(temp,\
				    total,equal_var=False,axis=0)
			sig = np.where(lsm>=0.5, sig, 1)
			m.contourf(x,y,np.where(sig<=0.05, 1, 0), levels=[.5,1.5], \
				colors=["none", "grey"], hatches=["///////"], alpha=0)

			m.drawcoastlines()
			if cnt==15:
				cax = plt.axes([0.2, 0.075, 0.6, 0.01])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal",\
						extend="both")
				cb.set_label("Environments per MJO-day anomaly (%)")
			cnt=cnt+1
	plt.show()

def plot_aus_monthly_mjo_corr():

	#Define the number of bootstraps
	n_boot = 100
	#Define file and variable names for the four diagnostics of interest. 
	# For each diagnostic, there are two daily netcdf files, one for active MJO days only, 
	# the other for inactive MJO days
	f = ["era5_logit_is_conv_aws_daily.nc","era5_logit_is_sta_daily.nc","era5_dcp_0.03_daily.nc","era5_t_totals_48.16_daily.nc","era5_mlcape*s06_20000.0_daily.nc"]
	var = ["logit","logit","dcp","t_totals","mlcape*s06"]
	datasets = [read_aus_conv_wind_clim(f[i], daily = True) for i in np.arange(len(f))]
	seasons = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
	s_names = ["DJF","MAM","JJA","SON"]
	all_times = pd.date_range("1979-01-01","2018-12-31")
	active_times = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_t_totals_48.16_mjo_active.nc").time.values
	inactive_times = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5_t_totals_48.16_mjo_inactive.nc").time.values
	time_df = pd.DataFrame({"active":np.in1d(all_times,active_times) * 1,\
		    "inactive":np.in1d(all_times,inactive_times) * 1} , index=all_times).\
		resample("1M").sum()
    
	#Plotting settings
	vmax = [[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]
	titles = ["Logistic eq. (measured)","Logistic eq. (STA)","DCP","T-totals","ML-CS6"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" \
		for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	cnt=1
	for s in np.arange(len(seasons)):
		#For each diagnostic
		for i in np.arange(len(var)):
			data = datasets[i]
			plt.subplot(4,len(f),cnt)
			m.drawcoastlines()
			temp_seasonal = data[np.in1d(data["time.month"],seasons[s])]
			lat = temp_seasonal.lat.data
			lon = temp_seasonal.lon.data

			temp_seasonal_reshaped = np.reshape(temp_seasonal.values, \
				(temp_seasonal.time.shape[0],-1))
			spr_out = [pearson(temp_seasonal_reshaped[:,j], \
				np.squeeze(time_df.loc[np.in1d(time_df.index.month, \
				seasons[s]), "active"].values)) \
					for j in np.arange(temp_seasonal_reshaped.shape[1])]
			r = np.array([spr_out[j][0] for j in np.arange(len(spr_out))])
			p = np.array([spr_out[j][1] for j in np.arange(len(spr_out))])
			r = r.reshape((lat.shape[0], lon.shape[0]))	
			p = p.reshape((lat.shape[0], lon.shape[0]))	


			x,y = np.meshgrid(lon,lat)
			lsm = get_era5_mask(lon,lat)
			r = np.where(lsm>=0.5, r, np.nan)
			p = np.where(lsm>=0.5, p, np.nan)
			c = m.pcolor(x, y, r, cmap = plt.get_cmap("RdBu_r"), \
				vmin=vmax[i][0], vmax=vmax[i][1])
			m.contourf(x,y,np.where(p<=0.05, 1, 0), levels=[.5,1.5], \
				colors=["none", "grey"], hatches=["///////"], alpha=0.05)
			m.drawcoastlines()
			matplotlib.rcParams['hatch.linewidth'] = 0.5
			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if s == 0:
				plt.title(titles[i])
			if (s == 3) & (i == (len(datasets)-1) ):
				cax = plt.axes([0.2, 0.075, 0.6, 0.01])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal")
				cb.set_label("Pearson's r")
			if i == 0:
				if seasons[s] == [12,1,2]:
					plt.ylabel("DJF       ",rotation=0)
				if seasons[s] == [3,4,5]:
					plt.ylabel("MAM       ",rotation=0)
				if seasons[s] == [6,7,8]:
					plt.ylabel("JJA       ",rotation=0)
				if seasons[s] == [9,10,11]:
					plt.ylabel("SON       ",rotation=0)
			cnt=cnt+1
	plt.subplots_adjust(wspace=0.2)   

def plot_aus_mean_clim():

	#Plot the difference in the mean of a variable at each ERA5 grid point, from the first half to 
	# the second half of the study period

	matplotlib.rcParams['hatch.linewidth'] = 0.5
	fnames = ["era5_lr36_mean.nc","era5_lr_freezing_mean.nc","era5_mhgt_mean.nc",\
		    "era5_ml_cape_mean.nc","era5_ml_el_mean.nc","era5_qmean01_mean.nc",\
			"era5_qmeansubcloud_mean.nc","era5_s06_mean.nc","era5_srhe_left_mean.nc",\
			    "era5_Umean06_mean.nc",]
	vmax = [None, None]
	titles = ["LR36", "LR-Freezing", "MHGT", "MLCAPE", "MLEL", "Qmean01", "Qmean-Subcloud",\
		"S06", "SRHE", "Umean06"]
	
	datasets = [read_aus_conv_wind_clim(fnames[i], daily = True) for i in np.arange(len(fnames))]
	n_boot = 1000
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	cnt=1
	for i in np.arange(len(fnames)):
		print(fnames[i])
		temp = datasets[i]
		plt.subplot(5,2,cnt)
		m.drawcoastlines()
		temp_first = temp[temp["time.year"] < 1999]
		temp_second = temp[temp["time.year"] >= 1999]
		if titles[i] == "LR-Freezing":
			temp_first = np.where((temp_first > 50) | (temp_first < -50), np.nan, temp_first)
			temp_second = np.where((temp_second > 50) | (temp_second < -50), np.nan, temp_second)
			trend = ((np.nanmean(temp_second, axis=0) - np.nanmean(temp_first,axis=0)) /\
				np.nanmean(temp_first, axis=0)) * 100
		else:
			trend = ((temp_second.mean("time") - temp_first.mean("time")) / \
				temp_first.mean("time")) * 100
			temp_first = temp_first.to_masked_array()
			temp_second = temp_second.to_masked_array()
		lat = temp.lat.data
		lon = temp.lon.data
		x,y = np.meshgrid(lon,lat)
		lsm = get_era5_mask(lon,lat)
		trend = np.where(lsm>=0.5, trend, np.nan)
		c = m.contourf(x,y,trend,cmap = plt.get_cmap("RdBu_r"), \
			vmin=vmax[0], vmax=vmax[1],\
			locator=matplotlib.ticker.MaxNLocator(symmetric=True))
		plt.colorbar()
		sig = hypothesis_test(temp_first, \
			temp_second, n_boot)
		sig = np.where(lsm>=0.5, sig, 1)
		m.contourf(x,y,np.where(sig<=0.05, 1, 0), levels=[.5,1.5], \
			colors=["none", "grey"], hatches=["///////"], alpha=0)
		plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
		plt.title(titles[i])
		#if (i == (len(datasets)-1) ):
		#	cax = plt.axes([0.2, 0.075, 0.6, 0.01])
		#	cb = plt.colorbar(c, cax=cax, orientation = "horizontal", \
		#		extend="both")
		#	cb.set_label("Trend (%)")

		cnt = cnt+1
				

def plot_aus_conv_wind_clim():

	#Plotting function/driver for the above function

	#fnames = ["barra_logit_is_conv_aws_daily.nc","barra_logit_is_sta_daily.nc",\
	#	"barra_dcp_0.03_daily.nc","barra_t_totals_49.2_daily.nc"]
	fnames = ["era5_logit_is_conv_aws_daily.nc", "era5_logit_is_sta_daily.nc",\
		"era5_dcp_0.03_daily.nc", "era5_t_totals_48.16_daily.nc"]
	models = ["era5","era5","era5","era5"]
	datasets = [read_aus_conv_wind_clim(fnames[i], daily = True) for i in np.arange(len(fnames))]
	vmax = [0.5,0.5,0.5, 0.5]
	titles = ["Logistic eq. (measured)", "Logistic eq. (reported)", "DCP", "T-totals"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)]; alph = [alph[i]+")" for i in np.arange(len(alph))]
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")
	plt.figure(figsize=[10,8])
	cnt=1
	for s in [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]: 
		print(s)
		for i in np.arange(len(datasets)):
			data = datasets[i]
			plt.subplot(4,len(fnames),cnt)
			m.drawcoastlines()
			temp = data[np.in1d(data["time.month"],s)]
			temp = temp.sum("time") / (data.steps[np.in1d(data["time.month"],s)].sum() / 24)
			lat = temp.lat.data
			lon = temp.lon.data
			x,y = np.meshgrid(lon,lat)
			if models[i] == "barra":
				lsm = get_barra_mask(lon,lat)
				temp = temp.where(lsm==1, np.nan)
			elif models[i] == "era5":
				lsm = get_era5_mask(lon,lat)
				temp = temp.where(lsm>=0.5, np.nan)
			c = temp.plot(cmap = plt.get_cmap("Reds"), \
				vmax=vmax[i], vmin=0, add_colorbar = False,\
				add_labels=False, extend="max")
			plt.annotate(alph[cnt-1], xy=(0.05, 0.05), xycoords='axes fraction')
			if cnt <= len(fnames):
				plt.title(titles[i])
			if i == 0:
				if s == [12,1,2]:
					plt.ylabel("DJF       ",rotation=0)
				if s == [3,4,5]:
					plt.ylabel("MAM       ",rotation=0)
				if s == [6,7,8]:
					plt.ylabel("JJA       ",rotation=0)
				if s == [9,10,11]:
					plt.ylabel("SON       ",rotation=0)
			if (s == [9,10,11]) & (i == (len(datasets)-1) ):
				cax = plt.axes([0.2, 0.075, 0.6, 0.01])
				cb = plt.colorbar(c, cax=cax, orientation = "horizontal", extend="max")
				cb.set_label("Daily frequency")
			cnt = cnt+1
	plt.subplots_adjust(wspace=0.2)
		

if __name__ == "__main__":

	#daily_max_clim("sa_small","erai",[1979,2017],\
		#var_list=["scp"],\
		#threshold=[0.018],\
		#levels=[np.arange(0,55,5)],\
		#trend_levels = [np.arange(-50,60,10)],\
		#plot_trends=True, trend_on_cond_days="mf",n_boot = 1000, plot_corr = False)

	#FIGURE 7:
	#daily_max_clim("sa_small","erai_fc",[1979,2017],threshold=[21.5],var_list=["wg10"],\
	#		levels=[np.arange(0,2.2,.2)],plot_trends=True,plot_corr=True,\
	#		trend_levels=np.arange(-175,175,25),seasons = [np.arange(1,13,1),[11,12,1],[2,3,4],\
	#		[5,6,7],[8,9,10]])
	#99.6 percentile BARRA-R wind gust: 25.25. ERA-Interim: 21.5

	#FIGURE 20:
	#daily_max_clim("sa_small","erai",[1979,2017],threshold=[9443,0.05,False,False,False,120],\
	#		var_list=["cape*s06","dcp","cond","mf","sf","ml_cape"],\
	#		levels=[np.arange(0,55,5),np.arange(0,55,5),\
	#			np.arange(0,55,5),np.arange(0,55,5),np.arange(0,22.5,2.5),\
	#			np.arange(0,55,5)],\
	#		trend_levels = [np.arange(-50,60,10),np.arange(-50,60,10),\
	#			np.arange(-50,60,10),\
	#			np.arange(-50,60,10),np.arange(-50,60,10),np.arange(-50,60,10)],\
	#		plot_trends=True,plot_corr=True,n_boot = 1000)
	#daily_max_clim("sa_small","erai",[2003,2016],threshold=[9443,0.05,False,False,False],\
	#		var_list=["cape*s06","dcp","cond","mf","sf"],\
	#		levels=[np.arange(0,55,5),np.arange(0,55,5),\
	#			np.arange(0,55,5),np.arange(0,55,5),np.arange(0,22.5,2.5)])
	#daily_max_clim("sa_small","barra",[2003,2016],threshold=[9443,0.05,False,False,False],\
	#		var_list=["cape*s06","dcp","cond","mf","sf"],\
	#		levels=[np.arange(0,55,5),np.arange(0,55,5),\
	#			np.arange(0,55,5),np.arange(0,55,5),np.arange(0,22.5,2.5)])

	#Figure 4
	#daily_max_clim("sa_small","erai_fc",[1979,2017],threshold=[21.5],var_list=["wg10"],plot_trends=True,\
	#	trend_levels = [np.arange(-150,175,25)],plot_corr = True,n_boot=100,\
	#	levels=[np.linspace(0,2,11)],seasons=[[11,12,1],[2,3,4],[5,6,7],[8,9,10]],log_cscale=True)
	#daily_max_clim("sa_small","erai_fc",[2003,2016],threshold=[21.5],var_list=["wg10"],\
	#	levels=[np.linspace(0,2,11)],seasons=[[11,12,1],[2,3,4],[5,6,7],[8,9,10]],log_cscale=True)
	#daily_max_clim("sa_small","barra_r_fc",[2003,2016],threshold=[23.75],var_list=["max_wg10"],\
	#	levels=[np.linspace(0,2,11)],seasons=[[11,12,1],[2,3,4],[5,6,7],[8,9,10]],log_cscale=True)

	#Ingredients
	#daily_max_clim("aus","erai",[1979,2017],var_list=["dcp2"],\
			#plot_trends=True,\
			#threshold = [0.0038],\
			#n_boot=1000,\
			#seasons=[np.arange(1,13,1)])

	#plot_aus_conv_wind_trends()
	#plot_aus_conv_wind_corr()
	plot_aus_daily_mjo_by_phase()
