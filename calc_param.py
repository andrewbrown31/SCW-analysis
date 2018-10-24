import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import os


#Functions to take reanalysis data at a single time step over some spatial domain, and calculate TS parameters


def get_dp(ta,hur):
	#Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
	#Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
	a = 17.27
	b = 237.7
	alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
	return (b*alpha) / (a - alpha)

def get_point(point,lon,lat,ta,dp,hgt,ua,va,uas,vas):
	# Return 1d arrays for all variables, at a given spatial point (now a function of p-level only)
	lon_ind = np.argmin(abs(lon-point[0]))
	lat_ind = np.argmin(abs(lat-point[1]))
	ta = np.squeeze(ta[:,lat_ind,lon_ind])
	dp = np.squeeze(dp[:,lat_ind,lon_ind])
	hgt = np.squeeze(hgt[:,lat_ind,lon_ind])
	ua = np.squeeze(ua[:,lat_ind,lon_ind])
	va = np.squeeze(va[:,lat_ind,lon_ind])
	uas = np.squeeze(uas[lat_ind,lon_ind])
	vas = np.squeeze(vas[lat_ind,lon_ind])

	return [ta,dp,hgt,ua,va,uas,vas]

def calc_param(times,ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,param,model,save):
	#For each time in "times", loop over lat/lon points in domain and calculate:
	# 1) profile 2) parcel (if create_parcel is set) 3) parameter
	#NOTE the choice of parameter may affect both steps 2) and 3)
	#Output is a list of numpy arrays of length=len(params) with dimensions [time,lat,lon]
	#Option to save as a netcdf file

	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(times),len(lat),len(lon)))

	for t in np.arange(0,len(times)):
		#print(times[t])
		for x in np.arange(0,len(lon)):
			for y in np.arange(0,len(lat)):
				#Restrict to a single spatial point
				start=dt.datetime.now()
				point = [lon[x],lat[y]]   
				ta_p,dp_p,hgt_p,ua_p,va_p,uas_p,vas_p = get_point(point,\
					lon,lat,ta[t,:,:,:],dp[t,:,:,:],hgt[t,:,:,:],ua[t,:,:,:],\
					va[t,:,:,:],uas[t,:,:],vas[t,:,:])
				print("POINT SLICING: "+ str(dt.datetime.now() - start))
				#Convert u and v to kts for use in profile
				ua_p_kts = utils.MS2KTS(ua_p)
				va_p_kts = utils.MS2KTS(va_p)
				#Create profile
				start=dt.datetime.now()
				prof = profile.create_profile(pres=p, hght=hgt_p, tmpc=ta_p, \
						dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
				print("PROFILE" + str(dt.datetime.now() - start))
				#Create most unstable parcel
				start=dt.datetime.now()
				mu_parcel = params.parcelx(prof, flag=3,dp=-10)
				print("PARCEL" + str(dt.datetime.now() - start))
				if "relhum850-500" in param:
					param_ind = np.where(param=="relhum850-500")[0][0]
					param_out[param_ind][t,y,x] = \
						np.mean(hur_p[(p_p<=851) & (p_p>=499)])
				if "mu_cape" in param:
				#CAPE for most unstable parcel
					start=dt.datetime.now()
					param_ind = np.where(param=="mu_cape")[0][0]
					param_out[param_ind][t,y,x] = mu_parcel.bplus
					print("CAPE" + str(dt.datetime.now() - start))
				if "s06" in param:
				#Wind shear 10 m (sfc) to 6 km
					start=dt.datetime.now()
					ua_0km = uas_p
					ua_6km = np.interp(6000,hgt_p,ua_p)
					va_0km = vas_p
					va_6km = np.interp(6000,hgt_p,va_p)
					shear = np.sqrt(np.square(ua_6km-ua_0km)+\
						np.square(va_6km-va_0km))
					param_ind = np.where(param=="s06")[0][0]
					param_out[param_ind][t,y,x] = shear
					print("s06" + str(dt.datetime.now() - start))
				if "mu_cin" in param:
				#CIN for most unstable parcel
					param_ind = np.where(param=="mu_cin")[0][0]
					param_out[param_ind][t,y,x] = -1*mu_parcel.bminus
				if "hel01" in param:
				#Combined (+ve and -ve) rel. helicity from 0-1 km
					param_ind = np.where(param=="hel01")[0][0]
					param_out[param_ind][t,y,x] = winds.helicity(prof,0,1000)[0]
				if "hel03" in param:
				#Combined (+ve and -ve) rel. helicity from 0-3 km
					param_ind = np.where(param=="hel03")[0][0]
					param_out[param_ind][t,y,x] = winds.helicity(prof,0,3000)[0]
				if "hel06" in param:
				#Combined (+ve and -ve) rel. helicity from 0-6 km
					param_ind = np.where(param=="hel06")[0][0]
					param_out[param_ind][t,y,x] = winds.helicity(prof,0,6000)[0]
				if "ship" in param:
				#Significant hail parameter
					param_ind = np.where(param=="ship")[0][0]
					param_out[param_ind][t,y,x] = params.ship(prof,\
						mupcl=mu_parcel)
				if "lhp" in param:
				#Large Hail Paramerer; NOTE requires convective profile (costly).
					conf_prof = profile.create_profile(profile="convective",\
						pres=p, hght=hgt_p, tmpc=ta_p, \
						dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
					param_ind = np.where(param=="lhp")[0][0]
					param_out[param_ind][t,y,x] = params.lhp(prof)
				if "hgz_depth" in param:
				#Hail growzth zone (in hPa)
					param_ind = np.where(param=="hgz_depth")[0][0]
					param_out[param_ind][t,y,x] = abs(params.hgz(prof)[1]\
						 - params.hgz(prof)[0])
				if "dcp" in param:
				#Derecho Composite Parameter ~ cold pool driven wind events
					param_ind = np.where(param=="dcp")[0][0]
					param_out[param_ind][t,y,x] = params.dcp(prof)
				if "mburst" in param:
				#Microburst composite index
					param_ind = np.where(param=="mburst")[0][0]
					param_out[param_ind][t,y,x] = params.mburst(prof)
				if "mmp" in param:
				#Mesoscale Convective System Maintanance Probability
					param_ind = np.where(param=="mmp")[0][0]
					param_out[param_ind][t,y,x] = params.mmp(prof,\
						mupcl=mu_parcel)

	if save:
		fname = "/g/data/eg3/ab4502/ExtremeWind/"+model+"_"+\
			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(times[-1],"%Y%m%d")+".nc"
		if os.path.isfile(fname):
			os.remove(fname)
		param_file = nc.Dataset(fname,"w")
		time_dim = param_file.createDimension("time",len(times))
		lat_dim = param_file.createDimension("lat",len(lat))
		lon_dim = param_file.createDimension("lon",len(lon))

		date_list_str = np.array([dt.datetime.strftime(x,"%Y-%m-%d %H:%M") \
			for x in times])
		time_var = param_file.createVariable("time",date_list_str.dtype,("time",))
		lat_var = param_file.createVariable("lat",lat.dtype,("lat",))
		lon_var = param_file.createVariable("lon",lat.dtype,("lon",))
		time_var[:] = date_list_str
		lat_var[:] = lat
		lon_var[:] = lon

		for i in np.arange(0,len(param)):
			temp_var = param_file.createVariable(param[i],param_out[i].dtype,\
				("time","lat","lon"))
			temp_var[:] = param_out[i]
		
	return param_out


def calc_param_points(times,ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used,param,loc_id):

	#Same as calc_param except from inputs generated by read_*model*_points()
	#Variables are now of the form [time, pressure, points]
	#Lat/lon are of length = number of points
	#Output is a pandas data frame

	param = np.array(param)
	values = np.empty((len(lon)*len(times),len(param)))
	values_lat = []
	values_lon = []
	values_lon_used = []
	values_lat_used = []
	values_loc_id = []
	values_year = []; values_month = []; values_day = []; values_hour = []; values_minute = []
	cnt = 0
	for point in np.arange(0,len(lon)):
		print(lon[point],lat[point])
		for t in np.arange(len(times)):
			print(times[t])
	
			hgt_p = hgt[t,:,point]; ta_p = ta[t,:,point]; dp_p = dp[t,:,point];
			hur_p = hur[t,:,point];	ua_p = ua[t,:,point]; va_p = va[t,:,point]; 
			uas_p = uas[t,point]; vas_p = vas[t,point]; p_p = p[t,:,point]

			#Convert u and v to kts for use in profile
			ua_p_kts = utils.MS2KTS(ua_p)
			va_p_kts = utils.MS2KTS(va_p)
			#Create profile
			prof = profile.create_profile(pres=p_p, hght=hgt_p, tmpc=ta_p, \
					dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
			#Create most unstable parcel
			mu_parcel = params.parcelx(prof, flag=3)
			if "relhum850-500" in param:
				values[cnt,param=="relhum850-500"] = \
					np.mean(hur_p[(p_p<=851) & (p_p>=499)])
			if "mu_cape" in param:
			#CAPE for most unstable parcel
				values[cnt,param=="mu_cape"] = mu_parcel.bplus
			if "s06" in param:
			#Wind shear 10 m (sfc) to 6 km
				ua_0km = uas_p
				ua_6km = np.interp(6000,hgt_p,ua_p)
				va_0km = vas_p
				va_6km = np.interp(6000,hgt_p,va_p)
				shear = np.sqrt(np.square(ua_6km-ua_0km)+np.square(va_6km-va_0km))
				values[cnt,param=="s06"] = shear
			if "mu_cin" in param:
			#CIN for most unstable parcel
				values[cnt,param=="mu_cin"] = -1*mu_parcel.bminus				
			if "hel03" in param:
			#Combined (+ve and -ve) rel. helicity from 0-3 km
				values[cnt,param=="hel03"] = winds.helicity(prof,0,3000)[0]
			if "hel06" in param:
			#Combined (+ve and -ve) rel. helicity from 0-6 km
				values[cnt,param=="hel06"] = winds.helicity(prof,0,6000)[0]
			if "ship" in param:
			#Significant hail parameter
				values[cnt,param=="ship"] = params.ship(prof,mupcl=mu_parcel)
			if "lhp" in param:
			#Large Hail Paramerer; NOTE requires convective profile (costly).
				conf_prof = profile.create_profile(profile="convective",pres=p_p, hght=hgt_p, tmpc=ta_p, \
					dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
				values[cnt,param=="lhp"] = params.lhp(prof)
			if "hgz_depth" in param:
			#Hail growzth zone (in hPa)
				values[cnt,param=="hgz_depth"] = abs(params.hgz(prof)[1] - params.hgz(prof)[0])
			if "dcp" in param:
			#Derecho Composite Parameter ~ cold pool driven wind events
				values[cnt,param=="dcp"] = params.dcp(prof)
			if "mburst" in param:
			#Microburst composite index
				values[cnt,param=="mburst"] = params.mburst(prof)
			if "mmp" in param:
			#Mesoscale Convective System Maintanance Probability
				values[cnt,param=="mmp"] = params.mmp(prof,mupcl=mu_parcel)

			values_lat.append(lat[point])
			values_lon.append(lon[point])
			values_lat_used.append(lat_used[point])
			values_lon_used.append(lon_used[point])
			values_loc_id.append(loc_id[point])
			values_year.append(times[t].year)
			values_month.append(times[t].month)
			values_day.append(times[t].day)
			values_hour.append(times[t].hour)
			values_minute.append(times[t].minute)
			cnt = cnt+1

	df = pd.DataFrame(values,columns=param)
	df["lat"] = values_lat
	df["lon"] = values_lon
	df["lon_used"] = values_lon_used
	df["lat_used"] = values_lat_used
	df["loc_id"] = values_loc_id
	df["year"] = values_year
	df["month"] = values_month
	df["day"] = values_day
	df["hour"] = values_hour
	df["minute"] = values_minute
	return df	
