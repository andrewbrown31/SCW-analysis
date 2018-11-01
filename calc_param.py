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
from skewt import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import wrf

#-------------------------------------------------------------------------------------------------

#This file contains functions to extract thunderstorm/extreme-convective-wind-gust parameters 
# from gridded model data, provided by *model*_read.py

#There are currently three main functions:
#	- calc_param_wrf
#		Extract parameters in a vectorised fashion, such that calculations are done once
#		for a spatial domain at each time step. Uses wrf-python for CAPE and
#		user-written functions for winds.
#
#	- calc_param_sharppy
#		Extract parameters by looping over time, latitude and longitude, and creating 
#		"profiles" with the SHARPpy package. Too slow to run for the whole reanalysis 
#		period
#
#	- calc_param_points
#		Extract parameters by looping over time, at a set of points provided by the 
#		user to the function. Currently, both wrf-python and SHARPpy can be used to 
#		calculate the parameters, by commenting/un-commenting lines of code.
#
#NOTE: "metpy" by unicode is used in some places for calculating specific humidity from
# thermodynamic properties
#
#NOTE: Currently, user defined wind functions used in calc_param_wrf use the bottom available
# pressure level as "0km"
#
#NOTE: When taking mean wind over a layer, does a weighted mean need to be used? Currently, 
# get_mean_wind interpolates heights to be evenly spaced, which I think may get around the
# weighting issue anyway (problem is caused by values clustered towards lower levels).
#
#NOTE: Should add in dependencies to if statements within calc_param_wrf. For example, if "ship"
# is included in the parameter list, then "mu_cape" must also be present, etc. Add something 
# within "ship" block like: [is mu_cape in the parameter list? -> if not, throw error with 
# message "mu_cape must be in the parameter list to calculate ship"]
#
#NOTE: Because data is on pressure levels, much greater efficiency could be achieved by 
# approximating height-based indicies at equivalent standard pressure levels
#-------------------------------------------------------------------------------------------------


def get_dp(ta,hur):
	#Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
	#Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
	a = 17.27
	b = 237.7
	alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
	return (b*alpha) / (a - alpha)

def get_point(point,lon,lat,ta,dp,hgt,ua,va,uas,vas):
	# Return 1d arrays for all variables, at a given spatial point (now a function
	# of p-level only)
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

def get_mean_wind(u,v,hgt,hgt_bot,hgt_top,density_weighted,density):
	#Get mean wind components [lat, lon] based on 3d input of u, v, hgt and p [levels,lat,lon]
	#Note lowest possible height is equal to bottom pressure level (1000 hPa)
	u_mean = np.empty((u.shape[1],u.shape[2]))
	v_mean = np.empty((u.shape[1],u.shape[2]))
	for i in np.arange(0,u.shape[1]):
	    for j in np.arange(0,u.shape[2]):
		if hgt_bot == 0:
			x = np.arange(hgt[:,i,j].min().round(-1),hgt_top+10,10)
		else:
			x = np.arange(hgt_bot,hgt_top+10,10)
		xp = hgt[:,i,j]
		u_interp = np.interp(x,xp,u[:,i,j])
		v_interp = np.interp(x,xp,v[:,i,j])
		if density_weighted:
			temp_density = np.interp(x,xp,density[:,i,j])
			u_mean[i,j] = (np.sum(temp_density*u_interp)) / (np.sum(temp_density))
			v_mean[i,j] = (np.sum(temp_density*v_interp)) / (np.sum(temp_density))
		else:
			u_mean[i,j] = np.mean(u_interp)
			v_mean[i,j] = np.mean(v_interp)
	return [u_mean,v_mean]

def get_shear_hgt(u,v,hgt,hgt_bot,hgt_top):
	#Get bulk wind shear [lat, lon] between two heights, based on 3d input of u, v, and 
	#hgt [levels,lat,lon]
	#Note lowest possible height is equal to bottom pressure level (1000 hPa)
	shear = np.empty((u.shape[1],u.shape[2]))
	for i in np.arange(0,u.shape[1]):
	    for j in np.arange(0,u.shape[2]):
		xp = hgt[:,i,j]
		u_interp = np.interp([hgt_bot,hgt_top],xp,u[:,i,j])
		v_interp = np.interp([hgt_bot,hgt_top],xp,v[:,i,j])
		shear[i,j] = np.sqrt(np.square(u_interp[1]-u_interp[0])+np.square(v_interp[1]\
				-v_interp[0]))		
	return shear

def get_shear_p(u,v,p,p_bot,p_top):
	#Get bulk wind shear [lat, lon] between two pressure levels, based on 3d input of u, v, and 
	#p [levels,lat,lon]
	#p_bot and p_top given in hPa
	shear = np.empty((u.shape[1],u.shape[2]))
	for i in np.arange(0,u.shape[1]):
	    for j in np.arange(0,u.shape[2]):
		xp = np.flipud(p[:,i,j])
		u_interp = np.interp([p_top,p_bot],xp,np.flipud(u[:,i,j]))
		v_interp = np.interp([p_top,p_bot],xp,np.flipud(v[:,i,j]))
		shear[i,j] = np.sqrt(np.square(u_interp[1]-u_interp[0])+np.square(v_interp[1]\
				-v_interp[0]))		
	return shear

def get_srh(u,v,hgt,hgt_top):
	#Get storm relative helicity [lat, lon] based on 3d input of u, v, and storm motion u and
	# v components
	# Is between the bottom pressure level (1000 hPa), approximating 0 m, and hgt_top (m)
	#Storm motion approxmiated by using mean 0-6 km wind
	u_storm, v_storm = get_mean_wind(u,v,hgt,0,6000,False,None)
	srh = np.empty(u[0].shape)
	for i in np.arange(0,u[0].shape[0]):
	    for j in np.arange(0,u[0].shape[1]):
		u_hgt = u[(hgt[:,i,j] <= hgt_top),i,j]
		v_hgt = v[(hgt[:,i,j] <= hgt_top),i,j]
		sru = u_hgt - u_storm[i,j]
		srv = v_hgt - v_storm[i,j]
		layers = (sru[1:] * srv[:-1]) - (sru[:-1] * srv[1:])
		srh[i,j] = abs(np.sum(layers))
	return srh

def get_tornado_pot(mlcape,lcl,mlcin,u,v,p,hgt):
	#From SHARPpy, but using approximations from EWD. Mixed layer cape approximated by 
	#using 950 hPa parcel. Mixed layer lcl approximated by using maximum theta-e parcel
	#Include scaling/limits from SPC
	shear = get_shear_p(u,v,p,1000,500)
	srh = get_srh(u,v,hgt,1000)
	
	mlcape = mlcape/1500
	srh = srh/150

	shear[shear<12.5] = 0
	shear[shear>30] = 1.5
	shear[(shear<=30) & (shear>=20)] = shear[(shear<=30) & (shear>=20)] / 20

	lcl[lcl<1000] = 1
	lcl[lcl>2000] = 0
	lcl[(lcl<=2000) & (lcl>=1000)] = (2000 - lcl[(lcl<=2000) & (lcl>=1000)]) / 1500

	mlcin[mlcin<50] = 1
	mlcin[mlcin>200] = 0
	mlcin[(mlcin<=200) & (mlcin>=50)] = (200 - mlcin[(mlcin<=200) & (mlcin>=50)]) / 150

	return (mlcape*srh*shear*lcl*mlcin)

def get_supercell_pot(mucape,u,v,hgt,ta_unit,p_unit,q_unit):
	#From EWD. MUCAPE approximated by treating each vertical grid point as a parcel, 
	# finding the CAPE of each parcel, and taking the maximum CAPE

	srh03 = get_srh(u,v,hgt,3000)
	density = mpcalc.density(p_unit,ta_unit,q_unit)
	density = np.array(density)
	mean_u6000, mean_v6000 = get_mean_wind(u,v,hgt,0,6000,True,density)
	mean_u500, mean_v500 = get_mean_wind(u,v,hgt,0,500,True,density)

	len6000 = np.sqrt(np.square(mean_u6000)+np.square(mean_v6000))
	len500 = np.sqrt(np.square(mean_u500)+np.square(mean_v500))

	return (mucape/1000) * (srh03/100) * ((0.5*np.square((len500-len6000)))/40)
	
def get_lr_p(t,p_1d,hgt,p_bot,p_top):
	#Get lapse rate (C/km) between two pressure levels
	#p_bot and p_top (hPa) must correspond to reanalysis pressure levels

	hgt_pbot = hgt[p_1d==p_bot] / 1000
	hgt_ptop = hgt[p_1d==p_top] / 1000
	t_pbot = t[p_1d==p_bot]
	t_ptop = t[p_1d==p_top]
	
	return np.squeeze(- (t_ptop - t_pbot) / (hgt_ptop - hgt_pbot))

def get_lr_hgt(t,hgt,hgt_bot,hgt_top):
	#Get lapse rate (C/km) between two pressure levels
	#p_bot and p_top (hPa) must correspond to reanalysis pressure levels

	t_bot = np.empty((t.shape[1],t.shape[2]))
	t_top = np.empty((t.shape[1],t.shape[2]))
	for i in np.arange(0,t.shape[1]):
	    for j in np.arange(0,t.shape[2]):
		t_bot[i,j] = np.interp(hgt_bot,hgt[:,i,j],t[:,i,j])
		t_top[i,j] = np.interp(hgt_top,hgt[:,i,j],t[:,i,j])

	return np.squeeze(- (t_top - t_bot) / ((hgt_top - hgt_bot)/1000))

def get_t_hgt(t,hgt,t_value):
	#Get the height [lev,lat,lon] at which temperature [lev,lat,lon] is equal to t_value
	t_hgt = np.empty((t.shape[1],t.shape[2]))
	for i in np.arange(0,t.shape[1]):
	    for j in np.arange(0,t.shape[2]):
		t_hgt[i,j] = np.interp(t_value,np.flipud(t[:,i,j]),np.flipud(hgt[:,i,j]))
	return t_hgt

def get_ship(mucape,muq,t,u,v,hgt,p_1d):
	#From EWD (no freezing level involved), but using SPC intended values:
	# https://github.com/sharppy/SHARPpy/blob/master/sharppy/sharptab/params.py
	
	shr06 = get_srh(u,v,hgt,6000)
	lr75 = get_lr_p(t,p_1d,hgt,750,500)
	h5_temp = np.squeeze(t[p_1d == 500])
	muq = muq*1000		#This equation assumes mixing ratio as g/kg *I think*
	frz_lvl = get_t_hgt(t,hgt,0)

	#Restrict extreme values
	shr06[shr06>27] = 27
	shr06[shr06<7] = 7
	muq[muq>13.6] = 13.6
	muq[muq<11] = 11
	h5_temp[h5_temp>-5.5] -5.5

	#Calculate ship
	ship = (-1*(mucape * muq * lr75 * h5_temp * shr06) / 42000000)

	#Scaling
	ship[mucape<1300] = ship[mucape<1300]*(mucape[mucape<1300]/1300)
	ship[lr75<5.8] = ship[lr75<5.8]*(lr75[lr75<5.8]/5.8)
	ship[frz_lvl<2400] = ship[frz_lvl<2400]*(frz_lvl[frz_lvl<2400]/2400)

	return ship

def get_mmp(u,v,mu_cape,t,hgt):
	#From SCP/SHARPpy
	#NOTE: Is costly due to looping over each layer in 0-1 km and 6-10 km, and within this 
	# loop, calling function get_shear_hgt which interpolates over lat/lon

	#Get max wind shear
	lowers = np.arange(100,1000+100,100)
	uppers = np.arange(6000,10000+1000,1000)
	no_shears = len(lowers)*len(uppers)
	shear_3d = np.empty((no_shears,u.shape[1],u.shape[2]))
	cnt=0
	for low in lowers:
		for up in uppers:
			shear_3d[cnt,:,:] = get_shear_hgt(u,v,hgt,low,up)
			cnt=cnt+1
	max_shear = np.max(shear_3d,axis=0)

	lr38 = get_lr_hgt(t,hgt,3000,8000)

	u_mean, v_mean = get_mean_wind(u,v,hgt,3000,12000,False,None)
	mean_wind = np.sqrt(np.square(u_mean)+np.square(v_mean))

	a_0 = 13.0 # unitless
	a_1 = -4.59*10**-2 # m**-1 * s
	a_2 = -1.16 # K**-1 * km
	a_3 = -6.17*10**-4 # J**-1 * kg
	a_4 = -0.17 # m**-1 * s

	mmp = 1. / (1. + np.exp(a_0 + (a_1 * max_shear) + (a_2 * lr38) + (a_3 * mu_cape) + \
		(a_4 * mean_wind)))

	mmp[mu_cape<100] = 0

	return mmp

def calc_param_wrf(times,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,param,model,save,region):

	#NOTE: Consider the winds used for "0 km" in SRH, s06, etc. Could be 10 m sfc winds, 
	# bottom pressure level (1000 hPa)? Currently, storm motion, SRH and s06 use bottom 
	# pressure level

	#Use 3d_cape in wrf-python to calculate MUCAPE (vectorised). Use this to calculate other
	# params

	#Input vars are of shape [time, levels, lat, lon]
	#Output is a list of numpy arrays of length=len(params) with dimensions [time,lat,lon]
	#Boolean option to save as a netcdf file

	#Assign p levels to a 4d array
	p_3d = np.empty(ta.shape)
	for i in np.arange(0,ta.shape[0]):
		for j in np.arange(0,ta.shape[2]):
			for k in np.arange(0,ta.shape[3]):
				p_3d[i,:,j,k] = p

	#Initialise output list
	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(times),len(lat),len(lon)))

	#Check params
	if len(param) != len(np.unique(param)):
		ValueError("Each parameter can only appear once in parameter list")

	#For each time
	for t in np.arange(0,len(times)):
		print(times[t])

		#Calculate q
		start = dt.datetime.now()
		hur_unit = units.percent*hur[t,:,:,:]
		ta_unit = units.degC*ta[t,:,:,:]
		p_unit = units.hectopascals*p_3d[t,:,:,:]
		q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
			ta_unit,p_unit)
		q = np.array(q_unit)

		#Get CAPE
		cape_3d = wrf.cape_3d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q\
			,hgt[t,:,:,:],terrain,ps[t,:,:],False,meta=False,missing=0)
		cape_2d = wrf.cape_2d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q\
			,hgt[t,:,:,:],terrain,ps[t,:,:],False,meta=False,missing=0)
		mu_cape_inds = np.argmax(cape_3d.data[0],axis=0)
		lcl = cape_2d[2].data
		lfc = cape_2d[3].data
		mu_cape = mu_cape_inds.choose(cape_3d.data[0])
		mu_cin = mu_cape_inds.choose(cape_3d.data[1])
		ml_cape = cape_3d.data[0,2,:,:]
		ml_cin = cape_3d.data[1,2,:,:]
		if t == 0:
			print("CAPE/CIN: "+str(dt.datetime.now()-start))

		#Get other parameters...
		if "relhum850-500" in param:
			start = dt.datetime.now()
			param_ind = np.where(param=="relhum850-500")[0][0]
			param_out[param_ind][t,:,:] = \
				np.mean(hur[t,(p<=851) & (p>=499),:,:])
			if t == 0:
				print("RelHum: "+str(dt.datetime.now()-start))
		if "mu_cape" in param:
		#CAPE for most unstable parcel
		#cape.data has cape values for each pressure level, as if they were each parcels.
		# Taking the max gives MUCAPE approximation
			param_ind = np.where(param=="mu_cape")[0][0]
			param_out[param_ind][t,:,:] = mu_cape
		if "ml_cape" in param:
		#CAPE for mixed layer
		#Currently using a parcel at 950 hPa to define ml_cape
			param_ind = np.where(param=="ml_cape")[0][0]
			param_out[param_ind][t,:,:] = ml_cape
		if "s06" in param:
		#Wind shear bottom pressure level to 6 km
			start = dt.datetime.now()
			param_ind = np.where(param=="s06")[0][0]
			param_out[param_ind][t,:,:] = get_shear_hgt(ua[t],va[t],hgt[t],0,6000)
			if t == 0:
				print("S06: "+str(dt.datetime.now()-start))
		if "mu_cin" in param:
		#CIN for same parcel used for mu_cape
			param_ind = np.where(param=="mu_cin")[0][0]
			param_out[param_ind][t,:,:] = mu_cin
		if "ml_cin" in param:
		#CIN for same parcel used for ml_cape
			param_ind = np.where(param=="ml_cin")[0][0]
			param_out[param_ind][t,:,:] = ml_cin
		if "srh01" in param:
		#Combined (+ve and -ve) rel. helicity from 0-1 km
			param_ind = np.where(param=="srh01")[0][0]
			param_out[param_ind][t,:,:] = get_srh(ua[t],va[t],hgt[t],1000)
		if "srh03" in param:
		#Combined (+ve and -ve) rel. helicity from 0-3 km
			start = dt.datetime.now()
			param_ind = np.where(param=="srh03")[0][0]
			param_out[param_ind][t,:,:] = get_srh(ua[t],va[t],hgt[t],3000)
			if t == 0:
				print("SRH: "+str(dt.datetime.now()-start))
		if "srh06" in param:
		#Combined (+ve and -ve) rel. helicity from 0-6 km
			param_ind = np.where(param=="srh06")[0][0]
			param_out[param_ind][t,:,:] = get_srh(ua[t],va[t],hgt[t],6000)
		if "ship" in param:
		#Significant hail parameter
			start = dt.datetime.now()
			param_ind = np.where(param=="ship")[0][0]
			muq = mu_cape_inds.choose(q)
			ship = get_ship(mu_cape,muq,ta[t],ua[t],va[t],hgt[t],p)
			param_out[param_ind][t,:,:] = ship
			if t == 0:
				print("SHIP: "+str(dt.datetime.now()-start))
		if "lhp" in param:
		#Large Hail Paramerer; NOTE requires convective profile (costly).
			param_ind = np.where(param=="lhp")[0][0]
		if "hgz_depth" in param:
		#Hail growzth zone (in hPa)
			param_ind = np.where(param=="hgz_depth")[0][0]
		if "dcp" in param:
		#Derecho Composite Parameter ~ cold pool driven wind events
			param_ind = np.where(param=="dcp")[0][0]
		if "mburst" in param:
		#Microburst composite index
			param_ind = np.where(param=="mburst")[0][0]
		if "mmp" in param:
		#Mesoscale Convective System Maintanance Probability
			start = dt.datetime.now()
			param_ind = np.where(param=="mmp")[0][0]
			param_out[param_ind][t,:,:] = get_mmp(ua[t],va[t],mu_cape,ta[t],hgt[t])
			if t == 0:
				print("MMP: "+str(dt.datetime.now()-start))
		if "scp" in param:
		#Supercell composite parameter (EWD)
			start = dt.datetime.now()
			param_ind = np.where(param=="scp")[0][0]
			scell_pot = get_supercell_pot(mu_cape,ua[t],va[t],hgt[t],ta_unit,p_unit,\
					q_unit)
			param_out[param_ind][t,:,:] = scell_pot
			if t == 0:
				print("SCP: "+str(dt.datetime.now()-start))
		if "stp" in param:
		#Significant tornado parameter
		#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
		# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
		# lcl.
			start = dt.datetime.now()
			param_ind = np.where(param=="stp")[0][0]
			stp = get_tornado_pot(ml_cape,lcl,ml_cin,ua[t],va[t],p_3d[t],hgt[t])
			param_out[param_ind][t,:,:] = stp
			if t == 0:
				print("STP: "+str(dt.datetime.now()-start))

	if save:
		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"_"+\
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


def calc_param_sharppy(times,ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,param,model,save,region):
	#Calculate parameters based on the SHARPpy package for creating profiles

	#For each time in "times", loop over lat/lon points in domain and calculate:
	# 1) profile 2) parcel (if create_parcel is set) 3) parameter
	#NOTE the choice of parameter may affect both steps 2) and 3)
	#Input vars are of shape [time, levels, lat, lon]
	#Output is a list of numpy arrays of length=len(params) with dimensions [time,lat,lon]
	#Option to save as a netcdf file

	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(times),len(lat),len(lon)))

	for t in np.arange(0,len(times)):
		print(times[t])
		for x in np.arange(0,len(lon)):
			for y in np.arange(0,len(lat)):
				#Restrict to a single spatial point
				point = [lon[x],lat[y]]   
				ta_p,dp_p,hgt_p,ua_p,va_p,uas_p,vas_p = get_point(point,\
					lon,lat,ta[t,:,:,:],dp[t,:,:,:],hgt[t,:,:,:],ua[t,:,:,:],\
					va[t,:,:,:],uas[t,:,:],vas[t,:,:])

				#Convert u and v to kts for use in profile
				ua_p_kts = utils.MS2KTS(ua_p)
				va_p_kts = utils.MS2KTS(va_p)

				#Create profile
				prof = profile.create_profile(pres=p, hght=hgt_p, tmpc=ta_p, \
						dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
				#Create most unstable parcel
				mu_parcel = params.parcelx(prof, flag=4,dp=-10)

				#Get storm motion vectors
				u_storm, v_storm = winds.mean_wind(prof,pbot=1000,ptop=500)

				if "relhum850-500" in param:
					param_ind = np.where(param=="relhum850-500")[0][0]
					param_out[param_ind][t,y,x] = \
						np.mean(hur_p[(p_p<=851) & (p_p>=499)])
				if "mu_cape" in param:
				#CAPE for most unstable parcel
					param_ind = np.where(param=="mu_cape")[0][0]
					param_out[param_ind][t,y,x] = mu_parcel.bplus
				if "s06" in param:
				#Wind shear 10 m (sfc) to 6 km
					ua_0km = uas_p
					ua_6km = np.interp(6000,hgt_p,ua_p)
					va_0km = vas_p
					va_6km = np.interp(6000,hgt_p,va_p)
					shear = np.sqrt(np.square(ua_6km-ua_0km)+\
						np.square(va_6km-va_0km))
					param_ind = np.where(param=="s06")[0][0]
					param_out[param_ind][t,y,x] = shear
				if "mu_cin" in param:
				#CIN for most unstable parcel
					param_ind = np.where(param=="mu_cin")[0][0]
					param_out[param_ind][t,y,x] = -1*mu_parcel.bminus
				if "srh01" in param:
				#Combined (+ve and -ve) rel. helicity from 0-1 km
					param_ind = np.where(param=="srh01")[0][0]
					param_out[param_ind][t,y,x] = abs(winds.helicity(prof,0,\
							1000,stu=u_storm,stv=v_storm)[0])
				if "srh03" in param:
				#Combined (+ve and -ve) rel. helicity from 0-3 km
					param_ind = np.where(param=="srh03")[0][0]
					param_out[param_ind][t,y,x] = abs(winds.helicity(prof,\
							0,3000,stu=u_storm,stv=v_storm)[0])
				if "srh06" in param:
				#Combined (+ve and -ve) rel. helicity from 0-6 km
					param_ind = np.where(param=="srh06")[0][0]
					param_out[param_ind][t,y,x] = abs(winds.helicity(prof,0,\
							6000,stu=u_storm,stv=v_storm)[0])
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
				if "scp" in param:
				#Mesoscale Convective System Maintanance Probability
					param_ind = np.where(param=="scp")[0][0]
					param_out[param_ind][t,y,x] = params.mmp(prof)
				if "stp" in param:
				#Mesoscale Convective System Maintanance Probability
					param_ind = np.where(param=="stp")[0][0]
					param_out[param_ind][t,y,x] = params.stp(prof)

	if save:
		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"_"+\
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

def calc_param_points(times,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,param,loc_id):

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
			ps_p = ps[t,point]

			#------------------------------------------------------------------------
			#USING SHARPpy
			#------------------------------------------------------------------------
			#Convert u and v to kts for use in profile
			#ua_p_kts = utils.MS2KTS(ua_p)
			#va_p_kts = utils.MS2KTS(va_p)
			#Create profile
			#prof = profile.create_profile(pres=p_p, hght=hgt_p, tmpc=ta_p, \
					#dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
			#Create most unstable parcel
			#mu_parcel = params.parcelx(prof, flag=3)

			#------------------------------------------------------------------------
			#With wrf-python (for erai)
			#------------------------------------------------------------------------
			hur_unit = units.percent*hur_p
			ta_unit = units.degC*ta_p
			p_unit = units.hectopascals*p_p
			q = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,ta_unit,p_unit)
			q = np.array(q)
			cape = wrf.cape_3d(p_p,ta_p+273.15,q\
				,hgt_p,terrain[point],ps_p,False,meta=False,missing=0)

			if "relhum850-500" in param:
				values[cnt,param=="relhum850-500"] = \
					np.mean(hur_p[(p_p<=851) & (p_p>=499)])
			if "mu_cape" in param:
			#CAPE for most unstable parcel
				#values[cnt,param=="mu_cape"] = mu_parcel.bplus
				#values[cnt,param=="mu_cape"] = cape.data[0][0][0]
				values[cnt,param=="mu_cape"] = np.max(cape.data[0,:,0,0])
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
