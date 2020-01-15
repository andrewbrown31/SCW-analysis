import itertools
import multiprocessing
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import os
#For some reason, with numpy 1.16.0, metpy import doesn't work at first, but then works when you try it again
try:
	from SkewT import get_dcape
	import metpy.units as units
	import metpy.calc as mpcalc
except:
	pass
#from metpy.units import units as units
#import metpy.calc as mpcalc
#import metpy
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
#	- calc_param_points
#		Extract parameters by looping over time, at a set of points provided by the 
#		user to the function. Currently, both wrf-python and SHARPpy can be used to 
#		calculate the parameters, by commenting/un-commenting lines of code.
#
#NOTE: "metpy" by unicode is used in some places for calculating specific humidity from
# thermodynamic properties
#
#NOTE: When taking mean wind over a layer, does a weighted mean need to be used? Currently, 
# get_mean_wind takes a vertical average between two pressure levels. This could bias the mean 
# towards heights where there are more model pressure levels.
#
#NOTE: Should add in dependencies to if statements within calc_param_wrf. For example, if "ship"
# is included in the parameter list, then "mu_cape" must also be present, etc. Add something 
# within "ship" block like: [is mu_cape in the parameter list? -> if not, throw error with 
# message "mu_cape must be in the parameter list to calculate ship"]
#
#NOTE: Because data is on pressure levels, much greater efficiency could be achieved by 
# approximating height-based indicies at equivalent standard pressure levels
#-------------------------------------------------------------------------------------------------


def get_dp(ta,hur,dp_mask=True):
	#Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
	#Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
	#For points where RH is zero, set the dew point temperature to -85 deg C
	#EDIT: Leave points where RH is zero masked as NaNs. Hanlde them after creating a SHARPpy object (by 
	# making the missing dp equal to the mean of the above and below layers)
	#EDIT: Replace with metpy code

	#a = 17.27
	#b = 237.7
	#alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
	#dp = (b*alpha) / (a - alpha)

	dp = np.array(mpcalc.dewpoint_rh(ta * units.units.degC, hur * units.units.percent))

	if dp_mask:
		return dp
	else:
		dp = np.array(dp)
		dp[np.isnan(dp)] = -85.
		return dp

def get_point(point,lon,lat,ta,dp,hgt,ua,va,uas,vas,hur):
	# Return 1d arrays for all variables, at a given spatial point (now a function
	# of p-level only)
	lon_ind = np.argmin(abs(lon-point[0]))
	lat_ind = np.argmin(abs(lat-point[1]))
	ta = np.squeeze(ta[:,lat_ind,lon_ind])
	dp = np.squeeze(dp[:,lat_ind,lon_ind])
	hgt = np.squeeze(hgt[:,lat_ind,lon_ind])
	hur = np.squeeze(hur[:,lat_ind,lon_ind])
	ua = np.squeeze(ua[:,lat_ind,lon_ind])
	va = np.squeeze(va[:,lat_ind,lon_ind])
	uas = np.squeeze(uas[lat_ind,lon_ind])
	vas = np.squeeze(vas[lat_ind,lon_ind])

	return [ta,dp,hgt,ua,va,uas,vas,hur]

def get_mean_wind(u,v,hgt,hgt_bot,hgt_top,density_weighted,density,method,p=None):
	#Get mean wind components [lat, lon] based on 3d input of u, v, hgt and p [levels,lat,lon]

	#papprox - Find the pressure levels which correspond to avg height closest to hgt_bot and
	# height top, take arithmatic mean between pressure levels
	#interp - Interpolate u and v to evenly spaced height levels between the lowest height 
	# for each point (1000 hPa), and the top height. Then take arithmatic average 
	# (with option for density weighted.
	#papprox_hgt - Get all winds on pressure levels between hgt_bot and hgt_top (avg hgt for
	# whole domain), then add winds interpolated to hgt_bot and hgt_top. Arithmatic avg.
	#plevel - Use pressure levels
	#plevels - take mean on pressure levels, where hgt_bot and hgt top are known pressure levels

	if method=="papprox":
		bot_ind = np.argmin(abs(np.mean(hgt,axis=(1,2)) - hgt_bot))
		top_ind = np.argmin(abs(np.mean(hgt,axis=(1,2)) - hgt_top))
		hgt_inds = np.arange(bot_ind,top_ind+1,1)
		u_mean = np.mean(u[hgt_inds],axis=0)
		v_mean = np.mean(v[hgt_inds],axis=0)
	if method=="plevels":
		bot_ind = np.argmin(abs(p - hgt_bot))
		top_ind = np.argmin(abs(p - hgt_top))
		hgt_inds = np.arange(bot_ind,top_ind+1,1)
		print(hgt_inds)
		u_mean = np.mean(u[hgt_inds],axis=0)
		v_mean = np.mean(v[hgt_inds],axis=0)
	elif method=="papprox_hgt":
		if u.ndim == 1:
			hgt_inds = np.where((hgt<hgt_top)&(hgt>hgt_bot))[0]
			u_p = u[hgt_inds]; v_p = v[hgt_inds]
			u_top = np.interp(hgt_top,hgt,u)
			v_top = np.interp(hgt_top,hgt,v)
			u_bot = np.interp(hgt_bot,hgt,u)
			v_bot = np.interp(hgt_bot,hgt,v)
			total_u = np.concatenate((u_p,np.ones(1)*u_top,np.ones(1)*u_bot),axis=0)
			total_v = np.concatenate((v_p,np.ones(1)*v_top,np.ones(1)*v_bot),axis=0)
		else:
			avg_hgt = np.mean(hgt,axis=(1,2))
			hgt_inds = np.where((avg_hgt<hgt_top)&(avg_hgt>hgt_bot))[0]
			#Get u and v for pressure levels
			u_p = u[hgt_inds]; v_p = v[hgt_inds]
			#Get u and v at top and bottom height
			u_top = np.expand_dims(np.array(wrf.interplevel(u, hgt, hgt_top)),0)
			v_top = np.expand_dims(np.array(wrf.interplevel(v, hgt, hgt_top)),0)
			u_bot = np.expand_dims(np.array(wrf.interplevel(u, hgt, hgt_bot,missing=np.nan)),0)
			v_bot = np.expand_dims(np.array(wrf.interplevel(v, hgt, hgt_bot,missing=np.nan)),0)
			total_u = np.concatenate((u_p,u_top,u_bot),axis=0)
			total_v = np.concatenate((v_p,v_top,v_bot),axis=0)
		u_mean = np.nanmean(total_u,axis=0)
		v_mean = np.nanmean(total_v,axis=0)
	elif method=="interp":
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
					u_mean[i,j] = (np.sum(temp_density*u_interp)) / \
						(np.sum(temp_density))
					v_mean[i,j] = (np.sum(temp_density*v_interp)) / \
						(np.sum(temp_density))
				else:
					u_mean[i,j] = np.mean(u_interp)
					v_mean[i,j] = np.mean(v_interp)

	return [u_mean,v_mean]

def get_shear_hgt(u,v,hgt,hgt_bot,hgt_top,uas=None,vas=None,components=False):
	#Get bulk wind shear [lat, lon] between two heights, based on 3d input of u, v, and 
	#hgt [levels,lat,lon]
	#Note lowest possible height is equal to bottom pressure level (1000 hPa), unless hgt_bot
	# is specified as less than or equal to 10 m, and 10 m winds are provided, in which case
	# sfc winds are then used 

	if (hgt_bot <= 10) & ~(uas is None):
		u_bot = uas
		v_bot = vas
	else:
		if u.ndim == 1:
			u_bot = np.array(np.interp(hgt_bot,hgt,u))
			v_bot = np.array(np.interp(hgt_bot,hgt,v))
		else:
			u_bot = np.array(wrf.interplevel(u, hgt, hgt_bot))
			v_bot = np.array(wrf.interplevel(v, hgt, hgt_bot))
			u_bot[hgt[0] >= hgt_bot] = u[0,hgt[0] >= hgt_bot]
			v_bot[hgt[0] >= hgt_bot] = v[0,hgt[0] >= hgt_bot]
	
	if u.ndim == 1:
		u_top = np.array(np.interp(hgt_top,hgt,u))
		v_top = np.array(np.interp(hgt_top, hgt, v))
		u_top[hgt[-1] <= hgt_top] = u[-1]
		v_top[hgt[-1] <= hgt_top] = v[-1]
	else:
		u_top = np.array(wrf.interplevel(u, hgt, hgt_top))
		v_top = np.array(wrf.interplevel(v, hgt, hgt_top))
		u_top[hgt[-1] <= hgt_top] = u[-1,hgt[-1] <= hgt_bot]
		v_top[hgt[-1] <= hgt_top] = v[-1,hgt[-1] <= hgt_bot]
		u_top[np.where(hgt==hgt_top)[1],np.where(hgt==hgt_top)[2]] = u[np.where(hgt==hgt_top)]
		v_top[np.where(hgt==hgt_top)[1],np.where(hgt==hgt_top)[2]] = v[np.where(hgt==hgt_top)]

	if components:
		return [u_top-u_bot, v_top-v_bot]
	else:
		shear = np.array(np.sqrt(np.square(u_top-u_bot)+np.square(v_top-v_bot)))
		return shear

def get_shear_p(u,v,p,p_bot,p_top,lev,uas=None,vas=None):
	#Get bulk wind shear [lat, lon] between two pressure levels, based on 3d input of u, v, and 
	#p [levels,lat,lon]
	#p_bot and p_top given in hPa
	#p_bot can also be given as "sfc" to use 10 m winds

	if p_bot < 1000:
		ValueError("Bottom pressure level can't be below bottom model level (1000 hPa)")
	if p_bot > 100:
		ValueError("Top pressure level can't be above top model level (100 hPa)")

	if u.ndim == 1:
		if p_bot == "sfc":
			u_bot = uas
			v_bot = vas
		elif p_bot in lev:
			u_bot = u[np.where(p_bot==lev)]
			v_bot = v[np.where(p_bot==lev)]
		else:
			u_bot = np.interp(p_bot, p, u)
			v_bot = np.interp(p_bot, p, v)
		if p_top in lev:
			u_top = u[np.where(p_top==lev)]
			v_top = v[np.where(p_top==lev)]
		else:
			u_top = np.interp(p_top, p, u)
			v_top = np.interp(p_top, p, v)
	else:
		if p_bot == "sfc":
			u_bot = uas
			v_bot = vas
		elif p_bot in lev:
			u_bot = u[p_bot==lev,:,:]
			v_bot = v[p_bot==lev,:,:]
		else:
			u_bot = wrf.interpz3d(u, p, p_bot)
			v_bot = wrf.interpz3d(v, p, p_bot)
		if p_top in lev:
			u_top = u[p_top==lev,:,:]
			v_top = v[p_top==lev,:,:]
		else:
			u_top = wrf.interpz3d(u, p, p_top)
			v_top = wrf.interpz3d(v, p, p_top)
	
	shear = np.array(np.sqrt(np.square(u_top-u_bot)+np.square(v_top-v_bot)))

	return shear

def get_td_diff(t,td,p,p_level):
	#Difference between dew point temp and air temp at p_level
	#Represents downdraft potential. See diagram in Gilmore and Wicker (1998)

	if t.ndim == 1:
		t_plevel = np.interp(p_level,p,t)
	else:
		if p_level in p:
			t_plevel = t[p[:,0,0]==p_level]
		else:
			t_plevel = np.array(wrf.interpz3d(t, p, p_level))

	if t.ndim == 1:
		td_plevel = np.interp(p_level,p,td)
	else:
		if p_level in p:
			td_plevel = td[p[:,0,0]==p_level]
		else:
			td_plevel = np.array(wrf.interpz3d(td, p, p_level))
	
	return (t_plevel - td_plevel)

def get_srh(u,v,hgt,hgt_top,papprox,storm_bot,storm_top,p=None):
	#Get storm relative helicity [lat, lon] based on 3d input of u, v, and storm motion u and
	# v components
	# Is between the bottom pressure level (1000 hPa), approximating 0 m, and hgt_top (m)
	#Storm motion approxmiated by using mean 0-6 km wind

	#u_storm, v_storm = get_mean_wind(u,v,hgt,storm_bot,storm_top,False,None,"plevels",p)
	mnu6, mnv6 = get_mean_wind(u,v,hgt,0,6000,False,None,"papprox",p)
	us6, vs6 = get_shear_hgt(u,v,hgt,0,6000,components=True)
	tmp = 7.5 / (np.sqrt(np.square(us6) + np.square(vs6)))
	u_storm = mnu6 - (tmp * vs6)
	v_storm = mnv6 + (tmp * us6)
	print(u_storm.max(), u_storm.min(), v_storm.max(), v_storm.min())

	if papprox:
		if u.ndim == 1:
			top_ind = np.argmin(abs(hgt - hgt_top))
		else:
			top_ind = np.argmin(abs(np.mean(hgt,axis=(1,2)) - hgt_top))
		hgt_inds = np.arange(0,top_ind+1,1)
		u_hgt = u[hgt_inds]
		v_hgt = v[hgt_inds]
		sru = u_hgt - u_storm
		srv = v_hgt - v_storm
		layers = (sru[1:] * srv[:-1]) - (sru[:-1] * srv[1:])
		srh = abs(np.sum(layers,axis=0))

	else:
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

def get_tornado_pot(mlcape,lcl,mlcin,u,v,p,hgt,lev,srh01,exp=False):
	#From SHARPpy, but using approximations from EWD. Mixed layer cape approximated by 
	#using 950 hPa parcel. Mixed layer lcl approximated by using maximum theta-e parcel
	#Include scaling/limits from SPC
	shear = get_shear_p(u,v,p,1000,500,lev)
	
	mlcape = mlcape/1500
	srh01 = srh01/150

	shear[shear<12.5] = 0
	shear[shear>30] = 1.5
	shear[(shear<=30) & (shear>=20)] = shear[(shear<=30) & (shear>=20)] / 20

	lcl[(lcl<=2000) & (lcl>=1000)] = (2000 - lcl[(lcl<=2000) & (lcl>=1000)]) / 1500
	lcl[lcl<1000] = 1
	lcl[lcl>2000] = 0

	mlcin[mlcin<50] = 1
	mlcin[mlcin>200] = 0
	mlcin[(mlcin<=200) & (mlcin>=50)] = (200 - mlcin[(mlcin<=200) & (mlcin>=50)]) / 150

	if exp:
		#Was meant to raise each term to an exponent, instead just force cape to be non-zero
		return ((mlcape+1)*(srh01)*(shear)*(lcl)*(mlcin))
	else:
		return (mlcape*srh01*shear*lcl*mlcin)

def get_non_sc_tornado_pot(mlcape,mlcin,lcl,u,v,uas,vas,p,t,hgt,lev,vo,lr1000):
	#From EWD. Mixed layer cape approximated by 
	#using 950 hPa parcel. Mixed layer lcl approximated by using maximum theta-e parcel
	#Vorticity calculated within MetPy. X and y are calculated externlly as delta lat,lon 
	# meshgrids with x corresponding to the "u" direction 
	shear = get_shear_hgt(u,v,hgt,0,6000,uas,vas)

	lr = abs(lr1000)/9
	mlcape = mlcape/100
	mlcin = (225-mlcin)/200
	shear = (18-shear)/5
	lcl = (2000-lcl)/1500
	vo = abs(vo)/(8*10**-5)

	return (lr*mlcape*mlcin*shear*lcl*vo)

def get_conv(u,v,dx,dy):
	#10 m relative vo.
	return (-1 * np.array(mpcalc.divergence(u,v,dx,dy)))

def get_vo(uas,vas,dx,dy):
	#10 m relative vo.
	return (np.array(mpcalc.vorticity(uas,vas,dx,dy)))

def get_supercell_pot(mucape,u,v,hgt,ta_unit,p_unit,q_unit,srh03):
	#From EWD. MUCAPE approximated by treating each vertical grid point as a parcel, 
	# finding the CAPE of each parcel, and taking the maximum CAPE

	density = mpcalc.density(p_unit,ta_unit,q_unit)
	density = np.array(density)
	mean_u6000, mean_v6000 = get_mean_wind(u,v,hgt,0,6000,True,density,"papprox_hgt")
	mean_u500, mean_v500 = get_mean_wind(u,v,hgt,0,500,True,density,"papprox_hgt")

	len6000 = np.sqrt(np.square(mean_u6000)+np.square(mean_v6000))
	len500 = np.sqrt(np.square(mean_u500)+np.square(mean_v500))

	return (mucape/1000) * (srh03/100) * ((0.5*np.square((len500-len6000)))/40)
	
def get_lr_p(t,p_1d,hgt,p_bot,p_top):
	#Get lapse rate (C/km) between two pressure levels
	#No interpolation is done, so p_bot and p_top (hPa) should correspond to 
	#reanalysis pressure levels

	hgt_pbot = hgt[np.argmin(abs(p_1d-p_bot))] / 1000
	hgt_ptop = hgt[np.argmin(abs(p_1d-p_top))] / 1000
	t_pbot = t[np.argmin(abs(p_1d-p_bot))]
	t_ptop = t[np.argmin(abs(p_1d-p_top))]
	
	return np.squeeze(- (t_ptop - t_pbot) / (hgt_ptop - hgt_pbot))

def get_lr_hgt(t,hgt,hgt_bot,hgt_top):
	#Get lapse rate (C/km) between two height levels

	if t.ndim == 1:
		t_bot = np.interp(hgt_bot,hgt,t)
		t_top = np.interp(hgt_top,hgt,t)
	else:
		t_bot = wrf.interplevel(t,hgt,float(hgt_bot),meta=False)
		t_bot[hgt[0] >= hgt_bot] = t[0,hgt[0] >= hgt_bot]
		t_bot[(np.where(hgt==hgt_bot))[1],(np.where(hgt==hgt_bot))[2]] = t[hgt==hgt_bot]
		t_top = wrf.interplevel(t,hgt,float(hgt_top),meta=False)
		t_top[hgt[-1] <= hgt_top] = t[-1,hgt[-1] <= hgt_bot]
		t_top[(np.where(hgt==hgt_top))[1],(np.where(hgt==hgt_top))[2]] = t[hgt==hgt_top]

	return np.squeeze(- (t_top - t_bot) / ((hgt_top - hgt_bot)/1000))

def get_t_hgt(t,hgt,t_value):
	#Get the height [lev,lat,lon] at which temperature [lev,lat,lon] is equal to t_value

	if t.ndim == 1:
		t_hgt = np.interp(t_value,np.flipud(t),np.flipud(hgt))
	else:
		t_hgt = np.array(wrf.interplevel(hgt, t, t_value))

	return t_hgt

def get_ship(mucape,muq,t,u,v,hgt,p_1d,s06):
	#From EWD (no freezing level involved), but using SPC intended values:
	# https://github.com/sharppy/SHARPpy/blob/master/sharppy/sharptab/params.py
	
	lr75 = get_lr_p(t,p_1d,hgt,750,500)
	h5_temp = np.squeeze(t[abs(p_1d-500)<.0001])
	muq = muq*1000		#This equation assumes mixing ratio as g/kg *I think*
	frz_lvl = get_t_hgt(t,hgt,0)

	#Restrict extreme values
	s06[s06>27] = 27
	s06[s06<7] = 7
	muq[muq>13.6] = 13.6
	muq[muq<11] = 11
	h5_temp[h5_temp>-5.5] = -5.5

	#Calculate ship
	ship = (-1*(mucape * muq * lr75 * h5_temp * s06) / 42000000)

	if t.ndim == 1:
		lr75 = np.ones(ship.shape)*lr75
		frz_lvl = np.ones(ship.shape)*frz_lvl
		s06 = np.ones(ship.shape)*s06

	#Scaling
	ship[mucape<1300] = ship[mucape<1300]*(mucape[mucape<1300]/1300)
	ship[lr75<5.8] = ship[lr75<5.8]*(lr75[lr75<5.8]/5.8)
	ship[frz_lvl<2400] = ship[frz_lvl<2400]*(frz_lvl[frz_lvl<2400]/2400)

	return ship

def get_mmp(u,v,uas,vas,mu_cape,t,hgt):
	#From SCP/SHARPpy
	#NOTE: Is costly due to looping over each layer in 0-1 km and 6-10 km, and within this 
	# loop, calling function get_shear_hgt which interpolates over lat/lon

	#Get max wind shear
	lowers = np.arange(0,1000+250,250)
	uppers = np.arange(6000,10000+1000,1000)
	no_shears = len(lowers)*len(uppers)
	if u.ndim == 1:
		shear_3d = np.empty((no_shears))
		cnt=0
		for low in lowers:
			for up in uppers:
				shear_3d[cnt] = get_shear_hgt(u,v,hgt,low,up,uas,vas)
				cnt=cnt+1
	else:
		shear_3d = np.empty((no_shears,u.shape[1],u.shape[2]))
		cnt=0
		for low in lowers:
			for up in uppers:
				shear_3d[cnt,:,:] = get_shear_hgt(u,v,hgt,low,up,uas,vas)
				cnt=cnt+1
	max_shear = np.max(shear_3d,axis=0)

	lr38 = get_lr_hgt(t,hgt,3000,8000)

	u_mean, v_mean = get_mean_wind(u,v,hgt,3000,12000,False,None,"papprox_hgt")
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

def get_mean_var_p(var,p,p_bot,p_top):
	#Return the vertical average of variable "var" [levels, lat, lon] based on pressure levels.
	if var.ndim == 1:
		return (np.mean(var[(p<=(p_bot+1)) & (p>=(p_top-1))],axis=0))
	else:
		return (np.mean(var[(p<=(p_bot+1)) & (p>=(p_top-1)),:,:],axis=0))


def critical_angle(u,v,hgt,u_sfc,v_sfc):
	#From SHARPpy
	#Angle between storm relative winds at 10 m and 10-500 m wind shear vector
	#Used in junjuction with thunderstorm/supercell indices
	u_storm, v_storm = get_mean_wind(u,v,hgt,0,6000,False,None,"papprox_hgt")
	if u.ndim == 1:
		u_500 = np.interp(500,hgt,u)
		v_500 = np.interp(500,hgt,v)
	else:
		u_500 = wrf.interpz3d(u,hgt,500)
		v_500 = wrf.interpz3d(v,hgt,500)

	shear_u = u_500 - u_sfc
	shear_v = v_500 - v_sfc
	srw_u = u_storm - u_sfc
	srw_v = v_storm - v_sfc

	dot = shear_u * srw_u + shear_v * srw_v
	shear_mag = np.sqrt(np.square(shear_u)+np.square(shear_v))
	srw_mag = np.sqrt(np.square(srw_u)+np.square(srw_v))

	return (np.degrees(np.arccos(dot / (shear_mag * srw_mag))))

def calc_param_wrf(times,ta,dp,hur,hgt,terrain,p,ps,ua,va,tas,ta2d,\
			uas,vas,lon,lat,param,model,out_name,save,region,\
			wg=False):

	#NOTE: Consider the winds used for "0 km" in SRH, s06, etc. Could be 10 m sfc winds, 
	# bottom pressure level (1000 hPa)? Currently, storm motion, SRH and s06 use bottom 
	# pressure level

	#Use 3d_cape in wrf-python to calculate MUCAPE/MLCAPE (vectorised). Use this to calculate other
	# params

	#Input vars are of shape [time, levels, lat, lon]
	#Output is a list of numpy arrays of length=len(params) with elements having dimensions [time,lat,lon]
	#Boolean option to save as a netcdf file

	#Set OpenMP options for wrf-python (utilising all processers)
	wrf.omp_set_num_threads(wrf.omp_get_num_procs())

	#Assign p levels to a 4d array, with same dimensions as input variables (ta, hgt, etc.)
	p_3d = np.moveaxis(np.tile(p,[ta.shape[0],ta.shape[2],ta.shape[3],1]),[0,1,2,3],[0,2,3,1])

	#Insert surface arrays, creating new arrays with "sfc" prefix
	sfc_ta = np.insert(ta, 0, tas, axis=1) 
	sfc_hgt = np.insert(hgt, 0, terrain, axis=1) 
	sfc_dp = np.insert(dp, 0, ta2d, axis=1) 
	sfc_p_3d = np.insert(p_3d, 0, ps, axis=1) 
	sfc_ua = np.insert(ua, 0, uas, axis=1) 
	sfc_va = np.insert(va, 0, vas, axis=1) 

	#Initialise output list
	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(times),len(lat),len(lon)))

	#Check params
	if len(param) != len(np.unique(param)):
		ValueError("Each parameter can only appear once in parameter list")

	#For each time
	total_start = dt.datetime.now()
	for t in np.arange(0,len(times)):
		print(times[t])

		#Calculate q for pressure level arrays
		ta_unit = units.units.degC*ta[t,:,:,:]
		dp_unit = units.units.degC*dp[t,:,:,:]
		p_unit = units.units.hectopascals*p_3d[t,:,:,:]
		hur_unit = mpcalc.relative_humidity_from_dewpoint(ta_unit, dp_unit)*100*units.units.percent
		q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
			ta_unit,p_unit)
		theta_unit = mpcalc.potential_temperature(p_unit,ta_unit)
		q = np.array(q_unit)

		#Calculate q for pressure level arrays with surface values
		sfc_ta_unit = units.units.degC*sfc_ta[t,:,:,:]
		sfc_dp_unit = units.units.degC*sfc_dp[t,:,:,:]
		sfc_p_unit = units.units.hectopascals*sfc_p_3d[t,:,:,:]
		sfc_hur_unit = mpcalc.relative_humidity_from_dewpoint(sfc_ta_unit, sfc_dp_unit)*\
			100*units.units.percent
		sfc_q_unit = mpcalc.mixing_ratio_from_relative_humidity(sfc_hur_unit,\
			sfc_ta_unit,sfc_p_unit)
		sfc_theta_unit = mpcalc.potential_temperature(sfc_p_unit,sfc_ta_unit)
		sfc_q = np.array(sfc_q_unit)

		#New way of getting MLCAPE. Insert an avg sfc-100 hPa AGL layer.
		#First, find avg values for ta, p, hgt and q for ML (between the surface and 100 hPa AGL)
		ml_inds = ((sfc_p_3d[t] <= ps[t]) & (sfc_p_3d[t] >= (ps[t] - 100)))
		ml_ta_avg = np.ma.masked_where(~ml_inds, sfc_ta[t]).mean(axis=0).data
		ml_q_avg = np.ma.masked_where(~ml_inds, sfc_q).mean(axis=0).data
		ml_hgt_avg = np.ma.masked_where(~ml_inds, sfc_hgt[t]).mean(axis=0).data
		ml_p3d_avg = np.ma.masked_where(~ml_inds, sfc_p_3d[t]).mean(axis=0).data
		#Insert the mean values into the bottom of the 3d arrays pressure-level arrays
		ml_ta_arr = np.insert(ta[t],0,ml_ta_avg,axis=0)
		ml_q_arr = np.insert(q,0,ml_q_avg,axis=0)
		ml_hgt_arr = np.insert(hgt[t],0,ml_hgt_avg,axis=0)
		ml_p3d_arr = np.insert(p_3d[t],0,ml_p3d_avg,axis=0)
		#Calculate CAPE and extract for ML avg layer. Need to sort by ascending p first, placing the ml-avg
		# layer after a pressure level-layer if they are on the same pressure surface
		a,temp1,temp2 = np.meshgrid(np.arange(ml_p3d_arr.shape[0]) ,\
			 np.arange(ml_p3d_arr.shape[1]), np.arange(ml_p3d_arr.shape[2]))
		sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),ml_p3d_arr],axis=0))
		ml_ta_arr = np.take_along_axis(ml_ta_arr, sort_inds, axis=0)
		ml_p3d_arr = np.take_along_axis(ml_p3d_arr, sort_inds, axis=0)
		ml_hgt_arr = np.take_along_axis(ml_hgt_arr, sort_inds, axis=0)
		ml_q_arr = np.take_along_axis(ml_q_arr, sort_inds, axis=0)
		cape3d_mlavg = wrf.cape_3d(ml_p3d_arr,ml_ta_arr + 273.15,\
			ml_q_arr,ml_hgt_arr,terrain,ps[t,:,:],False,meta=False, missing=0)
		ml_cape = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
			cape3d_mlavg.data[0]).max(axis=0).filled(0)
		ml_cin = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
			cape3d_mlavg.data[1]).max(axis=0).filled(0)
		ml_lfc = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
			cape3d_mlavg.data[2]).max(axis=0).filled(0)
		ml_lcl = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
			cape3d_mlavg.data[3]).max(axis=0).filled(0)
		ml_el = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
			cape3d_mlavg.data[4]).max(axis=0).filled(0)

		#Now get MUCAPE (ensuring parcels used are AGL)
		cape3d = wrf.cape_3d(sfc_p_3d[t,:,:,:],sfc_ta[t,:,:,:]+273.15,sfc_q,sfc_hgt[t,:,:,:],\
				terrain,ps[t,:,:],\
				True,meta=False, missing=0)
		cape = cape3d.data[0]
		cin = cape3d.data[1]
		lfc = cape3d.data[2]
		lcl = cape3d.data[3]
		#Mask values which are below the surface, and values which are less than 12.5 hPa AGL. 
		#The need to exclude values between the surface and 12.5 hPa AGL, is due to the fact that the 
		#WRF-python routine defines "layers" using mid-points between pressure levels.
		cape[sfc_p_3d[t] > ps[t]] = np.nan
		cin[sfc_p_3d[t] > ps[t]] = np.nan
		lfc[sfc_p_3d[t] > ps[t]] = np.nan
		lcl[sfc_p_3d[t] > ps[t]] = np.nan
		#cape[sfc_p_3d[t] > ps[t]-25] = np.nan
		#cin[sfc_p_3d[t] > ps[t]-25] = np.nan
		#Now, for each grid point, define mlcape as the maximum CAPE on vertical levels which are between 
		# 0 and 100 hPa AGL. Define ml_cin as the maximum cin in the same region. Note, cin is only given
		#by wrf-python if cape is present
		#Define most unstable cape/cin. Cin is for the same parcel as mu_cape
		mu_cape_inds = np.nanargmax(cape,axis=0)
		mu_cape = mu_cape_inds.choose(cape)
		mu_cin = mu_cape_inds.choose(cin)
		mu_lfc = mu_cape_inds.choose(lfc)
		mu_lcl = mu_cape_inds.choose(lcl)

		#Lastly, get lcl and lfc for most unstable parcel (defined by highest theta-e)
		#Make terrain following equal true, as this seems to eliminate below-ground
		# parcels
		cape_2d = wrf.cape_2d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q\
			,hgt[t,:,:,:],terrain,ps[t,:,:],True,meta=False,missing=0)
		lcl = cape_2d[2].data
		lfc = cape_2d[3].data

		#Clean up CAPE objects
		del dp_unit, theta_unit, ml_inds, ml_ta_avg, ml_q_avg, \
			ml_hgt_avg, ml_p3d_avg, ml_ta_arr, ml_q_arr, ml_hgt_arr, ml_p3d_arr, a, temp1, temp2,\
			sort_inds, cape3d_mlavg, cape3d, cape, cin \
			#,cape_2d

		#Get other parameters...
		if "relhum850-500" in param:
			param_ind = np.where(param=="relhum850-500")[0][0]
			param_out[param_ind][t,:,:] = get_mean_var_p(hur_unit,p,850,500)
		if "relhum1000-700" in param:
			param_ind = np.where(param=="relhum1000-700")[0][0]
			param_out[param_ind][t,:,:] = get_mean_var_p(hur_unit,p,1000,700)
		if "mu_cape" in param:
		#CAPE for most unstable parcel
		#cape.data has cape values for each pressure level, as if they were each parcels.
		# Taking the max gives MUCAPE approximation
			param_ind = np.where(param=="mu_cape")[0][0]
			param_out[param_ind][t,:,:] = mu_cape
		if "ml_cape" in param:
		#CAPE for mixed layer
			param_ind = np.where(param=="ml_cape")[0][0]
			param_out[param_ind][t,:,:] = ml_cape
		if "s06" in param:
		#Wind shear 10 m (sfc) to 6 km
			param_ind = np.where(param=="s06")[0][0]
			s06 = get_shear_hgt(ua[t],va[t],hgt[t],0,6000,\
				uas[t],vas[t])
			param_out[param_ind][t,:,:] = s06
		if "s03" in param:
		#Wind shear 10 m (sfc) to 3 km
			param_ind = np.where(param=="s03")[0][0]
			s03 = get_shear_hgt(ua[t],va[t],hgt[t],0,3000,\
				uas[t],vas[t])
			param_out[param_ind][t,:,:] = s03
		if "s01" in param:
		#Wind shear 10 m (sfc) to 1 km
			param_ind = np.where(param=="s01")[0][0]
			s01 = get_shear_hgt(ua[t],va[t],hgt[t],0,1000,\
				uas[t],vas[t])
			param_out[param_ind][t,:,:] = s01
		if "s0500" in param:
		#Wind shear sfc to 500 m
			param_ind = np.where(param=="s0500")[0][0]
			param_out[param_ind][t,:,:] = get_shear_hgt(ua[t],va[t],hgt[t],0,500,\
				uas[t],vas[t])
		if "lr1000" in param:
		#Lapse rate bottom pressure level to 1000 m
			param_ind = np.where(param=="lr1000")[0][0]
			lr1000 = get_lr_hgt(ta[t],hgt[t],0,1000)
			param_out[param_ind][t,:,:] = lr1000
		if "mu_cin" in param:
		#CIN for same parcel used for mu_cape
			param_ind = np.where(param=="mu_cin")[0][0]
			param_out[param_ind][t,:,:] = mu_cin
		if "ml_lfc" in param:
		#LCL for same parcel used for mu_cape
			param_ind = np.where(param=="ml_lfc")[0][0]
			temp_ml_lfc = np.copy(ml_lfc)
			temp_ml_lfc[temp_ml_lfc<0] = np.nan
			param_out[param_ind][t,:,:] = temp_ml_lfc
		if "ml_el" in param:
		#LCL for same parcel used for mu_cape
			param_ind = np.where(param=="ml_el")[0][0]
			temp_ml_el = np.copy(ml_el)
			temp_ml_el[temp_ml_el<0] = np.nan
			param_out[param_ind][t,:,:] = temp_ml_el
		if "ml_lcl" in param:
		#LCL for same parcel used for mu_cape
			param_ind = np.where(param=="ml_lcl")[0][0]
			temp_ml_lcl = np.copy(ml_lcl)
			temp_ml_lcl[temp_ml_lcl<0] = np.nan
			param_out[param_ind][t,:,:] = temp_ml_lcl
		if "lfc" in param:
		#LCL for same parcel used for mu_cape
			param_ind = np.where(param=="lfc")[0][0]
			temp_lfc = np.copy(lfc)
			temp_lfc[temp_lfc<=0] = np.nan
			param_out[param_ind][t,:,:] = temp_lfc
		if "lcl" in param:
		#LCL for same parcel used for mu_cape
			param_ind = np.where(param=="lcl")[0][0]
			temp_lcl = np.copy(lcl)
			temp_lcl[temp_lcl<=0] = np.nan
			param_out[param_ind][t,:,:] = temp_lcl
		if "ml_cin" in param:
		#CIN for same parcel used for ml_cape
			param_ind = np.where(param=="ml_cin")[0][0]
			param_out[param_ind][t,:,:] = ml_cin
		if "srh01" in param:
		#Combined (+ve and -ve) rel. helicity from 0-1 km
			param_ind = np.where(param=="srh01")[0][0]
			srh01 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,1000,True,850,700,p)
			param_out[param_ind][t,:,:] = srh01
		if "srh03" in param:
		#Combined (+ve and -ve) rel. helicity from 0-3 km
			srh03 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,3000,True,850,700,p)
			param_ind = np.where(param=="srh03")[0][0]
			param_out[param_ind][t,:,:] = srh03
		if "srh06" in param:
		#Combined (+ve and -ve) rel. helicity from 0-6 km
			param_ind = np.where(param=="srh06")[0][0]
			srh06 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,6000,True,850,700,p)
			param_out[param_ind][t,:,:] = srh06
		if "ship" in param:
		#Significant hail parameter
			if "s06" not in param:
				raise NameError("To calculate ship, s06 must be included")
			param_ind = np.where(param=="ship")[0][0]
			muq = mu_cape_inds.choose(q)
			ship = get_ship(mu_cape,muq,ta[t],ua[t],va[t],hgt[t],p,s06)
			param_out[param_ind][t,:,:] = ship
		if "mmp" in param:
		#Mesoscale Convective System Maintanance Probability
			param_ind = np.where(param=="mmp")[0][0]
			param_out[param_ind][t,:,:] = get_mmp(ua[t],va[t],uas[t],vas[t],\
				mu_cape,ta[t],hgt[t])
		if "scp" in param:
		#Supercell composite parameter (EWD)
			if "srh03" not in param:
				raise NameError("To calculate ship, srh03 must be included")
			param_ind = np.where(param=="scp")[0][0]
			scell_pot = get_supercell_pot(mu_cape,ua[t],va[t],hgt[t],ta_unit,p_unit,\
					q_unit,srh03)
			param_out[param_ind][t,:,:] = scell_pot
			#if t == 0:
				#print("SCP: "+str(dt.datetime.now()-start))
		if "stp" in param:
		#Significant tornado parameter
		#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
		# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
		# lcl.
			if "srh01" not in param:
				raise NameError("To calculate stp, srh01 must be included")
			param_ind = np.where(param=="stp")[0][0]
			stp = get_tornado_pot(ml_cape,lcl,ml_cin,ua[t],va[t],p_3d[t],hgt[t],p,srh01)
			param_out[param_ind][t,:,:] = stp
		if "vo10" in param:
		#10 m relative vorticity
			param_ind = np.where(param=="vo10")[0][0]
			x,y = np.meshgrid(lon,lat)
			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
			vo10 = get_vo(uas[t],vas[t],dx,dy)
			param_out[param_ind][t,:,:] = vo10
		if "conv10" in param:
			param_ind = np.where(param=="conv10")[0][0]
			x,y = np.meshgrid(lon,lat)
			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
			param_out[param_ind][t,:,:] = get_conv(uas[t],vas[t],dx,dy)
		if "conv1000-850" in param:
			levs = np.where((p<=1001)&(p>=849))[0]
			param_ind = np.where(param=="conv1000-850")[0][0]
			x,y = np.meshgrid(lon,lat)
			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
			param_out[param_ind][t,:,:] = \
				np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
		if "conv800-600" in param:
			levs = np.where((p<=801)&(p>=599))[0]
			param_ind = np.where(param=="conv800-600")[0][0]
			x,y = np.meshgrid(lon,lat)
			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
			param_out[param_ind][t,:,:] = \
				np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
		if "non_sc_stp" in param:
		#Non-Supercell Significant tornado parameter
		#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
		# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
		# lcl.
			if "vo10" not in param:
				raise NameError("To calculate non_sc_stp, vo must be included")
			if "lr1000" not in param:
				raise NameError("To calculate non_sc_stp, lr1000 must be included")
			param_ind = np.where(param=="non_sc_stp")[0][0]
			non_sc_stp = get_non_sc_tornado_pot(ml_cape,ml_cin,lcl,ua[t],va[t],\
				uas[t],vas[t],p_3d[t],ta[t],hgt[t],p,vo10,lr1000)
			param_out[param_ind][t,:,:] = non_sc_stp
		if "cape*s06" in param:
			param_ind = np.where(param=="cape*s06")[0][0]
			cs6 = ml_cape * np.power(s06,1.67)
			param_out[param_ind][t,:,:] = cs6
		if "td850" in param:
			param_ind = np.where(param=="td850")[0][0]
			td850 = get_td_diff(ta[t],dp[t],p_3d[t],850)
			param_out[param_ind][t,:,:] = td850
		if "td800" in param:
			param_ind = np.where(param=="td800")[0][0]
			param_out[param_ind][t,:,:] = get_td_diff(ta[t],dp[t],p_3d[t],800)
		if "td950" in param:
			param_ind = np.where(param=="td950")[0][0]
			param_out[param_ind][t,:,:] = get_td_diff(ta[t],dp[t],p_3d[t],950)
		if "wg" in param:
			try:
				param_ind = np.where(param=="wg")[0][0]
				param_out[param_ind][t,:,:] = wg[t]
			except ValueError:
				print("wg field expected, but not parsed")
		if "dcape" in param:
			param_ind = np.where(param=="dcape")[0][0]
			dcape = np.nanmax(get_dcape(np.array(p_3d[t]),ta[t],hgt[t],np.array(p),ps[t]),axis=0)
			param_out[param_ind][t,:,:] = dcape
		if "mlm" in param:
			param_ind = np.where(param=="mlm")[0][0]
			mlm_u, mlm_v = get_mean_wind(ua[t],va[t],hgt[t],800,600,False,None,"plevels",p)
			mlm = np.sqrt(np.square(mlm_u) + np.square(mlm_v))
			param_out[param_ind][t,:,:] = mlm
		if "dlm" in param:
			param_ind = np.where(param=="dlm")[0][0]
			dlm_u, dlm_v = get_mean_wind(ua[t],va[t],hgt[t],1000,500,False,None,"plevels",p)
			dlm = np.sqrt(np.square(dlm_u) + np.square(dlm_v))
			param_out[param_ind][t,:,:] = dlm
		if "dlm+dcape" in param:
			param_ind = np.where(param=="dlm+dcape")[0][0]
			dlm_dcape = dlm + np.sqrt(2*dcape)
			param_out[param_ind][t,:,:] = dlm_dcape
		if "mlm+dcape" in param:
			param_ind = np.where(param=="mlm+dcape")[0][0]
			mlm_dcape = mlm + np.sqrt(2*dcape)
			param_out[param_ind][t,:,:] = mlm_dcape
		if "dcape*cs6" in param:
			param_ind = np.where(param=="dcape*cs6")[0][0]
			param_out[param_ind][t,:,:] = (dcape/980.) * (cs6/20000)
		if "dlm*dcape*cs6" in param:
			param_ind = np.where(param=="dlm*dcape*cs6")[0][0]
			param_out[param_ind][t,:,:] = (dlm_dcape/30.) * (cs6/20000)
		if "mlm*dcape*cs6" in param:
			param_ind = np.where(param=="mlm*dcape*cs6")[0][0]
			param_out[param_ind][t,:,:] = (mlm_dcape/30.) * (cs6/20000)
		if "dcp" in param:
		#Derecho Composite Parameter ~ cold pool driven wind events
			param_ind = np.where(param=="dcp")[0][0]
			param_out[param_ind][t,:,:] = (dcape/980)*(mu_cape/2000)*(s06/10)*(dlm/8)

	if save:
		save_netcdf(region,model,out_name,times,lat,lon,param,param_out)
	
	return param_out

def calc_param_wrf_par(it):

	#Same as calc_param_wrf(), except the times argument is parsed in parallel. times generally specifies
	#a one month period, so this has the effect of calculating a month-worth of CAPE in parallel (on somewhere
	#between 16 and 30 cores)

	#times,t,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,param,model,out_name,save,region,wg = it

	t = it
	wg=False
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","vo10","lr1000","lcl",\
		"relhum1000-700","s06","s0500","s01","s03",\
		"cape*s06","dcp","td850","td800","td950","dcape","mlm","dlm",\
		"dcape*cs6","mlm+dcape","mlm*dcape*cs6"]

	#Assign p levels to a 4d array, with same dimensions as input variables (ta, hgt, etc.)
	p_3d = np.moveaxis(np.tile(p,[ta.shape[0],ta.shape[2],ta.shape[3],1]),[0,1,2,3],[0,2,3,1])

	#Initialise output list
	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(lat),len(lon)))

	#Check params
	if len(param) != len(np.unique(param)):
		ValueError("Each parameter can only appear once in parameter list")

	print(times[t])

	#Calculate q
	start = dt.datetime.now()
	hur_unit = units.percent*hur[t,:,:,:]
	ta_unit = units.degC*ta[t,:,:,:]
	dp_unit = units.degC*dp[t,:,:,:]
	p_unit = units.hectopascals*p_3d[t,:,:,:]
	q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
		ta_unit,p_unit)
	theta_unit = mpcalc.potential_temperature(p_unit,ta_unit)
	q = np.array(q_unit)

	#New way of getting MLCAPE. Insert an avg sfc-100 hPa AGL layer.
	#First, find avg values for ta, p, hgt and q for ML (between the surface and 100 hPa AGL)
	ml_inds = ((p_3d[t] <= ps[t]) & (p_3d[t] >= (ps[t] - 100)))
	ml_ta_avg = np.ma.masked_where(~ml_inds, ta[t]).mean(axis=0).data
	ml_q_avg = np.ma.masked_where(~ml_inds, q).mean(axis=0).data
	ml_hgt_avg = np.ma.masked_where(~ml_inds, hgt[t]).mean(axis=0).data
	ml_p3d_avg = np.ma.masked_where(~ml_inds, p_3d[t]).mean(axis=0).data
	#Insert the mean values into the bottom of the 3d arrays pressure-level arrays
	ml_ta_arr = np.insert(ta[t],0,ml_ta_avg,axis=0)
	ml_q_arr = np.insert(q,0,ml_q_avg,axis=0)
	ml_hgt_arr = np.insert(hgt[t],0,ml_hgt_avg,axis=0)
	ml_p3d_arr = np.insert(p_3d[t],0,ml_p3d_avg,axis=0)
	#Calculate CAPE and extract for ML avg layer. Need to sort by ascending p first, placing the ml-avg
	# layer after a pressure level-layer if they are on the same pressure surface
	a,temp1,temp2 = np.meshgrid(np.arange(ml_p3d_arr.shape[0]) ,\
		 np.arange(ml_p3d_arr.shape[1]), np.arange(ml_p3d_arr.shape[2]))
	sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),ml_p3d_arr],axis=0))
	ml_ta_arr = np.take_along_axis(ml_ta_arr, sort_inds, axis=0)
	ml_p3d_arr = np.take_along_axis(ml_p3d_arr, sort_inds, axis=0)
	ml_hgt_arr = np.take_along_axis(ml_hgt_arr, sort_inds, axis=0)
	ml_q_arr = np.take_along_axis(ml_q_arr, sort_inds, axis=0)
	cape3d_mlavg = wrf.cape_3d(ml_p3d_arr,ml_ta_arr + 273.15,\
		ml_q_arr,ml_hgt_arr,terrain,ps[t,:,:],False,meta=False,missing=0)
	ml_cape = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
		cape3d_mlavg.data[0]).max(axis=0).filled(0)
	ml_cin = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
		cape3d_mlavg.data[1]).max(axis=0).filled(0)

	#New way of getting CAPE using wrf-python (ensuring parcels used are AGL)
	cape3d = wrf.cape_3d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q,hgt[t,:,:,:],terrain,ps[t,:,:],\
		True,meta=False,missing=0)
	cape = cape3d.data[0]
	cin = cape3d.data[1]
	#Mask values which are below the surface, and values which are less than 12.5 hPa AGL. 
	#The need to exclude values between the surface and 12.5 hPa AGL, is due to the fact that the 
	#WRF-python routine defines "layers" using mid-points between pressure levels.
	cape[p_3d[t] > ps[t]-25] = np.nan
	cin[p_3d[t] > ps[t]-25] = np.nan
	#Now, for each grid point, define mlcape as the maximum CAPE on vertical levels which are between 
	# 0 and 100 hPa AGL. Define ml_cin as the maximum cin in the same region. Note, cin is only given
	#by wrf-python if cape is present
	#Define most unstable cape/cin. Cin is for the same parcel as mu_cape
	mu_cape_inds = np.nanargmax(cape,axis=0)
	mu_cape = mu_cape_inds.choose(cape)
	mu_cin = mu_cape_inds.choose(cin)
	#Lastly, get lcl and lfc. Make terrain following equal true, as this seems to eliminate below-ground
	# parcels
	cape_2d = wrf.cape_2d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q\
		,hgt[t,:,:,:],terrain,ps[t,:,:],True,meta=False,missing=0)
	lcl = cape_2d[2].data
	lfc = cape_2d[3].data

	#Clean up CAPE objects
	del hur_unit, dp_unit, theta_unit, ml_inds, ml_ta_avg, ml_q_avg, \
		ml_hgt_avg, ml_p3d_avg, ml_ta_arr, ml_q_arr, ml_hgt_arr, ml_p3d_arr, a, temp1, temp2,\
		sort_inds, cape3d_mlavg, cape3d, cape, cin, cape_2d

	#Get other parameters...
	if "relhum850-500" in param:
		param_ind = np.where(param=="relhum850-500")[0][0]
		param_out[param_ind] = get_mean_var_p(hur[t],p,850,500)
	if "relhum1000-700" in param:
		param_ind = np.where(param=="relhum1000-700")[0][0]
		param_out[param_ind] = get_mean_var_p(hur[t],p,1000,700)
	if "mu_cape" in param:
	#CAPE for most unstable parcel
	#cape.data has cape values for each pressure level, as if they were each parcels.
	# Taking the max gives MUCAPE approximation
		param_ind = np.where(param=="mu_cape")[0][0]
		param_out[param_ind] = mu_cape
	if "ml_cape" in param:
	#CAPE for mixed layer
		param_ind = np.where(param=="ml_cape")[0][0]
		param_out[param_ind] = ml_cape
	if "s06" in param:
	#Wind shear 10 m (sfc) to 6 km
		param_ind = np.where(param=="s06")[0][0]
		s06 = get_shear_hgt(ua[t],va[t],hgt[t],0,6000,\
			uas[t],vas[t])
		param_out[param_ind] = s06
	if "s03" in param:
	#Wind shear 10 m (sfc) to 3 km
		param_ind = np.where(param=="s03")[0][0]
		s03 = get_shear_hgt(ua[t],va[t],hgt[t],0,3000,\
			uas[t],vas[t])
		param_out[param_ind] = s03
	if "s01" in param:
	#Wind shear 10 m (sfc) to 1 km
		param_ind = np.where(param=="s01")[0][0]
		s01 = get_shear_hgt(ua[t],va[t],hgt[t],0,1000,\
			uas[t],vas[t])
		param_out[param_ind] = s01
	if "s0500" in param:
	#Wind shear sfc to 500 m
		param_ind = np.where(param=="s0500")[0][0]
		param_out[param_ind] = get_shear_hgt(ua[t],va[t],hgt[t],0,500,\
			uas[t],vas[t])
	if "lr1000" in param:
	#Lapse rate bottom pressure level to 1000 m
		param_ind = np.where(param=="lr1000")[0][0]
		lr1000 = get_lr_hgt(ta[t],hgt[t],0,1000)
		param_out[param_ind] = lr1000
	if "mu_cin" in param:
	#CIN for same parcel used for mu_cape
		param_ind = np.where(param=="mu_cin")[0][0]
		param_out[param_ind] = mu_cin
	if "lcl" in param:
	#LCL for same parcel used for mu_cape
		param_ind = np.where(param=="lcl")[0][0]
		temp_lcl = np.copy(lcl)
		temp_lcl[temp_lcl<=0] = np.nan
		param_out[param_ind] = temp_lcl
	if "ml_cin" in param:
	#CIN for same parcel used for ml_cape
		param_ind = np.where(param=="ml_cin")[0][0]
		param_out[param_ind] = ml_cin
	if "srh01" in param:
	#Combined (+ve and -ve) rel. helicity from 0-1 km
		param_ind = np.where(param=="srh01")[0][0]
		srh01 = get_srh(ua[t],va[t],hgt[t],1000,True,850,700,p)
		param_out[param_ind] = srh01
	if "srh03" in param:
	#Combined (+ve and -ve) rel. helicity from 0-3 km
		srh03 = get_srh(ua[t],va[t],hgt[t],3000,True,850,700,p)
		param_ind = np.where(param=="srh03")[0][0]
		param_out[param_ind] = srh03
	if "srh06" in param:
	#Combined (+ve and -ve) rel. helicity from 0-6 km
		param_ind = np.where(param=="srh06")[0][0]
		srh06 = get_srh(ua[t],va[t],hgt[t],6000,True,850,700,p)
		param_out[param_ind] = srh06
	if "ship" in param:
	#Significant hail parameter
		if "s06" not in param:
			raise NameError("To calculate ship, s06 must be included")
		param_ind = np.where(param=="ship")[0][0]
		muq = mu_cape_inds.choose(q)
		ship = get_ship(mu_cape,muq,ta[t],ua[t],va[t],hgt[t],p,s06)
		param_out[param_ind] = ship
	if "mmp" in param:
	#Mesoscale Convective System Maintanance Probability
		param_ind = np.where(param=="mmp")[0][0]
		param_out[param_ind] = get_mmp(ua[t],va[t],uas[t],vas[t],\
			mu_cape,ta[t],hgt[t])
	if "scp" in param:
	#Supercell composite parameter (EWD)
		if "srh03" not in param:
			raise NameError("To calculate ship, srh03 must be included")
		param_ind = np.where(param=="scp")[0][0]
		scell_pot = get_supercell_pot(mu_cape,ua[t],va[t],hgt[t],ta_unit,p_unit,\
				q_unit,srh03)
		param_out[param_ind] = scell_pot
		#if t == 0:
			#print("SCP: "+str(dt.datetime.now()-start))
	if "stp" in param:
	#Significant tornado parameter
	#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
	# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
	# lcl.
		if "srh01" not in param:
			raise NameError("To calculate stp, srh01 must be included")
		param_ind = np.where(param=="stp")[0][0]
		stp = get_tornado_pot(ml_cape,lcl,ml_cin,ua[t],va[t],p_3d[t],hgt[t],p,srh01)
		param_out[param_ind] = stp
	if "vo10" in param:
	#10 m relative vorticity
		param_ind = np.where(param=="vo10")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		vo10 = get_vo(uas[t],vas[t],dx,dy)
		param_out[param_ind] = vo10
	if "conv10" in param:
		param_ind = np.where(param=="conv10")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = get_conv(uas[t],vas[t],dx,dy)
	if "conv1000-850" in param:
		levs = np.where((p<=1001)&(p>=849))[0]
		param_ind = np.where(param=="conv1000-850")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = \
			np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
	if "conv800-600" in param:
		levs = np.where((p<=801)&(p>=599))[0]
		param_ind = np.where(param=="conv800-600")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = \
			np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
	if "non_sc_stp" in param:
	#Non-Supercell Significant tornado parameter
	#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
	# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
	# lcl.
		if "vo10" not in param:
			raise NameError("To calculate non_sc_stp, vo must be included")
		if "lr1000" not in param:
			raise NameError("To calculate non_sc_stp, lr1000 must be included")
		param_ind = np.where(param=="non_sc_stp")[0][0]
		non_sc_stp = get_non_sc_tornado_pot(ml_cape,ml_cin,lcl,ua[t],va[t],\
			uas[t],vas[t],p_3d[t],ta[t],hgt[t],p,vo10,lr1000)
		param_out[param_ind] = non_sc_stp
	if "cape*s06" in param:
		param_ind = np.where(param=="cape*s06")[0][0]
		cs6 = ml_cape * np.power(s06,1.67)
		param_out[param_ind] = cs6
	if "td850" in param:
		param_ind = np.where(param=="td850")[0][0]
		td850 = get_td_diff(ta[t],dp[t],p_3d[t],850)
		param_out[param_ind] = td850
	if "td800" in param:
		param_ind = np.where(param=="td800")[0][0]
		param_out[param_ind] = get_td_diff(ta[t],dp[t],p_3d[t],800)
	if "td950" in param:
		param_ind = np.where(param=="td950")[0][0]
		param_out[param_ind] = get_td_diff(ta[t],dp[t],p_3d[t],950)
	if "wg" in param:
		try:
			param_ind = np.where(param=="wg")[0][0]
			param_out[param_ind] = wg[t]
		except ValueError:
			print("wg field expected, but not parsed")
	if "dcape" in param:
		param_ind = np.where(param=="dcape")[0][0]
		dcape = np.nanmax(get_dcape(p_3d[t],ta[t],hgt[t],p,ps[t]),axis=0)
		param_out[param_ind] = dcape
	if "mlm" in param:
		param_ind = np.where(param=="mlm")[0][0]
		mlm_u, mlm_v = get_mean_wind(ua[t],va[t],hgt[t],800,600,False,None,"plevels",p)
		mlm = np.sqrt(np.square(mlm_u) + np.square(mlm_v))
		param_out[param_ind] = mlm
	if "dlm" in param:
		param_ind = np.where(param=="dlm")[0][0]
		dlm_u, dlm_v = get_mean_wind(ua[t],va[t],hgt[t],1000,500,False,None,"plevels",p)
		dlm = np.sqrt(np.square(dlm_u) + np.square(dlm_v))
		param_out[param_ind] = dlm
	if "dlm+dcape" in param:
		param_ind = np.where(param=="dlm+dcape")[0][0]
		dlm_dcape = dlm + np.sqrt(2*dcape)
		param_out[param_ind] = dlm_dcape
	if "mlm+dcape" in param:
		param_ind = np.where(param=="mlm+dcape")[0][0]
		mlm_dcape = mlm + np.sqrt(2*dcape)
		param_out[param_ind] = mlm_dcape
	if "dcape*cs6" in param:
		param_ind = np.where(param=="dcape*cs6")[0][0]
		param_out[param_ind] = (dcape/980.) * (cs6/20000)
	if "dlm*dcape*cs6" in param:
		param_ind = np.where(param=="dlm*dcape*cs6")[0][0]
		param_out[param_ind] = (dlm_dcape/30.) * (cs6/20000)
	if "mlm*dcape*cs6" in param:
		param_ind = np.where(param=="mlm*dcape*cs6")[0][0]
		param_out[param_ind] = (mlm_dcape/30.) * (cs6/20000)
	if "dcp" in param:
	#Derecho Composite Parameter ~ cold pool driven wind events
		param_ind = np.where(param=="dcp")[0][0]
		param_out[param_ind] = (dcape/980)*(mu_cape/2000)*(s06/10)*(dlm/8)
	
	return param_out


def calc_param_points(times,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,param,loc_id,method):

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
			if method == "points_SHARPpy":
			#Convert u and v to kts for use in profile
				ua_p_kts = utils.MS2KTS(ua_p)
				va_p_kts = utils.MS2KTS(va_p)
				prof = profile.create_profile(pres=p_p, hght=hgt_p, tmpc=ta_p, \
						dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
				mu_parcel = params.parcelx(prof, flag=3)
				mu_cape = mu_parcel.bplus
				mu_cin = mu_parcel.bminus
			elif method == "points_wrf":
			#------------------------------------------------------------------------
			#With wrf-python (for erai)
			#------------------------------------------------------------------------
				hur_unit = units.percent*hur_p
				ta_unit = units.degC*ta_p
				p_unit = units.hectopascals*p_p
				q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
					ta_unit,p_unit)
				q = np.array(q_unit)
				cape_3d = wrf.cape_3d(p_p,ta_p+273.15,q\
					,hgt_p,terrain[point],ps_p,False,meta=False,missing=0)
				cape_2d = wrf.cape_2d(p_p,ta_p+273.15,q\
					,hgt_p,terrain[point],ps_p,False,meta=False,missing=0)
				mu_cape_inds = np.argmax(cape_3d.data[0],axis=0)
				lcl = cape_2d[2].data
				lfc = cape_2d[3].data
				#mu_cape = mu_cape_inds.choose(cape_3d.data[0])
				#mu_cin = mu_cape_inds.choose(cape_3d.data[1])
				#Bc we are only at a point, don't have to use "choose"
				mu_cape = cape_3d.data[0,mu_cape_inds[0],0,0]
				mu_cin = cape_3d.data[1,mu_cape_inds[0],0,0]
				ml_cape = cape_3d.data[0,2,:,:]
				ml_cin = cape_3d.data[1,2,:,:]
			else:
				raise NameError("Invalid points method")

			if "relhum850-500" in param:
				#values[cnt,param=="relhum850-500"] = \
				#	np.mean(hur_p[(p_p<=851) & (p_p>=499)])
				#param_ind = np.where(param=="relhum850-500")[0][0]
				values[cnt,param=="relhum850-500"] = get_mean_var_p(hur_p,p_p,850,500)
			if "relhum1000-700" in param:
				values[cnt,param=="relhum1000-700"] = get_mean_var_p(hur_p,p_p,1000,700)
			if "mu_cape" in param:
			#CAPE for most unstable parcel
				#values[cnt,param=="mu_cape"] = mu_parcel.bplus
				#values[cnt,param=="mu_cape"] = cape.data[0][0][0]
				values[cnt,param=="mu_cape"] = mu_cape
			if "ml_cape" in param:
				values[cnt,param=="ml_cape"] = ml_cape
			if "ssfc6" in param:
			#Wind shear 10 m (sfc) to 6 km
				ssfc6 = get_shear_hgt(ua_p,va_p,hgt_p,0,6000,\
					uas_p,vas_p)
				values[cnt,param=="ssfc6"] = ssfc6
			if "ssfc3" in param:
			#Wind shear 10 m (sfc) to 3 km
				ssfc3 = get_shear_hgt(ua_p,va_p,hgt_p,0,3000,\
					uas_p,vas_p)
				values[cnt,param=="ssfc3"] = ssfc3
			if "ssfc1" in param:
			#Wind shear 10 m (sfc) to 1 km
				ssfc1 = get_shear_hgt(ua_p,va_p,hgt_p,0,1000,\
					uas_p,vas_p)
				values[cnt,param=="ssfc1"] = ssfc1
			if "s06" in param:
			#Wind shear 100 hPa to 6 km
				s06 = get_shear_hgt(ua_p,va_p,hgt_p,0,6000)
				values[cnt,param=="s06"] = s06
			if "ssfc850" in param:
				values[cnt,param=="ssfc850"] = get_shear_p(ua_p,va_p,p_p,"sfc",850,\
					p_p,uas_p,vas_p)
			if "s0500" in param:
				values[cnt,param=="s0500"] = get_shear_p(ua_p,va_p,p_p,"sfc",500,\
					p_p,uas_p,vas_p)
			if "lr1000" in param:
			#Lapse rate bottom pressure level to 1000 m
				lr1000 = get_lr_hgt(ta_p,hgt_p,0,1000)
				values[cnt,param=="lr1000"] = lr1000
			if "mu_cin" in param:
			#CIN for most unstable parcel
				#values[cnt,param=="mu_cin"] = -1*mu_parcel.bminus	
				values[cnt,param=="mu_cin"] = mu_cin	
			if "lcl" in param:
				temp_lcl = np.copy(lcl)
				temp_lcl[temp_lcl<=0] = np.nan
				values[cnt,param=="lcl"] = temp_lcl
			if "ml_cin" in param:
			#CIN for most mixed layer parcel
				values[cnt,param=="ml_cin"] = ml_cin	
			if "srh01" in param:
			#Combined (+ve and -ve) rel. helicity from 0-1 km
				#values[cnt,param=="hel03"] = winds.helicity(prof,0,3000)[0]
				srh01 = get_srh(ua_p,va_p,hgt_p,1000,True,850,700,p_p)
				values[cnt,param=="srh01"] = srh01
			if "srh03" in param:
			#Combined (+ve and -ve) rel. helicity from 0-3 km
				#values[cnt,param=="hel03"] = winds.helicity(prof,0,3000)[0]
				srh03 = get_srh(ua_p,va_p,hgt_p,3000,True,850,700,p_p)
				values[cnt,param=="srh03"] = srh03
			if "srh06" in param:
			#Combined (+ve and -ve) rel. helicity from 0-6 km
				#values[cnt,param=="hel06"] = winds.helicity(prof,0,6000)[0]
				srh06 = get_srh(ua_p,va_p,hgt_p,6000,True,850,700,p_p)
				values[cnt,param=="srh06"] = srh06
			if "ship" in param:
			#Significant hail parameter
				#values[cnt,param=="ship"] = params.ship(prof,mupcl=mu_parcel)
				muq = q[mu_cape_inds[0]]
				values[cnt,param=="ship"] = get_ship(mu_cape,muq,ta_p,\
					ua_p,va_p,hgt_p,p_p,s06)
			if "lhp" in param:
			#Large Hail Paramerer; NOTE requires convective profile (costly).
				#conf_prof = profile.create_profile(profile="convective",pres=p_p, hght=hgt_p, tmpc=ta_p, \
				#	dwpc=dp_p, u=ua_p_kts, v=va_p_kts)
				#values[cnt,param=="lhp"] = params.lhp(prof)
				None
			if "hgz_depth" in param:
			#Hail growzth zone (in hPa)
				#values[cnt,param=="hgz_depth"] = abs(params.hgz(prof)[1] - params.hgz(prof)[0])
				None
			if "dcp" in param:
			#Derecho Composite Parameter ~ cold pool driven wind events
				#values[cnt,param=="dcp"] = params.dcp(prof)
				None
			if "mburst" in param:
			#Microburst composite index
				#values[cnt,param=="mburst"] = params.mburst(prof)
				None
			if "mmp" in param:
			#Mesoscale Convective System Maintanance Probability
				#values[cnt,param=="mmp"] = params.mmp(prof,mupcl=mu_parcel)
				values[cnt,param=="mmp"] = get_mmp(ua_p,va_p,uas_p,vas_p,\
				mu_cape,ta_p,hgt_p)
			if "scp" in param:
				values[cnt,param=="scp"] = get_supercell_pot(mu_cape,ua_p,va_p,hgt_p,ta_unit,\
				p_unit,q_unit,srh03)
			if "stp" in param:
				values[cnt,param=="stp"] = get_tornado_pot(ml_cape,lcl,ml_cin,ua_p,\
				va_p,p_p,hgt_p,p_p,srh01)
			if "crt" in param:
				values[cnt,param=="crt"] = critical_angle(ua_p,va_p,hgt_p,uas_p,vas_p)
			if "cape*s06" in param:
				values[cnt,param=="cape*s06"] = mu_cape * np.power(s06,1.67)
			if "cape*ssfc6" in param:
				values[cnt,param=="cape*ssfc6"] = mu_cape * np.power(ssfc6,1.67)

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

def save_netcdf(region,model,out_name,times,lat,lon,param,param_out,append=False,out_dtype="f8",compress=False):
	if model == "erai_fc":	#Last time of fc data is the first time of the following month
		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+out_name+"_"+\
			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(times[-2],"%Y%m%d")+".nc"
	else:
		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+out_name+"_"+\
			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(times[-1],"%Y%m%d")+".nc"
	if (os.path.isfile(fname)) & (append):
		param_file = nc.Dataset(fname,"a",format="NETCDF4_CLASSIC")
	else:
		if os.path.isfile(fname):
			os.remove(fname)
		param_file = nc.Dataset(fname,"w",format="NETCDF4_CLASSIC")
		time_dim = param_file.createDimension("time",None)
		lat_dim = param_file.createDimension("lat",len(lat))
		lon_dim = param_file.createDimension("lon",len(lon))
		time_var = param_file.createVariable("time",lat.dtype,("time",))
		time_var.units = "hours since 1970-01-01 00:00:00"
		time_var.long_name = "time"
		lat_var = param_file.createVariable("lat",lat.dtype,("lat",))
		lat_var.units = "degrees_north"
		lat_var.long_name = "latitude"
		lon_var = param_file.createVariable("lon",lat.dtype,("lon",))
		lon_var.units = "degrees_east"
		lon_var.long_name = "longitude"
		time_var[:] = nc.date2num(times,time_var.units)
		lat_var[:] = lat
		lon_var[:] = lon

	for i in np.arange(0,len(param)):
		if append:
			if param[i] not in param_file.variables.keys():
				var_units, var_long_name, least_significant_digit = nc_attributes(param[i])
				if compress:
					temp_var = param_file.createVariable(param[i],out_dtype,\
						("time","lat","lon"),zlib=True,least_significant_digit=\
						least_significant_digit, complevel=1)
				else:
					temp_var = param_file.createVariable(param[i],out_dtype,\
						("time","lat","lon"))
				temp_var[:] = param_out[i].astype(out_dtype)
				temp_var.units = var_units
				temp_var.long_name = var_long_name
			else:
				param_file[param[i]][:] = param_out[i]
		else:
			var_units, var_long_name, least_significant_digit = nc_attributes(param[i])
			if compress:
				temp_var = param_file.createVariable(param[i],out_dtype,\
					("time","lat","lon"),zlib=True,\
					least_significant_digit=least_significant_digit, complevel=1)
			else:
				temp_var = param_file.createVariable(param[i],out_dtype,\
					("time","lat","lon"))
			temp_var[:] = param_out[i].astype(out_dtype)
			temp_var.units = var_units
			temp_var.long_name = var_long_name
	param_file.close()

def nc_attributes(param):
	if param=="ml_cape":
		units = "J/kg"
		long_name = "mixed_layer_cape"
		least_significant_digit = 1
	elif param=="ml_cin":
		units = "J/kg"
		long_name = "mixed_layer_cin"
		least_significant_digit = 1
	elif param=="mu_cape":
		units = "J/kg"
		long_name = "most_unstable_cape"
		least_significant_digit = 1
	elif param=="mu_cin":
		units = "J/kg"
		long_name = "most_unstable_cin"
		least_significant_digit = 1
	elif param=="ssfc850":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-850hPa"
		least_significant_digit = 2
	elif param=="s010":
		units = "m/s"
		long_name = "bulk_wind_shear_0-10km"
		least_significant_digit = 2
	elif param=="s06":
		units = "m/s"
		long_name = "bulk_wind_shear_0-6km"
		least_significant_digit = 2
	elif param=="s03":
		units = "m/s"
		long_name = "bulk_wind_shear_0-3km"
		least_significant_digit = 2
	elif param=="s01":
		units = "m/s"
		long_name = "bulk_wind_shear_0-1km"
		least_significant_digit = 2
	elif param=="ssfc500":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-6km"
		least_significant_digit = 2
	elif param=="ssfc6":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-500m"
		least_significant_digit = 2
	elif param=="ssfc3":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-3km"
		least_significant_digit = 2
	elif param=="ssfc1":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-1km"
		least_significant_digit = 2
	elif param == "srhe_left":
		units = "m^2/s^2"
		long_name = "effective_layer_storm_relative_helicity_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh01_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-1km_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh03_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-3km_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh06_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-6 km_left_moving_storm"
		least_significant_digit = 2
	elif param == "srhe":
		units = "m^2/s^2"
		long_name = "effective_layer_storm_relative_helicity"
		least_significant_digit = 2
	elif param=="srh01":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-1km"
		least_significant_digit = 2
	elif param=="srh03":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-3km"
		least_significant_digit = 2
	elif param=="srh06":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-6 km"
		least_significant_digit = 2
	elif param=="scp":
		units = ""
		long_name = "supercell_composite_parameter"
		least_significant_digit = 8
	elif param=="scp_fixed":
		units = ""
		long_name = "supercell_composite_parameter_fixed_layer_srh"
		least_significant_digit = 8
	elif param=="estp":
		units = ""
		long_name = "non_zero_cape_significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="stp":
		units = ""
		long_name = "significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="ship":
		units = ""
		long_name = "significant_hail_parameter"
		least_significant_digit = 8
	elif param=="mmp":
		units = ""
		long_name = "mcs_maintanance_probability"
		least_significant_digit = 8
	elif param=="relhum850-500":
		units = "%"
		long_name = "avg_relative_humidity_850-500hPa"
		least_significant_digit = 3
	elif param=="relhum1000-700":
		units = "%"
		long_name = "avg_relative_humidity_850-500hPa"
		least_significant_digit = 3
	elif param=="crt":
		units = "degrees"
		long_name = "tornado_critical_angle"
		least_significant_digit = 3
	elif param=="non_sc_stp":
		units = ""
		long_name = "non-supercell_significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="vo10":
		units = "s^-1 * 1e5"
		long_name = "relative_vorticity_10m"
		least_significant_digit = 5
	elif param=="lr1000":
		units = "degC/km"
		long_name = "lapse_rate_sfc-1000m"
		least_significant_digit = 3
	elif param=="lcl":
		units = "m"
		long_name = "most_unstable_lifting_condensation_level"
		least_significant_digit = 1
	elif param == "cape*s06":
		units = ""
		long_name = "erai_cape_times_s06167"
		least_significant_digit = 1
	elif param == "cape*s06_2":
		units = ""
		long_name = "erai_cape_times_s06"
		least_significant_digit = 1
	elif param == "cape*ssfc6":
		units = ""
		long_name = "cape*ssfc6"
		least_significant_digit = 1
	elif param == "wg10":
		units = "m/s"
		long_name = "wind_gust_10m"
		least_significant_digit = 2
	elif param == "wg":
		units = "m/s"
		long_name = "max_wind_gust_10m"
		least_significant_digit = 2
	elif param == "dcp":
		units = ""
		long_name = "derecho_composite_parameter"
		least_significant_digit = 8
	elif param == "cape":
		units = "J/kg"
		long_name = "most_unstable_cape"
		least_significant_digit = 1
	elif param == "conv10":
		units = "s^-1 * 1e5"
		long_name = "convergence_10m"
		least_significant_digit = 5
	elif param == "conv1000-850":
		units = "s^-1"
		long_name = "mean_convergence_1000-850hPa"
		least_significant_digit = 5
	elif param == "conv800-600":
		units = "s^-1"
		long_name = "mean_convergence_1000-850hPa"
		least_significant_digit = 5
	elif param == "td950":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "td800":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "td850":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "dcape":
		units = "J/kg"
		long_name = "downwards_maximum_convective_available_potential_energy"
		least_significant_digit = 1
	elif param == "ddraft_temp":
		units = "J/kg"
		long_name = "temperature_defecit_downdraft_parcel_sfc"
		least_significant_digit = 3
	elif param == "dlm":
		units = "m/s"
		long_name = "deep_layer_mean_wind_speed"
		least_significant_digit = 2
	elif param == "sb_cape":
		units = "J/kg"
		long_name = "surface_based_cape"
		least_significant_digit = 1
	elif param == "sb_cin":
		units = "J/kg"
		long_name = "surface_based_cin"
		least_significant_digit = 1
	elif param == "eff_lcl":
		units = "m"
		long_name = "effective_layer_parcel_lcl"
		least_significant_digit = 1
	elif param == "ml_lcl":
		units = "m"
		long_name = "mixed_layer_parcel_lcl"
		least_significant_digit = 1
	elif param == "mu_lcl":
		units = "m"
		long_name = "most_unstable_parcel_lcl"
		least_significant_digit = 1
	elif param == "sb_lcl":
		units = "m"
		long_name = "sfc_based_parcel_lcl"
		least_significant_digit = 1
	elif param == "cp":
		units = "mm/hr"
		long_name = "convective_precipitation"
		least_significant_digit = 8
	elif param == "dcp2":
		units = ""
		long_name = "dcp_erai_cape"
		least_significant_digit = 8
	elif param == "ebwd":
		units = "m/s"
		long_name = "effective_layer_bulk_wind_fifference(shear)"
		least_significant_digit = 2
	elif param == "Umean01":
		units = "m/s"
		long_name = "mean_wind_0km_1km"
		least_significant_digit = 2
	elif param == "Umean03":
		units = "m/s"
		long_name = "mean_wind_0km_3km"
		least_significant_digit = 2
	elif param == "Umean06":
		units = "m/s"
		long_name = "mean_wind_0km_6km"
		least_significant_digit = 2
	elif param == "U500":
		units = "m/s"
		long_name = "wind_speed_500hPa"
		least_significant_digit = 2
	elif param == "U10":
		units = "m/s"
		long_name = "diagnostic_wind_speed_10m"
		least_significant_digit = 2
	elif param == "Uwindinf":
		units = "m/s"
		long_name = "wind_speed_top_of_effective_layer"
		least_significant_digit = 2
	elif param == "Umeanwindinf":
		units = "m/s"
		long_name = "mean_wind_speed_effective_layer"
		least_significant_digit = 2
	elif param == "Umean800_600":
		units = "m/s"
		long_name = "mean_wind_speed800_600hPa"
		least_significant_digit = 2
	elif param == "stp_cin_left":
		units = ""
		long_name = "significant_tornado_parameter_with_cin_left_moving_storm"
		least_significant_digit = 8
	elif param == "stp_fixed_left":
		units = ""
		long_name = "significant_tornado_parameter_with_fixed_layer_left_moving_storm"
		least_significant_digit = 8
	elif param == "stp_cin":
		units = ""
		long_name = "significant_tornado_parameter_with_cin"
		least_significant_digit = 8
	elif param == "stp_fixed":
		units = ""
		long_name = "significant_tornado_parameter_with_fixed_layer"
		least_significant_digit = 8
	elif param == "mlcape*s06":
		units = ""
		long_name = "mixed_layer_cape_times_s06167"
		least_significant_digit = 1
	elif param == "mlcape*s06_2":
		units = ""
		long_name = "mixed_layer_cape_times_s06"
		least_significant_digit = 1
	elif param == "mucape*s06":
		units = ""
		long_name = "most_unstable_cape_times_s06167"
		least_significant_digit = 1
	elif param == "mucape*s06_2":
		units = ""
		long_name = "most_unstable_cape_times_s06"
		least_significant_digit = 1
	elif param == "effcape*s06":
		units = ""
		long_name = "effective_cape_times_s06167"
		least_significant_digit = 1
	elif param == "effcape*s06_2":
		units = ""
		long_name = "effective_cape_times_s06"
		least_significant_digit = 1
	elif param == "sbcape*s06":
		units = ""
		long_name = "surface_based_cape_times_s06167"
		least_significant_digit = 1
	elif param == "sbcape*s06_2":
		units = ""
		long_name = "surface_based_cape_times_s06"
		least_significant_digit = 1
	elif param == "dmgwind":
		units = ""
		long_name = "damaging_wind_kuchera"
		least_significant_digit = 8
	elif param == "dmgwind_fixed":
		units = ""
		long_name = "damaging_wind_kuchera_fixed"
		least_significant_digit = 8
	elif param == "ducs6":
		units = ""
		long_name = "convgust_times_mlcape*s06"
		least_significant_digit = 8
	elif param == "convgust_dry":
		units = ""
		long_name = "convective_gust_dry_mburst_ewd"
		least_significant_digit = 3
	elif param == "convgust_wet":
		units = ""
		long_name = "convective_gust_wet_mburst_ewd"
		least_significant_digit = 3
	elif param == "windex":
		units = "m/s"
		long_name = "wind_index_mccann_microburst"
		least_significant_digit = 3
	elif param == "gustex":
		units = "m/s"
		long_name = "gust_index_geerts_original"
		least_significant_digit = 3
	elif param == "gustex2":
		units = "m/s"
		long_name = "gust_index_geerts_Umean06"
		least_significant_digit = 3
	elif param == "gustex3":
		units = ""
		long_name = "gust_index_geerts_times_mlcape*s06"
		least_significant_digit = 3
	elif param == "lr01":
		units = "deg/km"
		long_name = "lapse_rate_0_1km"
		least_significant_digit = 3
	elif param == "lr03":
		units = "deg/km"
		long_name = "lapse_rate_0_3km"
		least_significant_digit = 3
	elif param == "lr24":
		units = "deg/km"
		long_name = "lapse_rate_2_4km"
		least_significant_digit = 3
	elif param == "lr13":
		units = "deg/km"
		long_name = "lapse_rate_1_3km"
		least_significant_digit = 3
	elif param == "lr36":
		units = "deg/km"
		long_name = "lapse_rate_3_6km"
		least_significant_digit = 3
	elif param == "lr_subcloud":
		units = "deg/km"
		long_name = "lapse_rate_sfc_lcl"
		least_significant_digit = 3
	elif param == "lr_freezing":
		units = "deg/km"
		long_name = "lapse_rate_sfc_to_freezing"
		least_significant_digit = 3
	elif param == "qmean01":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_1km"
		least_significant_digit = 3
	elif param == "qmean03":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_3km"
		least_significant_digit = 3
	elif param == "qmean06":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_6km"
		least_significant_digit = 3
	elif param == "qmean13":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_1_3km"
		least_significant_digit = 3
	elif param == "qmean36":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_3_6km"
		least_significant_digit = 3
	elif param == "qmeansubcloud":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_sfc_to_mllcl"
		least_significant_digit = 3
	elif param == "q_melting":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_melting_level"
		least_significant_digit = 3
	elif param == "q1":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_1km"
		least_significant_digit = 3
	elif param == "q3":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_3km"
		least_significant_digit = 3
	elif param == "q6":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_6km"
		least_significant_digit = 3
	elif param == "rhmin01":
		units = "%"
		long_name = "minimum_rh_0_1km"
		least_significant_digit = 3
	elif param == "rhmin03":
		units = "%"
		long_name = "minimum_rh_0_3km"
		least_significant_digit = 3
	elif param == "rhmin06":
		units = "%"
		long_name = "minimum_rh_0_6km"
		least_significant_digit = 3
	elif param == "rhmin13":
		units = "%"
		long_name = "minimum_rh_1_3km"
		least_significant_digit = 3
	elif param == "rhmin36":
		units = "%"
		long_name = "minimum_rh_3_6km"
		least_significant_digit = 3
	elif param == "rhminsubcloud":
		units = "%"
		long_name = "minimum_rh_sfc_mllcl"
		least_significant_digit = 3
	elif param == "mhgt":
		units = "m"
		long_name = "melting_lvl_hgt"
		least_significant_digit = 1
	elif param == "mu_el":
		units = "m"
		long_name = "equilibrium_lvl_mu_parcel"
		least_significant_digit = 1
	elif param == "ml_el":
		units = "m"
		long_name = "equilibrium_lvl_ml_parcel"
		least_significant_digit = 1
	elif param == "sb_el":
		units = "m"
		long_name = "equilibrium_lvl_sb_parcel"
		least_significant_digit = 1
	elif param == "eff_el":
		units = "m"
		long_name = "equilibrium_lvl_eff_parcel"
		least_significant_digit = 1
	elif param == "s13":
		units = "m/s"
		long_name = "wind_shear_1_3km"
		least_significant_digit = 2
	elif param == "s36":
		units = "m/s"
		long_name = "wind_shear_3_6km"
		least_significant_digit = 2
	elif param == "scld":
		units = "m/s"
		long_name = "wind_shear_mllcl_to_half_the_el"
		least_significant_digit = 2
	elif param == "U1":
		units = "m/s"
		long_name = "wind_speed_1km"
		least_significant_digit = 2
	elif param == "U3":
		units = "m/s"
		long_name = "wind_speed_3km"
		least_significant_digit = 2
	elif param == "U6":
		units = "m/s"
		long_name = "wind_speed_6km"
		least_significant_digit = 2
	elif param == "Ust_left":
		units = "m/s"
		long_name = "non_parcel_bunkers_storm_motion_speed_left_moving_storm"
		least_significant_digit = 2
	elif param == "Ust":
		units = "m/s"
		long_name = "non_parcel_bunkers_storm_motion_speed"
		least_significant_digit = 2
	elif param == "Usr01_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_1km"
		least_significant_digit = 2
	elif param == "Usr03_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_3km"
		least_significant_digit = 2
	elif param == "Usr06_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_6km"
		least_significant_digit = 2
	elif param == "Usr01":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_1km"
		least_significant_digit = 2
	elif param == "Usr03":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_3km"
		least_significant_digit = 2
	elif param == "Usr06":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_6km"
		least_significant_digit = 2
	elif param == "Usr13":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_1_3km"
		least_significant_digit = 2
	elif param == "Usr36":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_3_6km"
		least_significant_digit = 2
	elif param == "mosh":
		units = ""
		long_name = "modified_sherb"
		least_significant_digit = 8
	elif param == "moshe":
		units = ""
		long_name = "modified_sherb_with_effective_shear"
		least_significant_digit = 8
	elif param == "sherb":
		units = ""
		long_name = "sever_hazards_with_reduced_buoyancy_parameter"
		least_significant_digit = 8
	elif param == "eff_sherb":
		units = ""
		long_name = "sever_hazards_with_reduced_buoyancy_parameter_effective_layer_form"
		least_significant_digit = 8
	elif param == "v_totals":
		units = ""
		long_name = "vertical_totals_index"
		least_significant_digit = 3
	elif param == "c_totals":
		units = ""
		long_name = "cross_totals_index"
		least_significant_digit = 3
	elif param == "t_totals":
		units = ""
		long_name = "total_totals_index"
		least_significant_digit = 3
	elif param == "pwat":
		units = "in"
		long_name = "precip_water_sfc_400_hPa"
		least_significant_digit = 3
	elif param == "eff_cape":
		units = "J/kg"
		long_name = "effective_inflow_layer_parcel_define_cape"
		least_significant_digit = 1
	elif param == "eff_cin":
		units = "J/kg"
		long_name = "effective_inflow_layer_parcel_define_cin"
		least_significant_digit = 1
	elif param == "eff_lcl":
		units = "m"
		long_name = "effective_inflow_layer_parcel_define_lcl"
		least_significant_digit = 1
	elif param == "wndg":
		units = ""
		long_name = "wind_damage_parameter_spc"
		least_significant_digit = 8
	elif param == "mburst":
		units = ""
		long_name = "microburst_composite_index_spc"
		least_significant_digit = 1
	elif param == "sweat":
		units = ""
		long_name = "severe_weather_threat_index_spc"
		least_significant_digit = 3
	elif param == "maxtevv":
		units = "K Pa / (km s)"
		long_name = "max_thetae_vertical_velocity_sherburn"
		least_significant_digit = 8
	elif param == "omega01":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-1km"
		least_significant_digit = 8
	elif param == "omega03":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-3km"
		least_significant_digit = 8
	elif param == "omega06":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-6km"
		least_significant_digit = 8
	elif param == "dmi":
		units = ""
		long_name = "dry_microburst_index_goes"
		least_significant_digit = 3
	elif param == "hmi":
		units = ""
		long_name = "hybrid_microburst_index_goes"
		least_significant_digit = 3
	elif param == "wmsi_mu":
		units = ""
		long_name = "wet_microburst_severity_index_goes_mucape"
		least_significant_digit = 3
	elif param == "wmsi_ml":
		units = ""
		long_name = "wet_microburst_severity_index_goes_mlcape"
		least_significant_digit = 3
	elif param == "mwpi_mu":
		units = ""
		long_name = "microburst_windspeed_potential_index_goes_mucape"
		least_significant_digit = 3
	elif param == "mwpi_ml":
		units = ""
		long_name = "microburst_windspeed_potential_index_goes_mlcape"
		least_significant_digit = 3
	elif param == "wmpi":
		units = ""
		long_name = "wet_microburst_potential_index_proctor"
		least_significant_digit = 3
	elif param == "sfc_thetae":
		units = "degC"
		long_name = "sfc_equivalent_potential_temperature"
		least_significant_digit = 3
	elif param == "dpd850":
		units = "degC"
		long_name = "dewpoint_depression_850hPa"
		least_significant_digit = 3
	elif param == "dpd700":
		units = "degC"
		long_name = "dewpoint_depression_700hPa"
		least_significant_digit = 3
	elif param == "tei":
		units = "degC"
		long_name = "thetae_index"
		least_significant_digit = 3
	elif param == "te_diff":
		units = "degC"
		long_name = "max_min_thetae_diff_3000m"
		least_significant_digit = 3
	elif param == "ml_lfc":
		units = "m"
		long_name = "mixed_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "mu_lfc":
		units = "m"
		long_name = "most_unstable_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "eff_lfc":
		units = "m"
		long_name = "effective_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "pbl_top":
		units = "m"
		long_name = "pbl_level"
		least_significant_digit = 3
	elif param == "sb_lfc":
		units = "m"
		long_name = "surface_parcel_lfc"
		least_significant_digit = 1
	elif param == "k_index":
		units = ""
		long_name = "k_index"
		least_significant_digit = 3
	elif param == "esp":
		units = ""
		long_name = "enhanced_stretching_potential"
		least_significant_digit = 8
	elif param == "wbz":
		units = "m"
		long_name = "wet_bulb_zero_height"
		least_significant_digit = 1
	elif param == "Vprime":
		units = "m/s"
		long_name = "miller_1972_wind_speed"
		least_significant_digit = 3
	elif param == "F10":
		units = "thetae/100km/3h"
		long_name = "frontogenesis_function_10m"
		least_significant_digit = 8
	elif param == "Fn10":
		units = "thetae/100km/3h"
		long_name = "frontogenetical_function_10m"
		least_significant_digit = 8
	elif param == "Fs10":
		units = "thetae/100km/3h"
		long_name = "rotational_frontogenesis_10m"
		least_significant_digit = 8
	elif param == "icon10":
		units = "s-1 * 1e5"
		long_name = "instantaneous_contraction_rate_10m"
		least_significant_digit = 8
	elif param == "vgt10":
		units = "s-1 * 1e5"
		long_name = "horizontal_velocity_gradient_tensor_magnitude_10m"
		least_significant_digit = 8
	else:
		units = ""
		long_name = ""
		least_significant_digit = 8
		print("WARNING: "+param+" HAS NO UNITS OR LONG NAME CODED IN NC_ATTRIBUTES()")

	return [units,long_name,least_significant_digit]
