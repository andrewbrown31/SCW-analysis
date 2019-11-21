import warnings
import sys
import itertools
import multiprocessing
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import os
try:
	import metpy.units as units
	import metpy.calc as mpcalc
except:
	pass
import wrf
from calc_param import save_netcdf, get_dp
import xarray as xr
from erai_read import read_erai
from erai_read import get_mask as  get_erai_mask
from barra_read import read_barra
from barra_read import get_mask as  get_barra_mask

#-------------------------------------------------------------------------------------------------

#This file contains functions to extract thunderstorm/extreme-convective-wind-gust parameters 
# from gridded model data, provided by *model*_read.py

#Function uses wrf-python and metpy. The SkewT package has been adapted for the DCAPE routine
#
#NOTES:
#
#	- CIN is only non-zero where CAPE is non-zero. CIN did not behave nicely otherwise.
#		Although, CIN is only relevant when CAPE is non-zero anyway
#	- Effective layer parcel is defined by mass-weighted effective layer properties. This is 
#		calculated using np.trapz, although I'm not sure this is correct.
#-------------------------------------------------------------------------------------------------


def get_dp(ta,hur,dp_mask=True):

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

def get_min_var_hgt(var3d, hgt, hgt_bot, hgt_top, terrain):

	hgt = hgt - terrain
	var3d[hgt < hgt_bot] = np.nan
	var3d[hgt > hgt_top] = np.nan
	result = np.nanmin(var3d, axis=0)
	return result

def get_mean_var_hgt(var3d, hgt, hgt_bot, hgt_top, terrain, pressure_weighted=False, p3d=None):

	hgt = hgt - terrain
	var3d_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) , var3d)
	if pressure_weighted:
		try:
			result = np.ma.average(var3d, axis=0, weights=p3d)
		except:
			raise ValueError("FUNCTION get_mean_var_hgt() IS FAILING TO TAKE A PRESSURE WEIGHTED"+\
				" AVERAGE. HAS A 3D PRESSURE FIELD BEEN PARSED?")
	else:
		result = np.ma.mean(var3d_ma, axis=0)
	return result

def get_mean_var_p(var3d, p3d, p_bot, p_top, ps, pressure_weighted=False):

	var3d_ma = np.ma.masked_where((p3d > p_bot) | (p3d < p_top) | (p3d > ps), var3d)
	if pressure_weighted:
		result = np.ma.average(var3d_ma, axis=0, weights=p3d)
	else:
		result = np.ma.mean(var3d_ma, axis=0)
	return result

def get_shear_hgt(u,v,hgt,hgt_bot,hgt_top,terrain,components=False):
	#Get bulk wind shear [lat, lon] between two heights, based on 3d input of u, v, and 
	#hgt [levels,lat,lon]

	ubot = get_var_hgt_lvl(u, np.copy(hgt), hgt_bot, terrain)
	vbot = get_var_hgt_lvl(v, np.copy(hgt), hgt_bot, terrain)
	utop = get_var_hgt_lvl(u, np.copy(hgt), hgt_top, terrain)
	vtop = get_var_hgt_lvl(v, np.copy(hgt), hgt_top, terrain)

	if components:
		return [utop-ubot, vtop-vbot]
	else:
		shear = np.array(np.sqrt(np.square(utop-ubot)+np.square(vtop-vbot)))
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

def get_storm_motion(u, v, hgt, terrain):

	#Get left and right storm motion vectors, using non-parcel bunkers storm motion (see SHARPpy)
	hgt = hgt-terrain
	mnu6 = get_mean_var_hgt(np.copy(u),np.copy(hgt),0,6000,terrain)
	mnv6 = get_mean_var_hgt(np.copy(v),np.copy(hgt),0,6000,terrain)
	us6, vs6 = get_shear_hgt(np.copy(u), np.copy(v), np.copy(hgt), 0, 6000, terrain, components=True)
	tmp = 7.5 / (np.sqrt(np.square(us6) + np.square(vs6)))
	u_storm_right = mnu6 + (tmp * vs6)
	v_storm_right = mnv6 - (tmp * us6)
	u_storm_left = mnu6 - (tmp * vs6)
	v_storm_left = mnv6 + (tmp * us6)

	return [u_storm_right, v_storm_right, u_storm_left, v_storm_left]

def get_srh(u,v,hgt,hgt_bot,hgt_top,terrain):
	#Get storm relative helicity [lat, lon] based on 3d input of u, v, and storm motion u and
	# v components
	# Is between the bottom pressure level (1000 hPa), approximating 0 m, and hgt_top (m)
	#Storm motion approxmiated by using mean 0-6 km wind

	u_storm_right, v_storm_right, u_storm_left, v_storm_left = \
		get_storm_motion(np.copy(u), np.copy(v), np.copy(hgt), terrain)

	hgt = hgt - terrain
	u_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) , u)
	v_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) , v)
	sru_left = u_ma - u_storm_left
	srv_left = v_ma - v_storm_left
	layers_left = (sru_left[1:] * srv_left[:-1]) - (sru_left[:-1] * srv_left[1:])
	srh_left = abs(np.sum(layers_left,axis=0))
	sru_right = u_ma - u_storm_right
	srv_right = v_ma - v_storm_right
	layers_right = (sru_right[1:] * srv_right[:-1]) - (sru_right[:-1] * srv_right[1:])
	srh_right = abs(np.sum(layers_right,axis=0))

	return srh_left, srh_right

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

def get_var_hgt_lvl(var, hgt, desired_hgt, terrain):
	#Interpolate 3d varibale ("var") to a desired hgt sfc ("desired hgt") which is AGL (hgt should be ASL)
	hgt = hgt - terrain
	interp_var = wrf.interplevel(var,hgt,desired_hgt,meta=False)
	interp_var[hgt[0] >= desired_hgt] = var[0,hgt[0] >= desired_hgt]
	interp_var[(np.where(hgt==desired_hgt))[1],(np.where(hgt==desired_hgt))[2]] = var[hgt==desired_hgt]
	return interp_var

def get_lr_hgt(t,hgt,hgt_bot,hgt_top,terrain):
	#Get lapse rate (C/km) between two height levels (in km)

	hgt = hgt - terrain

	if hgt_top == "freezing":
		hgt_top = get_t_hgt(t,hgt,0,0)
		t_top = np.zeros(hgt_top.shape)
		t_bot = wrf.interplevel(t,hgt,float(hgt_bot),meta=False)
		t_bot[hgt[0] >= hgt_bot] = t[0,hgt[0] >= hgt_bot]
		t_bot[(np.where(hgt==hgt_bot))[1],(np.where(hgt==hgt_bot))[2]] = t[hgt==hgt_bot]

	else:

		if t.ndim == 1:
			t_bot = np.interp(hgt_bot,hgt,t)
			t_top = np.interp(hgt_top,hgt,t)
		else:
			t_bot = wrf.interplevel(t,hgt,hgt_bot,meta=False)
			t_bot[hgt[0] >= hgt_bot] = t[0,hgt[0] >= hgt_bot]
			t_bot[(np.where(hgt==hgt_bot))[1],(np.where(hgt==hgt_bot))[2]] = t[hgt==hgt_bot]
			t_top = wrf.interplevel(t,hgt,hgt_top,meta=False)
			t_top[hgt[-1] <= hgt_top] = t[-1,hgt[-1] <= hgt_bot]
			t_top[(np.where(hgt==hgt_top))[1],(np.where(hgt==hgt_top))[2]] = t[hgt==hgt_top]

	return np.squeeze(- (t_top - t_bot) / ((hgt_top - hgt_bot)/1000))

def get_t_hgt(t,hgt,t_value,terrain):
	#Get the height [lev,lat,lon] at which temperature [lev,lat,lon] is equal to t_value

	hgt = hgt - terrain

	if t.ndim == 1:
		t_hgt = np.interp(t_value,np.flipud(t),np.flipud(hgt))
	else:
		t_hgt = np.array(wrf.interplevel(hgt, t, t_value))

	return t_hgt

def get_var_hgt(var,hgt,var_value,terrain):
	#Get the height [lev,lat,lon] at which a "var" [lev,lat,lon] is equal to "var_value"

	hgt = hgt - terrain

	if t.ndim == 1:
		var_hgt = np.interp(var_value,np.flipud(var),np.flipud(hgt))
	else:
		var_hgt = np.array(wrf.interplevel(hgt, var, var_value))

	return var_hgt

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

def maxtevv_fn(te, om, hgt, terrain):

	hgt = hgt-terrain
	te2km = np.where((hgt>=0) & (hgt<=2000), te, np.nan)
	hgt2km = np.where((hgt>=0) & (hgt<=2000), hgt, np.nan)
	te6km = np.where((hgt>=0) & (hgt<=6000), te, np.nan)
	hgt6km = np.where((hgt>=0) & (hgt<=6000), hgt, np.nan)

	maxtevv = np.zeros(te2km.shape)
	for i in np.arange(te2km.shape[0]):

		temp_dte = te6km - te2km[i]
		temp_dz = (hgt6km - hgt2km[i]) / 1000
		temp_dz[temp_dz < 0.25] = np.nan
		temp_tevv = (temp_dte / temp_dz)[1:] * om
		temp_maxtevv = np.nanmax(temp_tevv, axis=0)
		maxtevv[i] = temp_maxtevv

	return (np.nanmax(maxtevv,axis=0))

def thetae_diff(te, hgt, terrain):

	#Returns thetae difference (diff between max and min thetae in lowest 3000m) 

	hgt = hgt-terrain

	te_ma = np.ma.masked_where((hgt<0) | (hgt>3000), te)

	min_idx = np.ma.argmin(te_ma, axis=0)
	max_idx = np.ma.argmax(te_ma, axis=0)

	min_te = min_idx.choose(te_ma)
	max_te = max_idx.choose(te_ma)
	te_diff = (max_te - min_te)
	te_diff[te_diff<0] = 0
	te_diff[min_idx < max_idx] = 0

	return te_diff

def tei_fn(te, p3d, ps, hgt, terrain):

	#Return theta-e index. Defined by SPC as diff between sfc thetae and min thetae in sfc to 400 hPa AGL layer.
	#Note that SHARPpy has reverted to diff between max and min thetae in same layer, to get closer to SPC
	# operational output

	te[p3d > ps] = np.nan
	te[p3d < (ps - 400)] = np.nan
	min_te = np.nanmin(te, axis=0)
	sfc_te = get_var_hgt_lvl(te, hgt, 0, terrain)
	tei = sfc_te - min_te
	tei[tei < 0] = 0

	return tei

def kinematics(u, v, thetae, dx, dy, lats):

	#Use metpy functions to calculate various kinematics, given 2d arrays as inputs

	ddy_thetae = mpcalc.first_derivative( thetae, delta=dy, axis=0)
	ddx_thetae = mpcalc.first_derivative( thetae, delta=dx, axis=1)
	mag_thetae = np.sqrt( ddx_thetae**2 + ddy_thetae**2)
	div = mpcalc.divergence(u, v, dx, dy)
	strch_def = mpcalc.stretching_deformation(u, v, dx, dy)
	shear_def = mpcalc.shearing_deformation(u, v, dx, dy)
	tot_def = mpcalc.total_deformation(u, v, dx, dy)
	psi = 0.5 * np.arctan2(shear_def, strch_def)
	beta = np.arcsin((-ddx_thetae * np.cos(psi) - ddy_thetae * np.sin(psi)) / mag_thetae)
	vo = mpcalc.vorticity(u, v, dx, dy)
	conv = -div * 1e5

	F = 0.5 * mag_thetae * (tot_def * np.cos(2 * beta) - div) * 1.08e4 * 1e5
	Fn = 0.5 * mag_thetae * (div - tot_def * np.cos(2 * beta) ) * 1.08e4 * 1e5
	Fs = 0.5 * mag_thetae * (vo + tot_def * np.sin(2 * beta) ) * 1.08e4 * 1e5
	icon = 0.5 * (tot_def - div) * 1e5
	vgt = np.sqrt( div**2 + vo**2 + tot_def**2 ) * 1e5

	return [F, Fn, Fs, icon, vgt, conv, vo*1e5]


if __name__ == "__main__":

	warnings.simplefilter("ignore")

	#Parse arguments from cmd line and set up inputs (date region model)
	model = sys.argv[1]
	region = sys.argv[2]
	t1 = sys.argv[3]
	t2 = sys.argv[4]
	issave = sys.argv[5]
	out_name = sys.argv[6]
	if region == "sa_small":
		start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "aus":
		start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	else:
		raise ValueError("INVALID REGION\n")
	domain = [start_lat,end_lat,start_lon,end_lon]
	try:
		time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]
	except:
		raise ValueError("INVALID START OR END TIME. SHOULD BE YYYYMMDDHH\n")
	if issave=="True":
		issave = True
	elif issave=="False":
		issave = False
	else:
		raise ValueError("\n INVALID ISSAVE...SHOULD BE True OR False")

	#Set OpenMP options for wrf-python (utilising all processers)
	wrf.omp_set_num_threads(1)
	print("USING "+str(wrf.omp_get_num_procs())+" PROCESSERS")

	#Load data
	if model == "erai":
		ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,\
			cp,wg10,mod_cape,lon,lat,date_list = \
			read_erai(domain,time)
		dp = get_dp(hur=hur, ta=ta, dp_mask = False)
		lsm = np.repeat(get_erai_mask(lon,lat)[np.newaxis],ta.shape[0],0)
	elif model == "barra":
		ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
			read_barra(domain,time)
		dp = get_dp(hur=hur, ta=ta, dp_mask=False)
		lsm = np.repeat(get_barra_mask(lon,lat)[np.newaxis],ta.shape[0],0)
	x, y = np.meshgrid(lon,lat)
	dx, dy = mpcalc.lat_lon_grid_deltas(x,y)

	#Assign p levels to a 4d array, with same dimensions as input variables (ta, hgt, etc.)
	p_3d = np.moveaxis(np.tile(p,[ta.shape[0],ta.shape[2],ta.shape[3],1]),[0,1,2,3],[0,2,3,1])

	#Insert surface arrays, creating new arrays with "sfc" prefix
	sfc_ta = np.insert(ta, 0, tas, axis=1) 
	sfc_hgt = np.insert(hgt, 0, terrain, axis=1) 
	sfc_dp = np.insert(dp, 0, ta2d, axis=1) 
	sfc_p_3d = np.insert(p_3d, 0, ps, axis=1) 
	sfc_ua = np.insert(ua, 0, uas, axis=1) 
	sfc_va = np.insert(va, 0, vas, axis=1) 

	#Sort by ascending p
	temp0,a,temp1,temp2 = np.meshgrid(np.arange(sfc_p_3d.shape[0]) , np.arange(sfc_p_3d.shape[1]),\
		 np.arange(sfc_p_3d.shape[2]), np.arange(sfc_p_3d.shape[3]))
	sort_inds = np.flip(np.lexsort([np.swapaxes(a,1,0),sfc_p_3d],axis=1), axis=1)
	sfc_hgt = np.take_along_axis(sfc_hgt, sort_inds, axis=1)
	sfc_dp = np.take_along_axis(sfc_dp, sort_inds, axis=1)
	sfc_p_3d = np.take_along_axis(sfc_p_3d, sort_inds, axis=1)
	sfc_ua = np.take_along_axis(sfc_ua, sort_inds, axis=1)
	sfc_va = np.take_along_axis(sfc_va, sort_inds, axis=1)
	sfc_ta = np.take_along_axis(sfc_ta, sort_inds, axis=1)

	param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
			"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl",\
			"dcape", \
			"ml_lfc", "mu_lfc", "eff_lfc", "sb_lfc",\
			"lr01", "lr03", "lr13", "lr36", "lr24", "lr_freezing","lr_subcloud",\
			"qmean01", "qmean03", "qmean06", "qmean13", "qmean36",\
			"qmeansubcloud", "q_melting", "q1", "q3", "q6",\
			"rhmean01", "rhmean03", "rhmean06", "rhmean13", "rhmean36","rhmeansubcloud",\
			"rhmin01", "rhmin03", "rhmin06", "rhmin13", "rhmin36", \
			"rhminsubcloud", "tei", "wbz", \
			"mhgt", "mu_el", "ml_el", "sb_el", "eff_el", \
			"pwat", "v_totals", "c_totals", "t_totals", \
			"maxtevv", "te_diff", "dpd850", "dpd700", "pbl_top",\
			\
			"srhe", "srh01", "srh03", "srh06", \
			"srhe_left", "srh01_left", "srh03_left", "srh06_left", \
			"ebwd", "s010", "s06", "s03", "s01", "s13", "s36", "scld", \
			"omega01", "omega03", "omega06", \
			"U500", "U10", "U1", "U3", "U6", \
			"Ust", "Ust_left", "Usr01", "Usr03", "Usr06", "Usr01_left",\
			"Usr03_left", "Usr06_left", \
			"Uwindinf", "Umeanwindinf", "Umean800_600", "Umean06", \
			"Umean01", "Umean03", "wg10",\
			\
			"dcp", "stp_cin", "stp_cin_left", "stp_fixed", "stp_fixed_left",\
			"scp", "ship",\
			"mlcape*s06", "mucape*s06", "sbcape*s06", "effcape*s06", \
			"mlcape*s06_2", "mucape*s06_2", "sbcape*s06_2", \
			"effcape*s06_2", "dmgwind", "hmi", "wmsi_mu", "wmsi_ml",\
			"dmi", "mwpi_mu", "mwpi_ml", "ducs6", "convgust", "windex",\
			"gustex", "gustex2","gustex3", "eff_sherb", "sherb", \
			"moshe", "mosh", "wndg","mburst","sweat","k_index", "esp", "Vprime",\
			\
			"F10", "Fn10", "Fs10", "icon10", "vgt10", "conv10", "vo10",\
			"F01", "Fn01", "Fs01", "icon01", "vgt01", "conv01", "vo01",\
			"F03", "Fn03", "Fs03", "icon03", "vgt03", "conv03", "vo03",\
				])
	if model == "erai":
		param = np.concatenate([param, ["cape","cp","dcp2","cape*s06","cape*s062"]])
	output_data = np.zeros((len(param),ta.shape[0],ta.shape[2],ta.shape[3]))

	tot_start = dt.datetime.now()
	for t in np.arange(0,len(date_list)):
		print(date_list[t])

		cape_start = dt.datetime.now()

		#Calculate q and wet bulb for pressure level arrays
		ta_unit = units.units.degC*ta[t,:,:,:]
		dp_unit = units.units.degC*dp[t,:,:,:]
		p_unit = units.units.hectopascals*p_3d[t,:,:,:]
		hur_unit = mpcalc.relative_humidity_from_dewpoint(ta_unit, \
			dp_unit)*100*units.units.percent
		q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
			ta_unit,p_unit)
		theta_unit = mpcalc.potential_temperature(p_unit,ta_unit)
		thetae_unit = mpcalc.equivalent_potential_temperature(p_unit,ta_unit,dp_unit)
		q = np.array(q_unit)

		#Calculate q and wet bulb for pressure level arrays with surface values
		sfc_ta_unit = units.units.degC*sfc_ta[t,:,:,:]
		sfc_dp_unit = units.units.degC*sfc_dp[t,:,:,:]
		sfc_p_unit = units.units.hectopascals*sfc_p_3d[t,:,:,:]
		sfc_hur_unit = mpcalc.relative_humidity_from_dewpoint(sfc_ta_unit, sfc_dp_unit)*\
			100*units.units.percent
		sfc_q_unit = mpcalc.mixing_ratio_from_relative_humidity(sfc_hur_unit,\
			sfc_ta_unit,sfc_p_unit)
		sfc_theta_unit = mpcalc.potential_temperature(sfc_p_unit,sfc_ta_unit)
		sfc_thetae_unit = mpcalc.equivalent_potential_temperature(sfc_p_unit,sfc_ta_unit,sfc_dp_unit)
		sfc_q = np.array(sfc_q_unit)
		sfc_hur = np.array(sfc_hur_unit)
		#APPROXMIATED USING THE ONE-THIRD RULE: KNOX (2017)
		sfc_wb = sfc_ta[t] - (1/3 * (sfc_ta[t] - sfc_dp[t]))

		#Calculate mixed-layer parcel indices, based on avg sfc-100 hPa AGL layer parcel.
		#First, find avg values for ta, p, hgt and q for ML (between the surface
		# and 100 hPa AGL)
		ml_inds = ((sfc_p_3d[t] <= ps[t]) & (sfc_p_3d[t] >= (ps[t] - 100)))
		ml_ta_avg = np.ma.masked_where(~ml_inds, sfc_ta[t]).mean(axis=0).data
		ml_q_avg = np.ma.masked_where(~ml_inds, sfc_q).mean(axis=0).data
		ml_hgt_avg = np.ma.masked_where(~ml_inds, sfc_hgt[t]).mean(axis=0).data
		ml_p3d_avg = np.ma.masked_where(~ml_inds, sfc_p_3d[t]).mean(axis=0).data
		#Insert the mean values into the bottom of the 3d arrays pressure-level arrays
		ml_ta_arr = np.insert(sfc_ta[t],0,ml_ta_avg,axis=0)
		ml_q_arr = np.insert(sfc_q,0,ml_q_avg,axis=0)
		ml_hgt_arr = np.insert(sfc_hgt[t],0,ml_hgt_avg,axis=0)
		ml_p3d_arr = np.insert(sfc_p_3d[t],0,ml_p3d_avg,axis=0)
		#Sort by ascending p
		a,temp1,temp2 = np.meshgrid(np.arange(ml_p3d_arr.shape[0]) ,\
			 np.arange(ml_p3d_arr.shape[1]), np.arange(ml_p3d_arr.shape[2]))
		sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),ml_p3d_arr],axis=0))
		ml_ta_arr = np.take_along_axis(ml_ta_arr, sort_inds, axis=0)
		ml_p3d_arr = np.take_along_axis(ml_p3d_arr, sort_inds, axis=0)
		ml_hgt_arr = np.take_along_axis(ml_hgt_arr, sort_inds, axis=0)
		ml_q_arr = np.take_along_axis(ml_q_arr, sort_inds, axis=0)
		#Calculate CAPE using wrf-python. 
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

		#Now get most-unstable CAPE (max CAPE in vertical, ensuring parcels used are AGL)
		cape3d = wrf.cape_3d(sfc_p_3d[t,:,:,:],sfc_ta[t,:,:,:]+273.15,\
				sfc_q,sfc_hgt[t,:,:,:],\
				terrain,ps[t,:,:],\
				True,meta=False, missing=0)
		cape = cape3d.data[0]
		cin = cape3d.data[1]
		lfc = cape3d.data[2]
		lcl = cape3d.data[3]
		el = cape3d.data[4]
		#Mask values which are below the surface and above 500 hPa AGL
		cape[(sfc_p_3d[t] > ps[t]) | (sfc_p_3d[t]<(ps[t]-500))] = np.nan
		cin[(sfc_p_3d[t] > ps[t]) | (sfc_p_3d[t]<(ps[t]-500))] = np.nan
		lfc[(sfc_p_3d[t] > ps[t]) | (sfc_p_3d[t]<(ps[t]-500))] = np.nan
		lcl[(sfc_p_3d[t] > ps[t]) | (sfc_p_3d[t]<(ps[t]-500))] = np.nan
		el[(sfc_p_3d[t] > ps[t]) | (sfc_p_3d[t]<(ps[t]-500))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		mu_cape_inds = np.nanargmax(cape,axis=0)
		mu_cape = mu_cape_inds.choose(cape)
		mu_cin = mu_cape_inds.choose(cin)
		mu_lfc = mu_cape_inds.choose(lfc)
		mu_lcl = mu_cape_inds.choose(lcl)
		mu_el = mu_cape_inds.choose(el)

		#Now get surface based CAPE. Simply the CAPE defined by parcel 
		#with surface properties
		sb_cape = cape[0]
		sb_cin = cin[0]
		sb_lfc = lfc[0]
		sb_lcl = lcl[0]
		sb_el = el[0]

		#Now get the effective-inflow layer parcel CAPE. Layer defined as a parcel with
		# the mass-wegithted average conditions of the inflow layer; the layer between when the profile
		# has CAPE > 100 and cin < 250.
		#If no effective layer, effective layer CAPE is zero.
		#Only levels below 500 hPa AGL are considered
		eff_p = np.where((cape >= 100) & (cin <= 250) & (sfc_p_3d[t]>(ps[t]-500)) & (sfc_p_3d[t]<ps[t]),\
			sfc_p_3d[t], np.nan)
		eff_ta = np.where((cape >= 100) & (cin <= 250) & (sfc_p_3d[t]>(ps[t]-500)) & (sfc_p_3d[t]<ps[t]),\
			sfc_ta[t], np.nan)
		eff_hgt = np.where((cape >= 100) & (cin <= 250) & (sfc_p_3d[t]>(ps[t]-500)) & (sfc_p_3d[t]<ps[t]),\
			sfc_hgt[t], np.nan)
		eff_q = np.where((cape >= 100) & (cin <= 250) & (sfc_p_3d[t]>(ps[t]-500)) & (sfc_p_3d[t]<ps[t]),\
			sfc_q[t], np.nan)

		eff_avg_p = (np.nanmin(eff_p,axis=0) + np.nanmax(eff_p,axis=0)) / 2
		eff_avg_hgt = (np.nanmin(eff_hgt,axis=0) + np.nanmax(eff_hgt,axis=0)) / 2
		eff_avg_ta = np.nanmean(eff_ta, axis=0)
		eff_avg_q = np.nanmean(eff_q, axis=0)

		eff_avg_p = np.where(np.isnan(eff_avg_p), sfc_p_3d[t,0], eff_avg_p)
		eff_avg_hgt = np.where(np.isnan(eff_avg_p), sfc_hgt[t,0], eff_avg_hgt)
		eff_avg_ta = np.where(np.isnan(eff_avg_p), sfc_ta[t,0], eff_avg_ta)
		eff_avg_q = np.where(np.isnan(eff_avg_p), sfc_q[t,0], eff_avg_q)
		#Insert the mean values into the bottom of the 3d arrays pressure-level arrays
		eff_ta_arr = np.insert(sfc_ta[t],0,eff_avg_ta,axis=0)
		eff_q_arr = np.insert(sfc_q,0,eff_avg_q,axis=0)
		eff_hgt_arr = np.insert(sfc_hgt[t],0,eff_avg_hgt,axis=0)
		eff_p3d_arr = np.insert(sfc_p_3d[t],0,eff_avg_p,axis=0)
		#Sort by ascending p
		a,temp1,temp2 = np.meshgrid(np.arange(eff_p3d_arr.shape[0]) ,\
			 np.arange(eff_p3d_arr.shape[1]), np.arange(eff_p3d_arr.shape[2]))
		sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),eff_p3d_arr],axis=0))
		eff_ta_arr = np.take_along_axis(eff_ta_arr, sort_inds, axis=0)
		eff_p3d_arr = np.take_along_axis(eff_p3d_arr, sort_inds, axis=0)
		eff_hgt_arr = np.take_along_axis(eff_hgt_arr, sort_inds, axis=0)
		eff_q_arr = np.take_along_axis(eff_q_arr, sort_inds, axis=0)
		#Calculate CAPE using wrf-python. 
		cape3d_effavg = wrf.cape_3d(eff_p3d_arr,eff_ta_arr + 273.15,\
			eff_q_arr,eff_hgt_arr,terrain,ps[t,:,:],False,meta=False, missing=0)
		eff_cape = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
			(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[0]).max(axis=0).filled(0)
		eff_cin = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
			(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[1]).max(axis=0).filled(0)
		eff_lfc = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
			(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[2]).max(axis=0).filled(0)
		eff_lcl = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
			(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[3]).max(axis=0).filled(0)
		eff_el = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
			(eff_p3d_arr==eff_avg_p)),\
			cape3d_effavg.data[4]).max(axis=0).filled(0)
		if t == 0:
			print("CAPE TIME:")
			print(dt.datetime.now() - cape_start)

		#Calculate other parameters
		#Thermo
		thermo_start = dt.datetime.now()
		lr01 = get_lr_hgt(sfc_ta[t],sfc_hgt[t],0,1000,terrain)
		lr03 = get_lr_hgt(sfc_ta[t],sfc_hgt[t],0,3000,terrain)
		lr13 = get_lr_hgt(sfc_ta[t],sfc_hgt[t],1000,3000,terrain)
		lr24 = get_lr_hgt(sfc_ta[t],sfc_hgt[t],2000,4000,terrain)
		lr36 = get_lr_hgt(sfc_ta[t],sfc_hgt[t],3000,6000,terrain)
		lr_freezing = get_lr_hgt(sfc_ta[t],sfc_hgt[t],0,"freezing",terrain)
		lr_subcloud = get_lr_hgt(sfc_ta[t],sfc_hgt[t],0,ml_lcl,terrain)
		lr850_670 = get_lr_p(ta[t], p, hgt[t], 850, 670)
		melting_hgt = get_t_hgt(sfc_ta[t],sfc_hgt[t],0,terrain)
		hwb0 = get_var_hgt(sfc_wb,sfc_hgt[t],0,terrain)
		rhmean01 = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],0,1000,terrain)
		rhmean03 = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],0,3000,terrain)
		rhmean06 = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],0,6000,terrain)
		rhmean13 = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],1000,3000,terrain)
		rhmean36 = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],3000,6000,terrain)
		rhmeansubcloud = get_mean_var_hgt(np.copy(sfc_hur),sfc_hgt[t],0,ml_lcl,terrain)
		qmean01 = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],0,1000,terrain) * 1000
		qmean03 = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],0,3000,terrain) * 1000
		qmean06 = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],0,6000,terrain) * 1000
		qmean13 = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],1000,3000,terrain) * 1000
		qmean36 = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],3000,6000,terrain) * 1000
		qmeansubcloud = get_mean_var_hgt(np.copy(sfc_q),sfc_hgt[t],0,ml_lcl,terrain) * 1000
		q_melting = get_var_hgt_lvl(np.copy(sfc_q), sfc_hgt[t], melting_hgt, terrain) * 1000
		q1 = get_var_hgt_lvl(np.copy(sfc_q), sfc_hgt[t], 1000, terrain) * 1000
		q3 = get_var_hgt_lvl(np.copy(sfc_q), sfc_hgt[t], 3000, terrain) * 1000
		q6 = get_var_hgt_lvl(np.copy(sfc_q), sfc_hgt[t], 6000, terrain) * 1000
		rhmin01 = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 0, 1000, terrain)
		rhmin03 = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 0, 3000, terrain)
		rhmin06 = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 0, 6000, terrain)
		rhmin13 = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 1000, 3000, terrain)
		rhmin36 = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 3000, 6000, terrain)
		rhminsubcloud = get_min_var_hgt(np.copy(sfc_hur), sfc_hgt[t], 0, ml_lcl, terrain)
		v_totals = ta[t, np.argmin(abs(p-850))] - ta[t, np.argmin(abs(p-500))] 
		c_totals = dp[t, np.argmin(abs(p-850))] - ta[t, np.argmin(abs(p-500))] 
		t_totals = v_totals + c_totals
		pwat = (((sfc_q[:-1]+sfc_q[1:])*1000/2 * (sfc_p_3d[t,:-1]-sfc_p_3d[t,1:])) * 0.00040173)\
			.sum(axis=0)	#From SHARPpy
		maxtevv = maxtevv_fn(np.array(sfc_thetae_unit), wap[t], sfc_hgt[t], terrain)
		te_diff = thetae_diff(np.array(sfc_thetae_unit), sfc_hgt[t], terrain)
		tei = tei_fn(np.array(sfc_thetae_unit), sfc_p_3d[t], ps[t], sfc_hgt[t], terrain)
		dpd850 = ta[t, np.argmin(abs(p-850))] - dp[t, np.argmin(abs(p-850))]
		dpd700 = ta[t, np.argmin(abs(p-700))] - dp[t, np.argmin(abs(p-700))]
		if t == 0:
			print("THERMO TIME:")
			print(dt.datetime.now() - thermo_start)
		#Winds
		winds_start = dt.datetime.now()
		umeanwindinf = get_mean_var_hgt(sfc_ua[t], sfc_hgt[t], np.nanmin(eff_hgt,axis=0), \
					np.nanmax(eff_hgt,axis=0),terrain)
		vmeanwindinf = get_mean_var_hgt(sfc_va[t], sfc_hgt[t], np.nanmin(eff_hgt,axis=0),\
					np.nanmax(eff_hgt,axis=0),terrain)
		umean01 = get_mean_var_hgt(sfc_ua[t], sfc_hgt[t], 0, 1000, terrain)
		vmean01 = get_mean_var_hgt(sfc_va[t], sfc_hgt[t], 0, 1000, terrain)
		umean03 = get_mean_var_hgt(sfc_ua[t], sfc_hgt[t], 0, 3000, terrain)
		vmean03 = get_mean_var_hgt(sfc_va[t], sfc_hgt[t], 0, 3000, terrain)
		umean06 = get_mean_var_hgt(sfc_ua[t], sfc_hgt[t], 0, 6000, terrain)
		vmean06 = get_mean_var_hgt(sfc_va[t], sfc_hgt[t], 0, 6000, terrain)
		umean800_600 = get_mean_var_p(ua[t], p_3d[t], 800, 600, ps[t])
		vmean800_600 = get_mean_var_p(va[t], p_3d[t], 800, 600, ps[t])
		Umeanwindinf = np.sqrt( (umeanwindinf**2) + (vmeanwindinf**2) )
		Umean01 = np.sqrt( (umean01**2) + (vmean01**2) )
		Umean03 = np.sqrt( (umean03**2) + (vmean03**2) )
		Umean06 = np.sqrt( (umean06**2) + (vmean06**2) )
		Umean800_600 = np.sqrt( (umean800_600**2) + (vmean800_600**2) )
		uwindinf = get_var_hgt_lvl(sfc_ua[t], sfc_hgt[t], eff_avg_hgt, terrain)
		vwindinf = get_var_hgt_lvl(sfc_va[t], sfc_hgt[t], eff_avg_hgt, terrain)
		u10 = get_var_hgt_lvl(sfc_ua[t], sfc_hgt[t], 10, terrain)
		v10 = get_var_hgt_lvl(sfc_va[t], sfc_hgt[t], 10, terrain)
		u500 = ua[t,np.argmin(abs(p-500))]
		v500 = va[t,np.argmin(abs(p-500))]
		u1 = get_var_hgt_lvl(sfc_ua[t], sfc_hgt[t], 1000, terrain) 
		v1 = get_var_hgt_lvl(sfc_va[t], sfc_hgt[t], 1000, terrain) 
		u3 = get_var_hgt_lvl(sfc_ua[t], sfc_hgt[t], 3000, terrain) 
		v3 = get_var_hgt_lvl(sfc_va[t], sfc_hgt[t], 3000, terrain) 
		u6 = get_var_hgt_lvl(sfc_ua[t], sfc_hgt[t], 6000, terrain) 
		v6 = get_var_hgt_lvl(sfc_va[t], sfc_hgt[t], 6000, terrain) 
		Uwindinf = np.sqrt( (uwindinf**2) + (vwindinf**2) )
		U500 = np.sqrt( (u500**2) + (v500**2) )
		U10 = np.sqrt( (u10**2) + (v10**2) )
		U1 = np.sqrt( (u1**2) + (v1**2) )
		U3 = np.sqrt( (u3**2) + (v3**2) )
		U6 = np.sqrt( (u6**2) + (v6**2) )
		scld = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], ml_lcl, 0.5*ml_el, terrain)
		s01 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 1000, terrain)
		s03 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 3000, terrain)
		s06 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 6000, terrain)
		s010 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 10000, terrain)
		s13 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 1000, 3000, terrain)
		s36 = get_shear_hgt(sfc_ua[t], sfc_va[t], sfc_hgt[t], 3000, 6000, terrain)
		ebwd = get_shear_hgt(sfc_va[t], sfc_va[t], sfc_hgt[t], np.nanmin(eff_hgt,axis=0),\
					np.nanmax(eff_hgt,axis=0),terrain)
		srh01_left, srh01_right = get_srh(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 1000, terrain)
		srh03_left, srh03_right = get_srh(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 3000, terrain)
		srh06_left, srh06_right = get_srh(sfc_ua[t], sfc_va[t], sfc_hgt[t], 0, 6000, terrain)
		srhe_left, srhe_right = get_srh(sfc_ua[t], sfc_va[t], sfc_hgt[t], \
						np.nanmin(eff_hgt,axis=0), np.nanmax(eff_hgt,axis=0), terrain)
		ust_right, vst_right, ust_left, vst_left = \
			get_storm_motion(sfc_ua[t], sfc_va[t], sfc_hgt[t], terrain)
		sru01_right = umean01 - ust_right
		srv01_right = vmean01 - vst_right
		sru03_right = umean03 - ust_right
		srv03_right = vmean03 - vst_right
		sru06_right = umean06 - ust_right
		srv06_right = vmean06 - vst_right
		sru01_left = umean01 - ust_left
		srv01_left = vmean01 - vst_left
		sru03_left = umean03 - ust_left
		srv03_left = vmean03 - vst_left
		sru06_left = umean06 - ust_left
		srv06_left = vmean06 - vst_left
		Ust_right = np.sqrt( ust_right**2 + vst_right**2)
		Ust_left = np.sqrt( ust_left**2 + vst_left**2)
		Usr01_right = np.sqrt( sru01_right**2 + srv01_right**2)
		Usr03_right = np.sqrt( sru03_right**2 + srv03_right**2)
		Usr06_right = np.sqrt( sru06_right**2 + srv06_right**2)
		Usr01_left = np.sqrt( sru01_left**2 + srv01_left**2)
		Usr03_left = np.sqrt( sru03_left**2 + srv03_left**2)
		Usr06_left = np.sqrt( sru06_left**2 + srv06_left**2)
		omega01 = get_mean_var_hgt(wap[t], hgt[t], 0, 1000, terrain)
		omega03 = get_mean_var_hgt(wap[t], hgt[t], 0, 3000, terrain)
		omega06 = get_mean_var_hgt(wap[t], hgt[t], 0, 6000, terrain)
		if t == 0:
			print("WINDS TIME:")
			print(dt.datetime.now() - winds_start)
		#Kinematic
		kinematic_start = dt.datetime.now()
		thetae10 = get_var_hgt_lvl(np.array(sfc_thetae_unit), sfc_hgt[t], 10, terrain)
		thetae01 = get_mean_var_hgt(np.array(sfc_thetae_unit), sfc_hgt[t], 0, 1000, terrain)
		thetae03 = get_mean_var_hgt(np.array(sfc_thetae_unit), sfc_hgt[t], 0, 3000, terrain)
		F10, Fn10, Fs10, icon10, vgt10, conv10, vo10 = \
				kinematics(u10, v10, thetae10, dx, dy, y)
		F01, Fn01, Fs01, icon01, vgt01, conv01, vo01 = \
				kinematics(umean01, vmean01, thetae01, dx, dy, y)
		F03, Fn03, Fs03, icon03, vgt03, conv03, vo03 = \
				kinematics(umean03, vmean03, thetae03, dx, dy, y)
		if t == 0:
			print("KINEMATIC TIME:")
			print(dt.datetime.now() - kinematic_start)
		#Composites
		Rq = qmean01 / 12.
		windex = 5. * np.power( (melting_hgt/1000.) * Rq * (np.power( lr_freezing,2) - 30. + \
				qmean01 - 2. * q_melting), 0.5)
		gustex = (0.5 * windex) + (0.5 * U500)
		gustex2 = (0.5 * windex) + (0.5 * Umean06)
		gustex3 = (((0.5 * windex) + (0.5 * Umean06)) / 30.) * \
				((ml_cape * np.power((s06), 1.67)) / 20000.)
		#hmi = lr850_670 + dpd850 - dpd670
		#hmi[hmi<0] = 0
		#wmsi_ml
		#dmi
		#mwpi_ml
		#stp_fixed_left
		#stp_fixed
		#moshe
		#ship
		#stp_cin_left
		#stp_cin
		#scp
		#eff_sherb
		#k_index
		#mlcape*s06
		#mlcape*s062
		#mucape*s06
		#mucape*s062
		#sbcape*s06
		#sbcape*s062
		#effcape*s06
		#effcape*s062
		#cape*s06
		#sherb
		#mosh
		#wndg
		#sweat
		#esp
		#mmp
		

		#Fill output
		output_data[np.where(param=="ml_cape")[0][0], t] = ml_cape
		output_data[np.where(param=="mu_cape")[0][0], t] = mu_cape
		output_data[np.where(param=="eff_cape")[0][0], t] = eff_cape
		output_data[np.where(param=="sb_cape")[0][0], t] = sb_cape
		output_data[np.where(param=="ml_cin")[0][0], t] = ml_cin
		output_data[np.where(param=="mu_cin")[0][0], t] = mu_cin
		output_data[np.where(param=="eff_cin")[0][0], t] = eff_cin
		output_data[np.where(param=="sb_cin")[0][0], t] = sb_cin
		output_data[np.where(param=="ml_lcl")[0][0], t] = ml_lcl
		output_data[np.where(param=="mu_lcl")[0][0], t] = mu_lcl
		output_data[np.where(param=="eff_lcl")[0][0], t] = eff_lcl
		output_data[np.where(param=="sb_lcl")[0][0], t] = sb_lcl
		output_data[np.where(param=="ml_lfc")[0][0], t] = ml_lfc
		output_data[np.where(param=="mu_lfc")[0][0], t] = mu_lfc
		output_data[np.where(param=="eff_lfc")[0][0], t] = eff_lfc
		output_data[np.where(param=="sb_lfc")[0][0], t] = sb_lfc
		output_data[np.where(param=="ml_el")[0][0], t] = ml_el
		output_data[np.where(param=="mu_el")[0][0], t] = mu_el
		output_data[np.where(param=="eff_el")[0][0], t] = eff_el
		output_data[np.where(param=="sb_el")[0][0], t] = sb_el
		if model == "erai":
			output_data[np.where(param=="cape")[0][0], t] = mod_cape[t]
			output_data[np.where(param=="cp")[0][0], t] = cp[t]

		output_data[np.where(param=="lr01")[0][0], t] = lr01
		output_data[np.where(param=="lr03")[0][0], t] = lr03
		output_data[np.where(param=="lr13")[0][0], t] = lr13
		output_data[np.where(param=="lr24")[0][0], t] = lr24
		output_data[np.where(param=="lr36")[0][0], t] = lr36
		output_data[np.where(param=="lr_freezing")[0][0], t] = lr_freezing
		output_data[np.where(param=="mhgt")[0][0], t] = melting_hgt
		output_data[np.where(param=="wbz")[0][0], t] = hwb0
		output_data[np.where(param=="qmean01")[0][0], t] = qmean01
		output_data[np.where(param=="qmean03")[0][0], t] = qmean03
		output_data[np.where(param=="qmean06")[0][0], t] = qmean06
		output_data[np.where(param=="qmean13")[0][0], t] = qmean13
		output_data[np.where(param=="qmean36")[0][0], t] = qmean36
		output_data[np.where(param=="qmeansubcloud")[0][0], t] = qmeansubcloud
		output_data[np.where(param=="q_melting")[0][0], t] = q_melting
		output_data[np.where(param=="q1")[0][0], t] = q1
		output_data[np.where(param=="q3")[0][0], t] = q3
		output_data[np.where(param=="q6")[0][0], t] = q6
		output_data[np.where(param=="rhmean01")[0][0], t] = rhmean01
		output_data[np.where(param=="rhmean03")[0][0], t] = rhmean03
		output_data[np.where(param=="rhmean06")[0][0], t] = rhmean06
		output_data[np.where(param=="rhmean13")[0][0], t] = rhmean13
		output_data[np.where(param=="rhmean36")[0][0], t] = rhmean36
		output_data[np.where(param=="rhmeansubcloud")[0][0], t] = rhmeansubcloud
		output_data[np.where(param=="rhmin01")[0][0], t] = rhmin01
		output_data[np.where(param=="rhmin03")[0][0], t] = rhmin03
		output_data[np.where(param=="rhmin06")[0][0], t] = rhmin06
		output_data[np.where(param=="rhmin13")[0][0], t] = rhmin13
		output_data[np.where(param=="rhmin36")[0][0], t] = rhmin36
		output_data[np.where(param=="rhminsubcloud")[0][0], t] = rhminsubcloud
		output_data[np.where(param=="v_totals")[0][0], t] = v_totals
		output_data[np.where(param=="c_totals")[0][0], t] = c_totals
		output_data[np.where(param=="t_totals")[0][0], t] = t_totals
		output_data[np.where(param=="pwat")[0][0], t] = pwat
		output_data[np.where(param=="maxtevv")[0][0], t] = maxtevv
		output_data[np.where(param=="te_diff")[0][0], t] = te_diff
		output_data[np.where(param=="tei")[0][0], t] = tei
		output_data[np.where(param=="dpd700")[0][0], t] = dpd700
		output_data[np.where(param=="dpd850")[0][0], t] = dpd850

		output_data[np.where(param=="Umeanwindinf")[0][0], t] = Umeanwindinf
		output_data[np.where(param=="Umean01")[0][0], t] = Umean01
		output_data[np.where(param=="Umean03")[0][0], t] = Umean03
		output_data[np.where(param=="Umean06")[0][0], t] = Umean06
		output_data[np.where(param=="Umean800_600")[0][0], t] = Umean800_600
		output_data[np.where(param=="Uwindinf")[0][0], t] = Uwindinf
		output_data[np.where(param=="U500")[0][0], t] = U500
		output_data[np.where(param=="U1")[0][0], t] = U1
		output_data[np.where(param=="U3")[0][0], t] = U3
		output_data[np.where(param=="U6")[0][0], t] = U6
		output_data[np.where(param=="Ust")[0][0], t] = Ust_right
		output_data[np.where(param=="Ust_left")[0][0], t] = Ust_left
		output_data[np.where(param=="Usr01")[0][0], t] = Usr01_right
		output_data[np.where(param=="Usr01_left")[0][0], t] = Usr01_left
		output_data[np.where(param=="Usr03")[0][0], t] = Usr03_right
		output_data[np.where(param=="Usr03_left")[0][0], t] = Usr03_left
		output_data[np.where(param=="Usr06")[0][0], t] = Usr06_right
		output_data[np.where(param=="Usr06_left")[0][0], t] = Usr06_left
		output_data[np.where(param=="wg10")[0][0], t] = wg10[t]
		output_data[np.where(param=="U10")[0][0], t] = U10
		output_data[np.where(param=="scld")[0][0], t] = scld
		output_data[np.where(param=="s01")[0][0], t] = s01
		output_data[np.where(param=="s03")[0][0], t] = s03
		output_data[np.where(param=="s06")[0][0], t] = s06
		output_data[np.where(param=="s010")[0][0], t] = s010
		output_data[np.where(param=="s13")[0][0], t] = s13
		output_data[np.where(param=="s36")[0][0], t] = s36
		output_data[np.where(param=="ebwd")[0][0], t] = ebwd
		output_data[np.where(param=="srh01")[0][0], t] = srh01_right
		output_data[np.where(param=="srh01_left")[0][0], t] = srh01_left
		output_data[np.where(param=="srh03")[0][0], t] = srh03_right
		output_data[np.where(param=="srh03_left")[0][0], t] = srh03_left
		output_data[np.where(param=="srh06")[0][0], t] = srh06_right
		output_data[np.where(param=="srh06_left")[0][0], t] = srh06_left
		output_data[np.where(param=="srhe")[0][0], t] = srhe_right
		output_data[np.where(param=="srhe_left")[0][0], t] = srhe_left
		output_data[np.where(param=="omega01")[0][0], t] = omega01
		output_data[np.where(param=="omega03")[0][0], t] = omega03
		output_data[np.where(param=="omega06")[0][0], t] = omega06

		output_data[np.where(param=="F10")[0][0], t] = F10
		output_data[np.where(param=="Fn10")[0][0], t] = Fn10
		output_data[np.where(param=="Fs10")[0][0], t] = Fs10
		output_data[np.where(param=="icon10")[0][0], t] = icon10
		output_data[np.where(param=="vgt10")[0][0], t] = vgt10
		output_data[np.where(param=="conv10")[0][0], t] = conv10
		output_data[np.where(param=="vo10")[0][0], t] = vo10
		output_data[np.where(param=="F01")[0][0], t] = F01
		output_data[np.where(param=="Fn01")[0][0], t] = Fn01
		output_data[np.where(param=="Fs01")[0][0], t] = Fs01
		output_data[np.where(param=="icon01")[0][0], t] = icon01
		output_data[np.where(param=="vgt01")[0][0], t] = vgt01
		output_data[np.where(param=="conv01")[0][0], t] = conv01
		output_data[np.where(param=="vo01")[0][0], t] = vo01
		output_data[np.where(param=="F03")[0][0], t] = F03
		output_data[np.where(param=="Fn03")[0][0], t] = Fn03
		output_data[np.where(param=="Fs03")[0][0], t] = Fs03
		output_data[np.where(param=="icon03")[0][0], t] = icon03
		output_data[np.where(param=="vgt03")[0][0], t] = vgt03
		output_data[np.where(param=="conv03")[0][0], t] = conv03
		output_data[np.where(param=="vo03")[0][0], t] = vo03

		if t == 0:
			print("TOTAL TIME:")
			print(dt.datetime.now() - tot_start)

#		#Get other parameters...
#		if "relhum850-500" in param:
#			param_ind = np.where(param=="relhum850-500")[0][0]
#			param_out[param_ind][t,:,:] = get_mean_var_p(hur_unit,p,850,500)
#		if "relhum1000-700" in param:
#			param_ind = np.where(param=="relhum1000-700")[0][0]
#			param_out[param_ind][t,:,:] = get_mean_var_p(hur_unit,p,1000,700)
#		if "mu_cape" in param:
#		#CAPE for most unstable parcel
#		#cape.data has cape values for each pressure level, as if they were each parcels.
#		# Taking the max gives MUCAPE approximation
#			param_ind = np.where(param=="mu_cape")[0][0]
#			param_out[param_ind][t,:,:] = mu_cape
#		if "ml_cape" in param:
#		#CAPE for mixed layer
#			param_ind = np.where(param=="ml_cape")[0][0]
#			param_out[param_ind][t,:,:] = ml_cape
#		if "s06" in param:
#		#Wind shear 10 m (sfc) to 6 km
#			param_ind = np.where(param=="s06")[0][0]
#			s06 = get_shear_hgt(ua[t],va[t],hgt[t],0,6000,\
#				uas[t],vas[t])
#			param_out[param_ind][t,:,:] = s06
#		if "s03" in param:
#		#Wind shear 10 m (sfc) to 3 km
#			param_ind = np.where(param=="s03")[0][0]
#			s03 = get_shear_hgt(ua[t],va[t],hgt[t],0,3000,\
#				uas[t],vas[t])
#			param_out[param_ind][t,:,:] = s03
#		if "s01" in param:
#		#Wind shear 10 m (sfc) to 1 km
#			param_ind = np.where(param=="s01")[0][0]
#			s01 = get_shear_hgt(ua[t],va[t],hgt[t],0,1000,\
#				uas[t],vas[t])
#			param_out[param_ind][t,:,:] = s01
#		if "s0500" in param:
#		#Wind shear sfc to 500 m
#			param_ind = np.where(param=="s0500")[0][0]
#			param_out[param_ind][t,:,:] = get_shear_hgt(ua[t],va[t],hgt[t],0,500,\
#				uas[t],vas[t])
#		if "lr1000" in param:
#		#Lapse rate bottom pressure level to 1000 m
#			param_ind = np.where(param=="lr1000")[0][0]
#			lr1000 = get_lr_hgt(ta[t],hgt[t],0,1000)
#			param_out[param_ind][t,:,:] = lr1000
#		if "mu_cin" in param:
#		#CIN for same parcel used for mu_cape
#			param_ind = np.where(param=="mu_cin")[0][0]
#			param_out[param_ind][t,:,:] = mu_cin
#		if "ml_lfc" in param:
#		#LCL for same parcel used for mu_cape
#			param_ind = np.where(param=="ml_lfc")[0][0]
#			temp_ml_lfc = np.copy(ml_lfc)
#			temp_ml_lfc[temp_ml_lfc<0] = np.nan
#			param_out[param_ind][t,:,:] = temp_ml_lfc
#		if "ml_el" in param:
#		#LCL for same parcel used for mu_cape
#			param_ind = np.where(param=="ml_el")[0][0]
#			temp_ml_el = np.copy(ml_el)
#			temp_ml_el[temp_ml_el<0] = np.nan
#			param_out[param_ind][t,:,:] = temp_ml_el
#		if "ml_lcl" in param:
#		#LCL for same parcel used for mu_cape
#			param_ind = np.where(param=="ml_lcl")[0][0]
#			temp_ml_lcl = np.copy(ml_lcl)
#			temp_ml_lcl[temp_ml_lcl<0] = np.nan
#			param_out[param_ind][t,:,:] = temp_ml_lcl
#		if "lfc" in param:
#		#LCL for same parcel used for mu_cape
#			param_ind = np.where(param=="lfc")[0][0]
#			temp_lfc = np.copy(lfc)
#			temp_lfc[temp_lfc<=0] = np.nan
#			param_out[param_ind][t,:,:] = temp_lfc
#		if "lcl" in param:
#		#LCL for same parcel used for mu_cape
#			param_ind = np.where(param=="lcl")[0][0]
#			temp_lcl = np.copy(lcl)
#			temp_lcl[temp_lcl<=0] = np.nan
#			param_out[param_ind][t,:,:] = temp_lcl
#		if "ml_cin" in param:
#		#CIN for same parcel used for ml_cape
#			param_ind = np.where(param=="ml_cin")[0][0]
#			param_out[param_ind][t,:,:] = ml_cin
#		if "srh01" in param:
#		#Combined (+ve and -ve) rel. helicity from 0-1 km
#			param_ind = np.where(param=="srh01")[0][0]
#			srh01 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,1000,True,850,700,p)
#			param_out[param_ind][t,:,:] = srh01
#		if "srh03" in param:
#		#Combined (+ve and -ve) rel. helicity from 0-3 km
#			srh03 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,3000,True,850,700,p)
#			param_ind = np.where(param=="srh03")[0][0]
#			param_out[param_ind][t,:,:] = srh03
#		if "srh06" in param:
#		#Combined (+ve and -ve) rel. helicity from 0-6 km
#			param_ind = np.where(param=="srh06")[0][0]
#			srh06 = get_srh(sfc_ua[t],sfc_va[t],sfc_hgt[t]-terrain,6000,True,850,700,p)
#			param_out[param_ind][t,:,:] = srh06
#		if "ship" in param:
#		#Significant hail parameter
#			if "s06" not in param:
#				raise NameError("To calculate ship, s06 must be included")
#			param_ind = np.where(param=="ship")[0][0]
#			muq = mu_cape_inds.choose(q)
#			ship = get_ship(mu_cape,muq,ta[t],ua[t],va[t],hgt[t],p,s06)
#			param_out[param_ind][t,:,:] = ship
#		if "mmp" in param:
#		#Mesoscale Convective System Maintanance Probability
#			param_ind = np.where(param=="mmp")[0][0]
#			param_out[param_ind][t,:,:] = get_mmp(ua[t],va[t],uas[t],vas[t],\
#				mu_cape,ta[t],hgt[t])
#		if "scp" in param:
#		#Supercell composite parameter (EWD)
#			if "srh03" not in param:
#				raise NameError("To calculate ship, srh03 must be included")
#			param_ind = np.where(param=="scp")[0][0]
#			scell_pot = get_supercell_pot(mu_cape,ua[t],va[t],hgt[t],ta_unit,p_unit,\
#					q_unit,srh03)
#			param_out[param_ind][t,:,:] = scell_pot
#			#if t == 0:
#				#print("SCP: "+str(dt.datetime.now()-start))
#		if "stp" in param:
#		#Significant tornado parameter
#		#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
#		# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
#		# lcl.
#			if "srh01" not in param:
#				raise NameError("To calculate stp, srh01 must be included")
#			param_ind = np.where(param=="stp")[0][0]
#			stp = get_tornado_pot(ml_cape,lcl,ml_cin,ua[t],va[t],p_3d[t],hgt[t],p,srh01)
#			param_out[param_ind][t,:,:] = stp
#		if "vo10" in param:
#		#10 m relative vorticity
#			param_ind = np.where(param=="vo10")[0][0]
#			x,y = np.meshgrid(lon,lat)
#			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
#			vo10 = get_vo(uas[t],vas[t],dx,dy)
#			param_out[param_ind][t,:,:] = vo10
#		if "conv10" in param:
#			param_ind = np.where(param=="conv10")[0][0]
#			x,y = np.meshgrid(lon,lat)
#			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
#			param_out[param_ind][t,:,:] = get_conv(uas[t],vas[t],dx,dy)
#		if "conv1000-850" in param:
#			levs = np.where((p<=1001)&(p>=849))[0]
#			param_ind = np.where(param=="conv1000-850")[0][0]
#			x,y = np.meshgrid(lon,lat)
#			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
#			param_out[param_ind][t,:,:] = \
#				np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
#		if "conv800-600" in param:
#			levs = np.where((p<=801)&(p>=599))[0]
#			param_ind = np.where(param=="conv800-600")[0][0]
#			x,y = np.meshgrid(lon,lat)
#			dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
#			param_out[param_ind][t,:,:] = \
#				np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
#		if "non_sc_stp" in param:
#		#Non-Supercell Significant tornado parameter
#		#NOTE: LCL here is for "maximum" parcel. I.e., the parcel with heighest equivalent
#		# potential temperature in the lowest 3000 m. STP however, calls for mixed layer
#		# lcl.
#			if "vo10" not in param:
#				raise NameError("To calculate non_sc_stp, vo must be included")
#			if "lr1000" not in param:
#				raise NameError("To calculate non_sc_stp, lr1000 must be included")
#			param_ind = np.where(param=="non_sc_stp")[0][0]
#			non_sc_stp = get_non_sc_tornado_pot(ml_cape,ml_cin,lcl,ua[t],va[t],\
#				uas[t],vas[t],p_3d[t],ta[t],hgt[t],p,vo10,lr1000)
#			param_out[param_ind][t,:,:] = non_sc_stp
#		if "cape*s06" in param:
#			param_ind = np.where(param=="cape*s06")[0][0]
#			cs6 = ml_cape * np.power(s06,1.67)
#			param_out[param_ind][t,:,:] = cs6
#		if "td850" in param:
#			param_ind = np.where(param=="td850")[0][0]
#			td850 = get_td_diff(ta[t],dp[t],p_3d[t],850)
#			param_out[param_ind][t,:,:] = td850
#		if "td800" in param:
#			param_ind = np.where(param=="td800")[0][0]
#			param_out[param_ind][t,:,:] = get_td_diff(ta[t],dp[t],p_3d[t],800)
#		if "td950" in param:
#			param_ind = np.where(param=="td950")[0][0]
#			param_out[param_ind][t,:,:] = get_td_diff(ta[t],dp[t],p_3d[t],950)
#		if "wg" in param:
#			try:
#				param_ind = np.where(param=="wg")[0][0]
#				param_out[param_ind][t,:,:] = wg[t]
#			except ValueError:
#				print("wg field expected, but not parsed")
#		if "dcape" in param:
#			param_ind = np.where(param=="dcape")[0][0]
#			dcape = np.nanmax(get_dcape(np.array(p_3d[t]),ta[t],hgt[t],np.array(p),ps[t]),axis=0)
#			param_out[param_ind][t,:,:] = dcape
#		if "mlm" in param:
#			param_ind = np.where(param=="mlm")[0][0]
#			mlm_u, mlm_v = get_mean_wind(ua[t],va[t],hgt[t],800,600,False,None,"plevels",p)
#			mlm = np.sqrt(np.square(mlm_u) + np.square(mlm_v))
#			param_out[param_ind][t,:,:] = mlm
#		if "dlm" in param:
#			param_ind = np.where(param=="dlm")[0][0]
#			dlm_u, dlm_v = get_mean_wind(ua[t],va[t],hgt[t],1000,500,False,None,"plevels",p)
#			dlm = np.sqrt(np.square(dlm_u) + np.square(dlm_v))
#			param_out[param_ind][t,:,:] = dlm
#		if "dlm+dcape" in param:
#			param_ind = np.where(param=="dlm+dcape")[0][0]
#			dlm_dcape = dlm + np.sqrt(2*dcape)
#			param_out[param_ind][t,:,:] = dlm_dcape
#		if "mlm+dcape" in param:
#			param_ind = np.where(param=="mlm+dcape")[0][0]
#			mlm_dcape = mlm + np.sqrt(2*dcape)
#			param_out[param_ind][t,:,:] = mlm_dcape
#		if "dcape*cs6" in param:
#			param_ind = np.where(param=="dcape*cs6")[0][0]
#			param_out[param_ind][t,:,:] = (dcape/980.) * (cs6/20000)
#		if "dlm*dcape*cs6" in param:
#			param_ind = np.where(param=="dlm*dcape*cs6")[0][0]
#			param_out[param_ind][t,:,:] = (dlm_dcape/30.) * (cs6/20000)
#		if "mlm*dcape*cs6" in param:
#			param_ind = np.where(param=="mlm*dcape*cs6")[0][0]
#			param_out[param_ind][t,:,:] = (mlm_dcape/30.) * (cs6/20000)
#		if "dcp" in param:
#		#Derecho Composite Parameter ~ cold pool driven wind events
#			param_ind = np.where(param=="dcp")[0][0]
#			param_out[param_ind][t,:,:] = (dcape/980)*(mu_cape/2000)*(s06/10)*(dlm/8)

	if issave:
		save_netcdf(region,model,out_name,date_list,lat,lon,param,output_data)
	


