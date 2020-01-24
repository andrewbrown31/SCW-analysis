import argparse
from SkewT import get_dcape
import gc
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
	from mpi4py import MPI
except:
	pass
import wrf
from calc_param import save_netcdf, get_dp
import xarray as xr
from erai_read import read_erai
from era5_read import read_era5
from erai_read import get_mask as  get_erai_mask
from barra_read import read_barra, read_barra_fc
from barra_ad_read import read_barra_ad
from barra_read import get_mask as  get_barra_mask
from read_cmip import read_cmip5

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


def fill_output(output, t, param, ps_chunk, p, data):

	output[t*ps_chunk.shape[1]:(t+1)*ps_chunk.shape[1], np.where(param==p)[0][0]] = \
		data.reshape(ps_chunk.shape[1])
	return output

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

def get_eff_cape(cape, cin, sfc_p_3d, sfc_ta, sfc_hgt, sfc_q, ps, terrain):

	#Define the effective layer cape condition for the 3d grid
	cape_cond = (cape >= 100) & (cin <= 250) & (sfc_p_3d <= ps)
	eff_cape_cond = np.zeros(cape_cond.shape, dtype=bool)
	is_first = np.ones((cape_cond.shape[1], cape_cond.shape[2]), dtype=bool)
	eff_cape_cond[0] = cape_cond[0]
	for i in np.arange(1,cape_cond.shape[0]):
		eff_cape_cond[i] = cape_cond[i]
		is_first[is_first & (~cape_cond[i] & cape_cond[i-1])]=False
		eff_cape_cond[i, ~is_first] = False

	#Extract pressure and height for effective levels
	eff_p = np.where(eff_cape_cond,\
		sfc_p_3d, np.nan)
	eff_hgt = np.where(eff_cape_cond,\
		sfc_hgt, np.nan)

	#Define "average" conditions over the effective layer. For air temp and water vapour,
	# the pressure-weighted average is used. For height and pressure, use the halfway point. 
	#If the layer is of one-level depth, use that layer's conditions
	eff_avg_p = (np.nanmin(eff_p,axis=0) + np.nanmax(eff_p,axis=0)) / 2
	eff_avg_hgt = (np.nanmin(eff_hgt,axis=0) + np.nanmax(eff_hgt,axis=0)) / 2
	eff_avg_ta = trapz_int3d(sfc_ta, sfc_p_3d, eff_cape_cond)
	eff_avg_q = trapz_int3d(sfc_q, sfc_p_3d, eff_cape_cond)
	
	#If no effective layer is defined at a lat/lon point, assign the surface conditions as 
	# effective conditions, for use in the CAPE routine (these points are not used later anyway)
	eff_avg_p = np.where(np.isnan(eff_avg_p),\
		np.ma.masked_where(~((sfc_p_3d==ps)),\
		sfc_p_3d).max(axis=0).filled(0)\
		,eff_avg_p).astype(np.float32)
	eff_avg_hgt = np.where(np.isnan(eff_avg_p),\
		np.ma.masked_where(~((sfc_p_3d==ps)),\
		sfc_hgt).max(axis=0).filled(0)\
		,eff_avg_hgt).astype(np.float32)
	eff_avg_ta = np.where(np.isnan(eff_avg_p),\
		np.ma.masked_where(~((sfc_p_3d==ps)),\
		sfc_ta).max(axis=0).filled(0)\
		,eff_avg_ta).astype(np.float32)
	eff_avg_q = np.where(np.isnan(eff_avg_p),\
		np.ma.masked_where(~((sfc_p_3d==ps)),\
		sfc_q).max(axis=0).filled(0)\
		,eff_avg_q).astype(np.float32)

	#Insert the effective layer conditions into the bottom of the 3d arrays pressure-level arrays
	eff_ta_arr = np.insert(sfc_ta,0,eff_avg_ta,axis=0)
	eff_q_arr = np.insert(sfc_q,0,eff_avg_q,axis=0)
	eff_hgt_arr = np.insert(sfc_hgt,0,eff_avg_hgt,axis=0)
	eff_p3d_arr = np.insert(sfc_p_3d,0,eff_avg_p,axis=0)

	#Sort arrays by ascending pressure
	a,temp1,temp2 = np.meshgrid(np.arange(eff_p3d_arr.shape[0]) ,\
		 np.arange(eff_p3d_arr.shape[1]), np.arange(eff_p3d_arr.shape[2]))
	sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),eff_p3d_arr],axis=0))
	eff_ta_arr = np.take_along_axis(eff_ta_arr, sort_inds, axis=0)
	eff_p3d_arr = np.take_along_axis(eff_p3d_arr, sort_inds, axis=0)
	eff_hgt_arr = np.take_along_axis(eff_hgt_arr, sort_inds, axis=0)
	eff_q_arr = np.take_along_axis(eff_q_arr, sort_inds, axis=0)

	#Calculate CAPE using wrf-python. 
	cape3d_effavg = wrf.cape_3d(eff_p3d_arr,eff_ta_arr + 273.15,\
		eff_q_arr,eff_hgt_arr,terrain,ps,False,meta=False, missing=0)

	#From the 3d CAPE array, return just the effective layer vaues
	eff_cape = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[0]).max(axis=0).filled(0)
	eff_cin = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[1]).max(axis=0).filled(0)
	eff_lfc = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[2]).max(axis=0).filled(0)
	eff_lcl = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[3]).max(axis=0).filled(0)
	eff_el = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[4]).max(axis=0).filled(0)

	#Finally, mask points where there is no effective layer
	eff_cape[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_cin[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lfc[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lcl[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_el[eff_cape_cond.sum(axis=0) == 0] = 0

	return eff_cape, eff_cin, eff_lfc, eff_lcl, eff_el, eff_hgt, eff_avg_hgt

def get_pwat(q, p, sfcp3d):

	#p_ind = np.argmin(abs(p - 400)) + 1
	#pwat = (((q[:p_ind]+q[1:p_ind+1])*1000/2 * (sfcp3d[:p_ind]-sfcp3d[1:p_ind+1])) * 0.00040173)\
	#	.sum(axis=0)	#From SHARPpy
	sfcp3d[sfcp3d < 400] = 0
	pwat = (((q[:p_ind]+q[1:p_ind+1])*1000/2 * (sfcp3d[:p_ind]-sfcp3d[1:p_ind+1])) * 0.00040173)\
		.sum(axis=0)	#From SHARPpy
	return pwat


def get_min_var_hgt(var3d, hgt, hgt_bot, hgt_top, terrain):

	hgt = hgt - terrain
	var3d[hgt < hgt_bot] = np.nan
	var3d[hgt > hgt_top] = np.nan
	result = np.nanmin(var3d, axis=0)
	return result

def trapz_int3d(var3d, p3d, cond):

	#Vertical intergration using the trapezoidal rule for finite integration. Scaled by the total difference in 
	# pressure between the top and bottom layers, such that a mass-weighted mean is the output.
	
	#Cond is a boolean array which gives the layers of interest

	#If cond is false for the whole first dimension, then NaN is returned at that point

	#If there is only one level of interest, then var3d at that level is returned

	#See Dean S. documentation for more info.

	x,j,k = var3d.shape
	result = np.zeros((j,k))
	for i in np.arange(x-1):
		p_layer = p3d[i+1] - p3d[i]
		v_layer = var3d[i+1] + var3d[i]
		layer = np.where( (cond[i+1] & cond[i]), v_layer*p_layer, 0)
		result = result + layer
	p_masked = np.ma.masked_where(~cond, p3d)
	var3d_masked = np.ma.masked_where(~cond, var3d)
	ptop = p_masked.min(axis=0)
	pbot = p_masked.max(axis=0)
	return np.where( (cond.sum(axis=0) == 1), var3d_masked.max(axis=0),\
		    np.ma.filled(( 1 / (2 * (ptop - pbot) ) ) * result, np.nan) )

def get_mean_var_hgt(var3d, hgt, hgt_bot, hgt_top, terrain, mass_weighted=False, p3d=None):

	if mass_weighted:
		try:
			cond = ( (hgt-terrain) < hgt_bot) | ( (hgt-terrain) > hgt_top) | (np.isnan(hgt_bot)) | (np.isnan(hgt_top))
			result = trapz_int3d( var3d, p3d, ~cond)
		except:
			raise ValueError("FUNCTION get_mean_var_hgt() IS FAILING TO TAKE A PRESSURE WEIGHTED"+\
				" AVERAGE. HAS A 3D PRESSURE FIELD BEEN PARSED?")
	else:
		hgt = hgt - terrain
		var3d_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) , var3d)
		result = np.ma.mean(var3d_ma, axis=0)
	return result

def get_mean_var_p(var3d, p3d, p_bot, p_top, ps, mass_weighted=False):

	if mass_weighted:
		cond = ( p3d > p_bot) | ( p3d < p_top) | (p3d > ps)
		result = trapz_int3d( var3d, p3d, ~cond)
	else:
		var3d_ma = np.ma.masked_where((p3d > p_bot) | (p3d < p_top) | (p3d > ps), var3d)
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
	#Non-pressure weighted mean
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

def get_tornado_pot(mlcin, mllcl, sblcl, bwd6, ebwd, sbcape, mlcape, srh, esrh):
	#From SHARPpy

	mlcin = -mlcin

	mllcl_term = ( (2000. - mllcl) / 1000.) 
	mllcl_term[mllcl<1000] = 1
	mllcl_term[mllcl>1000] = 0

	sblcl_term = ( (2000. - sblcl) / 1000.) 
	sblcl_term[sblcl<1000] = 1
	sblcl_term[sblcl>1000] = 0

	bwd6[bwd6 > 30] = 30
	bwd6[bwd6 < 12.5] = 12.5
	bwd6_term = bwd6 / 20.

	ebwd_term = ( ebwd / 20.) 
	ebwd_term[ebwd<12.5] = 0
	ebwd_term[ebwd>30] = 1.5

	mlcin_term = ( mlcin + 200 ) / 150. 
	mlcin_term[mlcin>-50] = 1
	mlcin_term[mlcin<-200] = 0

	sbcape_term = sbcape / 1500.
	mlcape_term = mlcape / 1500.
	srh_term = srh / 150.
	esrh_term = esrh / 150.

	stp_fixed = sbcape_term * sblcl_term * srh_term * bwd6_term
	stp_cin = mlcape_term * mllcl_term * esrh_term * ebwd_term * mlcin_term
	stp_cin[stp_cin < 0] = 0

	return [stp_fixed, stp_cin]
   
def get_mburst(sb_cape, lr03, vt, dcape, pwat, tei, thetae, hgt, terrain):

	#SPC definition except; Lifted index term is set to zero and CAPE thresholds are lowered by 1000

	sfc_te = get_var_hgt_lvl(thetae, hgt, 0, terrain)
	sfc_te_term = np.where(sfc_te >= 355, 1, 0)

	sb_term = np.zeros(sb_cape.shape)
	sb_term[sb_cape < 1000] = -5
	sb_term[sb_cape >= 1000] = 0
	sb_term[sb_cape >= 2300] = 1
	sb_term[sb_cape >= 2700] = 2
	sb_term[sb_cape >= 3300] = 4

	pwat_term = np.where(pwat < 1.5, -3, 0)

	dcape_term = np.zeros(dcape.shape)
	dcape_term[(pwat > 1.7) & (dcape > 900)] = 1

	lr03_term = np.where(lr03 <= 8.4, 0, 1)

	vt_term = np.zeros(vt.shape)
	vt_term[(vt >= 27) & (vt < 28)] = 1
	vt_term[(vt >= 28) & (vt < 29)] = 2
	vt_term[(vt >= 29)] = 3

	tei_term = np.where(tei >= 35, 1, 0)

	return sfc_te_term + sb_term + pwat_term + dcape_term + lr03_term + vt_term + tei_term

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

def get_sherb(s03, ebwd, lr03, lr700_500):

	return [ (s03 / 27.) * (lr03 / 5.2) * (lr700_500 / 5.6) ,\
			(ebwd / 27.) * (lr03 / 5.2) * (lr700_500 / 5.6) ]

def get_wndg(ml_cape, ml_cin, lr03, u, v, hgt, terrain, p3d):

	ml_cin = -ml_cin
	umean = get_mean_var_hgt(u, hgt, 1000, 3500, terrain, True, p3d)
	vmean = get_mean_var_hgt(v, hgt, 1000, 3500, terrain, True, p3d)
	mean_wind = np.sqrt( umean**2 + vmean**2)

	lr03[lr03 < 7] = 0
	ml_cin[ml_cin < -50] = -50

	return (ml_cape / 2000.) * (lr03 / 9.) * (mean_wind / 15.) * ((50. + ml_cin)/40.)

def get_esp(ml_cape, lr03):

	esp = (ml_cape / 50.) * ((lr03 - 7) / 1.0)
	esp[lr03 < 7] = 0
	esp[ml_cape < 250] = 0

	return esp

def get_sweat(sfc_p3d, dp, t_totals, u, v):

	td850 = get_var_p_lvl(np.copy(dp), sfc_p3d, 850)
	u850 = get_var_p_lvl(np.copy(u), sfc_p3d, 850)
	v850 = get_var_p_lvl(np.copy(v), sfc_p3d, 850)
	u500 = get_var_p_lvl(np.copy(u), sfc_p3d, 500)
	v500 = get_var_p_lvl(np.copy(v), sfc_p3d, 500)
	U850 = np.sqrt( u850**2 + v850**2)
	U500 = np.sqrt( u500**2 + v500**2)
	dir850 = np.rad2deg(np.arctan2(v850, u850))
	dir500 = np.rad2deg(np.arctan2(v850, u850))

	td850[td850<0] = 0
	td850 = td850 * 12.

	term1 = td850
	term2 = np.copy(t_totals)
	term2 = 20 * (term2 - 49)
	term2[t_totals<49] = 0
	term3 = 2.*U850
	term4 = U500
	term5 = 125.*(np.sin(np.deg2rad(dir500-dir850)) + 0.2)

	term1[term1<0] = 0
	term2[term2<0] = 0
	term3[term3<0] = 0
	term4[term4<0] = 0
	term5[term5<0] = 0

	term5[(dir850 >= 130) & (dir850 <= 250)] = 0
	term5[(dir500 >= 210) & (dir500 <= 310)] = 0
	term5[(dir500 - dir850) > 0] = 0
	term5[(U850 >= 7.71) & (U500 >= 7.71)] = 0

	return term1 + term2 + term3 + term4 + term5

	
	
def get_supercell_pot(mucape,srhe,srh01,ebwd,s06):
	#From EWD. MUCAPE approximated by treating each vertical grid point as a parcel, 
	# finding the CAPE of each parcel, and taking the maximum CAPE

	ebwd[ ebwd > 20 ] = 20.
	ebwd[ ebwd < 10 ] = 0.
	s06[ s06 > 20 ] = 20.
	s06[ s06 < 10 ] = 0.
     
	mucape_term = mucape / 1000.
	srhe_term = srhe / 50.
	ebwd_term = ebwd / 20.
	srh01_term = srh01 / 50.
	s06_term = s06 / 20.

	scp = mucape_term * srhe_term * ebwd_term
	scp_fixed = mucape_term * srh01_term * s06_term
	return scp, scp_fixed
	
def get_lr_p(t,p3d,hgt,p_bot,p_top):
	#Get lapse rate (C/km) between two pressure levels
	#No interpolation is done, so p_bot and p_top (hPa) should correspond to 
	#reanalysis pressure levels

	hgt_pbot = get_var_p_lvl(hgt, p3d, p_bot) / 1000
	hgt_ptop = get_var_p_lvl(hgt, p3d, p_top) / 1000
	t_pbot = get_var_p_lvl(t, p3d, p_bot)
	t_ptop = get_var_p_lvl(t, p3d, p_top)
	
	return np.squeeze(- (t_ptop - t_pbot) / (hgt_ptop - hgt_pbot))

def get_var_p_lvl(var, p3d, desired_p):
	#Interpolate 3d varibale ("var") to a desired pres sfc ("desired_p") 
	interp_var = wrf.interplevel(var,p3d,desired_p,meta=False)
	interp_var[p3d[0] <= desired_p] = var[0,p3d[0] <= desired_p]
	interp_var[(np.where(p3d==desired_p))[1],(np.where(p3d==desired_p))[2]] = var[p3d==desired_p]
	return interp_var

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

	if var.ndim == 1:
		var_hgt = np.interp(var_value,np.flipud(var),np.flipud(hgt))
	else:
		var_hgt = np.array(wrf.interplevel(hgt, var, var_value))

	return var_hgt

def get_ship(mucape,muq,s06,lr75,h5_temp,frz_lvl):
	#From EWD (no freezing level involved), but using SPC intended values:
	# https://github.com/sharppy/SHARPpy/blob/master/sharppy/sharptab/params.py
	
	#Restrict extreme values
	s06[s06>27] = 27
	s06[s06<7] = 7
	muq[muq>13.6] = 13.6
	muq[muq<11] = 11
	h5_temp[h5_temp>-5.5] = -5.5

	#Calculate ship
	ship = (-1*(mucape * muq * lr75 * h5_temp * s06) / 42000000)

	#Scaling
	ship[mucape<1300] = ship[mucape<1300]*(mucape[mucape<1300]/1300)
	ship[lr75<5.8] = ship[lr75<5.8]*(lr75[lr75<5.8]/5.8)
	ship[frz_lvl<2400] = ship[frz_lvl<2400]*(frz_lvl[frz_lvl<2400]/2400)

	return ship

def get_mmp(u,v,mu_cape,t,hgt,terrain,p3d):
	#From SCP/SHARPpy
	#NOTE: Is costly due to looping over each layer in 0-1 km and 6-10 km, and within this 
	# loop, calling function get_shear_hgt which interpolates over lat/lon

	#Get max wind shear
	lowers = np.arange(0,1000+250,250)
	uppers = np.arange(6000,10000+1000,1000)
	no_shears = len(lowers)*len(uppers)
	shear_3d = np.empty((no_shears,u.shape[1],u.shape[2]))
	cnt=0
	for low in lowers:
		for up in uppers:
			shear_3d[cnt,:,:] = get_shear_hgt(u,v,hgt,low,up,terrain)
			cnt=cnt+1
	max_shear = np.max(shear_3d,axis=0)

	lr38 = get_lr_hgt(t,hgt,3000,8000,terrain)

	u_mean = get_mean_var_hgt(u,hgt,3000,12000,terrain,True,p3d)
	v_mean = get_mean_var_hgt(v,hgt,3000,12000,terrain,True,p3d)
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

def maxtevv_fn(te, om, hgt, terrain):

	#Calculate the max thetae x omega, as in Sherburn and Parker.

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
		temp_tevv = (temp_dte / temp_dz) * om
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

def get_uh(om, p3d, ta, q_unit, u, v, dx, dy, hgt, terrain, hgt_bot=2000, hgt_top=5000):

	#From Kain et al (2008)

	w = np.array( mpcalc.vertical_velocity( om * units.units.pascal / units.units.second,\
			p3d * units.units.hectopascal,\
			ta * units.units.degC,\
			q_unit) )
	vo3d = np.zeros(u.shape)
	for i in np.arange(vo3d.shape[0]):
		vo3d[i] = mpcalc.vorticity(u[i], v[i], dx, dy)
	levs = np.arange(hgt_bot, hgt_top+1000, 1000)
	w_comp = np.zeros( (len(levs)-1, om.shape[1], om.shape[2]) ) 
	vo_comp = np.zeros( (len(levs)-1, om.shape[1], om.shape[2]) ) 
	for l in np.arange(len(levs) - 1):
		vo_comp[l] = get_mean_var_hgt(vo3d, hgt, levs[l], levs[l+1], terrain, True, p3d)
		w_comp[l] = get_mean_var_hgt(w, hgt, levs[l], levs[l+1], terrain, True, p3d)
	uh = np.sum((vo_comp * w_comp), axis=0) * 1000  
	return uh

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

	#Get MPI communicator info
	#comm = MPI.COMM_WORLD
	#size = comm.Get_size()
	#rank = comm.Get_rank()

	#Try parsing arguments using argparse
	parser = argparse.ArgumentParser(description='wrf_parallel convective diagnostics processer')
	parser.add_argument("-m",help="Model name",required=True)
	parser.add_argument("-r",help="Region name",default="aus")
	parser.add_argument("-t1",help="Time start YYYYMMDDHH",required=True)
	parser.add_argument("-t2",help="Time end YYYYMMDDHH",required=True)
	parser.add_argument("-e", help="CMIP5 experiment name", default="")
	parser.add_argument("--ens", help="CMIP5 ensemble name", default="r1i1p1")
	parser.add_argument("--issave",help="Save output (1 or 0)", default=0)
	parser.add_argument("--outname",help="Name of saved output. In the form *outname*_*t1*_*t2*.nc",default=None)
	parser.add_argument("--is_dcape",help="Should DCAPE be calculated? (1 or 0)",default=1)
	args = parser.parse_args()

	if rank == 0:

		#Parse arguments from cmd line and set up inputs (date region model)
		model = args.m
		region = args.r
		t1 = args.t1
		t2 = args.t2
		issave = args.issave
		if args.outname==None:
			out_name = model
		is_dcape = args.is_dcape
		experiment = args.e
		ensemble = args.ens
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

		#Load data
		print("LOADING DATA ONTO ROOT NODE...")
		if model == "erai":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,\
				cp,wg10,mod_cape,lon,lat,date_list = \
				read_erai(domain,time)
			cp = cp.astype("float32", order="C")
			mod_cape = mod_cape.astype("float32", order="C")
		elif model == "era5":
			ta,temp1,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,\
				cp,wg10,mod_cape,lon,lat,date_list = \
				read_era5(domain,time)
			cp = cp.astype("float32", order="C")
			mod_cape = mod_cape.astype("float32", order="C")
			wap = np.zeros(hgt.shape)
		elif model == "barra":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
				read_barra(domain,time)
		elif model == "barra_fc":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
				read_barra_fc(domain,time)
		elif model == "barra_ad":
			wg10,temp2,ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,lon,lat,date_list = \
				read_barra_ad(domain, time, False)
		elif model in ["ACCESS1-0","ACCESS1-3"]:
			#Check that t1 and t2 are in the same year
			ta, hur, hgt, terrain, p3d, ps, ua, va, uas, vas, tas, ta2d, lon, lat, \
			    date_list = read_cmip5(model, experiment, ensemble, t1[0:4], domain)
		del temp1
		ta = ta.astype("float32", order="C")
		hur = hur.astype("float32", order="C")
		hgt = hgt.astype("float32", order="C")
		terrain = terrain.astype("float32", order="C")
		p = p.astype("float32", order="C")
		ps = ps.astype("float32", order="C")
		wap = wap.astype("float32", order="C")
		ua = ua.astype("float32", order="C")
		va = va.astype("float32", order="C")
		uas = uas.astype("float32", order="C")
		vas = vas.astype("float32", order="C")
		tas= tas.astype("float32", order="C")
		ta2d = ta2d.astype("float32", order="C")
		wg10 = wg10.astype("float32", order="C")
		lon = lon.astype("float32", order="C")
		lat = lat.astype("float32", order="C")

		gc.collect()

		param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
				"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl",\
				"lr01", "lr03", "lr13", "lr36", "lr24", "lr_freezing","lr_subcloud",\
				"qmean01", "qmean03", "qmean06", \
				"qmeansubcloud", "q_melting", "q1", "q3", "q6",\
				"rhmin01", "rhmin03", "rhmin13", \
				"rhminsubcloud", "tei", "wbz", \
				"mhgt", "mu_el", "ml_el", "sb_el", "eff_el", \
				"pwat", "v_totals", "c_totals", "t_totals", \
				"te_diff", "dpd850", "dpd700", "dcape", "ddraft_temp", "sfc_thetae", \
				\
				"srhe_left", "srh01_left", "srh03_left", "srh06_left", \
				"ebwd", "s010", "s06", "s03", "s01", "s13", "s36", "scld", \
				"U500", "U10", "U1", "U3", "U6", \
				"Ust_left", "Usr01_left",\
				"Usr03_left", "Usr06_left", \
				"Uwindinf", "Umeanwindinf", "Umean800_600", "Umean06", \
				"Umean01", "Umean03", "wg10",\
				\
				"dcp", "stp_cin_left", "stp_fixed_left",\
				"scp", "scp_fixed", "ship",\
				"mlcape*s06", "mucape*s06", "sbcape*s06", "effcape*s06", \
				"dmgwind", "dmgwind_fixed", "hmi", "wmsi_ml",\
				"dmi", "mwpi_ml", "convgust_wet", "convgust_dry", "windex",\
				"gustex", "eff_sherb", "sherb", "mmp", \
				"wndg","mburst","sweat","k_index","wmpi",\
				\
				"F10", "Fn10", "Fs10", "icon10", "vgt10", "conv10", "vo10",\
					])
		if model != "era5":
			param = np.concatenate([param, ["omega01", "omega03", "omega06", \
				"maxtevv", "mosh", "moshe"]])
		else:
			param = np.concatenate([param, ["cp"]])
		if model == "erai":
			param = np.concatenate([param, ["cape","cp","cape*s06"]])
		output_data = np.zeros((len(param),ta.shape[0],ta.shape[2],ta.shape[3]))

		#PARALLEL THNGS
		print("RESHAPING DATA...")
		orig_shape = ta.shape
		ta = ta.\
			reshape((ta.shape[0],ta.shape[1]*ta.shape[2]*ta.shape[3]))
		ua = ua.\
			reshape((ua.shape[0],ua.shape[1]*ua.shape[2]*ua.shape[3]))
		va = va.\
			reshape((va.shape[0],va.shape[1]*va.shape[2]*va.shape[3]))
		hgt = hgt.\
			reshape((hgt.shape[0],hgt.shape[1]*hgt.shape[2]*hgt.shape[3]))
		hur = hur.\
			reshape((hur.shape[0],hur.shape[1]*hur.shape[2]*hur.shape[3]))
		wap = wap.\
			reshape((wap.shape[0],wap.shape[1]*wap.shape[2]*wap.shape[3]))
		ps = ps.\
			reshape((ps.shape[0],ps.shape[1]*ps.shape[2]))
		tas = tas.\
			reshape((tas.shape[0],tas.shape[1]*tas.shape[2]))
		ta2d = ta2d.\
			reshape((ta2d.shape[0],ta2d.shape[1]*ta2d.shape[2]))
		uas = uas.\
			reshape((uas.shape[0],uas.shape[1]*uas.shape[2]))
		vas = vas.\
			reshape((vas.shape[0],vas.shape[1]*vas.shape[2]))
		wg10 = wg10.\
			reshape((wg10.shape[0],wg10.shape[1]*wg10.shape[2]))
		terrain = terrain.\
			reshape((terrain.shape[0]*terrain.shape[1]))
		if (model == "erai") | (model == "era5"):
			cp = cp.\
				reshape((cp.shape[0],cp.shape[1]*cp.shape[2]))
		if model == "erai":
			mod_cape = mod_cape.\
				reshape((mod_cape.shape[0],mod_cape.shape[1]*mod_cape.shape[2]))

		orig_length = ta.shape[0]

		#Set output array
		output_data = np.zeros((ps.shape[0]*ps.shape[1], len(param)))

		#Split/chunk the base arrays on the spatial-temporal grid point dimension, for parallel processing
		ta_split = np.array_split(ta, size, axis = 0)
		hgt_split = np.array_split(hgt, size, axis = 0)
		hur_split = np.array_split(hur, size, axis = 0)
		ua_split = np.array_split(ua, size, axis = 0)
		va_split = np.array_split(va, size, axis = 0)
		wap_split = np.array_split(wap, size, axis = 0)
		ps_split = np.array_split(ps, size, axis = 0)
		tas_split = np.array_split(tas, size, axis = 0)
		ta2d_split = np.array_split(ta2d, size, axis = 0)
		uas_split = np.array_split(uas, size, axis = 0)
		vas_split = np.array_split(vas, size, axis = 0)
		wg10_split = np.array_split(wg10, size, axis = 0)
		terrain_split = np.array_split(terrain, size, axis = 0)
		if (model == "erai") | (model == "era5"):
			cp_split = np.array_split(cp, size, axis = 0)
		if model == "erai":
			mod_cape_split = np.array_split(mod_cape, size, axis = 0)
		split_sizes = []
		for i in range(0,len(ta_split),1):
			split_sizes = np.append(split_sizes, ta_split[i].shape[0])

		#Remember the points at which splits occur on (noting that Gatherv and Scatterv act on a 
		# "C" style (row-major) flattened array). This will be different for pressure-level and sfc-level
		# variables
		split_sizes_input = split_sizes*ta.shape[1]
		displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
		split_sizes_output = split_sizes*ps.shape[1]*len(param)
		displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
		split_sizes_input_2d = split_sizes*ps.shape[1]
		displacements_input_2d = np.insert(np.cumsum(split_sizes_input_2d),0,0)[0:-1]
		print("SENDING TO OTHER NODES...")

	else:
		model = args.m
		split_sizes_input = None; displacements_input = None; split_sizes_output = None;\
			displacements_output = None; split_sizes_input_2d = None; displacements_input_2d = None;\
			split_sizes_sfc_input = None; displacements_sfc_input = None
		ta_split = None; hur_split = None; hgt_split = None; ua_split = None; p_3d_split = None;\
			va_split = None; uas_split = None; vas_split = None; lsm_split = None;\
			wg10_split = None; ps_split = None; tas_split = None; ta2d_split = None;\
			terrain_split = None; wap_split = None; sfc_ta_split = None; sfc_p_3d_split = None;\
			sfc_hgt_split = None; sfc_dp_split = None; sfc_ua_split = None; sfc_va_split = None
		ta = None; hur = None; hgt = None; ua = None; wap = None; p_3d = None;\
			va = None; uas = None; vas = None; lsm = None; wg10 = None; ps = None; tas = None; \
			ta2d = None; terrain = None; sfc_ta = None; sfc_p_3d = None; sfc_hgt = None; sfc_dp = None;\
			sfc_ua = None; sfc_va = None
		p = None
		output_data = None
		orig_shape = None; orig_sfc_shape = None;
		param = None
		is_dcape = None
		lon = None; lat = None
		if (model == "erai") | (model == "era5"):
			mod_cape_split = None; cp_split = None; mod_cape = None; cp = None

	#Broadcast split arrays to other cores
	ta_split = comm.bcast(ta_split, root=0)
	hgt_split = comm.bcast(hgt_split, root=0)
	hur_split = comm.bcast(hur_split, root=0)
	ua_split = comm.bcast(ua_split, root=0)
	va_split = comm.bcast(va_split, root=0)
	wap_split = comm.bcast(wap_split, root=0)
	ps_split = comm.bcast(ps_split, root=0)
	tas_split = comm.bcast(tas_split, root=0)
	ta2d_split = comm.bcast(ta2d_split, root=0)
	uas_split = comm.bcast(uas_split, root=0)
	vas_split = comm.bcast(vas_split, root=0)
	wg10_split = comm.bcast(wg10_split, root=0)
	terrain = comm.bcast(terrain, root=0)
	if (model == "erai") | (model == "era5"):
		cp_split = comm.bcast(cp_split, root=0)
	if model == "erai":
		mod_cape_split = comm.bcast(mod_cape_split, root=0)
	p = comm.bcast(p, root=0)
	lon = comm.bcast(lon, root=0)
	lat = comm.bcast(lat, root=0)
	split_sizes_input = comm.bcast(split_sizes_input, root = 0)
	displacements_input = comm.bcast(displacements_input, root = 0)
	split_sizes_input_2d = comm.bcast(split_sizes_input_2d, root = 0)
	displacements_input_2d = comm.bcast(displacements_input_2d, root = 0)
	split_sizes_output = comm.bcast(split_sizes_output, root = 0)
	displacements_output = comm.bcast(displacements_output, root = 0)
	orig_shape = comm.bcast(orig_shape, root = 0)
	param = comm.bcast(param, root = 0)
	is_dcape = comm.bcast(is_dcape, root = 0)

	#Create arrays to receive chunked/split data on each core, where rank specifies the core
	ta_chunk = np.zeros(np.shape(ta_split[rank]), dtype="float32")
	hgt_chunk = np.zeros(np.shape(hgt_split[rank]), dtype="float32")
	hur_chunk = np.zeros(np.shape(hur_split[rank]), dtype="float32")
	ua_chunk = np.zeros(np.shape(ua_split[rank]), dtype="float32")
	va_chunk = np.zeros(np.shape(va_split[rank]), dtype="float32")
	wap_chunk = np.zeros(np.shape(wap_split[rank]), dtype="float32")
	ps_chunk = np.zeros(np.shape(ps_split[rank]), dtype="float32")
	tas_chunk = np.zeros(np.shape(tas_split[rank]), dtype="float32")
	ta2d_chunk = np.zeros(np.shape(ta2d_split[rank]), dtype="float32")
	uas_chunk = np.zeros(np.shape(uas_split[rank]), dtype="float32")
	vas_chunk = np.zeros(np.shape(vas_split[rank]), dtype="float32")
	wg10_chunk = np.zeros(np.shape(wg10_split[rank]), dtype="float32")
	comm.Scatterv([ta,split_sizes_input, displacements_input, MPI.FLOAT],ta_chunk,root=0)
	comm.Scatterv([hgt,split_sizes_input, displacements_input, MPI.FLOAT],hgt_chunk,root=0)
	comm.Scatterv([hur,split_sizes_input, displacements_input, MPI.FLOAT],hur_chunk,root=0)
	comm.Scatterv([ua,split_sizes_input, displacements_input, MPI.FLOAT],ua_chunk,root=0)
	comm.Scatterv([va,split_sizes_input, displacements_input, MPI.FLOAT],va_chunk,root=0)
	comm.Scatterv([wap,split_sizes_input, displacements_input, MPI.FLOAT],wap_chunk,root=0)
	comm.Scatterv([ps,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],ps_chunk,root=0)
	comm.Scatterv([tas,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],tas_chunk,root=0)
	comm.Scatterv([ta2d,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],ta2d_chunk,root=0)
	comm.Scatterv([uas,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],uas_chunk,root=0)
	comm.Scatterv([vas,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],vas_chunk,root=0)
	comm.Scatterv([wg10,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],wg10_chunk,root=0)
	if (model == "erai") | (model == "era5"):
		cp_chunk = np.zeros(np.shape(cp_split[rank]), dtype="float32")
		comm.Scatterv([cp,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],cp_chunk,root=0)
	if model == "erai":
		mod_cape_chunk = np.zeros(np.shape(mod_cape_split[rank]), dtype="float32")
		comm.Scatterv([mod_cape,split_sizes_input_2d, displacements_input_2d, MPI.FLOAT],\
			mod_cape_chunk,root=0)

	comm.Barrier()

	#Print diagnostics
	if rank == 0:
		print("TOTAL TIMES: %s" %(ta.shape[0]))
		print("TIME CHUNKSIZE: %s" %(ta_chunk.shape,))
		print("CALCULATING CONVECTIVE PARAMETERS...")

#----------------------------------------------------------------------------------------------------------------
#
#		CALCULATE CAPE
#
#---------------------------------------------------------------------------------------------------------------

	#RESHAPE EVERYTHING BACK TO 4D/3D
	ta = ta_chunk.reshape((ta_chunk.shape[0],) + orig_shape[1:]) 
	hgt = hgt_chunk.reshape((hgt_chunk.shape[0],) + orig_shape[1:]) 
	hur = hur_chunk.reshape((hur_chunk.shape[0],) + orig_shape[1:]) 
	ua = ua_chunk.reshape((ua_chunk.shape[0],) + orig_shape[1:]) 
	va = va_chunk.reshape((va_chunk.shape[0],) + orig_shape[1:]) 
	wap = wap_chunk.reshape((wap_chunk.shape[0],) + orig_shape[1:]) 
	ps = ps_chunk.reshape((ps_chunk.shape[0],orig_shape[2], orig_shape[3]))
	tas = tas_chunk.reshape((tas_chunk.shape[0],orig_shape[2], orig_shape[3]))
	ta2d = ta2d_chunk.reshape((ta2d_chunk.shape[0],orig_shape[2], orig_shape[3]))
	uas = uas_chunk.reshape((uas_chunk.shape[0],orig_shape[2], orig_shape[3]))
	vas = vas_chunk.reshape((vas_chunk.shape[0],orig_shape[2], orig_shape[3]))
	wg10 = wg10_chunk.reshape((wg10_chunk.shape[0],orig_shape[2], orig_shape[3]))
	terrain = terrain.reshape((orig_shape[2], orig_shape[3]))
	if (model == "erai") | (model == "era5"):
		cp = cp_chunk.reshape((cp_chunk.shape[0],orig_shape[2], orig_shape[3]))
	if model == "erai":
		mod_cape = mod_cape_chunk.reshape((mod_cape_chunk.shape[0],orig_shape[2], orig_shape[3]))

	#Assign p levels to a 3d array, with same dimensions as input variables (ta, hgt, etc.)
	p_3d = np.moveaxis(np.tile(p,[ta.shape[2],ta.shape[3],1]),[0,1,2],[1,2,0]).astype(np.float32)

	tot_start = dt.datetime.now()
	output = np.zeros((ta_chunk.shape[0]*ps_chunk.shape[1],len(param)))
	for t in np.arange(0,ta_chunk.shape[0]):
		cape_start = dt.datetime.now()
	
		#t = 1
		#print(date_list[t])
		#p_3d = pres[t]

		dp = get_dp(hur=hur[t], ta=ta[t], dp_mask = False)

		#Insert surface arrays, creating new arrays with "sfc" prefix
		sfc_ta = np.insert(ta[t], 0, tas[t], axis=0) 
		sfc_hgt = np.insert(hgt[t], 0, terrain, axis=0) 
		sfc_dp = np.insert(dp, 0, ta2d[t], axis=0) 
		sfc_p_3d = np.insert(p_3d, 0, ps[t], axis=0) 
		sfc_ua = np.insert(ua[t], 0, uas[t], axis=0) 
		sfc_va = np.insert(va[t], 0, vas[t], axis=0) 
		#sfc_wap = np.insert(wap[t], 0, np.zeros(vas[t].shape), axis=0) 

		#Sort by ascending p
		a,temp1,temp2 = np.meshgrid(np.arange(sfc_p_3d.shape[0]) , np.arange(sfc_p_3d.shape[1]),\
			 np.arange(sfc_p_3d.shape[2]))
		sort_inds = np.flip(np.lexsort([np.swapaxes(a,1,0),sfc_p_3d],axis=0), axis=0)
		sfc_hgt = np.take_along_axis(sfc_hgt, sort_inds, axis=0)
		sfc_dp = np.take_along_axis(sfc_dp, sort_inds, axis=0)
		sfc_p_3d = np.take_along_axis(sfc_p_3d, sort_inds, axis=0)
		sfc_ua = np.take_along_axis(sfc_ua, sort_inds, axis=0)
		sfc_va = np.take_along_axis(sfc_va, sort_inds, axis=0)
		sfc_ta = np.take_along_axis(sfc_ta, sort_inds, axis=0)

		#Calculate q and wet bulb for pressure level arrays with surface values
		sfc_ta_unit = units.units.degC*sfc_ta
		sfc_dp_unit = units.units.degC*sfc_dp
		sfc_p_unit = units.units.hectopascals*sfc_p_3d
		sfc_hur_unit = mpcalc.relative_humidity_from_dewpoint(sfc_ta_unit, sfc_dp_unit)*\
			100*units.units.percent
		sfc_q_unit = mpcalc.mixing_ratio_from_relative_humidity(sfc_hur_unit,\
			sfc_ta_unit,sfc_p_unit)
		sfc_theta_unit = mpcalc.potential_temperature(sfc_p_unit,sfc_ta_unit)
		sfc_thetae_unit = mpcalc.equivalent_potential_temperature(sfc_p_unit,sfc_ta_unit,sfc_dp_unit)
		sfc_q = np.array(sfc_q_unit)
		sfc_hur = np.array(sfc_hur_unit)
		#sfc_wb = sfc_ta - (1/3 * (sfc_ta - sfc_dp))
		sfc_wb = np.array(wrf.wetbulb( sfc_p_3d*100, sfc_ta+273.15, sfc_q, units="degC"))

		#Calculate mixed-layer parcel indices, based on avg sfc-100 hPa AGL layer parcel.
		#First, find avg values for ta, p, hgt and q for ML (between the surface
		# and 100 hPa AGL)
		ml_inds = ((sfc_p_3d <= ps[t]) & (sfc_p_3d >= (ps[t] - 100)))
		#ml_p3d_avg = ps[t] - 50
		#ml_hgt_avg = get_var_p_lvl(sfc_hgt, sfc_p_3d, ml_p3d_avg)
		ml_p3d_avg = ( np.ma.masked_where(~ml_inds, sfc_p_3d).min(axis=0) + np.ma.masked_where(~ml_inds, sfc_p_3d).max(axis=0) ) / 2.
		ml_hgt_avg = ( np.ma.masked_where(~ml_inds, sfc_hgt).min(axis=0) + np.ma.masked_where(~ml_inds, sfc_hgt).max(axis=0) ) / 2.

		#ml_ta_avg = np.ma.average( np.ma.masked_where(~ml_inds, sfc_ta), axis=0).data.astype(np.float32)
		#ml_q_avg = np.ma.average( np.ma.masked_where(~ml_inds, sfc_q), axis=0).data.astype(np.float32)
		ml_ta_avg = trapz_int3d(sfc_ta, sfc_p_3d, ml_inds ).astype(np.float32)
		ml_q_avg = trapz_int3d(sfc_q, sfc_p_3d, ml_inds ).astype(np.float32)

		#Insert the mean values into the bottom of the 3d arrays pressure-level arrays
		ml_ta_arr = np.insert(sfc_ta,0,ml_ta_avg,axis=0)
		ml_q_arr = np.insert(sfc_q,0,ml_q_avg,axis=0)
		ml_hgt_arr = np.insert(sfc_hgt,0,ml_hgt_avg,axis=0)
		ml_p3d_arr = np.insert(sfc_p_3d,0,ml_p3d_avg,axis=0)
		#Sort by ascending p
		a,temp1,temp2 = np.meshgrid(np.arange(ml_p3d_arr.shape[0]) ,\
			 np.arange(ml_p3d_arr.shape[1]), np.arange(ml_p3d_arr.shape[2]))
		sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),ml_p3d_arr],axis=0))
		ml_ta_arr = np.take_along_axis(ml_ta_arr, sort_inds, axis=0)
		ml_p3d_arr = np.take_along_axis(ml_p3d_arr, sort_inds, axis=0)
		ml_hgt_arr = np.take_along_axis(ml_hgt_arr, sort_inds, axis=0)
		ml_q_arr = np.take_along_axis(ml_q_arr, sort_inds, axis=0)
		#Calculate CAPE using wrf-python. 
		cape3d_mlavg = wrf.cape_3d(ml_p3d_arr.astype(np.float64),\
			(ml_ta_arr + 273.15).astype(np.float64),\
			ml_q_arr.astype(np.float64),\
			ml_hgt_arr.astype(np.float64),terrain.astype(np.float64),\
			ps[t].astype(np.float64),False,meta=False, missing=0)
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
		cape3d = wrf.cape_3d(sfc_p_3d,sfc_ta+273.15,\
				sfc_q,sfc_hgt,\
				terrain,ps[t],\
				True,meta=False, missing=0)
		cape = cape3d.data[0]
		cin = cape3d.data[1]
		lfc = cape3d.data[2]
		lcl = cape3d.data[3]
		el = cape3d.data[4]
		#Mask values which are below the surface and above 500 hPa AGL
		cape[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		cin[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		lfc[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		lcl[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		el[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		mu_cape_inds = np.nanargmax(cape,axis=0)
		mu_cape = mu_cape_inds.choose(cape)
		mu_cin = mu_cape_inds.choose(cin)
		mu_lfc = mu_cape_inds.choose(lfc)
		mu_lcl = mu_cape_inds.choose(lcl)
		mu_el = mu_cape_inds.choose(el)
		muq = mu_cape_inds.choose(sfc_q)

		#Now get surface based CAPE. Simply the CAPE defined by parcel 
		#with surface properties
		sb_cape = np.ma.masked_where(~((sfc_p_3d==ps[t])),\
			cape).max(axis=0).filled(0)
		sb_cin = np.ma.masked_where(~((sfc_p_3d==ps[t])),\
			cin).max(axis=0).filled(0)
		sb_lfc = np.ma.masked_where(~((sfc_p_3d==ps[t])),\
			lfc).max(axis=0).filled(0)
		sb_lcl = np.ma.masked_where(~((sfc_p_3d==ps[t])),\
			lcl).max(axis=0).filled(0)
		sb_el = np.ma.masked_where(~((sfc_p_3d==ps[t])),\
			el).max(axis=0).filled(0)

		#Now get the effective-inflow layer parcel CAPE. Layer defined as a parcel with
		# the mass-wegithted average conditions of the inflow layer; the layer 
		# between when the profile has CAPE > 100 and cin < 250.
		#If no effective layer, effective layer CAPE is zero.
		#Only levels below 500 hPa AGL are considered

		#EDITS (23/01/2020)
		#Do not get surface-based values when eff_cape is not defined. Just leave as zero.
		#If an effective layer is only one level, the pacel is defined with quantities at 
		# that level. Previously, quantites were defined as zero, becuase of the averaging 
		# routine (i.e. bc pressure difference between the top of the effective layer and the 
		# bottom is zero). I assume this would result in zero CAPE (given q would be zero)
		eff_cape, eff_cin, eff_lfc, eff_lcl, eff_el, eff_hgt, eff_avg_hgt = get_eff_cape(\
			cape, cin, sfc_p_3d, sfc_ta, sfc_hgt, sfc_q, ps[t], terrain)
		eff_cape = np.where(np.isnan(eff_cape), 0, eff_cape)
		eff_cin = np.where(np.isnan(eff_cin), 0, eff_cin)
		eff_lfc = np.where(np.isnan(eff_lfc), 0, eff_lfc)
		eff_lcl = np.where(np.isnan(eff_lcl), 0, eff_lcl)
		eff_el = np.where(np.isnan(eff_el), 0, eff_el)

		#Calculate other parameters
		#Thermo
		thermo_start = dt.datetime.now()
		lr01 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,1000,terrain)
		lr03 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,3000,terrain)
		lr13 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),1000,3000,terrain)
		lr24 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),2000,4000,terrain)
		lr36 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),3000,6000,terrain)
		lr_freezing = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,"freezing",terrain)
		lr_subcloud = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,ml_lcl,terrain)
		lr850_670 = get_lr_p(ta[t], p_3d, hgt[t], 850, 670)
		lr750_500 = get_lr_p(ta[t], p_3d, hgt[t], 750, 500)
		lr700_500 = get_lr_p(ta[t], p_3d, hgt[t], 700, 500)
		melting_hgt = get_t_hgt(sfc_ta,np.copy(sfc_hgt),0,terrain)
		hwb0 = get_var_hgt(sfc_wb,np.copy(sfc_hgt),0,terrain)
		rhmean01 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,1000,terrain,True,np.copy(sfc_p_3d))
		rhmean03 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,3000,terrain,True,np.copy(sfc_p_3d))
		rhmean06 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,6000,terrain,True,np.copy(sfc_p_3d))
		rhmean13 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),1000,3000,terrain,True,np.copy(sfc_p_3d))
		rhmean36 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),3000,6000,terrain,True,np.copy(sfc_p_3d))
		rhmeansubcloud = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,ml_lcl,terrain,True,np.copy(sfc_p_3d))
		qmean01 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,1000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean03 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,3000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean06 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,6000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean13 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),1000,3000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean36 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),3000,6000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmeansubcloud = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,ml_lcl,terrain,True,np.copy(sfc_p_3d)) * 1000
		q_melting = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), melting_hgt, terrain) * 1000
		q1 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 1000, terrain) * 1000
		q3 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 3000, terrain) * 1000
		q6 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 6000, terrain) * 1000
		sfc_thetae = get_var_hgt_lvl(np.array(sfc_thetae_unit), np.copy(sfc_hgt), 0, terrain)
		rhmin01 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 1000, terrain)
		rhmin03 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 3000, terrain)
		rhmin06 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 6000, terrain)
		rhmin13 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 1000, 3000, terrain)
		rhmin36 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 3000, 6000, terrain)
		rhminsubcloud = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, ml_lcl, terrain)
		v_totals = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850) - \
				get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)
		c_totals = get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850) - \
				get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)
		t_totals = v_totals + c_totals
		pwat = get_pwat(sfc_q, p, sfc_p_3d, p_3d)
		if model != "era5":
			maxtevv = maxtevv_fn(np.array(sfc_thetae_unit), np.copy(sfc_wap), np.copy(sfc_hgt), terrain)
		te_diff = thetae_diff(np.array(sfc_thetae_unit), np.copy(sfc_hgt), terrain)
		tei = tei_fn(np.array(sfc_thetae_unit), sfc_p_3d, ps[t], np.copy(sfc_hgt), terrain)
		dpd850 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850)
		dpd700 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 700) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 700)
		dpd670 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 670) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 670)
		dpd500 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 500)
		if (int(is_dcape) == 1) & (ps[t].max() > 0):
			#Define DCAPE as the area between the moist adiabat of a descending parcel and the 
			# environmental temperature (w/o virtual temperature correction). Starting parcel 
			# chosen by the pressure level with minimum thetae
			dcape, ddraft_temp = get_dcape(np.array(sfc_p_3d[np.concatenate([[1100], p]) >= 300]), \
						sfc_ta[np.concatenate([[1100], p]) >= 300], \
						sfc_q[np.concatenate([[1100], p]) >= 300], \
						sfc_hgt[np.concatenate([[1100], p]) >= 300], \
						np.array(p[p>=300]), ps[t])
			sfc_thetae300 = sfc_thetae_unit[np.concatenate([[1100], p]) >= 300].data
			sfc_p300 = sfc_p_3d[np.concatenate([[1100], p]) >= 300]
			sfc_thetae300[(ps[t] - sfc_p300) > 400] = np.nan 
			sfc_thetae300[(sfc_p300 > ps[t])] = np.nan 
			#Calculate for all levels (sfc to 400 hPa agl), and then mask based on height agl 
			dcape = np.nanargmin(sfc_thetae300, axis=0).choose(dcape)
			ddraft_temp = tas[t] - \
				np.nanargmin(sfc_thetae300, axis=0).choose(ddraft_temp)
			ddraft_temp[(ddraft_temp<0) | (np.isnan(ddraft_temp))] = 0
		else:
			ddraft_temp = np.zeros(dpd500.shape)
			dcape = np.zeros(dpd500.shape)
		#Winds
		winds_start = dt.datetime.now()
		umeanwindinf = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0), \
					np.nanmax(eff_hgt,axis=0),0,False,sfc_p_3d)
		vmeanwindinf = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0),\
					np.nanmax(eff_hgt,axis=0),0,False,sfc_p_3d)
		umean01 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 1000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean01 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 1000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean03 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 3000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean03 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 3000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean06 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 6000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean06 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 6000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean800_600 = get_mean_var_p(ua[t], p_3d, 800, 600, ps[t], mass_weighted=True)
		vmean800_600 = get_mean_var_p(va[t], p_3d, 800, 600, ps[t], mass_weighted=True)
		Umeanwindinf = np.sqrt( (umeanwindinf**2) + (vmeanwindinf**2) )
		Umean01 = np.sqrt( (umean01**2) + (vmean01**2) )
		Umean03 = np.sqrt( (umean03**2) + (vmean03**2) )
		Umean06 = np.sqrt( (umean06**2) + (vmean06**2) )
		Umean800_600 = np.sqrt( (umean800_600**2) + (vmean800_600**2) )
		uwindinf = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), eff_avg_hgt, terrain)
		vwindinf = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), eff_avg_hgt, terrain)
		u10 = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), 10, terrain)
		v10 = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), 10, terrain)
		u500 = get_var_p_lvl(np.copy(sfc_ua), sfc_p_3d, 500)
		v500 = get_var_p_lvl(np.copy(sfc_va), sfc_p_3d, 500)
		u1 = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), 1000, terrain) 
		v1 = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), 1000, terrain) 
		u3 = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), 3000, terrain) 
		v3 = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), 3000, terrain) 
		u6 = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), 6000, terrain) 
		v6 = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), 6000, terrain) 
		Uwindinf = np.sqrt( (uwindinf**2) + (vwindinf**2) )
		U500 = np.sqrt( (u500**2) + (v500**2) )
		U10 = np.sqrt( (u10**2) + (v10**2) )
		U1 = np.sqrt( (u1**2) + (v1**2) )
		U3 = np.sqrt( (u3**2) + (v3**2) )
		U6 = np.sqrt( (u6**2) + (v6**2) )
		scld = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), ml_lcl, 0.5*mu_el, terrain)
		s01 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 1000, terrain)
		s03 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 3000, terrain)
		s06 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 6000, terrain)
		s010 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 10000, terrain)
		s13 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 1000, 3000, terrain)
		s36 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 3000, 6000, terrain)
		ebwd = get_shear_hgt(sfc_va, sfc_va, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0),\
					(mu_el - np.nanmin(eff_hgt,axis=0) ) / 2 + np.nanmin(eff_hgt,axis=0),\
					terrain)
		srh01_left, srh01_right = get_srh(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 1000, terrain)
		srh03_left, srh03_right = get_srh(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 3000, terrain)
		srh06_left, srh06_right = get_srh(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 6000, terrain)
		srhe_left, srhe_right = get_srh(sfc_ua, sfc_va, np.copy(sfc_hgt), \
						np.nanmin(eff_hgt,axis=0), np.nanmax(eff_hgt,axis=0), terrain)
		ust_right, vst_right, ust_left, vst_left = \
			get_storm_motion(sfc_ua, sfc_va, np.copy(sfc_hgt), terrain)
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
		if model != "era5":
			omega01 = get_mean_var_hgt(wap[t], hgt[t], 0, 1000, terrain, True, np.copy(p_3d))
			omega03 = get_mean_var_hgt(wap[t], hgt[t], 0, 3000, terrain, True, np.copy(p_3d))
			omega06 = get_mean_var_hgt(wap[t], hgt[t], 0, 6000, terrain, True, np.copy(p_3d))
		#Kinematic
		kinematic_start = dt.datetime.now()
		x, y = np.meshgrid(lon,lat)
		dx, dy = mpcalc.lat_lon_grid_deltas(x,y)
		thetae10 = get_var_hgt_lvl(np.array(sfc_thetae_unit), np.copy(sfc_hgt), 10, terrain)
		thetae01 = get_mean_var_hgt(np.array(sfc_thetae_unit), np.copy(sfc_hgt), 0, 1000, terrain, True, np.copy(sfc_p_3d))
		thetae03 = get_mean_var_hgt(np.array(sfc_thetae_unit), np.copy(sfc_hgt), 0, 3000, terrain, True, np.copy(sfc_p_3d))
		F10, Fn10, Fs10, icon10, vgt10, conv10, vo10 = \
				kinematics(u10, v10, thetae10, dx, dy, y)
		F01, Fn01, Fs01, icon01, vgt01, conv01, vo01 = \
				kinematics(umean01, vmean01, thetae01, dx, dy, y)
		F03, Fn03, Fs03, icon03, vgt03, conv03, vo03 = \
				kinematics(umean03, vmean03, thetae03, dx, dy, y)
		#Composites
		Rq = qmean01 / 12.
		windex = 5. * np.power( (melting_hgt/1000.) * Rq * (np.power( lr_freezing,2) - 30. + \
				qmean01 - 2. * q_melting), 0.5)
		windex[np.isnan(windex)] = 0
		gustex = (0.5 * windex) + (0.5 * Umean06)
		hmi = lr850_670 + dpd850 - dpd670
		wmsi_ml = (ml_cape * te_diff) / 1000
		dmi = lr750_500 + dpd700 - dpd500
		mwpi_ml = (ml_cape / 100.) + (lr850_670 + dpd850 - dpd670)
		wmpi = np.sqrt( np.power(melting_hgt,2) * (lr_freezing / 1000. - 5.5e-3) + \
				melting_hgt * (q1 - 1.5*q_melting) / 3.) /5.
		dmi[dmi<0] = 0
		hmi[hmi<0] = 0
		wmsi_ml[wmsi_ml<0] = 0
		mwpi_ml[wmsi_ml<0] = 0
		stp_fixed_left, stp_cin_left = get_tornado_pot( np.copy(ml_cin), np.copy(ml_lcl)\
					, np.copy(sb_lcl), np.copy(s06), np.copy(ebwd), \
					np.copy(sb_cape), np.copy(ml_cape), np.copy(srh01_left), \
					np.copy(srhe_left))		
		if model != "era5":
			moshe = ((lr03 - 4.)/4.) * ((s01 - 8)/10.) * \
				((ebwd - 8)/10.) * ((maxtevv + 10.)/9.)
			moshe[moshe<0] = 0
			mosh = ((lr03 - 4.)/4.) * ((s01 - 8)/10.) * ((maxtevv + 10.)/9.)
			mosh[mosh<0] = 0
		ship = get_ship(np.copy(mu_cape), np.copy(muq), np.copy(s06), np.copy(lr700_500), \
				get_var_p_lvl(sfc_ta, sfc_p_3d, 500), np.copy(melting_hgt) )
		scp, scp_fixed = get_supercell_pot(mu_cape, np.copy(srhe_left), np.copy(srh01_left), np.copy(ebwd),\
					np.copy(s06) )
		sherb, eff_sherb = get_sherb(np.copy(s03), np.copy(ebwd), np.copy(lr03), np.copy(lr700_500))
		k_index = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850) \
			- get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500) \
			+ get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850) - (dpd700)
		k_index[k_index<0] = 0
		mlcs6 = ml_cape * np.power(s06, 1.67)
		mucs6 = mu_cape * np.power(s06, 1.67)
		sbcs6 = sb_cape * np.power(s06, 1.67)
		effcs6 = eff_cape * np.power(s06, 1.67)
		if model == "erai":
			cs6 = mod_cape[t] * np.power(s06, 1.67)
		wndg = get_wndg(np.copy(ml_cape), np.copy(ml_cin), np.copy(lr03), sfc_ua, sfc_va, np.copy(sfc_hgt), terrain,\
			np.copy(sfc_p_3d))
		sweat = get_sweat(np.copy(sfc_p_3d), np.copy(sfc_dp), np.copy(t_totals), sfc_ua, sfc_va)
		mmp = get_mmp(sfc_ua, sfc_va, np.copy(mu_cape), sfc_ta, np.copy(sfc_hgt), terrain, np.copy(sfc_p_3d))
		dmgwind = (dcape/800.) * (Uwindinf / 8.)
		dmgwind_fixed = (dcape/800.) * (Umean800_600 / 8.)
		mburst = get_mburst(np.copy(sb_cape), np.copy(lr03), np.copy(v_totals), \
				np.copy(dcape), np.copy(pwat), np.copy(tei), \
				np.array(sfc_thetae_unit), \
				np.copy(sfc_hgt), terrain)
		mburst[mburst<0] = 0
		convgust_wet = np.sqrt( (Umean800_600**2) + (np.sqrt(2*dcape))**2 )
		convgust_dry = np.sqrt( (Umean800_600**2) + (np.sqrt(dcape))**2 )
		dcp = (dcape / 980.) * (mu_cape / 2000.) * (s06 / 20.) * (Umean06 / 16.)
	
		#Fill output
		output = fill_output(output, t, param, ps_chunk, "ml_cape", ml_cape)
		output = fill_output(output, t, param, ps_chunk, "mu_cape", mu_cape)
		output = fill_output(output, t, param, ps_chunk, "eff_cape", eff_cape)
		output = fill_output(output, t, param, ps_chunk, "sb_cape", sb_cape)
		output = fill_output(output, t, param, ps_chunk, "ml_cin", ml_cin)
		output = fill_output(output, t, param, ps_chunk, "mu_cin", mu_cin)
		output = fill_output(output, t, param, ps_chunk, "eff_cin", eff_cin)
		output = fill_output(output, t, param, ps_chunk, "sb_cin", sb_cin)
		output = fill_output(output, t, param, ps_chunk, "ml_lcl", ml_lcl)
		output = fill_output(output, t, param, ps_chunk, "mu_lcl", mu_lcl)
		output = fill_output(output, t, param, ps_chunk, "eff_lcl", eff_lcl)
		output = fill_output(output, t, param, ps_chunk, "sb_lcl", sb_lcl)
		output = fill_output(output, t, param, ps_chunk, "ml_el", ml_el)
		output = fill_output(output, t, param, ps_chunk, "mu_el", mu_el)
		output = fill_output(output, t, param, ps_chunk, "eff_el", eff_el)
		output = fill_output(output, t, param, ps_chunk, "sb_el", sb_el)
		if (model == "erai") | (model == "era5"):
			output = fill_output(output, t, param, ps_chunk, "cp", cp[t])
		if model == "erai":
			output = fill_output(output, t, param, ps_chunk, "cape", mod_cape[t])

		output = fill_output(output, t, param, ps_chunk, "lr01", lr01)
		output = fill_output(output, t, param, ps_chunk, "lr03", lr03)
		output = fill_output(output, t, param, ps_chunk, "lr13", lr13)
		output = fill_output(output, t, param, ps_chunk, "lr24", lr24)
		output = fill_output(output, t, param, ps_chunk, "lr36", lr36)
		output = fill_output(output, t, param, ps_chunk, "lr_subcloud", lr_subcloud)
		output = fill_output(output, t, param, ps_chunk, "lr_freezing", lr_freezing)
		output = fill_output(output, t, param, ps_chunk, "mhgt", melting_hgt)
		output = fill_output(output, t, param, ps_chunk, "wbz", hwb0)
		output = fill_output(output, t, param, ps_chunk, "qmean01", qmean01)
		output = fill_output(output, t, param, ps_chunk, "qmean03", qmean03)
		output = fill_output(output, t, param, ps_chunk, "qmean06", qmean06)
		output = fill_output(output, t, param, ps_chunk, "qmeansubcloud", qmeansubcloud)
		output = fill_output(output, t, param, ps_chunk, "q_melting", q_melting)
		output = fill_output(output, t, param, ps_chunk, "q1", q1)
		output = fill_output(output, t, param, ps_chunk, "q3", q3)
		output = fill_output(output, t, param, ps_chunk, "q6", q6)
		output = fill_output(output, t, param, ps_chunk, "sfc_thetae", sfc_thetae)
		output = fill_output(output, t, param, ps_chunk, "rhmin01", rhmin01)
		output = fill_output(output, t, param, ps_chunk, "rhmin03", rhmin03)
		output = fill_output(output, t, param, ps_chunk, "rhmin13", rhmin13)
		output = fill_output(output, t, param, ps_chunk, "rhminsubcloud", rhminsubcloud)
		output = fill_output(output, t, param, ps_chunk, "v_totals", v_totals)
		output = fill_output(output, t, param, ps_chunk, "c_totals", c_totals)
		output = fill_output(output, t, param, ps_chunk, "t_totals", t_totals)
		output = fill_output(output, t, param, ps_chunk, "pwat", pwat)
		output = fill_output(output, t, param, ps_chunk, "te_diff", te_diff)
		output = fill_output(output, t, param, ps_chunk, "tei", tei)
		output = fill_output(output, t, param, ps_chunk, "dpd700", dpd700)
		output = fill_output(output, t, param, ps_chunk, "dpd850", dpd850)
		output = fill_output(output, t, param, ps_chunk, "dcape", dcape)
		output = fill_output(output, t, param, ps_chunk, "ddraft_temp", ddraft_temp)

		output = fill_output(output, t, param, ps_chunk, "Umeanwindinf", Umeanwindinf)
		output = fill_output(output, t, param, ps_chunk, "Umean01", Umean01)
		output = fill_output(output, t, param, ps_chunk, "Umean03", Umean03)
		output = fill_output(output, t, param, ps_chunk, "Umean06", Umean06)
		output = fill_output(output, t, param, ps_chunk, "Umean800_600", Umean800_600)
		output = fill_output(output, t, param, ps_chunk, "Uwindinf", Uwindinf)
		output = fill_output(output, t, param, ps_chunk, "U500", U500)
		output = fill_output(output, t, param, ps_chunk, "U1", U1)
		output = fill_output(output, t, param, ps_chunk, "U3", U3)
		output = fill_output(output, t, param, ps_chunk, "U6", U6)
		output = fill_output(output, t, param, ps_chunk, "Ust_left", Ust_left)
		output = fill_output(output, t, param, ps_chunk, "Usr01_left", Usr01_left)
		output = fill_output(output, t, param, ps_chunk, "Usr03_left", Usr03_left)
		output = fill_output(output, t, param, ps_chunk, "Usr06_left", Usr06_left)
		output = fill_output(output, t, param, ps_chunk, "wg10", wg10[t])
		output = fill_output(output, t, param, ps_chunk, "U10", U10)
		output = fill_output(output, t, param, ps_chunk, "scld", scld)
		output = fill_output(output, t, param, ps_chunk, "s01", s01)
		output = fill_output(output, t, param, ps_chunk, "s03", s03)
		output = fill_output(output, t, param, ps_chunk, "s06", s06)
		output = fill_output(output, t, param, ps_chunk, "s010", s010)
		output = fill_output(output, t, param, ps_chunk, "s13", s13)
		output = fill_output(output, t, param, ps_chunk, "s36", s36)
		output = fill_output(output, t, param, ps_chunk, "ebwd", ebwd)
		output = fill_output(output, t, param, ps_chunk, "srh01_left", srh01_left)
		output = fill_output(output, t, param, ps_chunk, "srh03_left", srh03_left)
		output = fill_output(output, t, param, ps_chunk, "srh06_left", srh06_left)
		output = fill_output(output, t, param, ps_chunk, "srhe_left", srhe_left)

		output = fill_output(output, t, param, ps_chunk, "F10", F10)
		output = fill_output(output, t, param, ps_chunk, "Fn10", Fn10)
		output = fill_output(output, t, param, ps_chunk, "Fs10", Fs10)
		output = fill_output(output, t, param, ps_chunk, "icon10", icon10)
		output = fill_output(output, t, param, ps_chunk, "vgt10", vgt10)
		output = fill_output(output, t, param, ps_chunk, "conv10", conv10)
		output = fill_output(output, t, param, ps_chunk, "vo10", vo10)

		output = fill_output(output, t, param, ps_chunk, "stp_cin_left", stp_cin_left)
		output = fill_output(output, t, param, ps_chunk, "stp_fixed_left", stp_fixed_left)
		output = fill_output(output, t, param, ps_chunk, "windex", windex)
		output = fill_output(output, t, param, ps_chunk, "gustex", gustex)
		output = fill_output(output, t, param, ps_chunk, "hmi", hmi)
		output = fill_output(output, t, param, ps_chunk, "wmsi_ml", wmsi_ml)
		output = fill_output(output, t, param, ps_chunk, "dmi", dmi)
		output = fill_output(output, t, param, ps_chunk, "mwpi_ml", mwpi_ml)
		output = fill_output(output, t, param, ps_chunk, "wmpi", wmpi)
		output = fill_output(output, t, param, ps_chunk, "ship", ship)
		output = fill_output(output, t, param, ps_chunk, "scp", scp)
		output = fill_output(output, t, param, ps_chunk, "scp_fixed", scp_fixed)
		output = fill_output(output, t, param, ps_chunk, "eff_sherb", eff_sherb)
		output = fill_output(output, t, param, ps_chunk, "sherb", sherb)
		output = fill_output(output, t, param, ps_chunk, "k_index", k_index)
		output = fill_output(output, t, param, ps_chunk, "mlcape*s06", mlcs6)
		output = fill_output(output, t, param, ps_chunk, "mucape*s06", mucs6)
		output = fill_output(output, t, param, ps_chunk, "sbcape*s06", sbcs6)
		output = fill_output(output, t, param, ps_chunk, "effcape*s06", effcs6)
		if model == "erai":
			output = fill_output(output, t, param, ps_chunk, "cape*s06", cs6)
		output = fill_output(output, t, param, ps_chunk, "wndg", wndg)
		output = fill_output(output, t, param, ps_chunk, "sweat", sweat)
		output = fill_output(output, t, param, ps_chunk, "mmp", mmp)
		output = fill_output(output, t, param, ps_chunk, "mburst", mburst)
		output = fill_output(output, t, param, ps_chunk, "convgust_wet", convgust_wet)
		output = fill_output(output, t, param, ps_chunk, "convgust_dry", convgust_dry)
		output = fill_output(output, t, param, ps_chunk, "dcp", dcp)
		output = fill_output(output, t, param, ps_chunk, "dmgwind", dmgwind)
		output = fill_output(output, t, param, ps_chunk, "dmgwind_fixed", dmgwind_fixed)

		if model != "era5":
			output = fill_output(output, t, param, ps_chunk, "mosh", mosh)
			output = fill_output(output, t, param, ps_chunk, "moshe", moshe)
			output = fill_output(output, t, param, ps_chunk, "maxtevv", maxtevv)
			output = fill_output(output, t, param, ps_chunk, "omega01", omega01)
			output = fill_output(output, t, param, ps_chunk, "omega03", omega03)
			output = fill_output(output, t, param, ps_chunk, "omega06", omega06)


	#Print diagnostics
	if rank == 0:
		print("Time taken for each element on processor 1: %s" \
			%((dt.datetime.now() - tot_start)/float(ta_chunk.shape[0])), )
		print("GATHERING DATA...")

	#Gather output data together to root node
	comm.Gatherv(output, \
		[output_data, split_sizes_output, displacements_output, MPI.DOUBLE], \
		root=0)


	if rank == 0:
		print("SAVING DATA...")
		param_out = []
		for param_name in param:
			temp_data = output_data[:,np.where(param==param_name)[0][0]]
			param_out.append(temp_data.reshape((orig_shape[0],orig_shape[2],orig_shape[3])))

		#If the U1 variable is zero everywhere, then it is likely that data has not been read.
		#In this case, all values are missing, set to zero.
		for t in np.arange(param_out[0].shape[0]):
			if param_out[np.where(param=="U1")[0][0]][t].max() == 0:
				for p in np.arange(len(param_out)):
					param_out[p][t] = np.nan

		if issave:
			save_netcdf(region, model, out_name, date_list, lat, lon, param, param_out, \
				out_dtype = "f4", compress=True)

