import numpy as np
try:
	import metpy.units as units
	import metpy.calc as mpcalc
except:
	pass
import wrf

#-------------------------------------------------------------------------------------------------

#This file contains functions to extract thunderstorm/extreme-convective-wind-gust parameters 
# from gridded model data

#Function uses wrf-python and metpy. The SkewT package has been adapted for the DCAPE routine
# (see SkewT.py)
#
#NOTES:
#
#	- CIN is only non-zero where CAPE is non-zero. CIN did not behave nicely otherwise.
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
	eff_avg_p = ((np.nanmin(eff_p,axis=0) + np.nanmax(eff_p,axis=0)) / 2).astype(np.float32)
	eff_avg_hgt = ((np.nanmin(eff_hgt,axis=0) + np.nanmax(eff_hgt,axis=0)) / 2).astype(np.float32)
	eff_avg_ta = trapz_int3d(sfc_ta, sfc_p_3d, eff_cape_cond).astype(np.float32)
	eff_avg_q = trapz_int3d(sfc_q, sfc_p_3d, eff_cape_cond).astype(np.float32)

	#So that the wrf-python code behaves nicely, fill the points with no effective layer, using surface conditions.
	#These points will be masked later
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

	#Finally, make sure to mask points where there is no effective layer
	eff_cape[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_cin[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lfc[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lcl[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_el[eff_cape_cond.sum(axis=0) == 0] = 0

	return eff_cape, eff_cin, eff_lfc, eff_lcl, eff_el, eff_hgt, eff_avg_hgt

def get_pwat(q, sfcp3d):

	#p_ind = np.argmin(abs(p - 400)) + 1
	#pwat = (((q[:p_ind]+q[1:p_ind+1])*1000/2 * (sfcp3d[:p_ind]-sfcp3d[1:p_ind+1])) * 0.00040173)\
	#	.sum(axis=0)	#From SHARPpy
	sfcp3d[(sfcp3d < 400)] = 0
	pwat = np.nansum( (((q[:-1]+q[1:])*1000/2 * (sfcp3d[:-1]-sfcp3d[1:])) * 0.00040173), axis=0)	#From SHARPpy
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
			cond = ( (hgt-terrain) < hgt_bot) | ( (hgt-terrain) > hgt_top) | (np.isnan(hgt)) | (np.isnan(hgt_bot)) | (np.isnan(hgt_top)) \
				| (np.isnan(var3d))
			result = trapz_int3d( var3d, p3d, ~cond)
		except:
			raise ValueError("FUNCTION get_mean_var_hgt() IS FAILING TO TAKE A PRESSURE WEIGHTED"+\
				" AVERAGE. HAS A 3D PRESSURE FIELD BEEN PARSED?")
	else:
		hgt = hgt - terrain
		var3d_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) | (np.isnan(hgt)) , var3d)
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
	u_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | (np.isnan(u)) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) | (np.isnan(hgt)), u)
	v_ma = np.ma.masked_where((hgt < hgt_bot) | (hgt > hgt_top) | (np.isnan(v)) | \
			(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) | (np.isnan(hgt)) , v)
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
	mllcl_term[mllcl>2000] = 0

	sblcl_term = ( (2000. - sblcl) / 1000.) 
	sblcl_term[sblcl<1000] = 1
	sblcl_term[sblcl>2000] = 0

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
	term3 = 2.*(U850*1.944)
	term4 = U500 * 1.944
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

	if t.ndim == 1:
		t_bot = np.interp(hgt_bot,hgt,t)
		t_top = np.interp(hgt_top,hgt,t)
	else:
		t_bot = wrf.interplevel(t,hgt,hgt_bot,meta=False)
		t_bot[hgt[0] >= hgt_bot] = t[0,hgt[0] >= hgt_bot]
		t_bot[(np.where(hgt==hgt_bot))[1],(np.where(hgt==hgt_bot))[2]] = t[hgt==hgt_bot]
		t_top = wrf.interplevel(t,hgt,hgt_top,meta=False)
		t_top[hgt[-1] <= hgt_top] = t[-1,hgt[-1] <= hgt_top]
		t_top[(np.where(hgt==hgt_top))[1],(np.where(hgt==hgt_top))[2]] = t[hgt==hgt_top]

	return np.squeeze(- (t_top - t_bot) / ((hgt_top - hgt_bot)/1000))

def get_t_hgt(t,hgt,t_value,terrain):
	#Get the height [lev,lat,lon] at which temperature [lev,lat,lon] is equal to t_value

	hgt = hgt - terrain

	if t.ndim == 1:
		t_hgt = np.interp(t_value,np.flipud(t),np.flipud(hgt))
	else:
		t_hgt = np.array(wrf.interplevel(np.flipud(hgt), np.flipud(t), t_value))

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

	te_ma = np.ma.masked_where((hgt<0) | (hgt>3000) | (np.isnan(hgt)), te)

	#min_idx = np.ma.argmin(te_ma, axis=0)
	#max_idx = np.ma.argmax(te_ma, axis=0)
	min_idx = np.tile( np.ma.argmin(te_ma, axis=0), (te_ma.shape[0],1,1) )
	max_idx = np.tile( np.ma.argmax(te_ma, axis=0), (te_ma.shape[0],1,1) )

	min_te = np.take_along_axis( te_ma, min_idx, 0 )[0]
	max_te = np.take_along_axis( te_ma, max_idx, 0 )[0]
	#min_te = min_idx.choose(te_ma)
	#max_te = max_idx.choose(te_ma)
	te_diff = (max_te - min_te)
	te_diff[te_diff<0] = 0
	te_diff[min_idx[0] < max_idx[0]] = 0

	return te_diff

def tei_fn(te, sfc_te, p3d, ps, hgt, terrain):

	#Return theta-e index. Defined by SPC as diff between sfc thetae and min thetae in sfc to 400 hPa AGL layer.
	#Note that SHARPpy has reverted to diff between max and min thetae in same layer, to get closer to SPC
	# operational output

	te[p3d > ps] = np.nan
	te[p3d < (ps - 400)] = np.nan
	min_te = np.nanmin(te, axis=0)
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