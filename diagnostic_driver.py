from SkewT import get_dcape
import warnings
import numpy as np
import datetime as dt
try:
	import metpy.units as units
	import metpy.calc as mpcalc
except:
	pass
import wrf
from utils import save_netcdf, get_dp
from diagnostic_functions import *

#-------------------------------------------------------------------------------------------------

#This file is the same as wrf_parallel.py, except that the work is not done in parallel

#Note that changes in diagnostic definitions within wrf_parallel.py, which are not contained within
#   functions, will need to be copied here.

#-------------------------------------------------------------------------------------------------

def fill_output(output, t, param, ps, p, data):

	"""
	Fill an output array over a parameter dimension
	"""

	output[:,:,:,np.where(param==p)[0][0]] = data

	return output

def run_diagnostics(ta,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list,
					params="full",mdl_lvl=False,is_dcape=True,
					issave=False,out_name="",out_path=""):
	
	"""
	This function calculates and saves a suite of environmental diagnostics relevant for thunderstorms 
	and severe convective gusts. Some of these parameters (wg10) can be set to zero if unknown.
	Vertical coordinate can be pressure or height (see below)

	Parameters
	----------
	ta : (time,vertical,lat,lon) 'numpy.array'
		Air temperature (degrees C)
	hur : (time,vertical,lat,lon) 'numpy.array'
		Relative humidity (percent)
	hgt : (time,vertical,lat,lon) 'numpy.array'
		Geopotential height (meters)
	terrain : (lat,lon) 'numpy.array'
		Model terrain height above sea level (meters)
	p : (vertical) or (time,vertical,lat,lon) 'numpy.array'
		Air pressure (hPa). If one-dimensional, this corresponds to the vertical coordinate of other data (default),
		and mdl_lvl should be False. If multi-dimensional, then the input data is probably on model levels,
		so mdl_lvl should be true.
	ps : (time,lat,lon) 'numpy.array'
		Surface air pressure (hPa)
	ua : (time,vertical,lat,lon) 'numpy.array'
		Zonal wind component (m/s)
	va : (time,height,lat,lon) 'numpy.array'
		Meridional wind component (m/s)		
	uas : (time,lat,lon) 'numpy.array'
		Near-surface zonal wind component (m/s)
	vas : (time,lat,lon) 'numpy.array'
		Near-surface meridional wind component (m/s)			
	tas : (time,lat,lon) 'numpy.array'
		Near-surface air temperature (degrees C)
	ta2d : (time,lat,lon) 'numpy.array'	
		Near-surface dewpoint temperature (degrees C)
	wg10 : (time,lat,lon) 'numpy.array'	
		10-m surface wind gust (m/s)
	lon : (lon) 'numpy.array'
		Longitude coordinate (degrees East)
	lat : (lon) 'numpy.array'
		Latitude coordinate (degrees North)
	date_list : (time) 'list of datetime.datetime objects'
		Time coordinates
	params : 'str'
		Set of params to calculate. Either "full", "reduced", or "min". Note that this doesn't save
		on compute time, but saves on the output volume
	mdl_lvl : 'bool'
		Is data on model levels? Otherwise, pressure level data is assumed. See 'p' parameter.
	is_dcape : 'bool'
		Perform dcape calculation?
	issave : 'bool'
		Save output as netcdf?
	out_name : 'str'
		Name of output netcdf file. In the form <outname>_<datelist[0]>_<datelist[-1]>.nc
	out_path : 'str'
		Path to save output netcdf file
	"""

	warnings.simplefilter("ignore")

	if params=="full":
		param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
			"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl",\
			"lr01", "lr03", "lr13", "lr36", "lr24", "lr_freezing","lr_subcloud","lr700_500",\
			"qmean01", "qmean03", "qmean06", "muq", "ta500","ta850","dp850",\
			#NEW VARIABLES FOR ETSA TESTING
			"qmean0500","q2m",\
			"qmeansubcloud", "q_melting", "q1", "q3", "q6",\
			"rhmin01", "rhmin03", "rhmin13", "rhmean01",\
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
			"wndg","mburst","sweat","k_index","wmpi","bdsd",\
			\
			"F10", "Fn10", "Fs10", "icon10", "vgt10", "conv10", "vo10",\
				])
	elif params == "reduced":
		param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
			"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl",\
			"lr36", "lr13","lr700_500","lr_subcloud",\
			"qmean01","q_melting","rhmin01","rhmin03","rhmin13","dpd700",  "muq",\
			"mhgt", "ta500", "mu_el", "ml_el", "sb_el", "eff_el", \
			"pwat", "t_totals", \
			"dcape", \
			\
			"srhe_left", "srh01_left", \
			"ebwd", "s06", "s03", "scld",\
			"U10", "U1", \
			"Umean800_600", "Umean06", "Umean03",\
			"wg10",\
			\
			"dcp", "stp_cin_left", "stp_fixed_left",\
			"scp", "scp_fixed", "ship",\
			"mlcape*s06", "mucape*s06", "sbcape*s06", "effcape*s06", \
			"dmgwind", "dmgwind_fixed", \
			"convgust_wet", "convgust_dry", "windex",\
			"gustex", "mmp", \
			"wndg","sweat","k_index","bdsd"\

				])
	elif params == "min":
		param = np.array(["mu_cape", "mu_cin", "s06","wg10","dcape","convgust_dry","convgust_wet","gustex",\
				    "bdsd","qmean01","Umean06","lr13","mucape*s06"])

	#Convert input to float32. This is because to ensure that == operations work between variables 
	ta = ta.astype("float32", order="C")
	hur = hur.astype("float32", order="C")
	hgt = hgt.astype("float32", order="C")
	terrain = terrain.astype("float32", order="C")
	p = p.astype("float32", order="C")
	ps = ps.astype("float32", order="C")
	ua = ua.astype("float32", order="C")
	va = va.astype("float32", order="C")
	uas = uas.astype("float32", order="C")
	vas = vas.astype("float32", order="C")
	tas= tas.astype("float32", order="C")
	ta2d = ta2d.astype("float32", order="C")
	wg10 = wg10.astype("float32", order="C")
	lon = lon.astype("float32", order="C")
	lat = lat.astype("float32", order="C")

	#Set output array
	output_data = np.zeros((ps.shape[0], ps.shape[1], ps.shape[2], len(param)))


	#If mdl_lvl data is false (and data is on regular pressure levels), assign p
	# to a 3d array, with same dimensions as input variables (ta, hgt, etc.)
	#If the data is already on model levels (mdl_lvl=True), then do nothing
	if mdl_lvl:
		full_p3d = p_3d
	else:
		p_3d = np.moveaxis(np.tile(p,[ta.shape[2],ta.shape[3],1]),[0,1,2],[1,2,0]).\
			astype(np.float32)

	tot_start = dt.datetime.now()
	for t in np.arange(0,ta.shape[0]):
		output = np.zeros((1, ps.shape[1], ps.shape[2], len(param)))
	
		print(date_list[t])

		if mdl_lvl:
			p_3d = full_p3d[t]

		dp = get_dp(hur=hur[t], ta=ta[t], dp_mask = False)

		#Insert surface arrays, creating new arrays with "sfc" prefix
		sfc_ta = np.insert(ta[t], 0, tas[t], axis=0) 
		sfc_hgt = np.insert(hgt[t], 0, terrain, axis=0) 
		sfc_dp = np.insert(dp, 0, ta2d[t], axis=0) 
		sfc_p_3d = np.insert(p_3d, 0, ps[t], axis=0) 
		sfc_ua = np.insert(ua[t], 0, uas[t], axis=0) 
		sfc_va = np.insert(va[t], 0, vas[t], axis=0) 

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
		sfc_thetae_unit = mpcalc.equivalent_potential_temperature(sfc_p_unit,sfc_ta_unit,sfc_dp_unit)
		sfc_thetae = np.array(mpcalc.equivalent_potential_temperature(ps[t]*units.units.hectopascals,tas[t]*units.units.degC,\
				    ta2d[t]*units.units.degC))
		sfc_q = np.array(sfc_q_unit)
		sfc_hur = np.array(sfc_hur_unit)
		sfc_wb = np.array(wrf.wetbulb( sfc_p_3d*100, sfc_ta+273.15, sfc_q, units="degC"))

		#Calculate mixed-layer parcel indices, based on avg sfc-100 hPa AGL layer parcel.
		#First, find avg values for ta, p, hgt and q for ML (between the surface
		# and 100 hPa AGL)
		ml_inds = ((sfc_p_3d <= ps[t]) & (sfc_p_3d >= (ps[t] - 100)))
		ml_p3d_avg = ( np.ma.masked_where(~ml_inds, sfc_p_3d).min(axis=0) + np.ma.masked_where(~ml_inds, sfc_p_3d).max(axis=0) ) / 2.
		ml_hgt_avg = ( np.ma.masked_where(~ml_inds, sfc_hgt).min(axis=0) + np.ma.masked_where(~ml_inds, sfc_hgt).max(axis=0) ) / 2.
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
		cape[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		cin[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		lfc[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		lcl[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		el[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		mu_cape_inds = np.tile(np.nanargmax(cape,axis=0), (cape.shape[0],1,1))
		mu_cape = np.take_along_axis(cape, mu_cape_inds, 0)[0]
		mu_cin = np.take_along_axis(cin, mu_cape_inds, 0)[0]
		mu_lfc = np.take_along_axis(lfc, mu_cape_inds, 0)[0]
		mu_lcl = np.take_along_axis(lcl, mu_cape_inds, 0)[0]
		mu_el = np.take_along_axis(el, mu_cape_inds, 0)[0]
		muq = np.take_along_axis(sfc_q, mu_cape_inds, 0)[0] * 1000

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
		lr01 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,1000,terrain)
		lr03 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,3000,terrain)
		lr13 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),1000,3000,terrain)
		lr24 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),2000,4000,terrain)
		lr36 = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),3000,6000,terrain)
		lr_subcloud = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,ml_lcl,terrain)
		lr850_670 = get_lr_p(ta[t], p_3d, hgt[t], 850, 670)
		lr750_500 = get_lr_p(ta[t], p_3d, hgt[t], 750, 500)
		lr700_500 = get_lr_p(ta[t], p_3d, hgt[t], 700, 500)
		melting_hgt = get_t_hgt(sfc_ta,np.copy(sfc_hgt),0,terrain)
		melting_hgt = np.where((melting_hgt < 0) | (np.isnan(melting_hgt)), 0, melting_hgt)
		lr_freezing = get_lr_hgt(sfc_ta,np.copy(sfc_hgt),0,melting_hgt,terrain)
		lr_freezing = np.where((melting_hgt < 0) | (np.isnan(melting_hgt)), 0, lr_freezing)
		hwb0 = get_var_hgt(np.flipud(sfc_wb),np.flipud(np.copy(sfc_hgt)),0,terrain)
		rhmean01 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,1000,terrain,True,np.copy(sfc_p_3d))
		rhmean03 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,3000,terrain,True,np.copy(sfc_p_3d))
		rhmean06 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,6000,terrain,True,np.copy(sfc_p_3d))
		rhmean13 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),1000,3000,terrain,True,np.copy(sfc_p_3d))
		rhmean36 = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),3000,6000,terrain,True,np.copy(sfc_p_3d))
		rhmeansubcloud = get_mean_var_hgt(np.copy(sfc_hur),np.copy(sfc_hgt),0,ml_lcl,terrain,True,np.copy(sfc_p_3d))
		qmean0500 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,500,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean01 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,1000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean03 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,3000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean06 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,6000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean13 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),1000,3000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmean36 = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),3000,6000,terrain,True,np.copy(sfc_p_3d)) * 1000
		qmeansubcloud = get_mean_var_hgt(np.copy(sfc_q),np.copy(sfc_hgt),0,ml_lcl,terrain,True,np.copy(sfc_p_3d)) * 1000
		q_melting = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), melting_hgt, terrain) * 1000
		q_melting = np.where((melting_hgt < 0) | (np.isnan(melting_hgt)),\
			get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 0, terrain) * 100, q_melting)
		q1 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 1000, terrain) * 1000
		q3 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 3000, terrain) * 1000
		q6 = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 6000, terrain) * 1000
		q2m = get_var_hgt_lvl(np.copy(sfc_q), np.copy(sfc_hgt), 2, terrain) * 1000
		rhmin01 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 1000, terrain)
		rhmin03 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 3000, terrain)
		rhmin06 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, 6000, terrain)
		rhmin13 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 1000, 3000, terrain)
		rhmin36 = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 3000, 6000, terrain)
		rhminsubcloud = get_min_var_hgt(np.copy(sfc_hur), np.copy(sfc_hgt), 0, ml_lcl, terrain)
		ta850 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850)
		ta500 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)
		dp850 = get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850)
		v_totals = ta850 - ta500
		c_totals = dp850 - ta500
		t_totals = v_totals + c_totals
		pwat = get_pwat(sfc_q, np.copy(sfc_p_3d))
		te_diff = thetae_diff(np.array(sfc_thetae_unit), np.copy(sfc_hgt), terrain)
		tei = tei_fn(np.array(sfc_thetae_unit), sfc_thetae, sfc_p_3d, ps[t], np.copy(sfc_hgt), terrain)
		dpd850 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850)
		dpd700 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 700) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 700)
		dpd670 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 670) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 670)
		dpd500 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500) - \
				get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 500)
		if (int(is_dcape) == 1) & (ps[t].max() > 0):
			#Define DCAPE as the area between the moist adiabat of a descending parcel 
			# and the environmental temperature (w/o virtual temperature correction). 
			#Starting parcel chosen by the pressure level with minimum thetae below 
			# 400 hPa AGL

			if mdl_lvl:
				sfc_thetae300 = np.copy(sfc_thetae_unit)
				sfc_thetae300[(ps[t] - sfc_p_3d) > 400] = np.nan 
				sfc_thetae300[(sfc_p_3d > ps[t])] = np.nan 
				dcape, ddraft_temp = get_dcape( sfc_p_3d, sfc_ta, sfc_q, sfc_hgt,\
					ps[t], p_lvl=False, \
					minthetae_inds=np.argmin(sfc_thetae300, axis=0))

			else:
				#Get 3d DCAPE for every point below 300 hPa, and then mask points above 400 hPa AGL
				#For each lat/lon point, calculate the minimum thetae, and use
				# DCAPE for that point
				dcape, ddraft_temp = get_dcape(\
							np.array(sfc_p_3d[np.concatenate([[1100], \
								p]) >= 300]), \
							sfc_ta[np.concatenate([[1100], p]) >= 300], \
							sfc_q[np.concatenate([[1100], p]) >= 300], \
							sfc_hgt[np.concatenate([[1100], p]) >= 300], \
							ps[t], p=np.array(p[p>=300]))
				sfc_thetae300 = np.array(sfc_thetae_unit[np.concatenate([[1100], \
					p]) >= 300])
				sfc_p300 = sfc_p_3d[np.concatenate([[1100], p]) >= 300]
				sfc_thetae300[(ps[t] - sfc_p300) > 400] = np.nan 
				sfc_thetae300[(sfc_p300 > ps[t])] = np.nan 
				dcape_inds = np.tile(np.nanargmin(sfc_thetae300, axis=0), \
					    (sfc_thetae300.shape[0],1,1) )
				dcape = np.take_along_axis(dcape, dcape_inds, 0)[0]
				ddraft_temp = tas[t] - \
					np.take_along_axis(ddraft_temp, dcape_inds, 0)[0]

				ddraft_temp[(ddraft_temp<0) | (np.isnan(ddraft_temp))] = 0
		else:
			ddraft_temp = np.zeros(dpd500.shape)
			dcape = np.zeros(dpd500.shape)
		#Winds
		umeanwindinf = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0), \
					np.nanmax(eff_hgt,axis=0),0,True,sfc_p_3d)
		vmeanwindinf = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0),\
					np.nanmax(eff_hgt,axis=0),0,True,sfc_p_3d)
		umean01 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 1000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean01 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 1000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean03 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 3000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean03 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 3000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean06 = get_mean_var_hgt(sfc_ua, np.copy(sfc_hgt), 0, 6000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		vmean06 = get_mean_var_hgt(sfc_va, np.copy(sfc_hgt), 0, 6000, terrain, mass_weighted=True, p3d=np.copy(sfc_p_3d))
		umean800_600 = get_mean_var_p(ua[t], p_3d, 800, 600, ps[t], mass_weighted=True)
		vmean800_600 = get_mean_var_p(va[t], p_3d, 800, 600, ps[t], mass_weighted=True)
		Umeanwindinf = np.sqrt( (umeanwindinf**2) + (vmeanwindinf**2) )
		Umeanwindinf = np.where(np.isnan(Umeanwindinf), 0, Umeanwindinf)
		Umean01 = np.sqrt( (umean01**2) + (vmean01**2) )
		Umean03 = np.sqrt( (umean03**2) + (vmean03**2) )
		Umean06 = np.sqrt( (umean06**2) + (vmean06**2) )
		Umean800_600 = np.sqrt( (umean800_600**2) + (vmean800_600**2) )
		uwindinf = get_var_hgt_lvl(sfc_ua, np.copy(sfc_hgt), np.nanmax(eff_hgt,axis=0), 0)
		vwindinf = get_var_hgt_lvl(sfc_va, np.copy(sfc_hgt), np.nanmax(eff_hgt,axis=0), 0)
		u10 = uas[t]
		v10 = vas[t]
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
		ebwd = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), np.nanmin(eff_hgt,axis=0),\
					(mu_el * 0.5), 0)
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

		#Kinematic
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
		gustex = (0.5 * windex) + (0.5 * (U500*1.944))
		hmi = lr850_670 + dpd850 - dpd670
		wmsi_ml = (ml_cape * te_diff) / 1000
		dmi = lr750_500 + dpd700 - dpd500
		mwpi_ml = (sb_cape / 1000.) + (lr850_670/5. + ((dpd850 - dpd670)/5.))
		wmpi = np.sqrt( np.power(melting_hgt,2) * (lr_freezing / 1000. - 5.5e-3) + \
				melting_hgt * (q1 - 1.5*q_melting) / 3.) /5.
		dmi[dmi<0] = 0
		hmi[hmi<0] = 0
		wmsi_ml[wmsi_ml<0] = 0
		mwpi_ml[mwpi_ml<0] = 0
		stp_fixed_left, stp_cin_left = get_tornado_pot( np.copy(ml_cin), np.copy(ml_lcl)\
					, np.copy(sb_lcl), np.copy(s06), np.copy(ebwd), \
					np.copy(sb_cape), np.copy(ml_cape), np.copy(srh01_left), \
					np.copy(srhe_left))		

		t500 = get_var_p_lvl(sfc_ta, sfc_p_3d, 500)
		ship = get_ship(np.copy(mu_cape), np.copy(muq), np.copy(s06), np.copy(lr700_500), \
				np.copy(t500), np.copy(melting_hgt) )
		scp, scp_fixed = get_supercell_pot(np.copy(mu_cape), np.copy(srhe_left), np.copy(srh01_left), np.copy(ebwd),\
					np.copy(s06) )
		sherb, eff_sherb = get_sherb(np.copy(s03), np.copy(ebwd), np.copy(lr03), np.copy(lr700_500))
		k_index = (get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850) \
			- get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)) \
			+ get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850) - (dpd700)
		k_index[k_index<0] = 0
		mlcs6 = ml_cape * np.power(s06, 1.67)
		mucs6 = mu_cape * np.power(s06, 1.67)
		sbcs6 = sb_cape * np.power(s06, 1.67)
		effcs6 = eff_cape * np.power(s06, 1.67)

		wndg = get_wndg(np.copy(ml_cape), np.copy(ml_cin), np.copy(lr03), np.copy(sfc_ua), np.copy(sfc_va), np.copy(sfc_hgt), terrain,\
			np.copy(sfc_p_3d))
		sweat = get_sweat(np.copy(sfc_p_3d), np.copy(sfc_dp), np.copy(t_totals), np.copy(sfc_ua), np.copy(sfc_va))
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
		dcp = (dcape / 980.) * (mu_cape / 2000.) * ( (s06*1.944) / 20.) * ((Umean06*1.944) / 16.)
		z = (ebwd * 6.1e-2) + (Umean800_600 * 1.5e-1) + (lr13 * 9.4e-1) + (rhmin13 * 3.9e-2) + (srhe_left.data * 1.7e-2) + (q_melting * 3.8e-1) + (eff_lcl * 4.7e-4) - 1.3e+1
		bdsd = 1. / ( 1. + np.exp( -z ) )
	
		#Fill output
		if (params == "min") | (params == "full") | (params == "reduced"):
			output = fill_output(output, t, param, ps, "mu_cape", mu_cape)
			output = fill_output(output, t, param, ps, "mu_cin", mu_cin)
			output = fill_output(output, t, param, ps, "s06", s06)
			output = fill_output(output, t, param, ps, "wg10", wg10[t])
			output = fill_output(output, t, param, ps, "bdsd", bdsd)
			output = fill_output(output, t, param, ps, "qmean01", qmean01)
			output = fill_output(output, t, param, ps, "Umean06", Umean06)
			output = fill_output(output, t, param, ps, "lr13", lr13)
			output = fill_output(output, t, param, ps, "mucape*s06", mucs6)
			output = fill_output(output, t, param, ps, "dcape", dcape)
			output = fill_output(output, t, param, ps, "gustex", gustex)
			output = fill_output(output, t, param, ps, "convgust_wet", convgust_wet)
			output = fill_output(output, t, param, ps, "convgust_dry", convgust_dry)
			#output = fill_output(output, t, param, ps, "rhmin13", rhmin13)
			#output = fill_output(output, t, param, ps, "q_melting", q_melting)

		if (params == "reduced") | (params == "full"):
			output = fill_output(output, t, param, ps, "muq", muq)
			output = fill_output(output, t, param, ps, "lr700_500", lr700_500)
			output = fill_output(output, t, param, ps, "ta500", ta500)
			output = fill_output(output, t, param, ps, "mhgt", melting_hgt)
			output = fill_output(output, t, param, ps, "ml_cape", ml_cape)
			output = fill_output(output, t, param, ps, "eff_cape", eff_cape)
			output = fill_output(output, t, param, ps, "sb_cape", sb_cape)
			output = fill_output(output, t, param, ps, "ml_cin", ml_cin)
			output = fill_output(output, t, param, ps, "eff_cin", eff_cin)
			output = fill_output(output, t, param, ps, "sb_cin", sb_cin)
			output = fill_output(output, t, param, ps, "ml_lcl", ml_lcl)
			output = fill_output(output, t, param, ps, "mu_lcl", mu_lcl)
			output = fill_output(output, t, param, ps, "eff_lcl", eff_lcl)
			output = fill_output(output, t, param, ps, "sb_lcl", sb_lcl)
			output = fill_output(output, t, param, ps, "ml_el", ml_el)
			output = fill_output(output, t, param, ps, "mu_el", mu_el)
			output = fill_output(output, t, param, ps, "eff_el", eff_el)
			output = fill_output(output, t, param, ps, "sb_el", sb_el)
			output = fill_output(output, t, param, ps, "lr36", lr36)
			output = fill_output(output, t, param, ps, "muq", muq)
			output = fill_output(output, t, param, ps, "pwat", pwat)
			output = fill_output(output, t, param, ps, "srh01_left", srh01_left)
			output = fill_output(output, t, param, ps, "srhe_left", srhe_left)
			output = fill_output(output, t, param, ps, "s03", s03)
			output = fill_output(output, t, param, ps, "ebwd", ebwd)
			output = fill_output(output, t, param, ps, "Umean800_600", Umean800_600)
			output = fill_output(output, t, param, ps, "U10", U10)
			output = fill_output(output, t, param, ps, "stp_cin_left", stp_cin_left)
			output = fill_output(output, t, param, ps, "stp_fixed_left", stp_fixed_left)
			output = fill_output(output, t, param, ps, "windex", windex)
			output = fill_output(output, t, param, ps, "ship", ship)
			output = fill_output(output, t, param, ps, "scp", scp)
			output = fill_output(output, t, param, ps, "scp_fixed", scp_fixed)
			output = fill_output(output, t, param, ps, "k_index", k_index)
			output = fill_output(output, t, param, ps, "mlcape*s06", mlcs6)
			output = fill_output(output, t, param, ps, "sbcape*s06", sbcs6)
			output = fill_output(output, t, param, ps, "effcape*s06", effcs6)
			output = fill_output(output, t, param, ps, "wndg", wndg)
			output = fill_output(output, t, param, ps, "sweat", sweat)
			output = fill_output(output, t, param, ps, "mmp", mmp)
			output = fill_output(output, t, param, ps, "dcp", dcp)
			output = fill_output(output, t, param, ps, "dmgwind", dmgwind)
			output = fill_output(output, t, param, ps, "dmgwind_fixed", dmgwind_fixed)
			output = fill_output(output, t, param, ps, "t_totals", t_totals)
			output = fill_output(output, t, param, ps, "rhmean01", rhmean01)
			output = fill_output(output, t, param, ps, "rhmin01", rhmin01)
			output = fill_output(output, t, param, ps, "rhmin03", rhmin03)
			output = fill_output(output, t, param, ps, "dpd700", dpd700)
			output = fill_output(output, t, param, ps, "scld", scld)
			output = fill_output(output, t, param, ps, "Umean03", Umean03)
			output = fill_output(output, t, param, ps, "U1", U1)
	    
		if params == "full":
			output = fill_output(output, t, param, ps, "lr_freezing", lr_freezing)
			output = fill_output(output, t, param, ps, "lr_subcloud", lr_subcloud)
			output = fill_output(output, t, param, ps, "qmeansubcloud", qmeansubcloud)
			output = fill_output(output, t, param, ps, "lr01", lr01)
			output = fill_output(output, t, param, ps, "lr03", lr03)
			output = fill_output(output, t, param, ps, "lr24", lr24)
			output = fill_output(output, t, param, ps, "wbz", hwb0)
			output = fill_output(output, t, param, ps, "qmean0500", qmean0500)
			output = fill_output(output, t, param, ps, "qmean03", qmean03)
			output = fill_output(output, t, param, ps, "qmean06", qmean06)
			output = fill_output(output, t, param, ps, "q_melting", q_melting)
			output = fill_output(output, t, param, ps, "q2m", q2m)
			output = fill_output(output, t, param, ps, "q1", q1)
			output = fill_output(output, t, param, ps, "q3", q3)
			output = fill_output(output, t, param, ps, "q6", q6)
			output = fill_output(output, t, param, ps, "sfc_thetae", sfc_thetae)
			output = fill_output(output, t, param, ps, "rhmin13", rhmin13)
			output = fill_output(output, t, param, ps, "rhminsubcloud", rhminsubcloud)
			output = fill_output(output, t, param, ps, "ta850", ta850)
			output = fill_output(output, t, param, ps, "dp850", dp850)
			output = fill_output(output, t, param, ps, "v_totals", v_totals)
			output = fill_output(output, t, param, ps, "c_totals", c_totals)
			output = fill_output(output, t, param, ps, "te_diff", te_diff)
			output = fill_output(output, t, param, ps, "tei", tei)
			output = fill_output(output, t, param, ps, "dpd850", dpd850)
			output = fill_output(output, t, param, ps, "ddraft_temp", ddraft_temp)
			output = fill_output(output, t, param, ps, "Umeanwindinf", Umeanwindinf)
			output = fill_output(output, t, param, ps, "Umean01", Umean01)
			output = fill_output(output, t, param, ps, "Uwindinf", Uwindinf)
			output = fill_output(output, t, param, ps, "U500", U500)
			output = fill_output(output, t, param, ps, "U3", U3)
			output = fill_output(output, t, param, ps, "U6", U6)
			output = fill_output(output, t, param, ps, "Ust_left", Ust_left)
			output = fill_output(output, t, param, ps, "Usr01_left", Usr01_left)
			output = fill_output(output, t, param, ps, "Usr03_left", Usr03_left)
			output = fill_output(output, t, param, ps, "Usr06_left", Usr06_left)
			output = fill_output(output, t, param, ps, "s01", s01)
			output = fill_output(output, t, param, ps, "s010", s010)
			output = fill_output(output, t, param, ps, "s13", s13)
			output = fill_output(output, t, param, ps, "s36", s36)
			output = fill_output(output, t, param, ps, "srh03_left", srh03_left)
			output = fill_output(output, t, param, ps, "srh06_left", srh06_left)

			output = fill_output(output, t, param, ps, "F10", F10)
			output = fill_output(output, t, param, ps, "Fn10", Fn10)
			output = fill_output(output, t, param, ps, "Fs10", Fs10)
			output = fill_output(output, t, param, ps, "icon10", icon10)
			output = fill_output(output, t, param, ps, "vgt10", vgt10)
			output = fill_output(output, t, param, ps, "conv10", conv10)
			output = fill_output(output, t, param, ps, "vo10", vo10)

			output = fill_output(output, t, param, ps, "hmi", hmi)
			output = fill_output(output, t, param, ps, "wmsi_ml", wmsi_ml)
			output = fill_output(output, t, param, ps, "dmi", dmi)
			output = fill_output(output, t, param, ps, "mwpi_ml", mwpi_ml)
			output = fill_output(output, t, param, ps, "wmpi", wmpi)
			output = fill_output(output, t, param, ps, "eff_sherb", eff_sherb)
			output = fill_output(output, t, param, ps, "sherb", sherb)

		output_data[t] = output

	print("SAVING DATA...")
	param_out = []
	for param_name in param:
		temp_data = output_data[:,:,:,np.where(param==param_name)[0][0]]
		param_out.append(temp_data)

	#If the s06 variable is zero everywhere, then it is likely that data has not been read.
	#In this case, all values are missing, set to zero.
	for t in np.arange(param_out[0].shape[0]):
		if param_out[np.where(param=="lr13")[0][0]][t].max() == 0:
			for p in np.arange(len(param_out)):
				param_out[p][t] = np.nan

	if issave:
		save_netcdf(out_path, out_name, date_list, lat, lon, param, param_out, \
			out_dtype = "f4", compress=True)

	print(dt.datetime.now() - tot_start)
