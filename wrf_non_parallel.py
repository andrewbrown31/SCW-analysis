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
from read_cmip import read_cmip
from wrf_parallel import *

#-------------------------------------------------------------------------------------------------

#This file is the same as wrf_parallel.py, except that the work is not done in parallel

#Note that changes in diagnostic definitions within wrf_parallel.py, which are not contained within
#   functions, will need to be copied here.

#-------------------------------------------------------------------------------------------------

def fill_output(output, t, param, ps, p, data):

	output[:,:,:,np.where(param==p)[0][0]] = data

	return output

def main():
	load_start = dt.datetime.now()
	#Try parsing arguments using argparse
	parser = argparse.ArgumentParser(description='wrf non-parallel convective diagnostics processer')
	parser.add_argument("-m",help="Model name",required=True)
	parser.add_argument("-r",help="Region name (default is aus)",default="aus")
	parser.add_argument("-t1",help="Time start YYYYMMDDHH",required=True)
	parser.add_argument("-t2",help="Time end YYYYMMDDHH",required=True)
	parser.add_argument("-e", help="CMIP5 experiment name (not required if using era5, erai or barra)", default="")
	parser.add_argument("--ens", help="CMIP5 ensemble name (not required if using era5, erai or barra)", default="r1i1p1")
	parser.add_argument("--group", help="CMIP6 modelling group name", default="")
	parser.add_argument("--project", help="CMIP6 modelling intercomparison project", default="CMIP")
	parser.add_argument("--ver6hr", help="Version on al33 for 6hr data", default="")
	parser.add_argument("--ver3hr", help="Version on al33 for 3hr data", default="")
	parser.add_argument("--issave",help="Save output (True or False, default is False)", default="False")
	parser.add_argument("--outname",help="Name of saved output. In the form *outname*_*t1*_*t2*.nc. Default behaviour is the model name",default=None)
	parser.add_argument("--is_dcape",help="Should DCAPE be calculated? (1 or 0. Default is 1)",default=1)
	parser.add_argument("--al33",help="Should data be gathered from al33? Default is False, and data is gathered from r87. If True, then group is required",default="False")
	args = parser.parse_args()

	#Parse arguments from cmd line and set up inputs (date region model)
	model = args.m
	region = args.r
	t1 = args.t1
	t2 = args.t2
	issave = args.issave
	al33 = args.al33
	if args.outname==None:
		out_name = model
	else:
		out_name = args.outname
	is_dcape = args.is_dcape
	experiment = args.e
	ensemble = args.ens
	group = args.group
	project = args.project
	ver6hr = args.ver6hr
	ver3hr = args.ver3hr
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
	if al33=="True":
		al33 = True
	elif al33=="False":
		al33 = False
	else:
		raise ValueError("\n INVALID al33...SHOULD BE True OR False")

	#Load data
	print("LOADING DATA...")
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
	elif model in ["ACCESS1-0","ACCESS1-3","GFDL-CM3","GFDL-ESM2M","CNRM-CM5","MIROC5",\
		    "MRI-CGCM3","IPSL-CM5A-LR","IPSL-CM5A-MR","GFDL-ESM2G","bcc-csm1-1","MIROC-ESM",\
		    "BNU-ESM"]:
		#Check that t1 and t2 are in the same year
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, lon, lat, \
		    date_list = read_cmip(model, experiment, \
		    ensemble, year, domain, cmip_ver=5, al33=al33, group=group, ver6hr=ver6hr, ver3hr=ver3hr)
		wap = np.zeros(hgt.shape)
		wg10 = np.zeros(ps.shape)
		p = np.zeros(p_3d[0,:,0,0].shape)
		#date_list = pd.to_datetime(date_list).to_pydatetime()
		temp1 = None
	elif model in ["ACCESS-ESM1-5", "ACCESS-CM2"]:
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, lon, lat, \
		    date_list = read_cmip(model, experiment,\
		    ensemble, year, domain, cmip_ver=6, group=group, project=project)
		wap = np.zeros(hgt.shape)
		wg10 = np.zeros(ps.shape)
		p = np.zeros(p_3d[0,:,0,0].shape)
		#date_list = pd.to_datetime(date_list).to_pydatetime()
		temp1 = None
	else:
		raise ValueError("Model not recognised")
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

	#Set output array
	output_data = np.zeros((ps.shape[0], ps.shape[1], ps.shape[2], len(param)))


	#Assign p levels to a 3d array, with same dimensions as input variables (ta, hgt, etc.)
	#If the 3d p-lvl array already exists, then declare the variable "mdl_lvl" as true. 
	try:
		p_3d;
		mdl_lvl = True
		full_p3d = p_3d
	except:
		mdl_lvl = False
		p_3d = np.moveaxis(np.tile(p,[ta.shape[2],ta.shape[3],1]),[0,1,2],[1,2,0]).\
			astype(np.float32)

	print("LOAD TIME..."+str(dt.datetime.now()-load_start))
	tot_start = dt.datetime.now()
	for t in np.arange(0,ta.shape[0]):
		output = np.zeros((1, ps.shape[1], ps.shape[2], len(param)))
		cape_start = dt.datetime.now()
	
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
		sfc_wap = np.insert(wap[t], 0, np.zeros(vas[t].shape), axis=0) 

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
		cape[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		cin[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		lfc[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		lcl[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		el[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-500))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		#mu_cape_inds = np.nanargmax(cape,axis=0)
		#mu_cape = mu_cape_inds.choose(cape)
		#mu_cin = mu_cape_inds.choose(cin)
		#mu_lfc = mu_cape_inds.choose(lfc)
		#mu_lcl = mu_cape_inds.choose(lcl)
		#mu_el = mu_cape_inds.choose(el)
		#muq = mu_cape_inds.choose(sfc_q)
		mu_cape_inds = np.tile(np.nanargmax(cape,axis=0), (cape.shape[0],1,1))
		mu_cape = np.take_along_axis(cape, mu_cape_inds, 0)[0]
		mu_cin = np.take_along_axis(cin, mu_cape_inds, 0)[0]
		mu_lfc = np.take_along_axis(lfc, mu_cape_inds, 0)[0]
		mu_lcl = np.take_along_axis(lcl, mu_cape_inds, 0)[0]
		mu_el = np.take_along_axis(el, mu_cape_inds, 0)[0]
		muq = np.take_along_axis(sfc_q, mu_cape_inds, 0)[0]

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
		hwb0 = get_var_hgt(np.flipud(sfc_wb),np.flipud(np.copy(sfc_hgt)),0,terrain)
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
		pwat = get_pwat(sfc_q, np.copy(sfc_p_3d))
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
			#Define DCAPE as the area between the moist adiabat of a descending parcel 
			# and the environmental temperature (w/o virtual temperature correction). 
			#Starting parcel chosen by the pressure level with minimum thetae below 
			# 300 hPa

			if mdl_lvl:
				sfc_thetae300 = np.copy(sfc_thetae_unit)
				sfc_thetae300[(ps[t] - sfc_p_3d) > 400] = np.nan 
				sfc_thetae300[(sfc_p_3d > ps[t])] = np.nan 
				dcape, ddraft_temp = get_dcape( sfc_p_3d, sfc_ta, sfc_q, sfc_hgt,\
					ps[t], p_lvl=False, \
					minthetae_inds=np.argmin(sfc_thetae300, axis=0))

			else:
				#Get 3d DCAPE for every point below 300 hPa
				#For each lat/lon point, calculate the minimum thetae, and use
				# DCAPE for that point
				dcape, ddraft_temp = get_dcape(\
							np.array(sfc_p_3d[np.concatenate([[1100], \
								p]) >= 300]), \
							sfc_ta[np.concatenate([[1100], p]) >= 300], \
							sfc_q[np.concatenate([[1100], p]) >= 300], \
							sfc_hgt[np.concatenate([[1100], p]) >= 300], \
							ps[t], p=np.array(p[p>=300]))
				sfc_thetae300 = sfc_thetae_unit[np.concatenate([[1100], \
					p]) >= 300].data
				sfc_p300 = sfc_p_3d[np.concatenate([[1100], p]) >= 300]
				sfc_thetae300[(ps[t] - sfc_p300) > 400] = np.nan 
				sfc_thetae300[(sfc_p300 > ps[t])] = np.nan 
				#dcape = np.nanargmin(sfc_thetae300, axis=0).choose(dcape)
				#ddraft_temp = tas[t] - \
				#	np.nanargmin(sfc_thetae300, axis=0).choose(ddraft_temp)
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
		output = fill_output(output, t, param, ps, "ml_cape", ml_cape)
		output = fill_output(output, t, param, ps, "mu_cape", mu_cape)
		output = fill_output(output, t, param, ps, "eff_cape", eff_cape)
		output = fill_output(output, t, param, ps, "sb_cape", sb_cape)
		output = fill_output(output, t, param, ps, "ml_cin", ml_cin)
		output = fill_output(output, t, param, ps, "mu_cin", mu_cin)
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
		if (model == "erai") | (model == "era5"):
			output = fill_output(output, t, param, ps, "cp", cp[t])
		if model == "erai":
			output = fill_output(output, t, param, ps, "cape", mod_cape[t])

		output = fill_output(output, t, param, ps, "lr01", lr01)
		output = fill_output(output, t, param, ps, "lr03", lr03)
		output = fill_output(output, t, param, ps, "lr13", lr13)
		output = fill_output(output, t, param, ps, "lr24", lr24)
		output = fill_output(output, t, param, ps, "lr36", lr36)
		output = fill_output(output, t, param, ps, "lr_subcloud", lr_subcloud)
		output = fill_output(output, t, param, ps, "lr_freezing", lr_freezing)
		output = fill_output(output, t, param, ps, "mhgt", melting_hgt)
		output = fill_output(output, t, param, ps, "wbz", hwb0)
		output = fill_output(output, t, param, ps, "qmean01", qmean01)
		output = fill_output(output, t, param, ps, "qmean03", qmean03)
		output = fill_output(output, t, param, ps, "qmean06", qmean06)
		output = fill_output(output, t, param, ps, "qmeansubcloud", qmeansubcloud)
		output = fill_output(output, t, param, ps, "q_melting", q_melting)
		output = fill_output(output, t, param, ps, "q1", q1)
		output = fill_output(output, t, param, ps, "q3", q3)
		output = fill_output(output, t, param, ps, "q6", q6)
		output = fill_output(output, t, param, ps, "sfc_thetae", sfc_thetae)
		output = fill_output(output, t, param, ps, "rhmin01", rhmin01)
		output = fill_output(output, t, param, ps, "rhmin03", rhmin03)
		output = fill_output(output, t, param, ps, "rhmin13", rhmin13)
		output = fill_output(output, t, param, ps, "rhminsubcloud", rhminsubcloud)
		output = fill_output(output, t, param, ps, "v_totals", v_totals)
		output = fill_output(output, t, param, ps, "c_totals", c_totals)
		output = fill_output(output, t, param, ps, "t_totals", t_totals)
		output = fill_output(output, t, param, ps, "pwat", pwat)
		output = fill_output(output, t, param, ps, "te_diff", te_diff)
		output = fill_output(output, t, param, ps, "tei", tei)
		output = fill_output(output, t, param, ps, "dpd700", dpd700)
		output = fill_output(output, t, param, ps, "dpd850", dpd850)
		output = fill_output(output, t, param, ps, "dcape", dcape)
		output = fill_output(output, t, param, ps, "ddraft_temp", ddraft_temp)

		output = fill_output(output, t, param, ps, "Umeanwindinf", Umeanwindinf)
		output = fill_output(output, t, param, ps, "Umean01", Umean01)
		output = fill_output(output, t, param, ps, "Umean03", Umean03)
		output = fill_output(output, t, param, ps, "Umean06", Umean06)
		output = fill_output(output, t, param, ps, "Umean800_600", Umean800_600)
		output = fill_output(output, t, param, ps, "Uwindinf", Uwindinf)
		output = fill_output(output, t, param, ps, "U500", U500)
		output = fill_output(output, t, param, ps, "U1", U1)
		output = fill_output(output, t, param, ps, "U3", U3)
		output = fill_output(output, t, param, ps, "U6", U6)
		output = fill_output(output, t, param, ps, "Ust_left", Ust_left)
		output = fill_output(output, t, param, ps, "Usr01_left", Usr01_left)
		output = fill_output(output, t, param, ps, "Usr03_left", Usr03_left)
		output = fill_output(output, t, param, ps, "Usr06_left", Usr06_left)
		output = fill_output(output, t, param, ps, "wg10", wg10[t])
		output = fill_output(output, t, param, ps, "U10", U10)
		output = fill_output(output, t, param, ps, "scld", scld)
		output = fill_output(output, t, param, ps, "s01", s01)
		output = fill_output(output, t, param, ps, "s03", s03)
		output = fill_output(output, t, param, ps, "s06", s06)
		output = fill_output(output, t, param, ps, "s010", s010)
		output = fill_output(output, t, param, ps, "s13", s13)
		output = fill_output(output, t, param, ps, "s36", s36)
		output = fill_output(output, t, param, ps, "ebwd", ebwd)
		output = fill_output(output, t, param, ps, "srh01_left", srh01_left)
		output = fill_output(output, t, param, ps, "srh03_left", srh03_left)
		output = fill_output(output, t, param, ps, "srh06_left", srh06_left)
		output = fill_output(output, t, param, ps, "srhe_left", srhe_left)

		output = fill_output(output, t, param, ps, "F10", F10)
		output = fill_output(output, t, param, ps, "Fn10", Fn10)
		output = fill_output(output, t, param, ps, "Fs10", Fs10)
		output = fill_output(output, t, param, ps, "icon10", icon10)
		output = fill_output(output, t, param, ps, "vgt10", vgt10)
		output = fill_output(output, t, param, ps, "conv10", conv10)
		output = fill_output(output, t, param, ps, "vo10", vo10)

		output = fill_output(output, t, param, ps, "stp_cin_left", stp_cin_left)
		output = fill_output(output, t, param, ps, "stp_fixed_left", stp_fixed_left)
		output = fill_output(output, t, param, ps, "windex", windex)
		output = fill_output(output, t, param, ps, "gustex", gustex)
		output = fill_output(output, t, param, ps, "hmi", hmi)
		output = fill_output(output, t, param, ps, "wmsi_ml", wmsi_ml)
		output = fill_output(output, t, param, ps, "dmi", dmi)
		output = fill_output(output, t, param, ps, "mwpi_ml", mwpi_ml)
		output = fill_output(output, t, param, ps, "wmpi", wmpi)
		output = fill_output(output, t, param, ps, "ship", ship)
		output = fill_output(output, t, param, ps, "scp", scp)
		output = fill_output(output, t, param, ps, "scp_fixed", scp_fixed)
		output = fill_output(output, t, param, ps, "eff_sherb", eff_sherb)
		output = fill_output(output, t, param, ps, "sherb", sherb)
		output = fill_output(output, t, param, ps, "k_index", k_index)
		output = fill_output(output, t, param, ps, "mlcape*s06", mlcs6)
		output = fill_output(output, t, param, ps, "mucape*s06", mucs6)
		output = fill_output(output, t, param, ps, "sbcape*s06", sbcs6)
		output = fill_output(output, t, param, ps, "effcape*s06", effcs6)
		if model == "erai":
			output = fill_output(output, t, param, ps, "cape*s06", cs6)
		output = fill_output(output, t, param, ps, "wndg", wndg)
		output = fill_output(output, t, param, ps, "sweat", sweat)
		output = fill_output(output, t, param, ps, "mmp", mmp)
		output = fill_output(output, t, param, ps, "mburst", mburst)
		output = fill_output(output, t, param, ps, "convgust_wet", convgust_wet)
		output = fill_output(output, t, param, ps, "convgust_dry", convgust_dry)
		output = fill_output(output, t, param, ps, "dcp", dcp)
		output = fill_output(output, t, param, ps, "dmgwind", dmgwind)
		output = fill_output(output, t, param, ps, "dmgwind_fixed", dmgwind_fixed)

		if model != "era5":
			output = fill_output(output, t, param, ps, "mosh", mosh)
			output = fill_output(output, t, param, ps, "moshe", moshe)
			output = fill_output(output, t, param, ps, "maxtevv", maxtevv)
			output = fill_output(output, t, param, ps, "omega01", omega01)
			output = fill_output(output, t, param, ps, "omega03", omega03)
			output = fill_output(output, t, param, ps, "omega06", omega06)

		output_data[t] = output



	print("SAVING DATA...")
	param_out = []
	for param_name in param:
		temp_data = output_data[:,:,:,np.where(param==param_name)[0][0]]
		param_out.append(temp_data)

	#If the U1 variable is zero everywhere, then it is likely that data has not been read.
	#In this case, all values are missing, set to zero.
	for t in np.arange(param_out[0].shape[0]):
		if param_out[np.where(param=="U1")[0][0]][t].max() == 0:
			for p in np.arange(len(param_out)):
				param_out[p][t] = np.nan

	if issave:
		save_netcdf(region, model, out_name, date_list, lat, lon, param, param_out, \
			out_dtype = "f4", compress=True)

	print(dt.datetime.now() - tot_start)

if __name__ == "__main__":


	warnings.simplefilter("ignore")

	main()
