#import getcape
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
from era5_read import read_era5, read_era5_rt52
from erai_read import get_mask as  get_erai_mask
from barra_read import read_barra, read_barra_fc
from barra_ad_read import read_barra_ad
from barra_read import get_mask as  get_barra_mask
from barpa_read import read_barpa
from read_cmip import read_cmip
from wrf_parallel import *
from scipy.ndimage import uniform_filter

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
	elif region == "global":
		start_lat = -70; end_lat = 70; start_lon = -180; end_lon = 179.75
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
	if model in ["ACCESS1-0","ACCESS1-3","GFDL-CM3","GFDL-ESM2M","CNRM-CM5","MIROC5",\
		    "MRI-CGCM3","IPSL-CM5A-LR","IPSL-CM5A-MR","GFDL-ESM2G","bcc-csm1-1","MIROC-ESM",\
		    "BNU-ESM"]:
		#Check that t1 and t2 are in the same year
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, tp, lon, lat, \
		    date_list = read_cmip(model, experiment, \
		    ensemble, year, domain, cmip_ver=5, al33=al33, group=group, ver6hr=ver6hr, ver3hr=ver3hr)
		p = np.zeros(p_3d[0,:,0,0].shape)
		tp = tp.astype("float32", order="C")
	elif model in ["ACCESS-ESM1-5", "ACCESS-CM2"]:
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, lon, lat, \
		    date_list = read_cmip(model, experiment,\
		    ensemble, year, domain, cmip_ver=6, group=group, project=project)
		p = np.zeros(p_3d[0,:,0,0].shape)
	else:
		raise ValueError("Model not recognised")
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
	lon = lon.astype("float32", order="C")
	lat = lat.astype("float32", order="C")

	gc.collect()

	#This param list was originally given to AD for ERA5 global lightning report
	#param = np.array(["mu_cape", "eff_cape","ncape","mu_cin", "muq", "s06", "s0500", "lr700_500", "mhgt", "ta500","tp","cp","laplacian","t_totals"])
	#This param list is intended for application to GCMs based on AD ERA5 lightning report
	param = np.array(["mu_cape","s06","laplacian","t_totals","ta850","ta500","dp850","tp",\
		"muq","lr700_500","mhgt","z500"])

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
		sfc_q = np.array(sfc_q_unit)

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
		#Mask values which are below the surface and above 350 hPa AGL
		cape[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		cin[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		lfc[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		lcl[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		el[(sfc_p_3d > ps[t]) | (sfc_p_3d<(ps[t]-350))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		mu_cape_inds = np.tile(np.nanargmax(cape,axis=0), (cape.shape[0],1,1))
		mu_cape = np.take_along_axis(cape, mu_cape_inds, 0)[0]
		muq = np.take_along_axis(sfc_q, mu_cape_inds, 0)[0] * 1000

		#Calculate other parameters
		#Thermo
		thermo_start = dt.datetime.now()
		lr700_500 = get_lr_p(ta[t], p_3d, hgt[t], 700, 500)
		melting_hgt = get_t_hgt(sfc_ta,np.copy(sfc_hgt),0,terrain)
		melting_hgt = np.where((melting_hgt < 0) | (np.isnan(melting_hgt)), 0, melting_hgt)
		ta500 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)
		ta850 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850)
		dp850 = get_var_p_lvl(np.copy(sfc_dp), sfc_p_3d, 850)
		v_totals = ta850 - ta500
		c_totals = dp850 - ta500
		t_totals = v_totals + c_totals
		#Winds
		winds_start = dt.datetime.now()
		s06 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 6000, terrain)

		#Laplacian
		x, y = np.meshgrid(lon,lat)
		dx, dy = mpcalc.lat_lon_grid_deltas(x,y)
		if mdl_lvl:
			z500 = get_var_p_lvl(hgt[t], p_3d, 500)
			laplacian = np.array(mpcalc.laplacian(z500,deltas=[dy,dx])*1e9)
		else:
			z500 = np.squeeze(hgt[t,p==500])
			laplacian = np.array(mpcalc.laplacian(uniform_filter(z500, 4),deltas=[dy,dx])*1e9)
        

		#Fill output
		output = fill_output(output, t, param, ps, "mu_cape", mu_cape)
		output = fill_output(output, t, param, ps, "muq", muq)
		output = fill_output(output, t, param, ps, "s06", s06)
		output = fill_output(output, t, param, ps, "lr700_500", lr700_500)
		output = fill_output(output, t, param, ps, "ta500", ta500)
		output = fill_output(output, t, param, ps, "ta850", ta850)
		output = fill_output(output, t, param, ps, "dp850", dp850)
		output = fill_output(output, t, param, ps, "mhgt", melting_hgt)
		output = fill_output(output, t, param, ps, "tp", tp[t])
		output = fill_output(output, t, param, ps, "laplacian", laplacian)
		output = fill_output(output, t, param, ps, "t_totals", t_totals)
		output = fill_output(output, t, param, ps, "z500", z500)

		output_data[t] = output

	print("SAVING DATA...")
	param_out = []
	for param_name in param:
		temp_data = output_data[:,:,:,np.where(param==param_name)[0][0]]
		param_out.append(temp_data)

	#If the mhgt variable is zero everywhere, then it is likely that data has not been read.
	#In this case, all values are missing, set to zero.
	for t in np.arange(param_out[0].shape[0]):
		if param_out[np.where(param=="mhgt")[0][0]][t].max() == 0:
			for p in np.arange(len(param_out)):
				param_out[p][t] = np.nan

	if issave:
		save_netcdf(region, model, out_name, date_list, lat, lon, param, param_out, \
			out_dtype = "f4", compress=True)

	print(dt.datetime.now() - tot_start)

if __name__ == "__main__":


	warnings.simplefilter("ignore")

	main()
