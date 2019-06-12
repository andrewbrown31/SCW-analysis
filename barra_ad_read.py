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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

from calc_param import *

def read_barra_ad(domain,times,wg_only):
	#Open BARRA_AD netcdf files and extract variables needed for a range of times and given
	# spatial domain

	#ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times,"hours",6)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100)
	pres = pres[p_ind]
	lon,lat = get_lat_lon()
	lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
	lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
	lon = lon[lon_ind]
	lat = lat[lat_ind]
	terrain = get_terrain(lat_ind,lon_ind)

	#Initialise arrays
	ta = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	dp = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	hur = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	hgt = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((0,len(lat_ind),len(lon_ind)))
	vas = np.empty((0,len(lat_ind),len(lon_ind)))
	ps = np.empty((0,len(lat_ind),len(lon_ind)))
	max_max_wg = np.empty((0,len(lat_ind),len(lon_ind)))
	max_wg = np.empty((0,len(lat_ind),len(lon_ind)))
	date_times = np.empty((0))

	for t in np.arange(0,len(date_list)):
		year = dt.datetime.strftime(date_list[t],"%Y")
		month =	dt.datetime.strftime(date_list[t],"%m")
		day = dt.datetime.strftime(date_list[t],"%d")
		hour = dt.datetime.strftime(date_list[t],"%H")

		#Load BARRA forecast files
		max_max_wg_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/"\
			+"max_max_wndgust10m/"+year+"/"+month+"/max_max_wndgust10m-fc-spec-PT1H-BARRA_AD-v*-"\
			+year+month+day+"T"+hour+"*.nc")[0])
		max_wg_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/"\
			+"max_wndgust10m/"+year+"/"+month+"/max_wndgust10m-fc-spec-PT1H-BARRA_AD-v*-"\
			+year+month+day+"T"+hour+"*.nc")[0])
		if not wg_only:
		    ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/"\
			+year+"/"+month+"/air_temp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    z_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/geop_ht/"\
			+year+"/"+month+"/geop_ht-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    ua_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/wnd_ucmp/"\
			+year+"/"+month+"/wnd_ucmp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    va_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/wnd_vcmp/"\
			+year+"/"+month+"/wnd_vcmp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    hur_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/relhum/"\
			+year+"/"+month+"/relhum-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    uas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/uwnd10m/"\
			+year+"/"+month+"/uwnd10m-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    vas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/vwnd10m/"\
			+year+"/"+month+"/vwnd10m-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
		    ps_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/sfc_pres/"\
			+year+"/"+month+"/sfc_pres-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])

		#Get times to load in from file
		date_times = np.append(date_times,\
			nc.num2date(np.round(max_max_wg_file["time"][:]),\
				max_max_wg_file["time"].units))
		print(date_list[t])

		temp_max_max_wg = max_max_wg_file["max_max_wndgust10m"][:,lat_ind,lon_ind]
		temp_max_wg = max_wg_file["max_wndgust10m"][:,lat_ind,lon_ind]
		max_max_wg = np.append(max_max_wg,temp_max_max_wg,axis=0)
		max_wg = np.append(max_wg,temp_max_wg,axis=0)
		max_max_wg_file.close()
		max_wg_file.close()
		
		if not wg_only:
			#Load data
			temp_ta = ta_file["air_temp"][:,p_ind,lat_ind,lon_ind] - 273.15
			temp_ua = ua_file["wnd_ucmp"][:,p_ind,lat_ind,lon_ind]
			temp_va = va_file["wnd_vcmp"][:,p_ind,lat_ind,lon_ind]
			temp_hgt = z_file["geop_ht"][:,p_ind,lat_ind,lon_ind]
			temp_hur = hur_file["relhum"][:,p_ind,lat_ind,lon_ind]
			temp_hur[temp_hur<0] = 0
			temp_dp = get_dp(temp_ta,temp_hur)
			temp_uas = uas_file["uwnd10m"][:,lat_ind,lon_ind]
			temp_vas = vas_file["vwnd10m"][:,lat_ind,lon_ind]
			temp_ps = ps_file["sfc_pres"][:,lat_ind,lon_ind]/100 

			#Flip pressure axes for compatibility with SHARPpy
			temp_ta = np.fliplr(temp_ta)
			temp_dp = np.fliplr(temp_dp)
			temp_hur = np.fliplr(temp_hur)
			temp_hgt = np.fliplr(temp_hgt)
			temp_ua = np.fliplr(temp_ua)
			temp_va = np.fliplr(temp_va)
	
			#Fill arrays with current time steps
			ta = np.append(ta,temp_ta,axis=0)
			ua = np.append(ua,temp_ua,axis=0)
			va = np.append(va,temp_va,axis=0)
			hgt = np.append(hgt,temp_hgt,axis=0)
			hur = np.append(hur,temp_hur,axis=0)
			dp = np.append(dp,temp_dp,axis=0)
			uas = np.append(uas,temp_uas,axis=0)
			vas = np.append(vas,temp_vas,axis=0)
			ps = np.append(ps,temp_ps,axis=0)

			ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();ps_file.close()
	
	#Flip pressure	
	pres = np.flipud(pres)
	
	if wg_only:
		return [max_max_wg,max_wg,lon,lat,date_times]
	else:
		return [max_max_wg,max_wg,ta,dp,hur,hgt,terrain,pres,ps,ua,va,uas,vas,lon,lat,date_times]
	
#IF WANTING TO LOOP OVER TIME, CHANGE READ_BARRA TO READ ALL TIMES IN A RANGE, THEN LOOP OVER TIME DIMENSION WITHIN CALC_PARAM

def date_seq(times,delta_type,delta):
	start_time = times[0]
	end_time = times[1]
	current_time = times[0]
	date_list = [current_time]
	while (current_time < end_time):
		if delta_type == "hours":
			current_time = current_time + dt.timedelta(hours = delta)	
		date_list.append(current_time)
	return date_list

def get_pressure(top):
	ta_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/2012/12/air_temp-fc-prs-PT1H-BARRA_AD-v1-20121201T0000Z.sub.nc")
	p =ta_file["pressure"][:]
	p_ind = np.where(p>=top)[0]
	return [len(p_ind), p, p_ind]

def get_lat_lon():
	ta_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/2012/12/air_temp-fc-prs-PT1H-BARRA_AD-v1-20121201T0000Z.sub.nc")
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def get_lat_lon_inds(points,lon,lat):
	lsm = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/lnd_mask-fc-slv-PT0H-BARRA_AD-v1.nc").variables["lnd_mask"][:]
	x,y = np.meshgrid(lon,lat)
	x[lsm==0] = np.nan
	y[lsm==0] = np.nan
	lat_ind = np.empty(len(points))
	lon_ind = np.empty(len(points))
	lat_used = np.empty(len(points))
	lon_used = np.empty(len(points))
	for point in np.arange(0,len(points)):
		dist = np.sqrt(np.square(x-points[point][0]) + \
				np.square(y-points[point][1]))
		dist_lat,dist_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
		lat_ind[point] = dist_lat
		lon_ind[point] = dist_lon
		lon_used[point] = lon[dist_lon]
		lat_used[point] = lat[dist_lat]
	return [lon_ind, lat_ind, lon_used, lat_used]

def get_terrain(lat_ind,lon_ind):
	terrain_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/topog-fc-slv-PT0H-BARRA_AD-v1.nc")
	terrain = terrain_file.variables["topog"][lat_ind,lon_ind]
	terrain_file.close()
	return terrain


def remove_corrupt_dates(date_list):
	corrupt_dates = [dt.datetime(2014,11,22,6,0)]
	date_list = np.array(date_list)
	for i in np.arange(0,len(corrupt_dates)):
		date_list = date_list[~(date_list==corrupt_dates[i])]
	return date_list
