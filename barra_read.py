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
from matplotlib.mlab import griddata
import pandas as pd

from plot_param import plot_param
from calc_param import *

def read_barra(domain,times):
	#Open BARRA netcdf files and extract variables needed for a range of times and given
	# spatial domain
	#NOTE, currently this uses analysis files, with no time dimension length=1. For
	# use with forecast files (with time dimension length >1), will need to be changed.

	ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times)
	if len(times) > 1:
		date_list = date_seq(times)
	else:
		date_list = times
	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)

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
	ta = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	dp = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	hur = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	hgt = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	vas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ps = np.empty((len(date_list),len(lat_ind),len(lon_ind)))

	for t in np.arange(0,len(date_list)):
		year = dt.datetime.strftime(date_list[t],"%Y")
		month =	dt.datetime.strftime(date_list[t],"%m")
		day = dt.datetime.strftime(date_list[t],"%d")
		hour = dt.datetime.strftime(date_list[t],"%H")
		print(date_list[t])

		#Load BARRA analysis files
		ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+year+"/"+month+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/geop_ht/"\
	+year+"/"+month+"/geop_ht-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		ua_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_ucmp/"\
	+year+"/"+month+"/wnd_ucmp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		va_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_vcmp/"\
	+year+"/"+month+"/wnd_vcmp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		hur_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/relhum/"\
	+year+"/"+month+"/relhum-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		uas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/uwnd10m/"\
	+year+"/"+month+"/uwnd10m-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		vas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/vwnd10m/"\
	+year+"/"+month+"/vwnd10m-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		ps_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/sfc_pres/"\
	+year+"/"+month+"/sfc_pres-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]

		#Load data
		temp_ta = ta_file["air_temp"][p_ind,lat_ind,lon_ind] - 273.15
		temp_ua = ua_file["wnd_ucmp"][p_ind,lat_ind,lon_ind]
		temp_va = va_file["wnd_vcmp"][p_ind,lat_ind,lon_ind]
		temp_hgt = z_file["geop_ht"][p_ind,lat_ind,lon_ind]
		temp_hur = hur_file["relhum"][p_ind,lat_ind,lon_ind]
		temp_hur[temp_hur<0] = 0
		temp_dp = get_dp(temp_ta,temp_hur)
		uas[t,:,:] = uas_file["uwnd10m"][lat_ind,lon_ind]
		vas[t,:,:] = vas_file["vwnd10m"][lat_ind,lon_ind]
		ps[t,:,:] = ps_file["sfc_pres"][lat_ind,lon_ind]/100 

		#Flip pressure axes for compatibility with SHARPpy
		ta[t,:,:,:] = np.flipud(temp_ta)
		dp[t,:,:,:] = np.flipud(temp_dp)
		hur[t,:,:,:] = np.flipud(temp_hur)
		hgt[t,:,:,:] = np.flipud(temp_hgt)
		ua[t,:,:,:] = np.flipud(temp_ua)
		va[t,:,:,:] = np.flipud(temp_va)

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();ps_file.close()
		
	p = np.flipud(pres)

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list]
	
#IF WANTING TO LOOP OVER TIME, CHANGE READ_BARRA TO READ ALL TIMES IN A RANGE, THEN LOOP OVER TIME DIMENSION WITHIN CALC_PARAM

def read_barra_points(points,times):

	#NOTE might have to change to read forecast (hourly) data instead of analysis
	#NOTE if directly comparing with ERA-Interim, need to coarsen/interpolate first

	# points = [(x1,y1),(x2,y2),...,(xn,yn)]
	# times = [start_time, end_time]

	#Open BARRA netcdf files and extract variables needed for a multiple files/times given set of spatial points
	
	ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times)

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100)
	lon,lat = get_lat_lon()
	[lon_ind, lat_ind, lon_used, lat_used] = get_lat_lon_inds(points,lon,lat)

	ta = np.empty((len(date_list),no_p,len(points)))
	dp = np.empty((len(date_list),no_p,len(points)))
	hur = np.empty((len(date_list),no_p,len(points)))
	hgt = np.empty((len(date_list),no_p,len(points)))
	p = np.empty((len(date_list),no_p,len(points)))
	ua = np.empty((len(date_list),no_p,len(points)))
	va = np.empty((len(date_list),no_p,len(points)))
	uas = np.empty((len(date_list),len(points)))
	vas = np.empty((len(date_list),len(points)))
	ps = np.empty((len(date_list),len(points)))

	for t in np.arange(0,len(date_list)):
		year = dt.datetime.strftime(date_list[t],"%Y")
		month =	dt.datetime.strftime(date_list[t],"%m")
		day = dt.datetime.strftime(date_list[t],"%d")
		hour = dt.datetime.strftime(date_list[t],"%H")
		print(year,month,day,hour)

		#Load BARRA analysis files
		ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+year+"/"+month+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/geop_ht/"\
	+year+"/"+month+"/geop_ht-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		ua_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_ucmp/"\
	+year+"/"+month+"/wnd_ucmp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		va_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_vcmp/"\
	+year+"/"+month+"/wnd_vcmp-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		hur_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/relhum/"\
	+year+"/"+month+"/relhum-an-prs-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		uas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/uwnd10m/"\
	+year+"/"+month+"/uwnd10m-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		vas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/vwnd10m/"\
	+year+"/"+month+"/vwnd10m-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])
		ps_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/sfc_pres/"\
	+year+"/"+month+"/sfc_pres-an-spec-PT0H-BARRA_R-v1-"+year+month+day+"T"+hour+"*.nc")[0])

		times = ta_file["time"][:]
		
		for point in np.arange(0,len(points)):

			#NOTE THIS COULD BE MORE SOPHISTICATED THAN JUST NEAREST POINT
			# SHOULD PROABLY MAKE SURE IT'S NOT AN OCEAN POINT AT LEAST
			# IF COMPARING WITH ERA-INTERIM DIRECTLY, MIGHT NEED TO COARSEN GRID FIRST
			#lon_ind = np.argmin(abs(lon-points[point][0]))
			#lat_ind = np.argmin(abs(lat-points[point][1]))
			ta[t,:,point] = ta_file["air_temp"][p_ind,lat_ind,lon_ind] - 273.15
			ua[t,:,point] = ua_file["wnd_ucmp"][p_ind,lat_ind,lon_ind]
			va[t,:,point] = va_file["wnd_vcmp"][p_ind,lat_ind,lon_ind]
			hgt[t,:,point] = z_file["geop_ht"][p_ind,lat_ind,lon_ind]
			hur[t,:,point] = hur_file["relhum"][p_ind,lat_ind,lon_ind]
			hur[hur<0] = 0
			dp[t,:,point] = get_dp(ta[t,:,point],hur[t,:,point])
			uas[t,point] = uas_file["uwnd10m"][lat_ind,lon_ind]
			vas[t,point] = vas_file["vwnd10m"][lat_ind,lon_ind]
			ps[t,point] = ps_file["sfc_pres"][lat_ind,lon_ind]/100 
			p[t,:,point] = pres[p_ind]

		ta_file.close();z_file.close();ua_file.close();va_file.close();
		hur_file.close();uas_file.close();vas_file.close();ps_file.close()
	
	#Flip pressure axes for compatibility with SHARPpy
	ta = np.fliplr(ta)
	dp = np.fliplr(dp)
	hur = np.fliplr(hur)
	hgt = np.fliplr(hgt)
	ua = np.fliplr(ua)
	va = np.fliplr(va)
	p = np.fliplr(p)

	#Save lat/lon as array
	lon = np.empty((len(points)))
	lat = np.empty((len(points)))
	terrain = np.empty((len(points)))
	for point in np.arange(0,len(points)):
		lon[point] = points[point][0]
		lat[point] = points[point][1]
		terrain[point] = get_terrain(lat_ind[point],lon_ind[point])

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list]

def date_seq(times):
	start_time = times[0]
	end_time = times[1]
	current_time = times[0]
	date_list = [current_time]
	while (current_time < end_time):
		current_time = current_time + dt.timedelta(hours = 6)
		date_list.append(current_time)
	return date_list

def get_pressure(top):
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2012"+"/"+"12"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2012"+"12"+"01"+"T"+"00"+"*.nc")[0])
	p =ta_file["pressure"][:]
	p_ind = np.where(p>=top)[0]
	return [len(p_ind), p, p_ind]

def get_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2012"+"/"+"12"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2012"+"12"+"01"+"T"+"00"+"*.nc")[0])
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def get_lat_lon_inds(points,lon,lat):
	lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
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
	terrain_file = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/topog-an-slv-PT0H-BARRA_R-v1.nc")
	terrain = terrain_file.variables["topog"][lat_ind,lon_ind]
	terrain_file.close()
	return terrain

