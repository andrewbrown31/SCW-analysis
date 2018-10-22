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
from barra_read import date_seq

def read_erai(domain,time):
	#Open ERA-Interim netcdf files and extract variables needed for a single time point and given spatial domain

	ref = dt.datetime(1900,1,1,0,0,0)
	time_hours = (time - ref).total_seconds() / (3600)

	#Load ERA-Interim reanalysis files
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	z_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/z/\
z_6hrs_ERAI_historical_an-pl_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	ua_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ua/\
ua_6hrs_ERAI_historical_an-pl_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	va_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/va/\
va_6hrs_ERAI_historical_an-pl_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	hur_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/hur/\
hur_6hrs_ERAI_historical_an-pl_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	uas_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/uas/\
uas_6hrs_ERAI_historical_an-sfc_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])
	vas_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/vas/\
vas_6hrs_ERAI_historical_an-sfc_"+dt.datetime.strftime(time,"%Y%m")+"*.nc")[0])

	lon = ta_file["lon"][:]
	lat = ta_file["lat"][:]
	times = ta_file["time"][:]
	p = ta_file["lev"][:]/100
	p_ind = np.where(p>=100)[0]
	lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
	lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
	time_ind = np.where(time_hours==times)[0]
	ta = ta_file["ta"][time_ind,p_ind,lat_ind,lon_ind] - 273.15
	ua = ua_file["ua"][time_ind,p_ind,lat_ind,lon_ind]
	va = va_file["va"][time_ind,p_ind,lat_ind,lon_ind]
	hgt = z_file["z"][time_ind,p_ind,lat_ind,lon_ind] / 9.8
	hur = hur_file["hur"][time_ind,p_ind,lat_ind,lon_ind]
	hur[hur<0] = 0
	dp = get_dp(ta,hur)
	uas = uas_file["uas"][time_ind,lat_ind,lon_ind]
	vas = vas_file["vas"][time_ind,lat_ind,lon_ind]
	lon = lon[lon_ind]
	lat = lat[lat_ind]
	p = p[p_ind]
	ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close()
	
	return [ta,dp,hgt,p,ua,va,uas,vas,lon,lat]
	
def read_erai_mf(points,times):

	#Open ERA-Interim netcdf files and extract variables needed for a multiple files/times given set of spatial points

	#Format dates and times
	ref = dt.datetime(1900,1,1,0,0,0)
	date_list = date_seq(times)
	formatted_dates = [format_dates(x) for x in date_list]
	unique_dates = np.unique(formatted_dates)
	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100)
	lon,lat = get_lat_lon()
	[lon_ind, lat_ind, lon_used, lat_used] = get_lat_lon_inds(points,lon,lat)

	#Initialise arrays for each variable
	ta = np.empty((len(date_list),no_p,len(points)))
	dp = np.empty((len(date_list),no_p,len(points)))
	hur = np.empty((len(date_list),no_p,len(points)))
	hgt = np.empty((len(date_list),no_p,len(points)))
	p = np.empty((len(date_list),no_p,len(points)))
	ua = np.empty((len(date_list),no_p,len(points)))
	va = np.empty((len(date_list),no_p,len(points)))
	uas = np.empty((len(date_list),len(points)))
	vas = np.empty((len(date_list),len(points)))

	for date in unique_dates:
		print(date)

		#Load ERA-Interim reanalysis files
		ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/z/\
z_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		ua_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ua/\
ua_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		va_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/va/\
va_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		hur_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/hur/\
hur_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		uas_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/uas/\
uas_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])
		vas_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/vas/\
vas_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]
		time_ind = [np.where(x==times)[0][0] for x in time_hours if (x in times)]
		date_ind = np.where(np.array(formatted_dates) == date)[0]

		#Load data for each spatial point given to function
		for point in np.arange(0,len(points)):
			ta[date_ind,:,point] = ta_file["ta"][time_ind,p_ind,lat_ind,lon_ind] - 273.15
			ua[date_ind,:,point] = ua_file["ua"][time_ind,p_ind,lat_ind,lon_ind]
			va[date_ind,:,point] = va_file["va"][time_ind,p_ind,lat_ind,lon_ind]
			hgt[date_ind,:,point] = z_file["z"][time_ind,p_ind,lat_ind,lon_ind] / 9.8
			hur[date_ind,:,point] = hur_file["hur"][time_ind,p_ind,lat_ind,lon_ind]
			hur[hur<0] = 0
			dp[date_ind,:,point] = get_dp(ta[date_ind,:,point],hur[date_ind,:,point])
			uas[date_ind,point] = uas_file["uas"][time_ind,lat_ind,lon_ind]
			vas[date_ind,point] = vas_file["vas"][time_ind,lat_ind,lon_ind]
			p[date_ind,:,point] = pres[p_ind]

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close()

	#Save lat/lon as array
	lon = np.empty((len(points)))
	lat = np.empty((len(points)))
	for point in np.arange(0,len(points)):
		lon[point] = points[point][0]
		lat[point] = points[point][1]
	
	return [ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list]

def get_pressure(top):
	#Returns [no of levels, levels, indices below "top"]
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+"201201"+"*.nc")[0])
	p =ta_file["lev"][:]/100
	p_ind = np.where(p>=top)[0]
	return [len(p_ind), p, p_ind]

def get_lsm():
	lsm_file = nc.Dataset("/short/eg3/ab4502/erai_lsm.nc")
	lsm = np.squeeze(lsm_file.variables["lsm"][:])
	lsm_lon = np.squeeze(lsm_file.variables["longitude"][:])
	lsm_lat = np.squeeze(lsm_file.variables["latitude"][:])
	return [lsm,lsm_lon,lsm_lat]

def reform_lsm(lon,lat):
	[lsm,lsm_lon,lsm_lat] = get_lsm()
	lsm_new = np.empty(lsm.shape)

	lsm_lon[lsm_lon>=180] = lsm_lon[lsm_lon>=180]-360
	for i in np.arange(0,len(lat)):
		for j in np.arange(0,len(lon)):
			lsm_new[i,j] = lsm[lat[i]==lsm_lat, lon[j]==lsm_lon]
	return lsm_new

def format_dates(x):
	return dt.datetime.strftime(x,"%Y") + dt.datetime.strftime(x,"%m")

def get_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+"201201"+"*.nc")[0])
	lon = ta_file["lon"][:]
	lat = ta_file["lat"][:]
	return [lon,lat]

def get_lat_lon_inds(points,lon,lat):
	lsm_new = reform_lsm(lon,lat)
	x,y = np.meshgrid(lon,lat)
	x[lsm_new==0] = np.nan
	y[lsm_new==0] = np.nan
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
		

