import gc
from dask.diagnostics import ProgressBar
import sys
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import xarray as xr

from metpy.calc import vertical_velocity_pressure as omega
import metpy.calc as mpcalc
from metpy.units import units

def read_barra(domain,times):
	#Open BARRA netcdf files and extract variables needed for a range of times and given
	# spatial domain
	#NOTE, currently this uses analysis files, with no time dimension length=1. For
	# use with forecast files (with time dimension length >1), will need to be changed.

	ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times,"hours",6)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times
	date_list = remove_corrupt_dates(date_list)
	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100, times[0])
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
	wap = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	vas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	tas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ta2d = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ps = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	wg10 = np.zeros(ps.shape)
	p_3d = np.moveaxis(np.tile(pres,[ta.shape[2],ta.shape[3],1]),2,0)

	for t in np.arange(0,len(date_list)):
		year = dt.datetime.strftime(date_list[t],"%Y")
		month =	dt.datetime.strftime(date_list[t],"%m")
		day = dt.datetime.strftime(date_list[t],"%d")
		hour = dt.datetime.strftime(date_list[t],"%H")
		#print(date_list[t])

		#Load BARRA analysis files
		ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+year+"/"+month+"/air_temp-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/geop_ht/"\
	+year+"/"+month+"/geop_ht-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		ua_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_ucmp/"\
	+year+"/"+month+"/wnd_ucmp-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		va_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/wnd_vcmp/"\
	+year+"/"+month+"/wnd_vcmp-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		w_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/vertical_wnd/"\
	+year+"/"+month+"/vertical_wnd-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		hur_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/relhum/"\
	+year+"/"+month+"/relhum-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		uas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/uwnd10m/"\
	+year+"/"+month+"/uwnd10m-an-spec-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		vas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/vwnd10m/"\
	+year+"/"+month+"/vwnd10m-an-spec-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		ta2d_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/slv/dewpt_scrn/"\
	+year+"/"+month+"/dewpt_scrn-an-slv-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		tas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/temp_scrn/"\
	+year+"/"+month+"/temp_scrn-an-spec-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
		ps_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/sfc_pres/"\
	+year+"/"+month+"/sfc_pres-an-spec-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]

		#Load data
		temp_ta = ta_file["air_temp"][p_ind,lat_ind,lon_ind] - 273.15
		temp_ua = ua_file["wnd_ucmp"][p_ind,lat_ind,lon_ind]
		temp_va = va_file["wnd_vcmp"][p_ind,lat_ind,lon_ind]
		temp_hgt = z_file["geop_ht"][p_ind,lat_ind,lon_ind]
		temp_hur = hur_file["relhum"][p_ind,lat_ind,lon_ind]
		temp_hur[temp_hur<0] = 0
		temp_hur[temp_hur>100] = 100
		temp_dp = get_dp(temp_ta,temp_hur)
		temp_wap = omega( w_file["vertical_wnd"][p_ind,lat_ind,lon_ind] * (units.metre / units.second),\
			p_3d * (units.hPa), \
			temp_ta * units.degC )
		uas[t,:,:] = uas_file["uwnd10m"][lat_ind,lon_ind]
		vas[t,:,:] = vas_file["vwnd10m"][lat_ind,lon_ind]
		tas[t,:,:] = tas_file["temp_scrn"][lat_ind,lon_ind] - 273.15
		ta2d[t,:,:] = ta2d_file["dewpt_scrn"][lat_ind,lon_ind] - 273.15
		ps[t,:,:] = ps_file["sfc_pres"][lat_ind,lon_ind]/100 

		#Flip pressure axes for compatibility with SHARPpy
		ta[t,:,:,:] = np.flipud(temp_ta)
		dp[t,:,:,:] = np.flipud(temp_dp)
		hur[t,:,:,:] = np.flipud(temp_hur)
		hgt[t,:,:,:] = np.flipud(temp_hgt)
		wap[t,:,:,:] = np.flipud(temp_wap)
		ua[t,:,:,:] = np.flipud(temp_ua)
		va[t,:,:,:] = np.flipud(temp_va)


		#Load forecast data
		fc_year = dt.datetime.strftime(date_list[t] - dt.timedelta(hours=6),"%Y")
		fc_month = dt.datetime.strftime(date_list[t] - dt.timedelta(hours=6),"%m")
		fc_day = dt.datetime.strftime(date_list[t] - dt.timedelta(hours=6),"%d")
		fc_hour = dt.datetime.strftime(date_list[t]- dt.timedelta(hours=6),"%H")
		#try:
		wg10_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/forecast/spec/"\
				+"max_wndgust10m/"+fc_year+"/"+fc_month+"/max_wndgust10m-fc-spec-PT1H-BARRA_R-*-"\
				+fc_year+fc_month+fc_day+"T"+fc_hour+"*.nc")[0])
		fc_times = nc.num2date(wg10_file["time"][:], wg10_file["time"].units)
		an_times = nc.num2date(ps_file["time"][:], ps_file["time"].units)
		wg10[t] = wg10_file.variables["max_wndgust10m"]\
			[np.where(np.array(an_times) == np.array(fc_times))[0][0],lat_ind,lon_ind]
		wg10_file.close()
		#except:
		#wg10[t][:] = np.nan

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();ps_file.close();tas_file.close();ta2d_file.close();w_file.close()
		
	p = np.flipud(pres)

	return [ta,dp,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list]
	
def read_barra_fc(domain,times,mslp=False):
	#If mslp=True, then replace ps with mslp

	#Open BARRA netcdf files and extract variables needed for a range of times and given
	# spatial domain
	#NOTE that 10 m v and v winds are not being de-staggered. There is therefore a 0.05 degree
	# mis-match between the 10 m u and v

	ref = dt.datetime(1970,1,1,0,0,0)
	if len(times) > 1:
		date_list = date_seq(times,"hours",1)
	else:
		date_list = times

	#Get time-invariant pressure and spatial info for first time in "times"
	no_p, pres_full, p_ind = get_pressure(100, times[0])
	pres = pres_full[p_ind]
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
	wap = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	vas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	tas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ta2d = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ps = np.zeros((len(date_list),len(lat_ind),len(lon_ind)))
	p_3d = np.moveaxis(np.tile(pres,[ta.shape[2],ta.shape[3],1]),2,0)
	wg10 = np.zeros(ps.shape)

	spec_fnames = []
	slv_fnames = []
	prs_fnames = []
	an_times = np.array([0,6,12,18])
	for t in np.arange(len(date_list)):

		temp_year = date_list[t].year
		temp_month = date_list[t].month
		temp_day = date_list[t].day

		diff = date_list[t].hour - an_times
		diff[diff<0] = 24
		temp_hour = an_times[np.argmin(diff)]

		temp_time = dt.datetime(temp_year, temp_month, temp_day, temp_hour)

		if np.min(diff) == 0:
			temp_time = temp_time + dt.timedelta(hours=-6)

		if temp_time.year >= 1990:

			#EDIT TO GET DATA FROM THE "PROD" DIRECTORY
			temp_fname_prs = glob.glob("/g/data/ma05/prod/BARRA_R/v1/forecast/prs/air_temp/"+\
				temp_time.strftime("%Y")+"/"+temp_time.strftime("%m")+\
				"/air_temp-fc-prs-PT1H-BARRA_R-v1*"+temp_time.strftime("%Y")+\
				temp_time.strftime("%m")+temp_time.strftime("%d")+"T"+\
				temp_time.strftime("%H")+"*.sub.nc")[0]
			temp_fname_slv = glob.glob("/g/data/ma05/BARRA_R/v1/forecast/slv/dewpt_scrn/"+\
				temp_time.strftime("%Y")+"/"+temp_time.strftime("%m")+\
				"/dewpt_scrn-fc-slv-PT1H-BARRA_R-v1*"+temp_time.strftime("%Y")+\
				temp_time.strftime("%m")+temp_time.strftime("%d")+"T"+\
				temp_time.strftime("%H")+"*.sub.nc")[0]
			temp_fname_spec = glob.glob("/g/data/ma05/BARRA_R/v1/forecast/spec/uwnd10m/"+\
				temp_time.strftime("%Y")+"/"+temp_time.strftime("%m")+\
				"/uwnd10m-fc-spec-PT1H-BARRA_R-v1*"+temp_time.strftime("%Y")+\
				temp_time.strftime("%m")+temp_time.strftime("%d")+"T"+\
				temp_time.strftime("%H")+"*.sub.nc")[0]
			if temp_fname_prs not in prs_fnames:
				prs_fnames.append(temp_fname_prs)
				slv_fnames.append(temp_fname_slv)
				spec_fnames.append(temp_fname_spec)

	for t in np.arange(0,len(prs_fnames)):
		#Load BARRA analysis files
		ta_file = nc.Dataset(prs_fnames[t])
		z_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","geop_ht").replace("v1.1","*").replace("v1","*").replace(".nc","*"))[0])
		ua_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","wnd_ucmp").replace("v1.1","*").replace("v1","*"))[0])
		va_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","wnd_vcmp").replace("v1.1","*").replace("v1","*"))[0])
		try:
			w_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","vertical_wnd").\
				replace("v1.1","*").replace("v1","*"))[0])
		except:
			try:
				w_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","vertical_wnd").\
					replace("v1.1","*").replace("v1","*").replace("prod/",""))[0])
			except:
				raise OSError("W file not found")
		hur_file = nc.Dataset(glob.glob(prs_fnames[t].replace("air_temp","relhum").replace("v1.1","*").replace("v1","*"))[0])
	
		uas_file = nc.Dataset(spec_fnames[t])
		vas_file = nc.Dataset(spec_fnames[t].replace("uwnd10m","vwnd10m"))
		tas_file = nc.Dataset(spec_fnames[t].replace("uwnd10m","temp_scrn"))
		if mslp:
		    ps_file = nc.Dataset(spec_fnames[t].replace("uwnd10m","mslp"))
		else:
		    ps_file = nc.Dataset(spec_fnames[t].replace("uwnd10m","sfc_pres"))
		wg_file = nc.Dataset(spec_fnames[t].replace("uwnd10m","max_wndgust10m"))

		ta2d_file = nc.Dataset(slv_fnames[t])

		#Get times to load in from file
		times = nc.num2date(ta_file["time"][:], ta_file["time"].units)

		#Check the vertical coordinates for the time step. In some cases they will be different
		# (e.g. for 2010-01-01 00:00, which exists in the 2009-12-31 18:00 file, there will be
		# 16 levels below 100 hPa, whereas 22 are expected)
		if pres_full.shape[0] == ta_file["air_temp"].shape[1]:

			#Load data
			temp_ta = ta_file["air_temp"][np.in1d(times, date_list), ta_file["pressure"][:] >= 100,\
				lat_ind,lon_ind] - 273.15
			temp_ua = ua_file["wnd_ucmp"][np.in1d(times, date_list), ua_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_va = va_file["wnd_vcmp"][np.in1d(times, date_list),va_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_hgt = z_file["geop_ht"][np.in1d(times, date_list),z_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_hur = hur_file["relhum"][np.in1d(times, date_list),hur_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_hur[temp_hur<0] = 0
			temp_hur[temp_hur>100] = 100
			temp_dp = get_dp(temp_ta,temp_hur)
			temp_wap = omega( w_file["vertical_wnd"][np.in1d(times, date_list),\
				(np.in1d(w_file["pressure"][:].astype(np.float32), \
					pres_full.astype(np.float32))) \
				& (w_file["pressure"][:] >= 100),\
				lat_ind,lon_ind] * (units.metre / units.second),\
				p_3d * (units.hPa), \
				temp_ta * units.degC )

			#Flip pressure axes for compatibility with SHARPpy
			ta[np.in1d(date_list, times),:,:,:] = np.flip(temp_ta, axis=1)
			dp[np.in1d(date_list, times),:,:,:] = np.flip(temp_dp, axis=1)
			hur[np.in1d(date_list, times),:,:,:] = np.flip(temp_hur, axis=1)
			hgt[np.in1d(date_list, times),:,:,:] = np.flip(temp_hgt, axis=1)
			wap[np.in1d(date_list, times),:,:,:] = np.flip(temp_wap, axis=1)
			ua[np.in1d(date_list, times),:,:,:] = np.flip(temp_ua, axis=1)
			va[np.in1d(date_list, times),:,:,:] = np.flip(temp_va, axis=1)

			ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();w_file.close()

		else:
			ta[np.in1d(date_list, times),:,:,:] = np.nan
			dp[np.in1d(date_list, times),:,:,:] = np.nan
			hur[np.in1d(date_list, times),:,:,:] = np.nan
			hgt[np.in1d(date_list, times),:,:,:] = np.nan
			wap[np.in1d(date_list, times),:,:,:] = np.nan
			ua[np.in1d(date_list, times),:,:,:] = np.nan
			va[np.in1d(date_list, times),:,:,:] = np.nan

		
		uas[np.in1d(date_list, times),:,:] = uas_file["uwnd10m"][np.in1d(times, date_list),lat_ind,lon_ind]
		vas[np.in1d(date_list, times),:,:] = vas_file["vwnd10m"][np.in1d(times, date_list),lat_ind,lon_ind]
		tas[np.in1d(date_list, times),:,:] = tas_file["temp_scrn"][np.in1d(times, date_list),lat_ind,lon_ind] - 273.15
		ta2d[np.in1d(date_list, times),:,:] = ta2d_file["dewpt_scrn"][np.in1d(times, date_list),lat_ind,lon_ind] - 273.15
		if mslp:
		    ps[np.in1d(date_list, times),:,:] = ps_file["mslp"][np.in1d(times, date_list),lat_ind,lon_ind]/100 
		else:
		    ps[np.in1d(date_list, times),:,:] = ps_file["sfc_pres"][np.in1d(times, date_list),lat_ind,lon_ind]/100 
		wg10[np.in1d(date_list, times),:,:] = wg_file["max_wndgust10m"][np.in1d(times, date_list),lat_ind,lon_ind]

		uas_file.close();vas_file.close();ps_file.close();tas_file.close();ta2d_file.close()
		
	p = np.flipud(pres)

	return [ta,dp,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list]
	
def read_barra_xarray():

	start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275

	#Load temperature data and combine
	ta = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/prs/air_temp/1990/01/*.nc", concat_dim="time").\
		sel(latitude=slice(start_lat, end_lat), longitude=slice(start_lon, end_lon))
	ta = ta.sel(pressure=(ta.pressure>=100))
	p = ta.pressure
	tas = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/spec/temp_scrn/1990/01/*.nc", concat_dim="time").\
		rename({"temp_scrn":"air_temp"}).sel(latitude=slice(start_lat, end_lat), \
		longitude=slice(start_lon, end_lon))
	ta = ta.reset_index("pressure").assign_coords(pressure=(np.arange(1,len(ta["pressure"])+1)))
	sfc_ta = xr.auto_combine([tas.expand_dims({"pressure":[0]}).drop("height"), ta.drop("pressure_")], "pressure")

	#Create 4d pressure array from surface pressure and pressure level coordinates
	p4d = np.tile(np.moveaxis(np.tile(p,[len(ta.latitude),len(ta.longitude),1]),[0,1,2],[1,2,0])[np.newaxis],\
		(len(ta.time),1,1,1) )
	pds = xr.Dataset({"p3d": (["time", "pressure", "latitude", "longitude"], p4d)},\
		coords={"time":ta.time, "pressure":ta.pressure, "latitude":ta.latitude, "longitude":ta.longitude})
	ps = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/spec/sfc_pres/1990/01/*.nc", concat_dim="time").\
		rename({"sfc_pres":"p3d"}).sel(latitude=slice(start_lat, end_lat), \
		longitude=slice(start_lon, end_lon))
	sfc_p_3d = xr.auto_combine([ps.expand_dims({"pressure":[0]}), pds], "pressure")

	#Load relative humidity on pressure levels. Load dewpoint on the surface. Convert both to mixing ratio and 
	# combine
	#Get mixing ratio on pressure levels...
	hur = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/prs/relhum/1990/01/*.nc", concat_dim="time").\
		sel(latitude=slice(start_lat, end_lat), longitude=slice(start_lon, end_lon))
	hur = hur.sel(pressure=(hur.pressure>=100))
	hur = hur.reset_index("pressure").assign_coords(pressure=(np.arange(1,len(hur["pressure"])+1)))
	sat_vap_pres = 6.112 * np.exp( 17.67 * (ta.air_temp - 273.15) / (ta.air_temp - 29.65) ) 
	q = (hur.relhum / 100 ) * (0.62198 * sat_vap_pres / (pds.p3d - sat_vap_pres))
	#Get mixing ratio on sfc
	sdp = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/slv/dewpt_scrn/1990/01/*.nc", concat_dim="time").\
		sel(latitude=slice(start_lat, end_lat), \
		longitude=slice(start_lon, end_lon))
	sat_vap_pres_dp = 6.112 * np.exp( 17.67 * (sdp.dewpt_scrn - 273.15) / (sdp.dewpt_scrn - 29.65) ) 
	sat_vap_pres_ta = 6.112 * np.exp( 17.67 * (tas.air_temp - 273.15) / (tas.air_temp - 29.65) ) 
	temp_sfc_hur = sat_vap_pres_dp / sat_vap_pres_ta
	q_sfc = (temp_sfc_hur) * (0.62198 * sat_vap_pres_ta / (ps.p3d - sat_vap_pres_ta))
	sfc_q = xr.concat([q_sfc.expand_dims({"pressure":[0]}).drop("height"), q.drop("pressure_")], dim="pressure") 

	hgt = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/prs/geop_ht/1990/01/*.nc", concat_dim="time").\
		sel(latitude=slice(start_lat, end_lat), longitude=slice(start_lon, end_lon))
	hgt = hgt.sel(pressure=(hgt.pressure>=100))
	ter = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/static/topog-an-slv-PT0H-BARRA_R-v1.nc").\
		rename({"topog":"geop_ht"}).sel(latitude=slice(start_lat, end_lat), \
		longitude=slice(start_lon, end_lon))
	
	hgt = hgt.reset_index("pressure").reset_index("time").assign_coords(pressure=(np.arange(1,len(hgt["pressure"])+1)))
	sfc_hgt = xr.concat([ter.expand_dims({"time":len(hgt.time)}).assign_coords(time=hgt.time).expand_dims({"pressure":[0]}), hgt.drop("pressure_")], "pressure")

	ds = xr.Dataset({"p3d": (["time", "pressure", "latitude", "longitude"], sfc_p_3d.p3d),\
		"ta": (["time", "pressure", "latitude", "longitude"], sfc_ta.air_temp),\
		"hgt": (["time", "pressure", "latitude", "longitude"], sfc_hgt.geop_ht),\
		"q": (["time", "pressure", "latitude", "longitude"], sfc_q),\
		"hgt": (["time", "pressure", "latitude", "longitude"], sfc_hgt.geop_ht)})
	#Now have to work out how to sort by descending pressure...


def read_barra_points(points,times):

	#NOTE might have to change to read forecast (hourly) data instead of analysis

	# points = [(x1,y1),(x2,y2),...,(xn,yn)]
	# times = [start_time, end_time]

	#Open BARRA netcdf files and extract variables needed for a multiple files/times given set of spatial points
	
	ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times,"hours",6)
	date_list = remove_corrupt_dates(date_list)

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

		#Some BARRA files have different vertical resolution
		p_ind = np.in1d(pres,ta_file.variables["pressure"][:])
		
		times = ta_file["time"][:]
		
		for point in np.arange(0,len(points)):

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

def get_pressure(top, date):
	#NOTE THAT WE GET THE PRESSURE COORDINATES OF THE FIRST DATE
	year = dt.datetime.strftime(date,"%Y")
	month =	dt.datetime.strftime(date,"%m")
	day = dt.datetime.strftime(date,"%d")
	hour = dt.datetime.strftime(date,"%H")

	#Load BARRA analysis files
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/prod/BARRA_R/v1/analysis/prs/air_temp/"\
+year+"/"+month+"/air_temp-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])

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

def get_mask(lon,lat):

	#Take 1d lat lon data from an already-loaded BARRA-R domain (e.g. sa_small or aus) and return a land-sea mask
	nat_lon,nat_lat = get_lat_lon()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[0]) & (nat_lat <= lat[-1]))[0]
	lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
	lsm_domain = lsm[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
	
	return lsm_domain

def remove_corrupt_dates(date_list):
	corrupt_dates = [dt.datetime(2014,11,22,6,0)]
	date_list = np.array(date_list)
	for i in np.arange(0,len(corrupt_dates)):
		date_list = date_list[~(date_list==corrupt_dates[i])]
	return date_list

def to_points():

	#Read in all BARRA netcdf convective parameters, and extract point data.
	#(Hopefuly) a faster version of event_analysis.load_netcdf_points_mf()

	#Read netcdf data
	from dask.diagnostics import ProgressBar
	ProgressBar().register()
	#f=xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc*.nc", parallel=True)
	f=xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/barra_fc/barra_fc_20150101_20150131.nc",\
		chunks={"lat":100, "lon":100, "time":100})

	#JUST WANTING TO APPEND A COUPLE OF NEW VARIABLES TO THE DATAFRAME
	#f=f[["Vprime", "wbz"]]

	#Setup lsm
	lat = f.coords.get("lat").values
	lon = f.coords.get("lon").values
	lsm = get_mask(lon,lat)
	x,y = np.meshgrid(lon,lat)
	x[lsm==0] = np.nan
	y[lsm==0] = np.nan

	#Load info for the 35 AWS stations around Australia
	loc_id, points = get_aus_stn_info()

	dist_lon = []
	dist_lat = []
	for i in np.arange(len(loc_id)):

		dist = np.sqrt(np.square(x-points[i][0]) + \
			np.square(y-points[i][1]))
		temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
		dist_lon.append(temp_lon)
		dist_lat.append(temp_lat)

	df = f.isel(lat = xr.DataArray(dist_lat, dims="points"), \
			lon = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()

	df = df.reset_index()

	for p in np.arange(len(loc_id)):
		df.loc[df.points==p,"loc_id"] = loc_id[p]

	df.drop("points",axis=1).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_wrfpython_aus_1979_2017.pkl")
	#df = df.drop("points",axis=1)
	#df_orig = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")
	#df_new = pd.merge(df_orig, df[["time","loc_id","wbz","Vprime"]], on=["time","loc_id"])
	#df_new.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")

def to_points_loop_rad(loc_id,points,fname,start_year,end_year,rad=50,lsm=True,\
		variables=False,pb=False):

	#Register progress bar for xarray if desired
	if pb:
		ProgressBar().register()

	#Create monthly dates from start_year to end_year to iterate over
	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	#Initialise dataframe for point data
	max_df = pd.DataFrame()
	mean_df = pd.DataFrame()

	#For each month from start_year to end_year
	for t in np.arange(len(dates)):
		#Read convective diagnostics from eg3
		print(dates[t])
		f=xr.open_dataset(glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/"+\
			"barra_fc/barra_fc_"+dates[t].strftime("%Y%m")+"*.nc")[0],\
			chunks={"lat":100, "lon":100, "time":100}, engine="h5netcdf")

		#For each location (lat/lon pairing), get the distance (km) to each BARRA grid point
		lat = f.coords.get("lat").values
		lon = f.coords.get("lon").values
		x,y = np.meshgrid(lon,lat)
		if lsm:
			mask = get_mask(lon,lat)
			x[mask==0] = np.nan
			y[mask==0] = np.nan
		dist_km = []
		for i in np.arange(len(loc_id)):
			dist_km.append(latlon_dist(points[i][1], points[i][0], y, x) )

		#Subset netcdf data to a list of variables, if available
		try:
			f=f[variables]
		except:
			pass

		#Subset netcdf data based on lat and lon, and convert to a dataframe
		#Get all points (regardless of LSM) within 100 km radius
		max_temp_df = pd.DataFrame()
		mean_temp_df = pd.DataFrame()
		for i in np.arange(len(loc_id)):
			a,b = np.where(dist_km[i] <= rad)
			subset = f.isel_points("points",lat=a, lon=b).persist()
			max_point_df = subset.max("points").to_dataframe()
			mean_point_df = subset.mean("points").to_dataframe()
			max_point_df["points"] = i
			mean_point_df["points"] = i
			max_temp_df = pd.concat([max_temp_df, max_point_df], axis=0)
			mean_temp_df = pd.concat([mean_temp_df, mean_point_df], axis=0)

		#Manipulate dataframe for nice output
		max_temp_df = max_temp_df.reset_index()
		for p in np.arange(len(loc_id)):
			max_temp_df.loc[max_temp_df.points==p,"loc_id"] = loc_id[p]
		max_temp_df = max_temp_df.drop("points",axis=1)
		max_df = pd.concat([max_df, max_temp_df])

		mean_temp_df = mean_temp_df.reset_index()
		for p in np.arange(len(loc_id)):
			mean_temp_df.loc[mean_temp_df.points==p,"loc_id"] = loc_id[p]
		mean_temp_df = mean_temp_df.drop("points",axis=1)
		mean_df = pd.concat([mean_df, mean_temp_df])

		#Clean up
		f.close()
		gc.collect()

	#Save point output to disk
	max_df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+"_max.pkl")
	mean_df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+"_mean.pkl")

def to_points_loop(loc_id,points,fname,start_year,end_year,variables=False):

	#As in to_points(), but by looping over monthly data
	from dask.diagnostics import ProgressBar
	import gc
	ProgressBar().register()

	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		f=xr.open_dataset(glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/"+\
			"barra_fc/barra_fc_"+dates[t].strftime("%Y%m")+"*.nc")[0],\
			chunks={"lat":100, "lon":100, "time":100}, engine="h5netcdf")

		#Setup lsm
		lat = f.coords.get("lat").values
		lon = f.coords.get("lon").values
		lsm = get_mask(lon,lat)
		x,y = np.meshgrid(lon,lat)
		x[lsm==0] = np.nan
		y[lsm==0] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		try:
			f=f[variables]
		except:
			pass

		temp_df = f.isel(lat = xr.DataArray(dist_lat, dims="points"), \
				lon = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()

		temp_df = temp_df.reset_index()

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop("points",axis=1)
		df = pd.concat([df, temp_df])
		f.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def to_points_loop_wind_dir(loc_id,points,fname,start_year,end_year):

	#As in to_points_loop(), but just for 10 m wind direction, from the ma07 directory
	from dask.diagnostics import ProgressBar
	import gc
	#ProgressBar().register()

	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	lsm = xr.open_dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc")

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		year = dt.datetime.strftime(dates[t],"%Y")
		month =	dt.datetime.strftime(dates[t],"%m")
		av_u_file = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/slv/av_uwnd10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")
		av_v_file = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/slv/av_vwnd10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")
		u_file = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/spec/uwnd10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")
		v_file = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/spec/vwnd10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")
		av_u_int = av_u_file.av_uwnd10m.interp({"latitude":lsm.latitude.values, \
			"longitude":lsm.longitude.values})
		av_v_int = av_v_file.av_vwnd10m.interp({"latitude":lsm.latitude.values, \
			"longitude":lsm.longitude.values})
		u_int = u_file.uwnd10m.interp({"latitude":lsm.latitude.values, \
			"longitude":lsm.longitude.values})
		v_int = v_file.vwnd10m.interp({"latitude":lsm.latitude.values, \
			"longitude":lsm.longitude.values})
		av_wd = 180 + ( 180/np.pi ) * \
			(np.arctan2(av_u_int, av_v_int))
		wd = 180 + ( 180/np.pi ) * \
			(np.arctan2(u_int, v_int))

		#Setup lsm
		lat = lsm.coords.get("latitude").values
		lon = lsm.coords.get("longitude").values
		x,y = np.meshgrid(lon,lat)
		x[lsm.lnd_mask==0] = np.nan
		y[lsm.lnd_mask==0] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		av_temp_df = av_wd.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
			longitude = xr.DataArray(dist_lon, dims="points")).persist().\
			to_dataframe("av_wd")
		temp_df = wd.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
			longitude = xr.DataArray(dist_lon, dims="points")).persist().\
			to_dataframe("wd")

		av_temp_df = av_temp_df.reset_index()
		temp_df = temp_df.reset_index()

		temp_df["av_wd"] = av_temp_df["av_wd"]

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop(["points", "height", \
			"forecast_period", "forecast_reference_time"],axis=1)
		df = pd.concat([df, temp_df])
		u_file.close()
		v_file.close()
		av_u_file.close()
		av_v_file.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def to_points_loop_verticalwnd(loc_id,points,fname,start_year,end_year):

	#As in to_points_loop(), but just for vertical velocity at 700 hPa, from the ma07 directory
	from dask.diagnostics import ProgressBar
	import gc
	ProgressBar().register()

	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	lsm = xr.open_dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc")

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		year = dt.datetime.strftime(dates[t],"%Y")
		month =	dt.datetime.strftime(dates[t],"%m")
		f = xr.open_mfdataset("/g/data/ma05/BARRA_R/v1/forecast/prs/vertical_wnd/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")

		#Setup lsm
		lat = f.coords.get("latitude").values
		lon = f.coords.get("longitude").values
		x,y = np.meshgrid(lon,lat)
		x[lsm.lnd_mask==0] = np.nan
		y[lsm.lnd_mask==0] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		if f.pressure.shape[0] > 37:
			temp_df = f.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
				longitude = xr.DataArray(dist_lon, dims="points"), \
				pressure = [56,57]).persist().to_dataframe().dropna()
		else:
			temp_df = f.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
				longitude = xr.DataArray(dist_lon, dims="points"), \
				pressure = 28).persist().to_dataframe()

		temp_df = temp_df.reset_index()

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop(["points", "pressure", "latitude_longitude",\
			"forecast_period", "forecast_reference_time"],axis=1)
		df = pd.concat([df, temp_df])
		f.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def to_points_loop_ctk(loc_id,points,fname,start_year,end_year,variable):

	#As in to_points_loop() but for Nathan's CTK data
	#loc_id is a list of strings with names for each point
	#points is a list of lat-lon pairings, same length as loc_id
	#fname is the name of the pickle output file
	#start_year and end_year are ints
	#variable is a string, which must be a key within the "codes" dictionay below

	from dask.diagnostics import ProgressBar
	import gc
	import warnings
	warnings.simplefilter("ignore")

	#Create monthly list of dates from start year to end year
	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	#Initialise output dataframe
	df = pd.DataFrame()

	#Explicitly set grib codes for convective ctk parameters. The values are the parameter codes within
	# the grib file, the key is just what you want to call the output
	codes = {"ml_cape":"39", "ml_cin":"41", "eff_cape":"99", "eff_cin":"101",\
			"s06":"132"}

	#Load BARRA-R LSM
	lsm = nc.Dataset("/g/data/ma05/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").\
		variables["lnd_mask"][:]

	#Loop over monthly dates, read netcdf data and extract point time series
	for t in np.arange(len(dates)):

		#Load grib files
		print(dates[t])
		files = glob.glob("/g/data/ma05/BARRA_R/v1/products/ctk/"+
			dates[t].strftime("%Y")+"/"+dates[t].strftime("%m")+"/*.grib2")
		files.sort()
		f=xr.open_mfdataset(files, engine="cfgrib", backend_kwargs={"read_keys":["parameterName"], \
			"filter_by_keys":{"parameterName":codes[variable]}})

		#Use LSM
		lat = f.latitude.data
		lon = f.longitude.data
		x,y = np.meshgrid(lon,lat)
		x[lsm==0] = np.nan
		y[lsm==0] = np.nan

		#For each point of interest, get the lat/lon index
		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):
			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		#Extract time series from xarray DataSet
		temp_df = f.unknown.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
				longitude = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()

		#Clean up time series dataframe
		temp_df = temp_df.reset_index()
		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]
		temp_df = temp_df.drop("points",axis=1).rename(columns={"unknown":variable})
		df = pd.concat([df, temp_df])
		f.close()
		gc.collect()

	#Save time series output
	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def get_aus_stn_info():
	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	
	df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
		names=names, header=0)
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}
	df = df.replace({"stn_name":renames})
	points = [(df.lon.iloc[i], df.lat.iloc[i]) for i in np.arange(df.shape[0])]
	return [df.stn_name.values,points]

def latlon_dist(lat, lon, lats, lons):

	#Calculate great circle distance (Harversine) between a lat lon point (lat, lon) and a list of lat lon
	# points (lats, lons)

	R = 6373.0

	lat1 = np.deg2rad(lat)
	lon1 = np.deg2rad(lon)
	lat2 = np.deg2rad(lats)
	lon2 = np.deg2rad(lons)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

	return (R * c)

def get_dp(ta,hur,dp_mask=True):
	#Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
	#Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
	#For points where RH is zero, set the dew point temperature to -85 deg C
	#EDIT: Leave points where RH is zero masked as NaNs. Hanlde them after creating a SHARPpy object (by 
	# making the missing dp equal to the mean of the above and below layers)
	#EDIT: Replace with metpy code

	dp = np.array(mpcalc.dewpoint_rh(ta * units.degC, hur * units.percent))

	if dp_mask:
		return dp
	else:
		dp = np.array(dp)
		dp[np.isnan(dp)] = -85.
		return dp

if __name__ == "__main__":

	if len(sys.argv) > 1:
		start_year = int(sys.argv[1])
		end_year = int(sys.argv[2])
	if len(sys.argv) > 3:
		variable = sys.argv[3]
	

	loc_id, points = get_aus_stn_info()

	to_points_loop_wind_dir(loc_id,points,"barra_wind_dir_"+str(start_year)+"_"+str(end_year),\
			start_year,end_year)
