import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd

from calc_param import get_dp

from metpy.calc import vertical_velocity_pressure as omega
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
		ta2d_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/spec/dewpt_scrn/"\
	+year+"/"+month+"/dewpt_scrn-an-spec-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])
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
	
#IF WANTING TO LOOP OVER TIME, CHANGE READ_BARRA TO READ ALL TIMES IN A RANGE, THEN LOOP OVER TIME DIMENSION WITHIN CALC_PARAM

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
