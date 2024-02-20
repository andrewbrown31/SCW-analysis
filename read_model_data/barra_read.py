from tqdm import tqdm
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import xarray as xr
from metpy.units import units
from utils import get_dp

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

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list]
	
def read_barra_fc(domain,times,mslp=False):
	#If mslp=True, then replace ps with mslp

	#Open BARRA netcdf files and extract variables needed for a range of times and given
	# spatial domain
	#Xarray is used to de-stagger the U and V grids

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

			temp_fname_prs = glob.glob("/g/data/cj37/BARRA/BARRA_R/v1/forecast/prs/air_temp/"+\
				temp_time.strftime("%Y")+"/"+temp_time.strftime("%m")+\
				"/air_temp-fc-prs-PT1H-BARRA_R-v1*"+temp_time.strftime("%Y")+\
				temp_time.strftime("%m")+temp_time.strftime("%d")+"T"+\
				temp_time.strftime("%H")+"*.sub.nc")[0]
			temp_fname_slv = glob.glob("/g/data/cj37/BARRA/BARRA_R/v1/forecast/slv/dewpt_scrn/"+\
				temp_time.strftime("%Y")+"/"+temp_time.strftime("%m")+\
				"/dewpt_scrn-fc-slv-PT1H-BARRA_R-v1*"+temp_time.strftime("%Y")+\
				temp_time.strftime("%m")+temp_time.strftime("%d")+"T"+\
				temp_time.strftime("%H")+"*.sub.nc")[0]
			temp_fname_spec = glob.glob("/g/data/cj37/BARRA/BARRA_R/v1/forecast/spec/uwnd10m/"+\
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
		ua_file = xr.open_dataset(glob.glob(prs_fnames[t].replace("air_temp","wnd_ucmp").replace("v1.1","*").replace("v1","*"))[0])["wnd_ucmp"]
		va_file = xr.open_dataset(glob.glob(prs_fnames[t].replace("air_temp","wnd_vcmp").replace("v1.1","*").replace("v1","*"))[0])["wnd_vcmp"]
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
	
		uas_file = xr.open_dataset(spec_fnames[t])["uwnd10m"]
		vas_file = xr.open_dataset(spec_fnames[t].replace("uwnd10m","vwnd10m"))["vwnd10m"]
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
			temp_ua = ua_file.isel({"longitude":np.insert(lon_ind,[0,len(lon_ind)],[lon_ind[0]-1, lon_ind[-1]+1]),\
				"latitude":np.insert(lat_ind,[0,len(lat_ind)],[lat_ind[0]-1, lat_ind[-1]+1]),\
				"pressure":ua_file["pressure"].values>=100, "time":np.in1d(times, date_list)}).\
				interp({"longitude":ta_file["longitude"][lon_ind], "latitude":ta_file["latitude"][lat_ind]}).values
			temp_va = va_file.isel({"longitude":np.insert(lon_ind,[0,len(lon_ind)],[lon_ind[0]-1, lon_ind[-1]+1]),\
				"latitude":np.insert(lat_ind,[0,len(lat_ind)],[lat_ind[0]-1, lat_ind[-1]+1]),\
				"pressure":va_file["pressure"].values>=100, "time":np.in1d(times, date_list)}).\
				interp({"longitude":ta_file["longitude"][lon_ind], "latitude":ta_file["latitude"][lat_ind]}).values
			temp_hgt = z_file["geop_ht"][np.in1d(times, date_list),z_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_hur = hur_file["relhum"][np.in1d(times, date_list),hur_file["pressure"][:] >= 100,lat_ind,lon_ind]
			temp_hur[temp_hur<0] = 0
			temp_hur[temp_hur>100] = 100
			temp_dp = get_dp(temp_ta,temp_hur)

			#Flip pressure axes for compatibility with SHARPpy
			ta[np.in1d(date_list, times),:,:,:] = np.flip(temp_ta, axis=1)
			dp[np.in1d(date_list, times),:,:,:] = np.flip(temp_dp, axis=1)
			hur[np.in1d(date_list, times),:,:,:] = np.flip(temp_hur, axis=1)
			hgt[np.in1d(date_list, times),:,:,:] = np.flip(temp_hgt, axis=1)
			ua[np.in1d(date_list, times),:,:,:] = np.flip(temp_ua, axis=1)
			va[np.in1d(date_list, times),:,:,:] = np.flip(temp_va, axis=1)

			ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();w_file.close()

		else:
			ta[np.in1d(date_list, times),:,:,:] = np.nan
			dp[np.in1d(date_list, times),:,:,:] = np.nan
			hur[np.in1d(date_list, times),:,:,:] = np.nan
			hgt[np.in1d(date_list, times),:,:,:] = np.nan
			ua[np.in1d(date_list, times),:,:,:] = np.nan
			va[np.in1d(date_list, times),:,:,:] = np.nan

		
		uas[np.in1d(date_list, times),:,:] = uas_file.isel({"longitude":np.insert(lon_ind,[0,len(lon_ind)],[lon_ind[0]-1, lon_ind[-1]+1]),\
				"latitude":np.insert(lat_ind,[0,len(lat_ind)],[lat_ind[0]-1, lat_ind[-1]+1]),\
				"time":np.in1d(times, date_list)}).\
				interp({"longitude":tas_file["longitude"][lon_ind], "latitude":tas_file["latitude"][lat_ind]}).values
		vas[np.in1d(date_list, times),:,:] = vas_file.isel({"longitude":np.insert(lon_ind,[0,len(lon_ind)],[lon_ind[0]-1, lon_ind[-1]+1]),\
				"latitude":np.insert(lat_ind,[0,len(lat_ind)],[lat_ind[0]-1, lat_ind[-1]+1]),\
				"time":np.in1d(times, date_list)}).\
				interp({"longitude":tas_file["longitude"][lon_ind], "latitude":tas_file["latitude"][lat_ind]}).values
		tas[np.in1d(date_list, times),:,:] = tas_file["temp_scrn"][np.in1d(times, date_list),lat_ind,lon_ind] - 273.15
		ta2d[np.in1d(date_list, times),:,:] = ta2d_file["dewpt_scrn"][np.in1d(times, date_list),lat_ind,lon_ind] - 273.15
		if mslp:
		    ps[np.in1d(date_list, times),:,:] = ps_file["mslp"][np.in1d(times, date_list),lat_ind,lon_ind]/100 
		else:
		    ps[np.in1d(date_list, times),:,:] = ps_file["sfc_pres"][np.in1d(times, date_list),lat_ind,lon_ind]/100 
		wg10[np.in1d(date_list, times),:,:] = wg_file["max_wndgust10m"][np.in1d(times, date_list),lat_ind,lon_ind]

		uas_file.close();vas_file.close();ps_file.close();tas_file.close();ta2d_file.close()
		
	p = np.flipud(pres)

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list]

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
	ta_file = nc.Dataset(glob.glob("/g/data/cj37/BARRA/BARRA_R/v1/analysis/prs/air_temp/"\
+year+"/"+month+"/air_temp-an-prs-PT0H-BARRA_R-v1*"+year+month+day+"T"+hour+"*.nc")[0])

	p =ta_file["pressure"][:]
	p_ind = np.where(p>=top)[0]
	return [len(p_ind), p, p_ind]

def get_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/cj37/BARRA/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2012"+"/"+"12"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2012"+"12"+"01"+"T"+"00"+"*.nc")[0])
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def get_lat_lon_inds(points,lon,lat):
	lsm = nc.Dataset("/g/data/cj37/BARRA/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
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
	terrain_file = nc.Dataset("/g/data/cj37/BARRA/BARRA_R/v1/static/topog-an-slv-PT0H-BARRA_R-v1.nc")
	terrain = terrain_file.variables["topog"][lat_ind,lon_ind]
	terrain_file.close()
	return terrain

def get_mask(lon,lat):

	#Take 1d lat lon data from an already-loaded BARRA-R domain (e.g. sa_small or aus) and return a land-sea mask
	nat_lon,nat_lat = get_lat_lon()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[0]) & (nat_lat <= lat[-1]))[0]
	lsm = nc.Dataset("/g/data/cj37/BARRA/BARRA_R/v1/static/lnd_mask-an-slv-PT0H-BARRA_R-v1.nc").variables["lnd_mask"][:]
	lsm_domain = lsm[lat_ind[0]:lat_ind[-1]+1,lon_ind[0]:lon_ind[-1]+1]
	
	return lsm_domain

def remove_corrupt_dates(date_list):
	corrupt_dates = [dt.datetime(2014,11,22,6,0)]
	date_list = np.array(date_list)
	for i in np.arange(0,len(corrupt_dates)):
		date_list = date_list[~(date_list==corrupt_dates[i])]
	return date_list

def fix_wg_spikes(da_vals):

	'''
	Take a lat, lon, time numpy array of daily maximum max_wndgust10m, and identify/smooth wind gust spikes
	Identify spikes by considering adjacent points. Spikes are where there is at least one adjacent point with 
	    a gust less than 50% of the potential spike. Potential spikes are gusts above 25 m/s.
	Replace spikes using the mean of adjacent points.
	'''
	
	ind = np.where(da_vals >= 25)
	for i in tqdm(np.arange(len(ind[0]))):
		pot_spike = da_vals[ind[0][i], ind[1][i], ind[2][i]]
		adj_gusts = []
		for ii in [-1, 1]:
			for jj in [-1, 1]:
				try:
					adj_gusts.append( da_vals[ind[0][i], ind[1][i]+ii, ind[2][i]+jj])
				except:
					pass
		if (np.array(adj_gusts) < (0.5*pot_spike)).any():
			pot_spike = np.median(adj_gusts)
		da_vals[ind[0][i], ind[1][i], ind[2][i]] = pot_spike
	return da_vals