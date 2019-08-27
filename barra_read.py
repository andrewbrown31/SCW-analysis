import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import xarray as xr

from metpy.calc import vertical_velocity_pressure as omega
from metpy.units import units

def get_dp(ta,hur,dp_mask=True):
        #Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
        #Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
        #For points where RH is zero, set the dew point temperature to -85 deg C
        #EDIT: Leave points where RH is zero masked as NaNs. Hanlde them after creating a SHARPpy object (by 
        # making the missing dp equal to the mean of the above and below layers)
        #EDIT: Replace with metpy code

        #a = 17.27
        #b = 237.7
        #alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
        #dp = (b*alpha) / (a - alpha)

        dp = np.array(mpcalc.dewpoint_rh(ta * units.units.degC, hur * units.units.percent))

        if dp_mask:
                return dp
        else:
                dp = np.array(dp)
                dp[np.isnan(dp)] = -85.
                return dp

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

def get_pressure(top):
	#NOTE THAT WE GET THE LESSER-RESULUTION PRESSURE COORDINATES (16 LEVELS BELOW 100 hPa)
	ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/"\
	+"2016"+"/"+"09"+"/air_temp-an-prs-PT0H-BARRA_R-v1-"+"2016"+"09"+"28"+"T"+"06"+"*.nc")[0])
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
	f=xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/barra/barra*", parallel=True)

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
			lon = xr.DataArray(dist_lon, dims="points")).to_dataframe()

	df = df.reset_index()

	for p in np.arange(len(loc_id)):
		df.loc[df.points==p,"loc_id"] = loc_id[p]

	df.drop("points",axis=1).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_points_wrfpython_aus_1979_2017.pkl")
	#df = df.drop("points",axis=1)
	#df_orig = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")
	#df_new = pd.merge(df_orig, df[["time","loc_id","wbz","Vprime"]], on=["time","loc_id"])
	#df_new.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")

def get_aus_stn_info():
	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	
	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
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



if __name__ == "__main__":

	to_points()
