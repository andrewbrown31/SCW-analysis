import xarray as xr
from event_analysis import get_aus_stn_info
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
from barra_read import date_seq
from calc_param import get_dp

def read_erai(domain,times):
	#Open ERA-Interim netcdf files and extract variables needed for a range of times 
	# and given spatial domain
	#Option to also use one time (include hour)

	ref = dt.datetime(1900,1,1,0,0,0)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times
	formatted_dates = [format_dates(x) for x in date_list]
	unique_dates = np.unique(formatted_dates)
	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)
	if (date_list[0].day==1) & (date_list[0].hour<3):
		fc_unique_dates = np.insert(unique_dates, 0, format_dates(date_list[0] - dt.timedelta(1)))
	else:
		fc_unique_dates = np.copy(unique_dates)

	#Get time-invariant pressure and spatial info
	no_p, p, p_ind = get_pressure(100)
	p = p[p_ind]
	lon,lat = get_lat_lon()
	lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
	lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
	lon = lon[lon_ind]
	lat = lat[lat_ind]
	terrain = reform_terrain(lon,lat)

	#Initialise arrays for each variable
	ta = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	dp = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	hur = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	hgt = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	wap = np.empty((len(date_list),no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	vas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ps = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	cp = np.zeros(ps.shape) * np.nan
	cape = np.zeros(ps.shape) * np.nan
	wg10 = np.zeros(ps.shape) * np.nan

	tas = np.empty((len(date_list),len(lat_ind),len(lon_ind)))
	ta2d = np.empty((len(date_list),len(lat_ind),len(lon_ind)))

	for date in unique_dates:

	#Load ERA-Interim reanalysis files
		ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/z/\
z_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
		wap_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/wap/\
wap_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0])
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
		ta2d_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/ta2d/\
ta2d_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])
		tas_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/tas/\
tas_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])
		ps_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/ps/\
ps_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]
		time_ind = [np.where(x==times)[0][0] for x in time_hours if (x in times)]
		date_ind = np.where(np.array(formatted_dates) == date)[0]

		#Load analysis data
		ta[date_ind,:,:,:] = ta_file["ta"][time_ind,p_ind,lat_ind,lon_ind] - 273.15
		wap[date_ind,:,:,:] = wap_file["wap"][time_ind,p_ind,lat_ind,lon_ind]
		ua[date_ind,:,:,:] = ua_file["ua"][time_ind,p_ind,lat_ind,lon_ind]
		va[date_ind,:,:,:] = va_file["va"][time_ind,p_ind,lat_ind,lon_ind]
		hgt[date_ind,:,:,:] = z_file["z"][time_ind,p_ind,lat_ind,lon_ind] / 9.8
		hur[date_ind,:,:,:] = hur_file["hur"][time_ind,p_ind,lat_ind,lon_ind]
		hur[hur<0] = 0
		hur[hur>100] = 100
		dp[date_ind,:,:,:] = get_dp(ta[date_ind,:,:,:],hur[date_ind,:,:,:])
		uas[date_ind,:,:] = uas_file["uas"][time_ind,lat_ind,lon_ind]
		vas[date_ind,:,:] = vas_file["vas"][time_ind,lat_ind,lon_ind]
		tas[date_ind,:,:] = tas_file["tas"][time_ind,lat_ind,lon_ind] - 273.15
		ta2d[date_ind,:,:] = ta2d_file["ta2d"][time_ind,lat_ind,lon_ind] - 273.15
		ps[date_ind,:,:] = ps_file["ps"][time_ind,lat_ind,lon_ind] / 100

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();tas_file.close();ta2d_file.close();ps_file.close();wap_file.close()
	
	for date in fc_unique_dates:
	
		if int(date) >= 197900:

			cp_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/cp/"\
	+"cp_3hrs_ERAI_historical_fc-sfc_"+date+"*.nc")[0])
			cape_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/cape/"\
	+"cape_3hrs_ERAI_historical_fc-sfc_"+date+"*.nc")[0])
			wg10_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/wg10/"\
	+"wg10_3hrs_ERAI_historical_fc-sfc_"+date+"*.nc")[0])

			#Load forecast data
			fc_times = nc.num2date(cp_file["time"][:], cp_file["time"].units)
			#an_times = nc.num2date(ps_file["time"][time_ind], ps_file["time"].units)
			an_times = date_list
			fc_cp = cp_file.variables["cp"][:,lat_ind,lon_ind]
			fc_cape = cape_file.variables["cape"][:,lat_ind,lon_ind]
			fc_wg10 = wg10_file.variables["wg10"][:,lat_ind,lon_ind]
			cnt = 0
			for an_t in an_times:
				try:
					fc_ind = np.where(an_t == np.array(fc_times))[0][0]
					cp[cnt] = ((fc_cp[fc_ind] - fc_cp[fc_ind - 1]) * 1000.) / 3.
					cape[cnt] = (fc_cape[fc_ind])
					wg10[cnt] = (fc_wg10[fc_ind])
				except:
					pass
				cnt = cnt + 1

			cp_file.close(); cape_file.close(); wg10_file.close()

	return [ta,dp,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,cp,wg10,cape,lon,lat,date_list]
	
def read_erai_fc(domain,times):
	#Open ERA-Interim forecast netcdf files and extract variables needed for a range of times 
	# and given spatial domain
	#Option to also use one time (include hour)

	ref = dt.datetime(1900,1,1,0,0,0)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times

	#If the last date in the list is the start of the next month, don't include in date stamp
	# list for file names
	if (date_list[-1].day==1) & (date_list[-1].hour==0):
		formatted_dates = [format_dates(x) for x in date_list[0:-1]]
	else:
		formatted_dates = [format_dates(x) for x in date_list]
	unique_dates = np.unique(formatted_dates)

	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)

	#Get time-invariant pressure and spatial info
	lon,lat = get_lat_lon()
	lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
	lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
	lon = lon[lon_ind]
	lat = lat[lat_ind]
	terrain = reform_terrain(lon,lat)

	#Initialise arrays for each variable
	wg10 = np.empty((0,len(lat_ind),len(lon_ind)))
	cape = np.empty((0,len(lat_ind),len(lon_ind)))

	for date in unique_dates:
		#print(date)

	#Load ERA-Interim reanalysis files
		wg10_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/wg10/\
wg10_3hrs_ERAI_historical_fc-sfc_"+date+"*.nc")[0])
		cape_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/cape/\
cape_3hrs_ERAI_historical_fc-sfc_"+date+"*.nc")[0])

		#Get times to load in from file
		times = wg10_file["time"][:]
		time_ind = [np.where(x==times)[0][0] for x in time_hours if (x in times)]

		#Load data
		wg10 = np.append(wg10,wg10_file["wg10"][time_ind,lat_ind,lon_ind],axis=0)
		cape = np.append(cape,cape_file["cape"][time_ind,lat_ind,lon_ind],axis=0)

		wg10_file.close();cape_file.close()
	
	return [wg10,cape,lon,lat,date_list]

def read_erai_points(points,times):

	#Open ERA-Interim netcdf files and extract variables needed for a range of 
	# times at a given set of spatial points

	#Format dates and times
	ref = dt.datetime(1900,1,1,0,0,0)
	date_list = date_seq(times,"hours",6)
	formatted_dates = [format_dates(x) for x in date_list]
	unique_dates = np.unique(formatted_dates)
	time_hours = np.empty(len(date_list))
	for t in np.arange(0,len(date_list)):
		time_hours[t] = (date_list[t] - ref).total_seconds() / (3600)

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100)
	lon,lat = get_lat_lon()
	[lon_ind, lat_ind, lon_used, lat_used] = get_lat_lon_inds(points,lon,lat)
	terrain_new = reform_terrain(lon,lat)

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
	ps = np.empty((len(date_list),len(points)))

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
		ps_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/ps/\
ps_6hrs_ERAI_historical_an-sfc_"+date+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]
		time_ind = [np.where(x==times)[0][0] for x in time_hours if (x in times)]
		date_ind = np.where(np.array(formatted_dates) == date)[0]

		#Load data for each spatial point given to function
		for point in np.arange(0,len(points)):
			ta[date_ind,:,point] = ta_file["ta"][time_ind,p_ind,lat_ind[point]\
						,lon_ind[point]] - 273.15
			ua[date_ind,:,point] = ua_file["ua"][time_ind,p_ind,lat_ind[point]\
						,lon_ind[point]]
			va[date_ind,:,point] = va_file["va"][time_ind,p_ind,lat_ind[point]\
						,lon_ind[point]]
			hgt[date_ind,:,point] = z_file["z"][time_ind,p_ind,lat_ind[point]\
						,lon_ind[point]] / 9.8
			hur[date_ind,:,point] = hur_file["hur"][time_ind,p_ind,lat_ind[point]\
						,lon_ind[point]]
			hur[hur<0] = 0
			dp[date_ind,:,point] = get_dp(ta[date_ind,:,point],hur[date_ind,:,point])
			uas[date_ind,point] = uas_file["uas"][time_ind,lat_ind[point]\
						,lon_ind[point]]
			vas[date_ind,point] = vas_file["vas"][time_ind,lat_ind[point]\
						,lon_ind[point]]
			ps[date_ind,point] = ps_file["ps"][time_ind,lat_ind[point]\
						,lon_ind[point]] /100
			p[date_ind,:,point] = pres[p_ind]

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();ps_file.close()

	#Save lat/lon as array
	lon = np.empty((len(points)))
	lat = np.empty((len(points)))
	terrain = np.empty((len(points)))
	for point in np.arange(0,len(points)):
		lon[point] = points[point][0]
		lat[point] = points[point][1]
		terrain[point] = terrain_new[lat_ind[point],lon_ind[point]]
	
	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list]

def get_pressure(top):
	#Returns [no of levels, levels, indices below "top"]
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+"201201"+"*.nc")[0])
	p =ta_file["lev"][:]/100
	p_ind = np.where(p>=top)[0]
	ta_file.close()
	return [len(p_ind), p, p_ind]

def get_lsm():
	#Load the ERA-Interim land-sea mask (land = 1)
	lsm_file = nc.Dataset("/short/eg3/ab4502/erai_lsm.nc")
	lsm = np.squeeze(lsm_file.variables["lsm"][:])
	lsm_lon = np.squeeze(lsm_file.variables["longitude"][:])
	lsm_lat = np.squeeze(lsm_file.variables["latitude"][:])
	lsm_file.close()
	return [lsm,lsm_lon,lsm_lat]

def reform_lsm(lon,lat):
	#Re-shape the land sea mask to go from longitude:[0,360] to [-180,180]
	[lsm,lsm_lon,lsm_lat] = get_lsm()
	lsm_new = np.empty(lsm.shape)

	lsm_lon[lsm_lon>=180] = lsm_lon[lsm_lon>=180]-360
	for i in np.arange(0,len(lat)):
		for j in np.arange(0,len(lon)):
			lsm_new[i,j] = lsm[lat[i]==lsm_lat, lon[j]==lsm_lon]
	return lsm_new

def get_terrain():
	#Load the ERA-Interim surface geopetential height as terrain height
	terrain_file = nc.Dataset("/short/eg3/ab4502/erai_sfc_geopt.nc")
	terrain = np.squeeze(terrain_file.variables["z"][:])/9.8
	terrain_lon = np.squeeze(terrain_file.variables["longitude"][:])
	terrain_lat = np.squeeze(terrain_file.variables["latitude"][:])
	terrain_file.close()
	return [terrain,terrain_lon,terrain_lat]

def reform_terrain(lon,lat):
	#Re-shape terrain height to go from longitude:[0,360] to [-180,180]
	[terrain,terrain_lon,terrain_lat] = get_terrain()
	terrain_new = np.empty((len(lat),len(lon)))

	terrain_lon[terrain_lon>=180] = terrain_lon[terrain_lon>=180]-360
	for i in np.arange(0,len(lat)):
		for j in np.arange(0,len(lon)):
			terrain_new[i,j] = terrain[lat[i]==terrain_lat, lon[j]==terrain_lon]
	return terrain_new

def format_dates(x):
	return dt.datetime.strftime(x,"%Y") + dt.datetime.strftime(x,"%m")

def get_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/\
ta_6hrs_ERAI_historical_an-pl_"+"201201"+"*.nc")[0])
	lon = ta_file["lon"][:]
	lat = ta_file["lat"][:]
	ta_file.close()
	return [lon,lat]

def get_mask(lon,lat):
	#Return lsm for a given domain (with lats=lat and lons=lon)
	nat_lon,nat_lat = get_lat_lon()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[-1]) & (nat_lat <= lat[0]))[0]
	lsm = reform_lsm(nat_lon,nat_lat)
	lsm_domain = lsm[(lat_ind[0]):(lat_ind[-1]+1),(lon_ind[0]):(lon_ind[-1]+1)]

	return lsm_domain

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
		
def drop_erai_fc_duplicates(arr,times):

	#ERAI forecast data has been saved with one day dupliaceted per year. Function to drop the first duplicate
	#day for each year from a 3d array

	u,idx = np.unique(times,return_index=True)
	arr = arr[idx]

	return (arr,u)

def to_points():

	#Read in all ERA-Interim netcdf convective parameters, and extract point data.
	#(Hopefuly) a faster version of event_analysis.load_netcdf_points_mf()

	#Read netcdf data
	from dask.diagnostics import ProgressBar
	ProgressBar().register()
	f=xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/erai/erai*", parallel=True)

	#JUST WANTING TO APPEND A COUPLE OF NEW VARIABLES TO THE DATAFRAME
	f=f[["Vprime", "wbz"]]

	#Setup lsm
	lon_orig,lat_orig = get_lat_lon()
	lsm = reform_lsm(lon_orig,lat_orig)
	lat = f.coords.get("lat").values
	lon = f.coords.get("lon").values
	x,y = np.meshgrid(lon,lat)
	lsm_new = lsm[((lat_orig<=lat[0]) & (lat_orig>=lat[-1]))]
	lsm_new = lsm_new[:,((lon_orig>=lon[0]) & (lon_orig<=lon[-1]))]
	x[lsm_new==0] = np.nan
	y[lsm_new==0] = np.nan

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

	#df.drop("points",axis=1).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")
	df = df.drop("points",axis=1)
	df_orig = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")
	df_new = pd.merge(df_orig, df[["time","loc_id","wbz","Vprime"]], on=["time","loc_id"])
	df_new.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sharppy_aus_1979_2017.pkl")

if __name__ == "__main__":

	to_points()
