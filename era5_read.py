import sys
import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
from barra_read import date_seq
from calc_param import get_dp

def read_era5(domain,times):
	#Open ERA5 netcdf files and extract variables needed for a range of times 
	# and given spatial domain

	ref = dt.datetime(1900,1,1,0,0,0)
	if len(times) > 1:
		date_list = date_seq(times,"hours",1)
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
	sfc_lon,sfc_lat = get_lat_lon_sfc()
	sfc_lon_ind = np.where((sfc_lon >= domain[2]) & (sfc_lon <= domain[3]))[0]
	sfc_lat_ind = np.where((sfc_lat >= domain[0]) & (sfc_lat <= domain[1]))[0]
	sfc_lon = sfc_lon[lon_ind]
	sfc_lat = sfc_lat[lat_ind]

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
		ta_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/T/"+date[0:4]+\
			"/T_era5_aus_"+date+"*.nc")[0])
		z_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/Z/"+date[0:4]+\
			"/Z_era5_aus_"+date+"*.nc")[0])
		ua_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/U/"+date[0:4]+\
			"/U_era5_aus_"+date+"*.nc")[0])
		va_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/V/"+date[0:4]+\
			"/V_era5_aus_"+date+"*.nc")[0])
		hur_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/R/"+date[0:4]+\
			"/R_era5_aus_"+date+"*.nc")[0])

		uas_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/10U/"+date[0:4]+\
			"/10U_era5_global_"+date+"*.nc")[0])
		vas_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/10V/"+date[0:4]+\
			"/10V_era5_global_"+date+"*.nc")[0])
		ta2d_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/2D/"+date[0:4]+\
			"/2D_era5_global_"+date+"*.nc")[0])
		tas_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/2T/"+date[0:4]+\
			"/2T_era5_global_"+date+"*.nc")[0])
		ps_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/SP/"+date[0:4]+\
			"/SP_era5_global_"+date+"*.nc")[0])
		cp_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/CP/"+date[0:4]+\
			"/CP_era5_global_"+date+"*.nc")[0])
		wg10_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/10FG/"+date[0:4]+\
			"/10FG_era5_global_"+date+"*.nc")[0])

		#Get times to load in from file
		times = ta_file["time"][:]
		time_ind = [np.where(x==times)[0][0] for x in time_hours if (x in times)]
		date_ind = np.where(np.array(formatted_dates) == date)[0]

		#Get times to load in from forecast files (wg10 and cp)
		fc_times = cp_file["time"][:]
		fc_time_ind = [np.where(x==fc_times)[0][0] for x in time_hours if (x in fc_times)]

		#Load analysis data
		ta[date_ind,:,:,:] = ta_file["t"][time_ind,p_ind,lat_ind,lon_ind] - 273.15
		#wap[date_ind,:,:,:] = wap_file["wap"][time_ind,p_ind,lat_ind,lon_ind]
		ua[date_ind,:,:,:] = ua_file["u"][time_ind,p_ind,lat_ind,lon_ind]
		va[date_ind,:,:,:] = va_file["v"][time_ind,p_ind,lat_ind,lon_ind]
		hgt[date_ind,:,:,:] = z_file["z"][time_ind,p_ind,lat_ind,lon_ind] / 9.8
		hur[date_ind,:,:,:] = hur_file["r"][time_ind,p_ind,lat_ind,lon_ind]
		hur[hur<0] = 0
		hur[hur>100] = 100
		dp[date_ind,:,:,:] = get_dp(ta[date_ind,:,:,:],hur[date_ind,:,:,:])
		uas[date_ind,:,:] = uas_file["u10"][time_ind,sfc_lat_ind,sfc_lon_ind]
		vas[date_ind,:,:] = vas_file["v10"][time_ind,sfc_lat_ind,sfc_lon_ind]
		tas[date_ind,:,:] = tas_file["t2m"][time_ind,sfc_lat_ind,sfc_lon_ind] - 273.15
		ta2d[date_ind,:,:] = ta2d_file["d2m"][time_ind,sfc_lat_ind,sfc_lon_ind] - 273.15
		ps[date_ind,:,:] = ps_file["sp"][time_ind,sfc_lat_ind,sfc_lon_ind] / 100
		fc_date_ind = np.in1d(date_list, nc.num2date(cp_file["time"][fc_time_ind], cp_file["time"].units))
		cp[fc_date_ind,:,:] = cp_file["cp"][fc_time_ind,sfc_lat_ind,sfc_lon_ind]
		wg10[fc_date_ind,:,:] = wg10_file["fg10"][fc_time_ind,sfc_lat_ind,sfc_lon_ind]

		ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();tas_file.close();ta2d_file.close();ps_file.close()
	

	p = np.flip(p)
	ta = np.flip(ta, axis=1)
	dp = np.flip(dp, axis=1)
	hur = np.flip(hur, axis=1)
	hgt = np.flip(hgt, axis=1)
	ua = np.flip(ua, axis=1)
	va = np.flip(va, axis=1)

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,cp,wg10,cape,lon,lat,date_list]

def get_pressure(top):
	#Returns [no of levels, levels, indices below "top"]
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/T/2012/"+\
"T_era5_aus_"+"201201"+"*.nc")[0])
	p =ta_file["level"][:]
	p_ind = np.where(p>=top)[0]
	ta_file.close()
	return [len(p_ind), p, p_ind]

def get_lsm():
	#Load the ERA-Interim land-sea mask (land = 1)
	lsm_file = nc.Dataset("/g/data/ub4/era5/netcdf/static_era5.nc")
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
	terrain_file = nc.Dataset("/g/data/ub4/era5/netcdf/static_era5.nc")
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

def get_lat_lon_sfc():
	uas_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/surface/10U/2012/"+\
"10U_era5_global_"+"201201"+"*.nc")[0])
	lon = uas_file["longitude"][:]
	lat = uas_file["latitude"][:]
	uas_file.close()
	return [lon,lat]

def get_lat_lon():
	ta_file = nc.Dataset(glob.glob("/g/data/ub4/era5/netcdf/pressure/T/2012/"+\
"T_era5_aus_"+"201201"+"*.nc")[0])
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	ta_file.close()
	return [lon,lat]

def get_mask(lon,lat):
	#Return lsm for a given domain (with lats=lat and lons=lon)
	lsm,nat_lon,nat_lat = get_lsm()
	lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
	lat_ind = np.where((nat_lat >= lat[-1]) & (nat_lat <= lat[0]))[0]
	lsm_domain = lsm[(lat_ind[0]):(lat_ind[-1]+1),(lon_ind[0]):(lon_ind[-1]+1)]
	lsm_domain = np.where(lsm_domain > .5, 1, 0)

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

def to_points_loop(loc_id,points,fname,start_year,end_year,variables=False):

	#As in to_points(), but by looping over monthly data
	from dask.diagnostics import ProgressBar
	import gc
	#ProgressBar().register()

	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		f=xr.open_dataset(glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/"+\
			"era5/era5_"+dates[t].strftime("%Y%m")+"*.nc")[0],\
			chunks={"lat":139, "lon":178, "time":50}, engine="h5netcdf")

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

	#lsm = get_lsm()[0]
	start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275 

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		year = dt.datetime.strftime(dates[t],"%Y")
		month =	dt.datetime.strftime(dates[t],"%m")
		u_file = xr.open_mfdataset("/g/data/ub4/era5/netcdf/surface/10U/"+\
			year+"/10U_era5_global_"+year+month+"*.nc", concat_dim="time").\
			sel({"latitude":slice(end_lat,start_lat), \
			    "longitude":slice(start_lon,end_lon)})
		v_file = xr.open_mfdataset("/g/data/ub4/era5/netcdf/surface/10V/"+\
			year+"/10V_era5_global_"+year+month+"*.nc", concat_dim="time").\
			sel({"latitude":slice(end_lat,start_lat), \
			    "longitude":slice(start_lon,end_lon)})
		wd = 180 + ( 180/np.pi ) * \
			(np.arctan2(u_file["u10"], v_file["v10"]))

		#Setup lsm
		lat = u_file.coords.get("latitude").values
		lon = u_file.coords.get("longitude").values
		lsm = get_mask(lon,lat)
		x,y = np.meshgrid(lon,lat)
		x[lsm<0.5] = np.nan
		y[lsm<0.5] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		temp_df = wd.isel(latitude = xr.DataArray(dist_lat, dims="points"), \
			longitude = xr.DataArray(dist_lon, dims="points")).persist().\
			to_dataframe("wd")

		temp_df = temp_df.reset_index()

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop(["points"],axis=1)
		df = pd.concat([df, temp_df])
		u_file.close()
		v_file.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")


def get_aus_stn_info():
	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	

	df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
		names=names, header=0)

	#Dict to map station names to
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

	if len(sys.argv) > 1:
		start_time = int(sys.argv[1])
		end_time = int(sys.argv[2])

	loc_id, points = get_aus_stn_info()

	to_points_loop(loc_id, points, "era5_allvars_v2_"+str(start_time)+"_"+str(end_time), \
			start_time, end_time)
