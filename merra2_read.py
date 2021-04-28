import metpy.calc as mpcalc
import metpy.units as units
import gc
from barra_read import latlon_dist
from dask.diagnostics import ProgressBar
import sys
import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
from barra_read import date_seq
from calc_param import get_dp

def read_merra2(domain,times,pres=True,delta_t=1):
	#Read 3-hourly MERRA2 pressure level/surface data

	if len(times) > 1:
		date_list = date_seq(times,"hours",delta_t)
	else:
		date_list = times

	files_3d = []; files_2d = []
	for d in date_list:
		files_3d.append(glob.glob("/g/data/rr7/MERRA2/raw/M2I3NPASM.5.12.4/"+d.strftime("%Y")+"/"+d.strftime("%m")+"/MERRA2*"+d.strftime("%Y%m%d")+"*.nc4")[0])
		files_2d.append(glob.glob("/g/data/ua8/MERRA2/1hr/M2I1NXASM.5.12.4/"+d.strftime("%Y")+"/"+d.strftime("%m")+"/MERRA2*"+d.strftime("%Y%m%d")+"*.nc4")[0])
	files_3d = np.unique(files_3d)
	files_2d = np.unique(files_2d)

	f3d = xr.open_mfdataset(files_3d, combine="by_coords").sel({"time":date_list, "lev":slice(1000,100), "lon":slice(domain[2], domain[3]), "lat":slice(domain[0], domain[1])})
	f2d = xr.open_mfdataset(files_2d, combine="by_coords").sel({"time":date_list, "lon":slice(domain[2], domain[3]), "lat":slice(domain[0], domain[1])})

	ta_file = f3d["T"]; z_file = f3d["H"]; ua_file = f3d["U"]; va_file = f3d["V"]; hur_file = f3d["RH"]
	uas_file = f2d["U10M"]; vas_file = f2d["V10M"]; hus_file = f2d["QV2M"]; tas_file = f2d["T2M"]; ps_file = f2d["PS"]

	ta = ta_file.values - 273.15
	ua = ua_file.values
	va = va_file.values
	hgt = z_file.values
	hur = hur_file.values * 100
	hur[hur<0] = 0
	hur[hur>100] = 100
	dp = get_dp(ta,hur)
	uas = uas_file.values
	vas = vas_file.values
	tas = tas_file.values - 273.15
	ps = ps_file.values / 100
	ta2d = np.array(mpcalc.dewpoint_from_specific_humidity(hus_file.values, tas*units.units.degC, \
                    ps*units.units.hectopascal))
	terrain = f3d["PHIS"].isel({"time":0}).values / 9.8 
	lon = f2d["lon"].values
	lat = f2d["lat"].values
	p = f3d["lev"].values

	return [ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,lon,lat,date_list]


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
			"era5/era5_"+dates[t].strftime("%Y%m")+"*.nc")[0],\
			chunks={"lat":10, "lon":10, "time":10}, engine="h5netcdf")

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

	dates = []
	for y in np.arange(start_year,end_year+1):
		for m in np.arange(1,13):
			dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		f=xr.open_dataset(glob.glob("/g/data/eg3/ab4502/ExtremeWind/aus/"+\
			"merra2/merra2_"+dates[t].strftime("%Y%m")+"*.nc")[0])

		#Setup lsm
		lat = f.coords.get("lat").values
		lon = f.coords.get("lon").values
		lsm = xr.where(xr.open_dataset("/g/data/eg3/ab4502/MERRA2_101.const_2d_asm_Nx.00000000.nc4")["FRLAND"].sel({"lat":lat, "lon":lon}) >= 0.5, 1, 0).values[0]
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

	to_keep = ["Adelaide","Darwin","Sydney","Woomera"]
	points = np.array(points)[np.in1d(loc_id,to_keep)]
	loc_id = np.array(loc_id)[np.in1d(loc_id,to_keep)]
	to_points_loop(loc_id,points,"merra2_"+str(start_time)+"_"+str(end_time),start_time,end_time,variables=["bdsd","eff_sherb","dcp","t_totals"])
