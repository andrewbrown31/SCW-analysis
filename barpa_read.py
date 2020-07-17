import warnings
import sys
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import xarray as xr
from barra_read import date_seq, get_aus_stn_info

from metpy.calc import vertical_velocity_pressure as omega
import metpy.calc as mpcalc
from metpy.units import units

def drop_duplicates(ds):

        #Drop time duplicates

        a, ind = np.unique(ds.time.values, return_index=True)
        return(ds.isel({"time":ind}))

def file_dates(files, query):

	is_in = []
	for i in np.arange(len(files)):
		t = dt.datetime.strptime(files[i].split("/")[11][:-1], "%Y%m%dT%H%M")
		t_list = date_seq([t + dt.timedelta(hours=6), t + dt.timedelta(days=10)], "hours", 6) 
		if any(np.in1d(query, t_list)):
			is_in.append(True)
		else:
			is_in.append(False)
	return is_in


def read_barpa(domain, time, experiment, forcing_mdl, ensemble):

	#NOTE: Data has been set to zero for below surface pressure.
	#But wrf_parallel doesn't use these levels anyway
	#TODO: The above statement I think is false. -273.15 K values may cause problems for some routines, 
	# even if below ground level. Mask these values to NaN

	#Create a list of 6-hourly "query" date-times, based on the start and end dates provided. 
	query_dates = date_seq(time, "hours", 6)

	#Get a list of all BARPA files in the du7 directory, for a given experiment/forcing model
	geopt_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp2/geop_ht_uv*"))
	hus_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp2/spec_hum*"))
	ta_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp2/air_temp*"))
	ua_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp2/wnd_ucmp*"))
	va_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp2/wnd_vcmp*"))
	huss_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp3/qsair_scrn*"))
	dewpt_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp26/dewpt_scrn*"))
	tas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp3/temp_scrn*"))
	uas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp3/uwnd10m_b*"))
	vas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp3/vwnd10m_b*"))
	ps_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp26/sfc_pres*"))
	wg_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/*/*/pp26/wndgust10m*"))

	#Get the files that we need
	geopt_files = geopt_files[file_dates(geopt_files, query_dates)]
	hus_files = hus_files[file_dates(hus_files, query_dates)]
	ta_files = ta_files[file_dates(ta_files, query_dates)]
	ua_files = ua_files[file_dates(ua_files, query_dates)]
	va_files = va_files[file_dates(va_files, query_dates)]
	huss_files = huss_files[file_dates(huss_files, query_dates)]
	dewpt_files = dewpt_files[file_dates(dewpt_files, query_dates)]
	tas_files = tas_files[file_dates(tas_files, query_dates)]
	uas_files = uas_files[file_dates(uas_files, query_dates)]
	vas_files = vas_files[file_dates(vas_files, query_dates)]
	ps_files = ps_files[file_dates(ps_files, query_dates)]
	wg_files = wg_files[file_dates(wg_files, query_dates)]

	#Load in these files, dropping duplicates
	#Drop the variable "realization", as it appears in some streams but not others, and is not used   
	geopt_ds = drop_duplicates(xr.open_mfdataset(geopt_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m
	hus_ds = drop_duplicates(xr.open_mfdataset(hus_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#1 (kg/kg?)
	ta_ds = drop_duplicates(xr.open_mfdataset(ta_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#K
	ua_ds = drop_duplicates(xr.open_mfdataset(ua_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m/s
	va_ds = drop_duplicates(xr.open_mfdataset(va_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m/s
	huss_ds = drop_duplicates(xr.open_mfdataset(huss_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#kg/kg
	dewpt_ds = drop_duplicates(xr.open_mfdataset(dewpt_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#K
	tas_ds = drop_duplicates(xr.open_mfdataset(tas_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#K
	uas_ds = drop_duplicates(xr.open_mfdataset(uas_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m/s
	vas_ds = drop_duplicates(xr.open_mfdataset(vas_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m/s
	ps_ds = drop_duplicates(xr.open_mfdataset(ps_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#Pa
	wg_ds = drop_duplicates(xr.open_mfdataset(wg_files, concat_dim="time", combine="nested", drop_variables=["realization"]))	#m/s

	#Slice to query times, spatial domain, convert to dataarray, restrict to below 100 hPa
	lons = slice(domain[2], domain[3])
	lats = slice(domain[0], domain[1])
	geopt_da = geopt_ds.sel({"time":query_dates, "pressure":geopt_ds["pressure"]>= 100, "latitude":lats, "longitude":lons})["geop_ht_uv"]
	hus_da = hus_ds.sel({"time":query_dates, "pressure":geopt_ds["pressure"]>= 100, "latitude":lats, "longitude":lons})["spec_hum_uv"]
	ta_da = ta_ds.sel({"time":query_dates, "pressure":geopt_ds["pressure"]>= 100, "latitude":lats, "longitude":lons})["air_temp_uv"]
	ua_da = ua_ds.sel({"time":query_dates, "pressure":geopt_ds["pressure"]>= 100, "latitude":lats, "longitude":lons})["wnd_ucmp_uv"]
	va_da = va_ds.sel({"time":query_dates, "pressure":geopt_ds["pressure"]>= 100, "latitude":lats, "longitude":lons})["wnd_vcmp_uv"]
	huss_da = huss_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["qsair_scrn"]
	dewpt_da = dewpt_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["dewpt_scrn"]
	tas_da = tas_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["temp_scrn"]
	uas_da = uas_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["uwnd10m_b"]
	vas_da = vas_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["vwnd10m_b"]
	ps_da = ps_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["sfc_pres"]
	wg_da = wg_ds.sel({"time":query_dates, "latitude":lats, "longitude":lons})["wndgust10m"]

	#As in read_cmip, make sure that all data arrays have the same times (take the union of the set of times).
	#If one of the dataarrays goes to size=0 on the time dimension, throw an error
	common_dates = np.array(list(set(hus_da.time.values) & set(ta_da.time.values) & set(ua_da.time.values)\
                 & set(va_da.time.values) & set(huss_da.time.values) & set(tas_da.time.values)\
                 & set(uas_da.time.values) & set(vas_da.time.values) & set(ps_da.time.values)\
		 & set(geopt_da.time.values) & set(wg_da.time.values) & set(dewpt_da.time.values)))
	geopt_da = geopt_da.isel({"time":np.in1d(geopt_da.time, common_dates)})
	hus_da = hus_da.isel({"time":np.in1d(hus_da.time, common_dates)})
	ta_da = ta_da.isel({"time":np.in1d(ta_da.time, common_dates)})
	ua_da = ua_da.isel({"time":np.in1d(ua_da.time, common_dates)})
	va_da = va_da.isel({"time":np.in1d(va_da.time, common_dates)})
	huss_da = huss_da.isel({"time":np.in1d(huss_da.time, common_dates)})
	dewpt_da = dewpt_da.isel({"time":np.in1d(dewpt_da.time, common_dates)})
	tas_da = tas_da.isel({"time":np.in1d(tas_da.time, common_dates)})
	uas_da = uas_da.isel({"time":np.in1d(uas_da.time, common_dates)})
	vas_da = vas_da.isel({"time":np.in1d(vas_da.time, common_dates)})
	ps_da = ps_da.isel({"time":np.in1d(ps_da.time, common_dates)})
	wg_da = wg_da.isel({"time":np.in1d(wg_da.time, common_dates)})
	for da in [geopt_da, hus_da, ta_da, ua_da, va_da, huss_da, dewpt_da, tas_da, uas_da, vas_da, ps_da, wg_da]:
		if len(da.time.values) == 0:
			varname=da.attrs["standard_name"]
			raise ValueError("ERROR: "+varname+" HAS BEEN SLICED IN TIME DIMENSION TO SIZE=0")

	#Now linearly interpolate pressure level data to match the BARRA pressure levels
	kwargs = {"fill_value":None, "bounds_error":False}
	#barra_levs = [100.0000000001, 150.0000000001, 175.0000000001, 
	#    200.0000000001, 225.0000000001, 250.0000000001, 275.0000000001, 
	#    300.0000000001, 350.0000000001, 400.0000000001, 450.0000000001, 
	#    500.0000000001, 600.0000000001, 700.0000000001, 750.0000000001, 
	#    800.0000000001, 850.0000000001, 900.0000000001, 925.0000000001, 
	#    950.0000000001, 975.0000000001, 1000.0000000001]
	#geopt_da = geopt_da.interp(coords={"pressure":barra_levs}, method="linear", kwargs=kwargs)
	#hus_da = hus_da.interp(coords={"pressure":barra_levs}, method="linear", kwargs=kwargs)
	#ta_da = ta_da.interp(coords={"pressure":barra_levs}, method="linear", kwargs=kwargs)
	#ua_da = ua_da.interp(coords={"pressure":barra_levs}, method="linear", kwargs=kwargs)
	#va_da = va_da.interp(coords={"pressure":barra_levs}, method="linear", kwargs=kwargs)

	#Linearly interpolate variables onto the same lat/lon grid (pressure level U/V grid). Extrapolate to staggered values outside the grid
	huss_da = huss_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	dewpt_da = dewpt_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	tas_da = tas_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	uas_da = uas_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	vas_da = vas_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	ps_da = ps_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)
	wg_da = wg_da.interp(coords={"latitude":hus_da.latitude, "longitude":hus_da.longitude}, method="linear", kwargs=kwargs)

	#Get numpy arrays of everything, and convert temperatures to degC and sfc pressure to hPa
	geopt = geopt_da.values
	hus = hus_da.values
	ta = ta_da.values - 273.15
	ua = ua_da.values
	va = va_da.values
	huss = huss_da.values
	dewpt = dewpt_da.values - 273.15
	tas = tas_da.values - 273.15
	uas = uas_da.values
	vas = vas_da.values
	ps = ps_da.values / 100.
	wg = wg_da.values

	#Mask -273.15 K values (these should only be values below surface)
	mask = ( ta==(-273.15) )
	geopt[mask] = np.nan
	hus[mask] = np.nan
	ta[mask] = np.nan
	ua[mask] = np.nan
	va[mask] = np.nan

	#Create 3d pressure variable
	p = np.moveaxis(np.tile(hus_da.pressure.values,[ta.shape[2],ta.shape[3],1]),2,0)

	#Get hur from hus, ta and p3d
	hur = np.array(mpcalc.relative_humidity_from_specific_humidity(hus, \
                    ta*units.degC, p*units.hectopascal) * 100)
	hur[hur<0] = 0
	hur[hur>100] = 100

	#Load terrain data
	terrain = xr.open_dataset("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/static/topog-BARPA-EASTAUS_12km.nc").\
		sel({"latitude":lats, "longitude":lons})["topog"].values

	#Get lat/lon
	lat = hus_da.latitude.values
	lon = hus_da.longitude.values

	#Flip the pressure dimension
	ta = np.flip(ta, axis=1)
	hur = np.flip(hur, axis=1)
	geopt = np.flip(geopt, axis=1)
	p = np.flip(p, axis=0)
	ua = np.flip(ua, axis=1)
	va = np.flip(va, axis=1)

	#Return times from one of the data arrays (they are identical in time). If it is different to the query date, then throw a warning
	query_times=pd.to_datetime(query_dates)
	times=pd.to_datetime(huss_da.time.values)
	if all(np.in1d(query_times,times)):
		pass
	else:
		message = "\n ".join(~query_times[np.in1d(query_times,times)].strftime("%Y%m%d %H:%M"))
		warnings.warn("WARNING: The following query dates were not loaded..."+message)
	
	#Format times for output (datetime objects)
	out_times = [dt.datetime.strptime(huss_da.time.dt.strftime("%Y-%m-%d %H:%M").values[i],"%Y-%m-%d %H:%M") for i in np.arange(huss_da.time.shape[0])]
	
	return [ta, hur, geopt, terrain, p[:,0,0], ps, ua, va, uas, vas, tas, dewpt, wg, lon,\
                    lat, out_times]

def to_points_wind_gust(loc_id, points, fname, start_year, end_year):

	#Load daily maximum wind gust data from du7, and extract point values
	        #As in to_points_loop(), but just for vertical velocity at 700 hPa, from the ma07 directory
        from dask.diagnostics import ProgressBar
        import gc
        ProgressBar().register()

        dates = []
        for y in np.arange(start_year,end_year+1):
                for m in np.arange(1,13):
                        dates.append(dt.datetime(y,m,1,12,0,0))
        last_date = dt.datetime(y+1,1,1,12,0,0)

        df = pd.DataFrame()

        lsm = xr.open_dataset("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/static/lnd_mask-BARPA-EASTAUS_12km.nc")

        #Read netcdf data
        for t in np.arange(len(dates)):
                print(dates[t])
                try:
                    query_dates = date_seq([dates[t], dates[t+1]+dt.timedelta(days=-1)], "hours", 24)
                except:
                    query_dates = date_seq([dates[t], last_date+dt.timedelta(days=-1)], "hours", 24)
                wg_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA-EASTAUS_12km/era/erai/r0/*/*/pp0/max_wndgust10m*.nc"))
                wg_files = wg_files[file_dates(wg_files, query_dates)]
                f = drop_duplicates(xr.open_mfdataset(wg_files, concat_dim="time", combine="nested")).sel({"time":query_dates})

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

                temp_df = f["max_wndgust10m"].isel(latitude = xr.DataArray(dist_lat, dims="points"), \
                                longitude = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()
                temp_df = temp_df.reset_index()
                temp_df["time"] = pd.DatetimeIndex(temp_df.time) + dt.timedelta(hours=-12) 
    
                for p in np.arange(len(loc_id)):
                        temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

                temp_df = temp_df.drop(["points",\
                        "forecast_period", "forecast_reference_time", "height"],axis=1)

                df = pd.concat([df, temp_df])
                f.close()
                gc.collect()

        df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

if __name__ == "__main__":

	loc_id, points = get_aus_stn_info()
	#REMOVE THESE LOCS AS THEY AREN'T IN THE EASTAUS DOMAIN
	removed = ['Broome', 'Port Hedland', 'Carnarvon', 'Meekatharra', 'Perth', 'Esperance', 'Kalgoorlie', 'Halls Creek']
	points = np.array(points)[np.in1d(loc_id, removed, invert=True)]
	loc_id = loc_id[np.in1d(loc_id, removed, invert=True)]
	to_points_wind_gust(loc_id, points, "barpa_erai_gusts", 1990, 2015)
