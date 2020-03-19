#Read in hybrid-height coordinate data from the NCI CMIP archive, for computation in wrf_parallel.py

import metpy.units as units
import metpy.calc as mpcalc
import sys
import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
from barra_read import date_seq
from calc_param import get_dp

def read_cmip5(model, experiment, ensemble, year, domain):

	#For the given model, institute, experiment, get the relevant file paths.
	hus_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/6hr/atmos/"+ensemble+"/hus/latest/*"))
	ta_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/6hr/atmos/"+ensemble+"/ta/latest/*"))
	ua_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/6hr/atmos/"+ensemble+"/ua/latest/*"))
	va_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/6hr/atmos/"+ensemble+"/va/latest/*"))

	huss_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/3hr/atmos/"+ensemble+"/huss/latest/*"))
	tas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/3hr/atmos/"+ensemble+"/tas/latest/*"))
	uas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/3hr/atmos/"+ensemble+"/uas/latest/*"))
	vas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/3hr/atmos/"+ensemble+"/vas/latest/*"))
	ps_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+model+"/"+experiment+\
		    "/3hr/atmos/"+ensemble+"/ps/latest/*"))

	#Isolate the files relevant for the current "year"
	#NOTE will have to change to incorperate months if there is more than one file per year
	lev_file_years = [ np.arange(int(hus_files[i].split("_")[-1][:-3].split("-")[0][0:4]), \
		int(hus_files[i].split("_")[-1][:-3].split("-")[1][0:4]) ) \
		    for i in np.arange(len(hus_files)) ]
	lev_fid = np.where([np.in1d(year, lev_file_years[i]).sum() \
		    for i in np.arange(len(hus_files))])[0]
	sfc_file_years = [ np.arange(int(huss_files[i].split("_")[-1][:-3].split("-")[0][0:4]), \
		int(huss_files[i].split("_")[-1][:-3].split("-")[1][0:4]) ) \
		    for i in np.arange(len(huss_files)) ]
	sfc_fid = np.where([np.in1d(year, sfc_file_years[i]) \
		    for i in np.arange(len(huss_files))])[0]

	#Load the data, match 3 hourly and 6 hourly data
	hus = xr.open_mfdataset([hus_files[i] for i in lev_fid])
	ta = xr.open_mfdataset([ta_files[i] for i in lev_fid])
	ua = xr.open_mfdataset([ua_files[i] for i in lev_fid])
	va = xr.open_mfdataset([va_files[i] for i in lev_fid])

	huss = xr.open_mfdataset([huss_files[i] for i in sfc_fid])
	huss = huss.sel({"time":np.in1d(huss.time, hus.time)})
	tas = xr.open_mfdataset([tas_files[i] for i in sfc_fid])
	tas = tas.sel({"time":np.in1d(tas.time, ta.time)})
	uas = xr.open_mfdataset([uas_files[i] for i in sfc_fid])
	uas = uas.sel({"time":np.in1d(uas.time, ua.time)})
	vas = xr.open_mfdataset([vas_files[i] for i in sfc_fid])
	vas = vas.sel({"time":np.in1d(vas.time, va.time)})
	ps = xr.open_mfdataset([ps_files[i] for i in sfc_fid])
	ps = ps.sel({"time":np.in1d(ps.time, hus.time)})

	#and trim to the domain given by "domain", as well as the year given by "year"
	hus = trim_cmip5(hus, domain, year)
	ta = trim_cmip5(ta, domain, year)
	ua = trim_cmip5(ua, domain, year)
	va = trim_cmip5(va, domain, year)

	huss = trim_cmip5(huss, domain, year)
	tas = trim_cmip5(tas, domain, year)
	uas = trim_cmip5(uas, domain, year)
	vas = trim_cmip5(vas, domain, year)
	ps = trim_cmip5(ps, domain, year)

	#Convert vertical coordinate to height 
	z = hus.lev + (hus.b * hus.orog)
	orog = hus.orog.values

	#Calculate pressure via hydrostatic equation
	q = hus.hus / (1 - hus.hus)
	tv = ta.ta * ( ( q + 0.622) / (0.622 * (1+q) ) )
	p = np.swapaxes(np.swapaxes(ps.ps * np.exp( -9.8*z / (287*tv)), 3, 2), 2, 1)

	#Convert quantities into those expected by wrf_parallel.py
	ta = ta.ta.values - 273.15
	hur = mpcalc.relative_humidity_from_specific_humidity(hus.hus.values, \
		    ta*units.units.degC, p.values*units.units.pascal) * 100
	z = np.tile(z.values, [ta.shape[0], 1, 1, 1])
	pres = p / 100.
	sfc_pres = ps.ps.values / 100.
	ua = ua.ua.interp({"lon":p.lon},method="linear", assume_sorted=True).values
	va = va.va.interp({"lat":p.lat},method="linear", assume_sorted=True).values
	uas = uas.uas.interp({"lat":p.lat,"lon":p.lon},method="linear", assume_sorted=True).values
	vas = vas.vas.interp({"lat":p.lat,"lon":p.lon},method="linear", assume_sorted=True).values
	tas = tas.tas.values - 273.15
	ta2d = mpcalc.dewpoint_from_specific_humidity(hus.hus.values, ta*units.units.degC, \
		    p.values*units.units.pascal)
	lon = p.lon.values
	lat = p.lat.values
	date_list = p.time.values

	return [ta, hur, z, orog, pres, sfc_pres, ua, va, uas, vas, tas, ta2d, lon,\
		    lat, date_list]


def trim_cmip5(dataset, domain, year):

	dataset = dataset.sel({"lon":slice(domain[2], domain[3]),\
		"lat":slice(domain[0],domain[1]),\
		"time":dataset["time.year"]==year})

	return dataset
