import sys
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import xarray as xr
from barra_read import date_seq 

from metpy.calc import vertical_velocity_pressure as omega
import metpy.calc as mpcalc
from metpy.units import units

def read_barpa(domain, time, experiment, forcing_mdl, ensemble, run_start):

	experiment="cmip5"
	forcing_mdl="ACCESS1-0"
	ensemble="r1i1p1"
	run_start="19980101T0000Z"
	time = [dt.datetime(2000,12,15,6), dt.datetime(2000,12,20,0)]

	#Given the start and end time quereyed (parameter=time), get the file paths for BARPA
	# containing these dates.
	date_list = 
	#Given the start and end time quereyed (parameter=time), get the file paths for BARPA
	# containing these dates.date_seq(time, "hours", 6)
	geopt_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	hus_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp26/spec_hum*"))
	ta_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	ua_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	va_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	
	huss_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	tas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	uas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	vas_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))
	ps_files = np.sort(glob.glob("/g/data/du7/barpa/trials/BARPA_12km/"+\
		    experiment+"/"+forcing_mdl+\
		    "/"+ensemble+"/"+run_start+"/*/pp2/geop_ht_uv*"))

	cycle_start = [dt.datetime.strptime(\
				geopt_files[i].split("/")[-1].split("-")[-1].split(".")[0],\
				"%Y%m%dT%H%MZ") \
			    for i in np.arange(len(geopt_files)) ] 
	file_times = [ date_seq([cycle_start[i]+dt.timedelta(hours=6), \
			    cycle_start[i]+dt.timedelta(days=10)], "hours", 6) \
			for i in np.arange(len(geopt_files)) ] 
	fidx = [ np.in1d( file_times[i], date_list).any() for i in np.arange(len(geopt_files)) ]
