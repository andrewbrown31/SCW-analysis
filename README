Python scripts which perform the following tasks as part of the ElectraNet/ESCI/NESP project

	- Read reanalysis files from three different datasets (ERA-Interim, ERA5, BARRA) stored on the raijin data server.
	  Output is a list of 4d or 3d variables [ time x (level) x lat x lon] which are relevant for convective indices (CAPE, wind shear, etc)
	  Also outputs relevant diagnostic variables when available (10 m max wind gust, model-derived CAPE, convective precipitation, etc.)
	  These scripts also have functions to extract point (time-series) output from 3d convective parameter netcdf files.

		barra_read.py (one/six-hourly at 12 km resolution over Australia)
		erai_read.py (six-hourly at 0.75 degrees resolution globally)
		era5_read.py (one-hourly at 0.25 degrees resolution globally)		


	- Calculate various thunderstorm/tornado parameters
	  These scripts call the above scripts to load data

		wrf_parallel.py
			Uses the NCAR "wrf-python" package to calculate CAPE/LCL/CIN/LFC/EL. Other relevant parameters are then calculated using a 
			combination of numpy, SkewT, metpy and wrf-python based on SHARPpy/SPC defintions, with some simplifying assumptions needed.
			Has been made parallel using mpi4py, and is able to be calculated across any number of nodes/cpus, so long as the input data
			is not too big to be sent.

		sharp_parallel_mpi.py
			Uses the SHARPpy package to calculate all relevant variables. Is also made parallel using mpi4py, by sending individual 
			spatial gridpoints out to any number of nodes/cpus. Memory is not an issue, and there shouldn't be an upper limit on the 
			number of cpus requested (excpet for raijin queue time)

		calc_param.py
			All routines for calculating parameters in this script are depreciated.
			However, this is still used for netCDF4 output/compression, and has a list of convective parameter output attributes.


	- Drivers for convective parameter scripts
	  All output is saved to /g/data/eg3/ab4502/ExtremeWind/{region}/{model}/{model}_{start date}_{end date}.nc

		jobs/barra_wrfpython/
			Driver scripts for calculating BARRA one-hourly convective indices using wrf-python. Submit one year per job. 
			Output is daily with extra code to concatenate to monthly (using CDO)
			All output is compressed using HDF5 lossless compression as well as lossy compression with least_significant_digit
			set for each variable

		jobs/era5_wrfpython/
			As above but for ERA5

		jobs/erai_wrfpython/
			As above but for ERA-Interim. Output is monthly

		jobs/erai_sharppy/
			Use sharp_parallel_mpi.py on the ERA-Interim dataset

	
	- Set up observational datasets using AWS daily maximum wind gusts, STA reports, the BoM best track TC database, and WWLLN/GPATS lightning data
	  Output is with respect to 35 AWS station locations for 11 years (2005 - 2015)

		obs_read.py

	- Plot things and analysis

		plot_param.py
		plot_timeseries.py
		event_analysis.py
		environmental_scatter.py
		plot_clim.py	(netcdf)
		plot_clim_xr.py (xarray)
		wind_gust.py
		

	- NESP TC-winds analysis
		
		tc_analysis.py
