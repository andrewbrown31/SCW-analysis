from erai_read import read_erai, read_erai_points, read_erai_fc
import glob
import datetime as dt
import numpy as np
import netCDF4 as nc

def append_capeS06():
	#Load each ERA-Interim sa_small netcdf file, and append cape*s06^1.67

	model = "erai"
	region = "sa_small"
	dates = []
	for y in np.arange(1979,2018):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(dates[t][0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(dates[t][-1],"%Y%m%d")+".nc"
		
		param_file = nc.Dataset(fname,"a")
		param_file.variables["cape*s06"][:] = param_file.variables["mu_cape"][:] * \
			np.power(param_file.variables["s06"][:],1.67)
		param_file.close()

def append_ducs3():
	#Load each ERA-Interim sa_small netcdf file, and append ducs3

	model = "erai"
	region = "sa_small"
	dates = []
	for y in np.arange(1979,2018):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(dates[t][0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(dates[t][-1],"%Y%m%d")+".nc"
		
		param_file = nc.Dataset(fname,"a")
		ducs3_var = param_file.createVariable("mlm*dcape*cs3",float,\
		("time","lat","lon"))
		ducs3_var.units = ""
		ducs3_var.long_name = ""
		ducs3_var[:] = (param_file.variables["mlm+dcape"][:]/30.) * \
			((param_file.variables["ml_cape"][:] * np.power(param_file.variables["s03"][:],1.67))/20000.)
		param_file.close()

def append_dp():
	#Load each ERA-Interim sa_small netcdf file, and append pressure-averaged dp

	start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	domain = [start_lat,end_lat,start_lon,end_lon]

	model = "erai"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	dates = []
	for y in np.arange(1979,2018):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_erai(domain,dates[t])	
		dp850_500 = np.nanmean(dp[:,(p <= 850) & (p>=500),:,:],axis=1)
		dp1000_700 = np.nanmean(dp[:,(p <= 1000) & (p>=700),:,:],axis=1)

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(dates[t][0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(dates[t][-1],"%Y%m%d")+".nc"
		
		param_file = nc.Dataset(fname,"a")
		
		if "dp850-500" not in param_file.variables.keys():
			dp850_500_var = param_file.createVariable("dp850-500",dp850_500.dtype,\
			("time","lat","lon"))
			dp850_500_var.units = "degC"
			dp850_500_var.long_name = "dew_point_850_500_hPa"
			dp850_500_var[:] = dp850_500
			dp1000_700_var = param_file.createVariable("dp1000-700",dp1000_700.dtype,\
				("time","lat","lon"))
			dp1000_700_var[:] = dp1000_700
			dp1000_700_var.units = "degC"
			dp1000_700_var.long_name = "dew_point_1000_700_hPa"
		else:
			param_file.variables["dp850-500"][:] = dp850_500
			param_file.variables["dp1000-700"][:] = dp1000_700
		
		param_file.close()

def rename_barra_wg():

	fnames = np.sort(glob.glob("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_r_fc/barra_r_fc_2003*"))
	for fname in fnames:
		print(fname)
		f = nc.Dataset(fname,"a")
		wg = f.variables["wg"]
		t = f.variables["time"]
		lat = f.variables["lat"]
		lon = f.variables["lon"]
		max_wg10_var = f.createVariable("max_wg10",wg[:].dtype,("time","lat","lon"))
		max_wg10_var.long_name = wg.long_name
		max_wg10_var.units = wg.units
		max_wg10_var[:] = wg[:]
		f.close()

def add_units():

	fnames = np.sort(glob.glob("/g/data/eg3/ab4502/ExtremeWind/sa_small/barra_r_fc/barra_r_fc_2003*"))
	for fname in fnames:
		f = nc.Dataset(fname,"a")
		max_wg10 = f.variables["max_wg10"]
		wg = f.variables["wg"]
		max_wg10.units = wg.units
		f.close()
	
def append_cond():
	#Load each ERA-Interim sa_small netcdf file, and append COND

	model = "erai"
	region = "aus"
	dates = []
	for y in np.arange(1979,2018):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(dates[t][0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(dates[t][-1],"%Y%m%d")+".nc"
		
		param_file = nc.Dataset(fname,"a")

		#LOAD VARIABLES FOR COND
		mlcape = param_file.variables["ml_cape"][:]
		s06 = param_file.variables["s06"][:]
		mlm = param_file.variables["mlm"][:]
		dcape = param_file.variables["dcape"][:]

		#Calculate COND
		sf = ((s06>=30) & (dcape<500) & (mlm>=26) )                
		mf = ((mlcape>120) & (dcape>350) & (mlm<26) )     
		cond = sf  | mf
		cond = cond * 1.0
		sf = sf * 1.0
		mf = mf * 1.0

		#Add to file		
		if "cond" in param_file.variables.keys():
			cond_var = param_file.variables["cond"]
			mf_var = param_file.variables["mf"]
			sf_var = param_file.variables["sf"]
		else:
			cond_var = param_file.createVariable("cond",cond.dtype,("time","lat","lon"))
			mf_var = param_file.createVariable("mf",mf.dtype,("time","lat","lon"))
			sf_var = param_file.createVariable("sf",sf.dtype,("time","lat","lon"))
		cond_var[:] = cond
		cond_var.units = "is_cond"
		cond_var.long_name = "conditional_parameter"
		mf_var[:] = mf
		mf_var.units = "is_mf"
		mf_var.long_name = "mesoscale_forcing"
		sf_var[:] = sf
		sf_var.units = "is_sf"
		sf_var.long_name = "synoptic_scale_forcing"

		param_file.close()

def append_cond_barra():
	#Load each ERA-Interim sa_small netcdf file, and append COND

	model = "barra"
	region = "sa_small"
	dates = []
	for y in np.arange(2003,2017):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(dates[t][0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(dates[t][-1],"%Y%m%d")+".nc"
		
		param_file = nc.Dataset(fname,"a")

		#LOAD VARIABLES FOR COND
		mlcape = param_file.variables["ml_cape"][:]
		s06 = param_file.variables["s06"][:]
		mlm = param_file.variables["mlm"][:]
		dcape = param_file.variables["dcape"][:]

		#Calculate COND
		sf = ((s06>30) & (dcape<500) & (mlm>=26) )                
		mf = ((mlcape>120) & (dcape>350) & (mlm<26) )     
		cond = sf  | mf
		cond = cond * 1.0
		sf = sf * 1.0
		mf = mf * 1.0

		#Add to file		
		if "cond" in param_file.variables.keys():
			cond_var = param_file.variables["cond"]
			mf_var = param_file.variables["mf"]
			sf_var = param_file.variables["sf"]
		else:
			cond_var = param_file.createVariable("cond",cond.dtype,("time","lat","lon"))
			mf_var = param_file.createVariable("mf",mf.dtype,("time","lat","lon"))
			sf_var = param_file.createVariable("sf",sf.dtype,("time","lat","lon"))
		cond_var[:] = cond
		cond_var.units = "is_cond"
		cond_var.long_name = "conditional_parameter"
		mf_var[:] = mf
		mf_var.units = "is_mf"
		mf_var.long_name = "mesoscale_forcing"
		sf_var[:] = sf
		sf_var.units = "is_sf"
		sf_var.long_name = "synoptic_scale_forcing"

		param_file.close()

if __name__ == "__main__":
	#append_dp()
	#append_capeS06()
	#rename_barra_wg()
	append_cond()
