from erai_read import read_erai, read_erai_points, read_erai_fc
import glob
import datetime as dt
import numpy as np
import netCDF4 as nc
from calc_param import get_dp
import sharppy.sharptab.profile as profile
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.thermo as thermo

def append_tq():
	#Load each ERA-Interim sa_small netcdf file, and append cape*s06^1.67

	start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	domain = [start_lat,end_lat,start_lon,end_lon]
	model = "erai"
	region = "aus"
	dates = []
	for y in np.arange(1980,2019):
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
		
		ta,dp,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,cp,wg10,cape,lon,lat,date_list = \
			read_erai(domain,dates[t])	

		dp = get_dp(ta, hur, dp_mask=False)

		param_file = nc.Dataset(fname,"a")
		tq_var = param_file.createVariable("tq",float,\
		("time","lat","lon"))
		tq_var.units = ""
		tq_var.long_name = "tq"
		tq_var[:] = ta[:,np.where(p==850)[0][0],:,:] + dp[:,np.where(p==850)[0][0],:,:] - \
				1.7*ta[:,np.where(p==700)[0][0],:,:]


		param_file.close()

def append_wbz():
	#Load each ERA-Interim netcdf file, and append wbz

	start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	domain = [start_lat,end_lat,start_lon,end_lon]
	model = "erai"
	region = "aus"
	dates = []
	for y in np.arange(1979,2019):
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
		
		ta,dp,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,cp,wg10,cape,lon,lat,date_list = \
			read_erai(domain,dates[t])	

		dp = get_dp(ta, hur, dp_mask=False)

		agl_idx = (p <= ps)

		#Replace masked dp values
		dp = replace_dp(dp)
		try:
			prof = profile.create_profile(pres = np.insert(p[agl_idx],0,ps), \
				hght = np.insert(hgt[agl_idx],0,terrain), \
				tmpc = np.insert(ta[agl_idx],0,tas), \
				dwpc = np.insert(dp[agl_idx],0,ta2d), \
				u = np.insert(ua[agl_idx],0,uas), \
				v = np.insert(va[agl_idx],0,vas), \
				strictqc=False, omeg=np.insert(wap[agl_idx],0,wap[agl_idx][0]) )
		except:
			p = p[agl_idx]; ua = ua[agl_idx]; va = va[agl_idx]; hgt = hgt[agl_idx]; ta = ta[agl_idx]; \
			dp = dp[agl_idx]
			p[0] = ps
			ua[0] = uas
			va[0] = vas
			hgt[0] = terrain
			ta[0] = tas
			dp[0] = ta2d
			prof = profile.create_profile(pres = p, \
				hght = hgt, \
				tmpc = ta, \
				dwpc = dp, \
				u = ua, \
				v = va, \
				strictqc=False, omeg=wap[agl_idx])



		pwb0 = params.temp_lvl(prof, 0, wetbulb=True)
		hwb0 = interp.to_agl(prof, interp.hght(prof, pwb0) )

		param_file = nc.Dataset(fname,"a")
		wbz_var = param_file.createVariable("wbz",float,\
		("time","lat","lon"))
		wbz_var.units = "m"
		wbz_var.long_name = "wet_bulb_zero_height"
		wbz_var[:] = hwb0

		T1 = abs( thermo.wetlift(prof.pres[0], prof.tmpc[0], 600) - interp.temp(prof, 600) )
		T2 = abs( thermo.wetlift(pwb0, interp.temp(prof, pwb0), sfc) - prof.tmpc[0] )
		Vprime = utils.KTS2MS( 13 * np.sqrt( (T1 + T2) / 2) + (1/3 * (Umean01) ) )

		Vprime_var = param_file.createVariable("Vprime",float,\
		("time","lat","lon"))
		Vprime_var.units = "m/s"
		Vprime_var.long_name = "miller_1972_wind_speed"
		Vprime_var[:] = Vprime

		param_file.close()

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
	append_wbz()
