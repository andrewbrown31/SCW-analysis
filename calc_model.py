#NOTE THAT TO CALCULATE DCAPE, NUMPY HAS BEEN UPGRADED TO 1.16.0. SO, TO RUN, SWAP TO ENVIRONMENT "PYCAT"
#source activate pycat

import numpy as np
from calc_param import calc_param_sharppy, calc_param_wrf, calc_param_wrf_par
from erai_read import read_erai, read_erai_points, read_erai_fc
from barra_read import read_barra, read_barra_points
from barra_ad_read import read_barra_ad
from barra_r_fc_read import read_barra_r_fc
from event_analysis import load_array_points
import datetime as dt
import itertools
import multiprocessing

#Functions to drive *model*_read.py (to extract variables from reanalysis) and calc_param.py (to calculate 
# convective parameters)

def calc_model(model,out_name,method,time,param,issave,region,cape_method):

# - Model is either "erai" or "barra"
# - "out_name" specifies the "model" part of the netcdf file saved if issave = True
# - Method is either "domain" or "points"
# - time is a list of the form [start,end] where start and end are both datetime objects
# - param is a list of strings corresponding to thunderstorm parameters to be calculated. 
#	See calc_params.py for a list
# - issave (boolean) - is the output to be saved to a netcdf file?
# - region is either "aus","sa_small", "sa_large" or "adelaideAP".
# - cape_method refers to the method of deriving CAPE from the reanalysis, and subsequently, the
# 	method for deriving other parameters. If "wrf", then wrf.cape3d is used, and the rest of
#	the parameters have been manually coded. If "SHARPpy", then the SHARPpy package is used
#	to create profile/parcel objects, which contain the parameters required. "wrf" is greatly
#	preferred for efficiency purposes (O(1 sec) to derive parameters for one time step
#	versus O(10 minutes)). However, "SHARPpy" can be used for verification of manually coded
#	parameters.

   if method == "domain":
	if region == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	else:
	    raise NameError("Region must be one of ""aus"", ""sa_small"" or ""sa_large""")
	domain = [start_lat,end_lat,start_lon,end_lon]

	#Extract variables for given domain and time period
	if model=="barra":
		print("\n	INFO: READING IN BARRA DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_barra(domain,time)
	elif model=="erai":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_erai(domain,time)
	elif model=="erai_fc":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		wg10,cape,lon,lat,date_list = \
			read_erai_fc(domain,time)
		param = ["wg10","cape"]
		param_out = [wg10,cape]
		save_netcdf(region,model,date_list,lat,lon,param,param_out)	
	elif model=="barra_ad":
		print("\n	INFO: READING IN BARRA-AD DATA...\n")
		max_max_wg,wg,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_barra_ad(domain,time,wg_only=False)
	elif model=="barra_r_fc":
		print("\n	INFO: READING IN BARRA-R FORECAST DATA...\n")
		wg,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_barra_r_fc(domain,time,wg_only=False)
	else:
		raise NameError("""model"" must be ""erai"", ""erai_fc"" or ""barra""")

	#Calculate parameters
	print("\n	INFO: CALCULATING PARAMETERS\n")
	if model != "erai_fc":
		if cape_method == "SHARPpy":
			param_out = calc_param_sharppy(date_list,ta,dp,hur,hgt,p,ua,va,uas,vas,\
				lon,lat,param,model,out_name,issave,region)
		elif cape_method == "wrf":
		    if (model == "barra_r_fc") or (model == "barra_ad"):
			#IF BARRA-R, include wind gust in the netcdf save function. Also, use every 6 hrs
			date_list_hrs = [t.hour for t in date_list]
			date_inds = np.in1d(np.array(date_list_hrs),np.array([0,6,12,18]))
			param_out = calc_param_wrf(date_list[date_inds],ta[date_inds],dp[date_inds],hur[date_inds],\
				hgt[date_inds],terrain,p,ps[date_inds],ua[date_inds],va[date_inds],\
				uas[date_inds],vas[date_inds],lon,lat,param,model,out_name,issave,region,\
				wg=wg[date_inds])
		    else:
			param_out = calc_param_wrf(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,\
				uas,vas,lon,lat,param,model,out_name,issave,region)
		elif cape_method == "wrf_par":
		    #USES WRF METHOD BUT PASSES DATES IN PARALLEL (USUALLY ONE MONTHS WORTH)
			it = itertools.product([date_list],np.arange(0,len(date_list)),\
				[ta],[dp],[hur],[hgt],[terrain],[p],[ps],[ua],[va],\
				[uas],[vas],[lon],[lat],[param],[model],[out_name],[issave],[region],[False])
			pool = multiprocessing.Pool()
			param_out = pool.map(calc_param_wrf_par,it)
			#Now have the param output. Need to check that it is in the right time order, and save
			pool.close()
			pool.join()
		else:
			raise NameError("""cape_method"" must be ""SHARPpy"" or ""wrf""")

   elif method == "points":
	if region == "adelaideAP":
		points = [(138.5204, -34.5924)]
		loc_id = ["Adelaide AP"]
	elif (model == "barra_ad") or (model == "barra_r_fc"):
		points = []
		loc_id = []
	else:
	    raise NameError("Region must be ""adelaideAP"", or model must be BARRA-AD or BARRA-R")

	if model=="barra":
		print("\n	INFO: READING IN BARRA DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list= \
			read_barra_points(points,times)
		df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
				lon_used,lat_used,param,loc_id,cape_method)
	elif model=="erai":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,\
			date_list = read_erai_points(points,times)
		df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
				lon_used,lat_used,param,loc_id,cape_method)
	elif model=="barra_ad":
		print("\n	INFO: READING IN BARRA-AD DATA...\n")
		start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
		domain = [start_lat,end_lat,start_lon,end_lon]
		max_max_wg10,max_wg10,lon,lat,date_list = \
			read_barra_ad(domain,time,wg_only=True)
		param = ["max_max_wg10","max_wg10"]
		param_out = [max_max_wg10,max_wg10]
	elif model=="barra_r_fc":
		print("\n	INFO: READING IN BARRA-R FORECAST DATA...\n")
		start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
		domain = [start_lat,end_lat,start_lon,end_lon]
		max_wg10,lon,lat,date_list = \
			read_barra_r_fc(domain,time,wg_only=True)
		param = ["max_wg10"]
		param_out = [max_wg10]
	else:
		raise NameError("""model"" must be ""erai"" or ""barra""")

	if issave:
		df.to_csv("/g/data/eg3/ab4502/ExtremeWind/"+region+"/data_"+out_name+"_"+\
			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(times[1],"%Y%m%d")+".csv",float_format="%.3f")	

   return [param,param_out,lon,lat,date_list]

def barra_ad_driver(points,loc_id):
	#Drive calc_model() for each month available in the BARRA dataset, for the sa_small domain

	model = "barra_ad"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	dates = []
	for y in np.arange(2006,2017):
		for m in np.arange(1,13):
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	df = pd.DataFrame()
	if dates[0][0] == dt.datetime(2006,01,01,0):
		dates[0][0] = dt.datetime(2006,01,01,06)
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		param,param_out,lon,lat,times = calc_model(model,model,"points",dates[t],"",False,\
			region,cape_method)	
		temp_df = load_array_points(param,param_out,lon,lat,times,points,loc_id,\
				model,smooth=False)
		df = df.append(temp_df)
		temp_fname_csv = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_"+\
			dates[t][0].strftime("%Y%m")+".csv"
		temp_fname_pkl = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
			"barra_ad_points_"+dates[t][0].strftime("%Y%m")+".pkl"
		if os.path.isfile(temp_fname_pkl):
			temp_exist = pd.read_pickle(temp_fname_pkl)
			temp_df = pd.concat([temp_exist,temp_df])
		temp_df.to_csv(temp_fname_csv)
		temp_df.to_pickle(temp_fname_pkl)
	outname_csv = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_"+\
		"2006_2016.csv"
	outname_pkl = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_"+\
		"2006_2016.pkl"
	if os.path.isfile(outname_pkl):
		exist_df = pd.read_pickle(outname_pkl)
		df = pd.concat([exist_df,df])
	df.to_csv(outname_csv)
	df.to_pickle(outname_pkl)

def barra_r_fc_driver(points,loc_id):
	#Drive calc_model() for each month available in the BARRA dataset, for the sa_small domain

	model = "barra_r_fc"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	smooth = False
	dates = []
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","non_sc_stp","vo","lr1000","lcl",\
		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
		"cape*s06","cape*ssfc6","td950","td850","td800","cape700","wg"]
	for y in np.arange(2003,2017,1):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	df = pd.DataFrame()
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		#To run convective parameters
		#param,param_out,lon,lat,times = calc_model(model,model,"domain",dates[t],param,True,\
		#	region,cape_method)	
		#To run wind gusts point extraction
		param,param_out,lon,lat,times = calc_model(model,model,"points",dates[t],param,False,\
			region,cape_method)
		temp_df = load_array_points(param,param_out,lon,lat,times,points,loc_id,\
				model,smooth=smooth)
		df = df.append(temp_df)
		temp_df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
			"barra_r_fc_points_"+dates[t][0].strftime("%Y%m")+".csv")
		temp_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
			"barra_r_fc_points_"+dates[t][0].strftime("%Y%m")+".pkl")
	df.to_csv("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/barra_r_fc_points_"+\
		"2006_2016.csv")
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/barra_r_fc_points_"+\
		"2006_2016.pkl")

def erai_fc_driver():
	#Drive calc_model() for 2010-2015 forcaset data in ERA-Interim, for the sa_small domain

	model = "erai_fc"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	param = ["wg10","cape"]
	dates = []
	for y in np.arange(1979,2010):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,6,0,0),\
				dt.datetime(y,m+1,1,0,0,0)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		calc_model(model,model,method,dates[t],param,True,region,cape_method)	

if __name__ == "__main__":

#--------------------------------------------------------------------------------------------------
# 	SETTINGS
#--------------------------------------------------------------------------------------------------

	model = "barra"
	cape_method = "wrf_par"

	method = "domain"
	region = "aus"

	experiment = "test"

	#ADELAIDE = UTC + 10:30
	time = [dt.datetime(2016,9,1,0,0,0),dt.datetime(2016,9,30,18,0,0)]
	issave = True

	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(136.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]

#--------------------------------------------------------------------------------------------------
#	RUN CODE
#--------------------------------------------------------------------------------------------------

	if (cape_method == "wrf") | (cape_method == "wrf_par"):
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","vo10","lr1000","lcl",\
			"relhum1000-700","s06","s0500","s01","s03",\
			"cape*s06","dcp","td850","td800","td950","dcape","mlm","dlm",\
			"dcape*cs6","mlm+dcape","mlm*dcape*cs6"]
	elif cape_method == "SHARPpy":
		param = ["ml_cape","mu_cape"]
	elif cape_method == "points_wrf":
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif cape_method == "points_SHARPpy":
		param = ["mu_cape","s06","cape*s06"]	

	#for time in times:
	start = dt.datetime.now()
	[param,param_out,lon,lat,date_list] = calc_model(model,experiment,method,time,\
			param,False,region,cape_method)	
	print(param_out[0][0])
	print(dt.datetime.now()-start)
	#erai_driver(param)
	#barra_driver(param)
	#barra_ad_driver(points,loc_id)
	#barra_r_fc_driver(points,loc_id)
	#param,param_out,lon,lat,date_list = calc_model(model,"test",method,time,param,True,region,cape_method)

