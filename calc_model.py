from calc_param import *
from erai_read import read_erai, read_erai_points, read_erai_fc
from barra_read import read_barra, read_barra_points

def calc_model(model,out_name,method,time,param,issave,region,cape_method):

# - Model is either "erai" or "barra"
# - "out_name" specifies the "model" part of the netcdf file saved if issave = True
# - Method is either "domain" or "points"
# - time is a list of the form [start,end] where start and end are both datetime objects
# - param is a list of strings corresponding to thunderstorm parameters to be calculated. 
#	See calc_params.py for a list
# - issave (boolean) - is the output to be saved to a netcdf file?
# - points is a list of coordinates in the form [(lon1,lat1),...,(lonN,latN)] at which
#	to extract parameters
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
	else:
		raise NameError("""model"" must be ""erai"", ""erai_fc"" or ""barra""")

	#Calculate parameters
	print("\n	INFO: CALCULATING PARAMETERS\n")
	if model != "erai_fc":
		if cape_method == "SHARPpy":
			param_out = calc_param_sharppy(date_list,ta,dp,hur,hgt,p,ua,va,uas,vas,\
				lon,lat,param,out_name,issave,region)
		elif cape_method == "wrf":
			param_out = calc_param_wrf(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,\
				uas,vas,lon,lat,param,out_name,issave,region)
		else:
			raise NameError("""cape_method"" must be ""SHARPpy"" or ""wrf""")

   elif method == "points":
	if region == "adelaideAP":
		points = [(138.5204, -34.5924)]
		loc_id = ["Adelaide AP"]
	else:
	    raise NameError("Region must be one of ""adelaideAP""")

	if model=="barra":
		print("\n	INFO: READING IN BARRA DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list= \
			read_barra_points(points,times)
	elif model=="erai":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,\
			date_list = read_erai_points(points,times)
	else:
		raise NameError("""model"" must be ""erai"" or ""barra""")

	df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
			lon_used,lat_used,param,loc_id,cape_method)

	if issave:
		df.to_csv("/g/data/eg3/ab4502/ExtremeWind/"+region+"/data_"+out_name+"_"+\
			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(times[1],"%Y%m%d")+".csv",float_format="%.3f")	

def barra_driver():
	#Drive calc_model() for each month available in the BARRA dataset, for the sa_small domain

	model = "barra"
	cape_method = "wrf"
	method = "domain"
	region = "sa_large"
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","non_sc_stp","vo","lr1000","lcl",\
		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
		"cape*s06","cape*ssfc6","td950","td850","td800","cape700"]
	dates = []
	for y in [2010,2011,2012,2013,2014,2015]:
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	dates[0][0] = dt.datetime(2010,1,1,6,0,0)
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		calc_model(model,model,method,dates[t],param,True,region,cape_method)	
	
def erai_driver():
	#Drive calc_model() for 2010-2015 in ERA-Interim, for the sa_small domain

	model = "erai"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","non_sc_stp","vo","lr1000","lcl",\
		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
		"cape*s06","cape*ssfc6","td950","td850","td800","cape700"]
	dates = []
	for y in np.arange(2010,2018):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
		    if (m != 12):
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
		    else:
			dates.append([dt.datetime(y,m,1,0,0,0),\
				dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		calc_model(model,model,method,dates[t],param,True,region,cape_method)	

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

	model = "erai_fc"
	cape_method = "wrf"

	method = "domain"
	region = "sa_small"

	experiment = ""

	#ADELAIDE = UTC + 10:30
	#times = [dt.datetime(2010,1,1,6,0,0),dt.datetime(2015,12,31,0,0,0)]
	times = [dt.datetime(2010,1,1,0,0,0),dt.datetime(2010,1,31,18,0,0)]
		#[dt.datetime(2014,9,28,0,0,0),dt.datetime(2014,9,29,0,0,0)]]
	issave = True

	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra","Renmark",\
			"Clare HS","Adelaide AP"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96)]

#--------------------------------------------------------------------------------------------------
#	RUN CODE
#--------------------------------------------------------------------------------------------------

	print("\n	CALCULATING THUNDERSTORM/TORNADO PARAMETERS FOR "+model+" USING "+cape_method+"\n 	FOR "+method+" IN "+region)

	out_name = model+"_"+cape_method+experiment

	if cape_method == "wrf":
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","non_sc_stp","vo","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif cape_method == "SHARPpy":
		param = ["mu_cin","mu_cape","s06","ssfc850","srh01","srh03","srh06","scp",\
			"ship","mmp","relhum850-500","stp"]
	elif cape_method == "points_wrf":
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif cape_method == "points_SHARPpy":
		param = ["mu_cape","s06","cape*s06"]	

	#for time in times:
	#calc_model(model,out_name,method,times,param,issave,region,cape_method)	
	#barra_driver()
	erai_fc_driver()
