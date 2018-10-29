from calc_param import *
from erai_read import read_erai, read_erai_points
from barra_read import read_barra, read_barra_points

def calc_model(model,out_name,method,time,param,issave,region):

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

   if method == "domain":
 	if region == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	else:
	    NameError("Region must be one of ""aus"", ""sa_small"" or ""sa_large""")
	domain = [start_lat,end_lat,start_lon,end_lon]

	#Extract variables for given domain and single time
	if model=="barra":
		print("\n	INFO: READING IN BARRA DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_barra(domain,time)
	elif model=="erai":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
			read_erai(domain,time)
	else:
		NameError("""model"" must be ""erai"" or ""barra""")

	#Calculate parameters
	print("\n	INFO: CALCULATING PARAMETERS\n")
	param_out = calc_param_wrf(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,\
		uas,vas,lon,lat,param,model,issave,region)

   elif method == "points":
	if region == "adelaideAP":
		points = [(138.5204, -34.5924)]
		loc_id = ["Adelaide AP"]
	else:
	    NameError("Region must be one of ""adelaideAP""")

	if model=="barra":
		print("\n	INFO: READING IN BARRA DATA...\n")
		#NOTE: read_barra_points needs to be fixed (see changes in read_erai_points)
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list= \
			read_barra_points(points,time)
	elif model=="erai":
		print("\n	INFO: READING IN ERA-Interim DATA...\n")
		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,\
			date_list = read_erai_points(region,time)
	else:
		NameError("""model"" must be ""erai"" or ""barra""")

	df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
			lon_used,lat_used,param,loc_id)

	if issave:
		df.to_csv("/g/data/eg3/ab4502/ExtremeWind/"+region+"/data_"+out_name+"_"+\
			dt.datetime.strftime(time[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(time[1],"%Y%m%d")+".csv",float_format="%.3f")	

if __name__ == "__main__":

	model = "barra"
	out_name = "barra_wrf3d"

	method = "points"
	region = "adelaideAP"

	time = [dt.datetime(2012,11,1,0,0,0),dt.datetime(2012,12,1,0,0,0)]
	param = ["mu_cape","s06"]
	issave = True
	calc_model(model,out_name,method,time,param,issave,region)	
