from plot_param import plot_param
from calc_param import *
from barra_read import read_barra, read_barra_points


model = "barra_wrf3d"	#Name of model (only for output filename purposes)
method = "domain"	#"points" or "domain"
region = "aus" 		#Only applicable if method="domain" - "aus", "sa_small", "sa_large"
time = [dt.datetime(2012,11,20,0,0,0),dt.datetime(2012,11,20,18,0,0)]	#Time range [start,end]



if method == "points":
	if region == "aus":
	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "sa_small":
	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "sa_large":
	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
	domain = [start_lat,end_lat,start_lon,end_lon]

	#Extract variables for given domain and single time
	print("\n	INFO: READING IN BARRA-R DATA...\n")
	ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = read_barra(domain,time)

	#SEE CALC_PARAM FOR LIST OF PARAMETERS
	print("\n	INFO: CALCULATING PARAMETERS\n")
	param = ["mu_cape","s06","srh06"]
	param_out = calc_param_wrf(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
				param,model,save=True,region)

elif method == "points":
	#Define sets of points
	points = [(138.5204, -34.5924)]
	loc_id = ["Adelaide AP"]

	#Read BARRA_R files (analysis)
	print("\n	INFO: READING IN BARRA DATA...\n")
	ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list = \
		read_barra_points(points,time)

	#SEE CALC_PARAM FOR LIST OF PARAMETERS
	param = ["mu_cape","s06","relhum850-500"]
	print("\n	INFO: CALCULATING PARAMETERS\n")
	df = calc_param_points(times,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,param,loc_id)

	df.to_csv("/home/548/ab4502/working/ExtremeWind/data_barra_"+dt.datetime.strftime(time[0],"%Y%m%d")+"_"+dt.datetime.strftime(time[1],"%Y%m%d")+".csv",float_format="%.3f")

