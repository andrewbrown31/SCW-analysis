from plot_param import plot_param
from calc_param import *
from erai_read import read_erai, read_erai_points

#Which model ("erai" or "barra"
model = "erai"

#"points" or "domain"? I.e. get parameters over some domain for a single time ("domain"), or get 
# a range of times at a list of points ("points")
method = "points"

if method == "domain":
	#Define spatial domain and time
	start_lat = -39
	end_lat =  -25
	start_lon = 132
	end_lon = 142
	domain = [start_lat,end_lat,start_lon,end_lon]
	time = [dt.datetime(2012,11,20,0,0,0),dt.datetime(2012,11,20,6,0,0)]

	#Extract variables for given domain and single time
	ta,dp,hgt,p,ua,va,uas,vas,lon,lat,date_list = read_erai(domain,time)

	#SEE CALC_PARAM FOR LIST OF PARAMETERS
	param = ["mu_cape","s06"]
	df = calc_param(date_list,ta,dp,hgt,p,ua,va,uas,vas,lon,lat,param,model)

	#Try plotting parameter contour
	time = time[0]
	plot_param(df,param,domain,time,model)

elif method == "points":
	#Define time range
	time = [dt.datetime(2012,11,1,0,0,0),dt.datetime(2012,12,1,0,0,0)]
	#Define list of points
	points = [(138.5204, -34.5924)]
	loc_id = ["Adelaide AP"]

	#Read ERA-Interim files
	print("\n	INFO: READING IN ERA-Interim DATA...\n")
	ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list = \
			read_erai_points(points,time)

	#SEE CALC_PARAM FOR LIST OF PARAMETERS
	print("\n	INFO: CALCULATING PARAMETERS\n")
	param = ["mu_cape","s06","relhum850-500"]
	df = calc_param_mf(date_list,ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used,\
			param,model,loc_id)

	df.to_csv("/home/548/ab4502/working/ExtremeWind/data_erai_"+dt.datetime.strftime(time[0],"%Y%m%d")+"_"+dt.datetime.strftime(time[1],"%Y%m%d")+".csv",float_format="%.3f")

