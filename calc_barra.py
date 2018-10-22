from plot_param import plot_param
from calc_param import *
from barra_read import read_barra, read_barra_mf

#Which model
model = "barra"

#"points" or "domain"? I.e. get parameters over some domain for a single time (domain), or get 
# a range of times at a select few points (points)
method = "points"

if method == "domain":
	#Define spatial domain and time
	start_lat = -39
	end_lat = -25
	start_lon = 132
	end_lon = 142
	domain = [start_lat,end_lat,start_lon,end_lon]
	time = dt.datetime(2012,11,28,12,0,0)

	#Extract variables for given domain and single time
	ta,dp,hgt,p,ua,va,uas,vas,lon,lat = read_barra(domain,time)

	#Calculate a given parameter for all grid points in domain
	#SEE CALC_PARAM FOR LIST OF PARAMETERS
	param = ["mu_cape","s06","mu_cin","hel03","hel06","ship","hgz_depth","dcp","mburst","mmp"]
	df = calc_param(ta,dp,hgt,p,ua,va,uas,vas,lon,lat,param,model)

	#Try plotting parameter contour
	plot_param(df,param,domain,time,model)

elif method == "points":
	#Define time range
	time = [dt.datetime(2012,11,1,0,0,0),dt.datetime(2012,11,4,0,0,0)]
	#Define sets of points
	points = [(138.5204, -34.5924)]
	loc_id = ["Adelaide AP"]

	#Read BARRA_R files (analysis)
	print("\n	INFO: READING IN BARRA DATA...\n")
	ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list = \
		read_barra_mf(points,time)

	param = ["mu_cape","s06","relhum850-500"]
	#Calculate parameters
	print("\n	INFO: CALCULATING PARAMETERS\n")
	df = calc_param_mf(date_list,ta,dp,hur,hgt,p,ua,va,uas,vas,lon,lat,lon_used,lat_used\
		,param,model,loc_id)

	df.to_csv("/home/548/ab4502/working/ExtremeWind/data_barra_"+dt.datetime.strftime(time[0],"%Y%m%d")+"_"+dt.datetime.strftime(time[1],"%Y%m%d")+".csv",float_format="%.3f")
#CHECK THAT VALUES OBTAINED FOR A SINGLE SPATIAL POINT OVER MFs IS THE SAME AS DOMAIN WIDE 
# 	- checked a few days against plots, but should probably do this more thoroughly
