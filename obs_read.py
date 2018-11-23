import numpy as np
import pandas as pd
import datetime as dt
import math
from calc_param import *

def calc_obs():

	#Read in upper air obs for 2300 UTC soundings 2010-2015 Adelaide AP

	names = ["record_id","stn_id","date_time","ta","ta_quality","dp","dp_quality",\
		"rh","rh_quality","ws","ws_quality","wd","wd_quality","p","p_quality",
		"z","z_quality","symbol"]
	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/adelaideAP_obs/UA01D_Data_023034_999999999555752.txt",names=names,header=0,dtype={"ta":np.float64},na_values = ["     ","      ","   ","          "," "])
	times = [dt.datetime.strptime(x,"%d/%m/%Y %H:%M") for x in df["date_time"]]
	df["date"] = times

	#Convert wind speed and direction to U and V
	df = uv(df)

	#Group by time/date of observation
	groups = df.groupby("date")

	#Loop over date/times/groups and keep ones which have >12 heights for all variables
	min_no_of_points = 12	#Set min no of points required to consider sounding
	params = ["ml_cape","ml_cin","mu_cin","mu_cape","s06","srh01","srh03","srh06","scp",\
		"stp","mmp","relhum850-500","crt","lr1000","lcl","cape*s06"]
	params_df_columns = params + ["lat","lon","lon_used","lat_used","loc_id","year",\
			"month","day","hour","minute"]
	params_df = pd.DataFrame(columns=params_df_columns)
	for name, group in groups:
		ta = group.ta[pd.notna(group.ta)]
		dp = group.dp[pd.notna(group.dp)]
		rh = group.rh[pd.notna(group.rh)]
		ua = group.ua[pd.notna(group.ua)]
		va = group.va[pd.notna(group.va)]
		p = group.p[pd.notna(group.p)]
		z = group.z[pd.notna(group.z)]
		dates = group.date_time[pd.notna(group.date_time)]
		var = [ta,dp,rh,ua,va,p,z]
		ind = get_min_var(var)
		length = len(var[ind])
		if ((length > min_no_of_points) & (group.p.min()<200)):
			obs_inds = (pd.notna(group.ta)) & (pd.notna(group.dp)) & \
				(pd.notna(group.rh)) & 	(pd.notna(group.ua)) & \
				(pd.notna(group.va)) & (pd.notna(group.p)) & \
				(pd.notna(group.z))
			temp_time = [dt.datetime.strptime(np.unique(dates[obs_inds])[0],\
				"%d/%m/%Y %H:%M")]
			temp_ta = np.swapaxes(np.array(ta[obs_inds],ndmin=3),1,2)
			temp_dp = np.swapaxes(np.array(dp[obs_inds],ndmin=3),1,2)
			temp_rh = np.swapaxes(np.array(rh[obs_inds],ndmin=3),1,2)
			temp_ua = np.swapaxes(np.array(ua[obs_inds],ndmin=3),1,2)
			temp_va = np.swapaxes(np.array(va[obs_inds],ndmin=3),1,2)
			temp_p = np.swapaxes(np.array(p[obs_inds],ndmin=3),1,2)
			temp_z = np.swapaxes(np.array(z[obs_inds],ndmin=3),1,2)
			temp_df = calc_param_points(temp_time,temp_ta,temp_dp,temp_rh,temp_z,\
					[8.2],temp_p,temp_p[:,0,:],temp_ua,temp_va,temp_ua[:,0,:],\
					temp_va[:,0,:],[138.5196],[-34.9524],[138.5196],[-34.9524]\
					,params,["Adelaide AP"],"points_wrf")
			params_df = params_df.append(temp_df)

	return params_df

def read_aws(loc):
	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality","aws_flag",\
			"#"]
	if loc == "Adelaide AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_023034_999999999557206.txt"
	elif loc == "Woomera":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_016001_999999999557206.txt"
	elif loc == "Coober Pedy AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_016090_999999999557206.txt"
	elif loc == "Port Augusta":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_018201_999999999557206.txt"
	elif loc == "Clare HS":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_021131_999999999557206.txt"
	elif loc == "Marree":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_017126_999999999557790.txt"
	elif loc == "Munkora":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_025557_999999999557790.txt"
	elif loc == "Robe":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026105_999999999557790.txt"
	elif loc == "Loxton":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_024024_999999999557790.txt"
	elif loc == "Coonawarra":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026091_999999999557790.txt"
	elif loc == "Renmark":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_024048_999999999557790.txt"
	elif loc == "Whyalla":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_018120_999999999557790.txt"
	elif loc == "Padthaway South":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026100_999999999557790.txt"
	elif loc == "Nuriootpa":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_023373_999999999557790.txt"
	elif loc == "Rayville Park":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_021133_999999999557790.txt"
	elif loc == "Mount Gambier":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026021_999999999557790.txt"
	elif loc == "Naracoorte":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026099_999999999557790.txt"
	elif loc == "The Limestone":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_026095_999999999557790.txt"
	elif loc == "Parafield":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_023013_999999999557790.txt"
	elif loc == "Austin Plains":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_025562_999999999557790.txt"
	elif loc == "Roseworthy":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_023122_999999999557790.txt"
	elif loc == "Tarcoola":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_016098_999999999557790.txt"
	elif loc == "Edinburgh":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/HM01X_Data_023083_999999999557790.txt"
	
	aws = pd.read_csv(fname\
				,header=1,names=names,dtype={"wind_gust":float},\
				na_values={"wind_gust":'     '})
	aws["day"] = aws.date_str.str.slice(0,2).astype("int")
	aws["month"] = aws.date_str.str.slice(3,5).astype("int")
	aws["year"] = aws.date_str.str.slice(6,10).astype("int")
	aws["hour"] = aws.date_str.str.slice(11,13).astype("int")
	aws["minute"] = aws.date_str.str.slice(14,16).astype("int")
	aws_dt = []
	for i in np.arange(0,aws.shape[0]):
		aws_dt.append(dt.datetime((aws["year"][i]),(aws["month"][i]),\
			(aws["day"][i]),(aws["hour"][i]),(aws["minute"][i])))
	aws["date"] = aws_dt
	return aws

def read_aws_all():
	#locs = ["Adelaide AP","Woomera","Coober Pedy AP","Port Augusta","Clare HS"]
	locs = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	aws = pd.DataFrame()
	for loc in locs:
		print(loc)
		temp_aws = read_aws(loc)
		temp_aws["stn_name"] = loc
		aws = aws.append(temp_aws)
	aws.to_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_wind_gusts_sa_2010_2015.pkl")
	return aws

def load_aws_all():
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_wind_gusts_sa_2010_2015.pkl")
	return aws

def uv(df):
	ua = np.empty(df.shape[0])
	va = np.empty(df.shape[0])
	for i in np.arange(0,df.shape[0]):
		ua[i] = df["ws"][i] * math.sin(math.radians(df["wd"][i]))
		va[i] = df["ws"][i] * math.cos(math.radians(df["wd"][i]))
	df["ua"] = ua
	df["va"] = va
	return df

def get_min_var(var):
	lengths = np.empty(len(var))
	for i in np.arange(0,len(var)):
		lengths[i] = len(var[i])
	ind = np.argmin(lengths)
	return(ind)

def read_synoptic_wind_gusts(loc):
	# Read .xlsx file provided by JDH containing "synoptic wind gust events"
	#loc corresponds to the names of the sheets in the spreadsheet
	#e.g. " Adelaide AP" 

	xl_file = pd.ExcelFile("/short/eg3/ab4502/ExtremeWind/sa_synoptic_gusts.xlsx")
	dfs = {sheet_name: xl_file.parse(sheet_name,header=3,skip_footer=5,skiprows=[4]) \
			for sheet_name in xl_file.sheet_names}
	ad_df = dfs[loc]
	ad_df_dates = []
	for i in np.arange(0,ad_df.shape[0]):
		string = str(ad_df["Year"][i])+"-"+str(ad_df["Month"][i])+"-"+str(ad_df["Day"][i])+\
			" "+str(ad_df["Time"][i])
		ad_df_dates.append(dt.datetime.strptime(string,"%Y-%m-%d %H:%M"))
	ad_df["dates"] = ad_df_dates
	ad_df["dates_utc"] = [x - dt.timedelta(hours=10,minutes=30) for x in ad_df_dates]
	
	return ad_df

def read_non_synoptic_wind_gusts():
	# Read .xlsx file provided by JDH containing "non-synoptic wind gust events"

	xl_file = pd.ExcelFile("/short/eg3/ab4502/ExtremeWind/jdh/sa_non_synoptic_gusts.xlsx")
	df = xl_file.parse(header=3,skiprows=[4]).reset_index()
	df_dates = []
	for i in np.arange(0,df.shape[0]):
		if type(df["Date"][i]) == dt.datetime:
			df_dates.append(df["Date"][i])
		else:
			df_dates.append(dt.datetime.strptime(df["Date"][i],"%d/%m/%Y"))
	df["dates"] = df_dates

	#Add gusts 2-4 to dataframe as extra rows
	ind1 = ~(df["station.1"].isna())
	ind2 = ~(df["station.2"].isna())
	ind3 = ~(df["station.3"].isna())
	ind4 = ~(df["station.4"].isna())
	df0 = df[["gust (m/s)","direction (deg.)","station","dates"]]
	df1 = df[["gust (m/s).1","direction (deg.).1","station.1","dates"]][ind1]
	df2 = df[["gust (m/s).2","direction (deg.).2","station.2","dates"]][ind2]
	df3 = df[["gust (m/s).3","direction (deg.).3","station.3","dates"]][ind3]
	df4 = df[["gust (m/s).4","direction (deg.).4","station.4","dates"]][ind4]
	df1.columns = df2.columns = df3.columns = df4.columns = df0.columns
	df_full = pd.concat([df0,df1,df2,df3,df4])

	#Fix data points (2010-2015) for which day and month are the wrong way around in JDH data
	#Checked for Woomera, Adelaide AP and Port Augusta
	df_full.dates[df_full["dates"]==dt.datetime(2010,7,12)] = dt.datetime(2010,12,7)
	df_full.dates[df_full["dates"]==dt.datetime(2014,6,10)] = dt.datetime(2014,10,6)
	df_full.dates[df_full["dates"]==dt.datetime(2015,7,12)] = dt.datetime(2015,12,7)

	#THINK JDH DATA MAY ALREADY BE IN UTC
	#df_full["dates_utc_start"] = [x - dt.timedelta(hours=10,minutes=30) \
	#		for x in df_full["dates"]]
	df_full.index = df_full["dates"]
	df_full = df_full.sort_index()
	#df_full["dates_utc_end"] = [x + dt.timedelta(hours=24) for x in df_full["dates_utc_start"]]

	return df_full

def read_lightning(smoothing=True):
	#Read Andrew Dowdy's lightning dataset for the following points (nearest grid point)
	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(138.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]
	
	path = "/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning/"
	years = [2010,2011,2012,2013,2014,2015]
	lightning = np.empty((1460*6,241,361))
	df = pd.DataFrame()
	for y in np.arange(0,len(years)):
		print("READING LIGHTNING DATA FOR YEAR "+str(years[y]))
		f = nc.Dataset(path+"lightning_Australasia0.250000degree_6.00000hr_"+\
			str(years[y])+".nc")
		times = f.variables["time"][:]
		time_dt = [dt.datetime(years[y],1,1,0,0,0) + dt.timedelta(hours=int(times[x])) \
				for x in np.arange(0,len(times))]
		for p in np.arange(0,len(points)):
			print(loc_id[p])
			lat_ind = np.argmin(abs(f.variables["lat"][:]-points[p][1]))
			lon_ind = np.argmin(abs(f.variables["lon"][:]-points[p][0]))
			if smoothing:
			#Sum all lightning counts within +/- 100 km of the grid point closest
			# to "point"
			    temp_lightning = np.sum(f.variables["Lightning_observed"][:,(lat_ind-4):(lat_ind+5),(lon_ind-4):(lon_ind+5)],axis=(1,2))
			else:
			    temp_lightning = f.variables["Lightning_observed"][:,lat_ind,lon_ind]
			
			temp_df = pd.DataFrame(\
				{"lightning":temp_lightning,\
				"loc_id":loc_id[p],"date":time_dt,"lon":points[p][0],\
				"lat":points[p][1],"lon_used":f.variables["lon"][lon_ind],\
				"lat_used":f.variables["lat"][lat_ind]})
			df = df.append(temp_df)
	if smoothing:
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed.pkl")
	else:
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa.pkl")
	return df

def load_lightning(smoothing=True):
	#Load csv created by read_lightning
	if smoothing:
		df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed.pkl")
	else:
		df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa.pkl")
	return df

if __name__ == "__main__":

	#df = calc_obs()
	
	#df.to_csv("/home/548/ab4502/working/ExtremeWind/data_obs_"+\
	#	"Nov2012"+".csv",float_format="%.3f")
	
	df = load_lightning()
