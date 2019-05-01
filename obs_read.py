import numpy as np
import pandas as pd
import datetime as dt
import math
import os
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

def read_aws(loc,resample=False):
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

	if resample:
		aws = aws.resample("6H",on="date",base=3,\
			loffset=dt.timedelta(hours=3),\
			closed="right").max()

	return aws

def read_aws_daily(loc):
	#Read daily AWS data which has been downloaded for 1979-2017

	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality",\
			"#"]
	if loc == "Adelaide AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_023034_999999999565266.txt"
	elif loc == "Woomera":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_016001_999999999565266.txt"
	elif loc == "Coober Pedy AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_016090_999999999565266.txt"
	elif loc == "Port Augusta":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_018201_999999999565266.txt"
	elif loc == "Clare HS":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_021131_999999999565266.txt"
	elif loc == "Marree":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_017126_999999999565266.txt"
	elif loc == "Munkora":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_025557_999999999565266.txt"
	elif loc == "Robe":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026105_999999999565266.txt"
	elif loc == "Loxton":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_024024_999999999565266.txt"
	elif loc == "Coonawarra":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026091_999999999565266.txt"
	elif loc == "Renmark":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_024048_999999999565266.txt"
	elif loc == "Whyalla":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_018120_999999999565266.txt"
	elif loc == "Padthaway South":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026100_999999999565266.txt"
	elif loc == "Nuriootpa":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_023373_999999999565266.txt"
	elif loc == "Rayville Park":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_021133_999999999565266.txt"
	elif loc == "Mount Gambier":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026021_999999999565266.txt"
	elif loc == "Naracoorte":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026099_999999999565266.txt"
	elif loc == "The Limestone":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_026095_999999999565266.txt"
	elif loc == "Parafield":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_023013_999999999565266.txt"
	elif loc == "Austin Plains":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_025562_999999999565266.txt"
	elif loc == "Roseworthy":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_023122_999999999565266.txt"
	elif loc == "Tarcoola":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_016098_999999999565266.txt"
	elif loc == "Edinburgh":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_023083_999999999565266.txt"
	elif loc == "Port Augusta Power Station":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/DC02D_Data_019066_999999999565467.txt"
	
	aws = pd.read_csv(fname\
				,names=names,dtype={"wind_gust":float},\
				na_values={"wind_gust":'     '})
	aws["day"] = aws.date_str.str.slice(0,2).astype("int")
	aws["month"] = aws.date_str.str.slice(3,5).astype("int")
	aws["year"] = aws.date_str.str.slice(6,10).astype("int")
	aws_dt = []
	for i in np.arange(0,aws.shape[0]):
		aws_dt.append(dt.datetime((aws["year"][i]),(aws["month"][i]),\
			(aws["day"][i])))
	aws["date"] = aws_dt
	
	return aws

def read_aws_daily_aus():
	#Read daily AWS data which has been downloaded for 35 stations Australia wide, 1979-2017
	#Remove suspect, wrong or inconsistent quality controlled data

	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality",\
			"#"]
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}

	path = "/short/eg3/ab4502/ExtremeWind/aws/daily_aus_1979_2018/"
	fnames = os.listdir(path)
	fnames = [path+f for f in fnames]
	
	df = pd.DataFrame()

	for f in fnames:
		print(f)
		
		temp = pd.read_csv(f, header=0
					,names=names,dtype={"wind_gust":float},\
					na_values={"wind_gust":'     '})
		temp["day"] = temp.date_str.str.slice(0,2).astype("int")
		temp["month"] = temp.date_str.str.slice(3,5).astype("int")
		temp["year"] = temp.date_str.str.slice(6,10).astype("int")
		temp_dt = []
		for i in np.arange(0,temp.shape[0]):
			temp_dt.append(dt.datetime((temp["year"][i]),(temp["month"][i]),\
				(temp["day"][i])))
		temp["date"] = temp_dt

		df = pd.concat([df,temp],axis=0)
	
	df = df.replace({"stn_name":renames})
	df.loc[np.in1d(df.quality,np.array(["S","W","I"])),"wind_gust"] = np.nan
	df = df[df.year<=2017]
	df.to_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_aus_1979_2017.pkl")

	return aws

def read_aws_1979(loc,resample=False):
	#Read half-hourly AWS data which has been downloaded for 1979-2017 (although, half 
	# hourly measurments for wind gusts only start in the 1990s)

	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality","aws_flag",\
			"#"]
	if loc == "Adelaide AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_023034_999999999565453.txt"
	elif loc == "Woomera":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_016001_999999999565453.txt"
	elif loc == "Coober Pedy AP":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_016090_999999999565453.txt"
	elif loc == "Port Augusta":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_018201_999999999565453.txt"
	elif loc == "Clare HS":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_021131_999999999565453.txt"
	elif loc == "Marree":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_017126_999999999565453.txt"
	elif loc == "Munkora":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_025557_999999999565453.txt"
	elif loc == "Robe":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026105_999999999565453.txt"
	elif loc == "Loxton":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_024024_999999999565453.txt"
	elif loc == "Coonawarra":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026091_999999999565453.txt"
	elif loc == "Renmark":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_024048_999999999565453.txt"
	elif loc == "Whyalla":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_018120_999999999565453.txt"
	elif loc == "Padthaway South":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026100_999999999565453.txt"
	elif loc == "Nuriootpa":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_023373_999999999565453.txt"
	elif loc == "Rayville Park":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_021133_999999999565453.txt"
	elif loc == "Mount Gambier":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026021_999999999565453.txt"
	elif loc == "Naracoorte":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026099_999999999565453.txt"
	elif loc == "The Limestone":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_026095_999999999565453.txt"
	elif loc == "Parafield":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_023013_999999999565453.txt"
	elif loc == "Austin Plains":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_025562_999999999565453.txt"
	elif loc == "Roseworthy":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_023122_999999999565453.txt"
	elif loc == "Tarcoola":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_016098_999999999565453.txt"
	elif loc == "Edinburgh":
		fname = "/short/eg3/ab4502/ExtremeWind/aws/half_hourly_1979_2017/HM01X_Data_023083_999999999565453.txt"
	
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

	if resample:
		aws = aws.resample("6H",on="date",base=3,\
			loffset=dt.timedelta(hours=3),\
			closed="right").max()

	return aws

def read_aws_all(resample=False):
	#locs = ["Adelaide AP","Woomera","Coober Pedy AP","Port Augusta","Clare HS"]
	locs = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	aws = pd.DataFrame()
	for loc in locs:
		print(loc)
		if resample:
			temp_aws = read_aws(loc,True)
		else:
			temp_aws = read_aws(loc,False)
		temp_aws["stn_name"] = loc
		aws = aws.append(temp_aws)
	if resample:
		aws.to_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_wind_gusts_sa_6hr_2010_2015.pkl")
	else:
		aws.to_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_wind_gusts_sa_2010_2015.pkl")
	return aws

def load_aws_all(resample=False):
	if resample:
		aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_wind_gusts_sa_6hr_2010_2015.pkl")
	else:
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

	#Fix data points for which day and month are the wrong way around in JDH data
	#Checked for Woomera, Adelaide AP and Port Augusta
	#NOTE THAT 28-11-2011 at Port Augusta is dubious (daily max AWS = 17 m/s)
	#NOTE THAT 30-12-2012 at Renmark is dubious (daily max AWS = 12.3 m/s)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2010,7,12)] = dt.datetime(2010,12,7)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2014,6,10)] = dt.datetime(2014,10,6)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2015,7,12)] = dt.datetime(2015,12,7)

	df_full.dates.loc[df_full["dates"]==dt.datetime(2004,12,10)] = dt.datetime(2004,10,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2005,10,06)] = dt.datetime(2005,06,10)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2008,05,12)] = dt.datetime(2008,12,05)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2010,01,12)] = dt.datetime(2010,12,01)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2011,9,11)] = dt.datetime(2011,11,9)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2012,5,9)] = dt.datetime(2012,9,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2015,7,1)] = dt.datetime(2015,1,7)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2016,12,7)] = dt.datetime(2016,7,12)

	df_full.dates.loc[df_full["dates"]==dt.datetime(1983,2,3)] = dt.datetime(1983,3,2)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1986,6,12)] = dt.datetime(1986,12,6)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1989,1,12)] = dt.datetime(1989,12,1)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1992,12,8)] = dt.datetime(1992,8,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1996,12,9)] = dt.datetime(1996,9,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1998,5,11)] = dt.datetime(1998,11,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2002,5,8)] = dt.datetime(2002,8,5)

	#Remove points which have been identified as incorrect
	wrong_points = [dt.datetime(1983,12,24),dt.datetime(2001,11,17)]
	df_full = df_full[~(np.in1d(np.array(df_full.dates),np.array(wrong_points).astype(np.datetime64)))]

	#THINK JDH DATA MAY ALREADY BE IN UTC
	#df_full["dates_utc_start"] = [x - dt.timedelta(hours=10,minutes=30) \
	#		for x in df_full["dates"]]
	df_full.index = df_full["dates"]
	df_full = df_full.sort_index()
	
	#Rename a few stations
	df_full.station.loc[df_full.station=="Adelaide Airport"] = "Adelaide AP"
	df_full.station.loc[df_full.station=="Mt Gambier"] = "Mount Gambier"
	df_full.station.loc[df_full.station=="Coober Pedy"] = "Coober Pedy AP"
	df_full.station.loc[df_full.station=="Port Augusta Power Station"] = "Port Augusta"

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

def load_wind_sa():
	#Load wind_SA.csv into a dataframe
	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/wind_sa.csv",usecols=np.arange(0,12)\
		,header=0,skiprows=[966])
	#Convert date
	df["date"] = [dt.datetime.strptime(df["Date time commenced"][i],"%Y-%m-%d %H:%M:%S") \
			for i in np.arange(0,df.shape[0])]
	return df

def read_clim_ind(ind):

	#Create annual time series' for each season

	seasons = [[2,3,4],[5,6,7],[8,9,10],[11,12,1]]
	names = ["FMA","MJJ","ASO","NDJ"]
	years = np.arange(1979,2017)
	if ind == "nino34":
		#NINO3.4
		df = pd.read_table("/g/data/eg3/ab4502/ird_tc_model/data/nino34.txt",names=np.arange(0,13,1),\
			index_col=0,sep="  ",skiprows=[0,1,2],skipfooter=3,engine="python")
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
		    for s in np.arange(0,len(seasons)):
			if s == 3:	#IF NDJ
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
			else:
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y,seasons[s][2]]])
			time_series.loc[y,names[s]] = mean
			time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])

	#DMI
	elif ind == "dmi":
		df = pd.read_table("/g/data/eg3/ab4502/ird_tc_model/data/dmi.txt",names=np.arange(0,13,1),\
			index_col=0,sep="  ",skiprows=[0,1,2],skipfooter=6,engine="python")
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
		    for s in np.arange(0,len(seasons)):
			if s == 3:	#IF NDJ
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
			else:
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y,seasons[s][2]]])
			time_series.loc[y,names[s]] = mean
			time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])

	#SAM
	elif ind == "sam":
		df = pd.read_table("/g/data/eg3/ab4502/ird_tc_model/data/sam.txt",names=np.arange(0,13,1),\
			index_col=0,sep="  | ",engine="python")
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
		    for s in np.arange(0,len(seasons)):
			if s == 3:	#IF NDJ
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
			else:
				mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
				df.loc[y,seasons[s][2]]])
			time_series.loc[y,names[s]] = mean
			time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])

	try:
		return time_series
	except:
		raise NameError("MUST BE ""sam"", ""nino34"" or ""dmi""")

if __name__ == "__main__":

	#df = calc_obs()
	
	#df.to_csv("/home/548/ab4502/working/ExtremeWind/data_obs_"+\
	#	"Nov2012"+".csv",float_format="%.3f")
	
	df = load_lightning()
