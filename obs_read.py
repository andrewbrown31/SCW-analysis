import numpy as np
import pandas as pd
import datetime as dt
import math
import os
import pytz
#from tzwhere import tzwhere
from event_analysis import get_aus_stn_info
import netCDF4 as nc

def calc_obs(start_date,end_date):

	#Read in upper air obs for 2300 UTC soundings 2010-2015 Adelaide AP

	names = ["record_id","stn_id","date_time","ta","ta_quality","dp","dp_quality",\
		"rh","rh_quality","ws","ws_quality","wd","wd_quality","p","p_quality",
		"z","z_quality","symbol"]
	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/upper_air/UA01D_Data_023034_999999999624286.txt",names=names,header=0,dtype={"ta":np.float64},na_values = ["     ","      ","   ","          "," "])
	times = [dt.datetime.strptime(x,"%d/%m/%Y %H:%M") for x in df["date_time"]]
	df["date"] = times
	df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index()

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

	params_df = params_df.reset_index()
	params_df["date"] = [dt.datetime(params_df["year"][t],params_df["month"][t],params_df["day"][t],\
			params_df["hour"][t]) for t in np.arange(params_df.shape[0])]
	params_df = params_df.set_index("date")

	return params_df

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

	#Load text file
	names = ["record_id","stn_no","stn_name","locality", "state","lat","lon","district","height","date_str",\
		"wind_gust","quality","wind_dir", "wind_dir_quality", "max_gust_str_lt", \
		"max_gust_time_quality", "eof"]
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
	data_types = dict(record_id=str, stn_no=int, stn_name=str, locality=str, state=str, lat=float, lon=float,\
				district=str, height=float, date_str=str, wind_gust=float, quality=str, \
				wind_dir=str, wind_dir_quality=str, max_gust_str_lt=str, max_gust_time_quality=str,\
				eof=str)

	print("LOADING TEXT FILE")
	f = "/short/eg3/ab4502/ExtremeWind/aws/daily_aus_full/DC02D_Data_999999999643799.txt"
	df = pd.read_csv(f, names=names, dtype=data_types, \
		na_values={"wind_gust":'     ', "max_gust_str_lt":"    "})
	df.loc[df["max_gust_str_lt"].isna(),"max_gust_str_lt"] = "0000"
	df = df.replace({"stn_name":renames})
	df["locality"] = df["locality"].str.strip()
	df["wind_dir"] = df["wind_dir"].str.strip()
	df.loc[np.in1d(df.quality,np.array(["S","W","I"])),"wind_gust"] = np.nan
	df["year"] = df.date_str.str.slice(6,10).astype("int")
	df = df[df.year<=2017].reset_index()
	
	#Get tz info
	print("GETTING TZ INFO...")
	tzwhere_mod = tzwhere.tzwhere()
	unique_locs = np.unique(df["stn_name"])
	tz_list = []
	for l in unique_locs:
		tz_str = tzwhere_mod.tzNameAt(df[df.stn_name==l].lat.unique()[0], \
			df[df.stn_name==l].lon.unique()[0]) 
		tz_list.append(pytz.timezone(tz_str))

	#Split the max gust date-time up into its components
	print("CONVERTING DATES TO DATETIME OBJECTS...")
	df["day"] = df.date_str.str.slice(0,2).astype("int")
	df["month"] = df.date_str.str.slice(3,5).astype("int")
	df["hour"] = df.max_gust_str_lt.str.slice(0,2).astype("int")
	df["min"] = df.max_gust_str_lt.str.slice(2,4).astype("int")
	df["daily_date"] = [dt.datetime(df["year"][i], df["month"][i], df["day"][i]) \
				for i in np.arange(df.shape[0])]
	#Convert to date-time object
	df["gust_time_lt"] = [dt.datetime(df["year"][i], df["month"][i], df["day"][i], \
				df["hour"][i], df["min"][i]) for i in np.arange(df.shape[0])]

	#Convert the date-time object to UTC. Needs to be done separately for each station (different time zones)
	df["gust_time_utc"] = 0
	print("\nCONVERTING FROM LT TO UTC...\n")
	for l in unique_locs:
		print(l)
		temp_df = df.loc[df.stn_name==l, "gust_time_lt"].reset_index()
		temp_df = [temp_df["gust_time_lt"][t] - \
			tz_list[np.where(np.array(unique_locs)==l)[0][0]].utcoffset(temp_df["gust_time_lt"][t]) \
				for t in np.arange(temp_df.shape[0])]
		df.loc[df.stn_name==l, "gust_time_utc"] = temp_df

	#Edit to be equal to the most recent analysis time (in order to compare to reanalysis, 00, 06, 12, 18 hours)
	print("\nCONVERTING FROM UTC TO MOST RECENT (RE)ANALYSIS TIME...")
	an = np.array([0,6,12,18])
	an_hour_utc = []
	year_utc = []
	month_utc = []
	day_utc = []
	for i in np.arange(df.shape[0]):
		t = an - df["gust_time_utc"][i].hour
		an_hour_utc.append(an[np.where(t<=0)[0][-1]])
		year_utc.append(df["gust_time_utc"][i].year)
		month_utc.append(df["gust_time_utc"][i].month)
		day_utc.append(df["gust_time_utc"][i].day)
	df["an_gust_time_utc"] = [dt.datetime(year_utc[i], month_utc[i], day_utc[i], \
				an_hour_utc[i]) for i in np.arange(df.shape[0])]
	df["an_hour"] = an_hour_utc

	#Drop duplicates which are formed by the same gust being recorded at the end of one day (e.g. 23:50 LT)
	# and the start of the next day (e.g. 02:00 LT), which then are given the same time when 
	# converted to UTC and placed at the most recent analysis time (e.g. 12:00 UTC)
	df = df.drop_duplicates(subset=["an_gust_time_utc","stn_name"])

	df[["stn_name","state","lat","lon","height","wind_gust","wind_dir","year","day","month","hour","min","daily_date","gust_time_lt","gust_time_utc","an_gust_time_utc","an_hour"]].to_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_6hrly_aus_1979_2017.pkl")


	return df

def read_aws_half_hourly_1979(loc,resample=False):
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
	df_full.dates.loc[df_full["dates"]==dt.datetime(2005,10,6)] = dt.datetime(2005,6,10)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2008,5,12)] = dt.datetime(2008,12,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2010,1,12)] = dt.datetime(2010,12,1)
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
	#Read Andrew Dowdy's lightning dataset for a list of points. Either nearest point (smoothing = False) or 
	# sum over +/- 1 degree in lat/lon
	loc_id,points = get_aus_stn_info()
	
	path = "/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning/"
	years = np.arange(2005,2016)
	lightning = np.empty((1460*len(years),241,361))
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
			#Sum all lightning counts within +/- 1 deg of the grid point closest
			# to "point"
			    temp_lightning = np.sum(f.variables["Lightning_observed"]\
					[:,(lat_ind-4):(lat_ind+5),(lon_ind-4):(lon_ind+5)],axis=(1,2))
			else:
			    temp_lightning = f.variables["Lightning_observed"][:,lat_ind,lon_ind]
			
			temp_df = pd.DataFrame(\
				{"lightning":temp_lightning,\
				"loc_id":loc_id[p],"date":time_dt,"lon":points[p][0],\
				"lat":points[p][1],"lon_used":f.variables["lon"][lon_ind],\
				"lat_used":f.variables["lat"][lat_ind]})
			df = df.append(temp_df)
	if smoothing:
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed.pkl")
	else:
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus.pkl")

	print("\n\n RESAMPLING TO DMAX...")
	df = df.set_index("date")
	df_daily = pd.DataFrame()
	for loc in np.unique(df.loc_id):
		print(loc)
		temp_df = pd.DataFrame(df[df.loc_id==loc][["lightning"]].resample("1D").max())
		temp_df["loc_id"] = loc
		df_daily = pd.concat([df_daily,temp_df])
	df_daily = df_daily.set_index("loc_id",append=True)

	if smoothing:
		df_daily.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed_daily.pkl")
	else:
		df_daily.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_daily.pkl")

	return df

def load_lightning(domain="aus",daily=True,smoothing=True):
	#Load csv created by read_lightning
	#Domain can be "aus" or "sa_small"
	if domain == "aus":
		if smoothing:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed.pkl")
		else:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus.pkl")
	elif domain == "sa_small":
		if smoothing:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed.pkl")
		else:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_daily.pkl")
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
	
	df = read_lightning(False)
	#read_aws_daily_aus()
