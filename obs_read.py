import numpy as np
import pandas as pd
import datetime as dt
import math
import os
import pytz
from tzwhere import tzwhere
#from event_analysis import get_aus_stn_info
import netCDF4 as nc
try:
	import sharppy.sharptab.profile as profile
	import sharppy.sharptab.utils as utils
	import sharppy.sharptab.params as params
	import sharppy.sharptab.interp as interp
	import sharppy.sharptab.winds as winds
except:
	pass

def analyse_events(event_type, domain, model=None, lightning_only=False, lightning_thresh=0, wg_thresh=0,\
			remove_tc_affected=False, add_sta=True):
        #Read data and combine

	#Load AWS, lightning and model data
	if domain == "sa_small":
		aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"\
                        +"all_daily_max_wind_gusts_sa_1979_2017.pkl")
		lightning = load_lightning(domain="sa_small",daily=False)

		if model == "erai":
			erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_points_sa_small_1979_2017.pkl")
			erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_fc_points_1979_2017.pkl")
		elif model == "barra":
			barra = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"barra_points_sa_small_2003_2016_daily_max.pkl")
			barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
				+"barra_r_fc_points_daily_2003_2016.pkl")
	elif domain == "aus":
		lightning = load_lightning(domain="aus",daily=False)
		aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/"\
                        	+"all_daily_max_wind_gusts_6hrly_aus_1979_2017.pkl")

		if model == "erai":
			erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_points_aus1979_2017.pkl")

	#If interested in the jdh dataset, read and combine with AWS, model and lightning
	if event_type == "jdh":
		jdh = read_non_synoptic_wind_gusts().set_index("station",append=True)
		if model == "erai":
			df = pd.concat([aws["wind_gust"],jdh["gust (m/s)"],erai,erai_fc["wg10"],\
				lightning["lightning"]] ,axis=1)
		elif model == "barra":
			df = pd.concat([aws["wind_gust"],jdh["gust (m/s)"],barra,barra_r_fc["max_wg10"],\
				lightning["lightning"]], axis=1)
			df = df.dropna(subset=["ml_cape"])
		df["jdh"] = 0
		df.loc[((~df["gust (m/s)"].isna()) & (~df["wind_gust"].isna()) & (df["wind_gust"]>=20)),"jdh"] = 1
		df = df.dropna(subset=["wind_gust"])
		print("No. of JDH events = "+str(df.jdh.sum()))

	#Else, combine just the AWS, model and lightning
	elif event_type == "aws":
		if domain == "sa_small":
			df = pd.concat([aws["wind_gust"].set_index(["an_gust_time_utc","stn_name"]),\
				erai.set_index(["loc_id"],append=True),\
				erai_fc.set_index(["date","loc_id"])["wg10"],\
				lightning.set_index(["date","loc_id"])["lightning"],\
                		],axis=1)
		elif domain == "aus":
			df = pd.concat([aws.set_index(["an_gust_time_utc","stn_name"]),\
				lightning.set_index(["date","loc_id"])["lightning"]],axis=1)
			if add_sta:
				df = verify_sta(df)

		if model == "erai":
			df = pd.concat([df,erai.set_index(["loc_id"],append=True)],axis=1)

		df = df.dropna(subset=["wind_gust"])
		df = df[df.wind_gust >= wg_thresh]

		if lightning_only:
			df = df.dropna(subset=["lightning"])
			df = df[df.lightning >= lightning_thresh]

		if remove_tc_affected:
			df = df[df.tc_affected==0]
	
	return df

def read_upperair_obs(start_date,end_date):

	#Read in upper air obs for 2300 UTC soundings 2010-2015 Adelaide AP

	names = ["record_id","stn_id","date_time","ta","ta_quality","dp","dp_quality",\
		"rh","rh_quality","ws","ws_quality","wd","wd_quality","p","p_quality",
		"z","z_quality","symbol"]
	path = "/short/eg3/ab4502/ExtremeWind/upper_air/"
	fnames = [path + "UA01D_Data_023034_999999999664589.txt", \
		path + "UA01D_Data_014015_999999999664589.txt", \
		path + "UA01D_Data_016001_999999999664589.txt", \
		path + "UA01D_Data_066037_999999999664589.txt" ]
	df = pd.DataFrame()
	for f in fnames:
		df = pd.concat([df, pd.read_csv(f ,\
			names=names,dtype={"ta":np.float64},na_values = ["     ","      ","   ","          "," "])] )
	times = [dt.datetime.strptime(x,"%d/%m/%Y %H:%M") for x in df["date_time"]]
	df["date"] = times
	df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index()

	#Convert wind speed and direction to U and V
	df = uv(df)

	#Group by time/date of observation
	groups = df.groupby(["date", "stn_id"])

	#Loop over date/times/groups and keep ones which have >12 heights for all variables
	min_no_of_points = 12	#Set min no of points required to consider sounding
	mu_cape = []
	times = []
	an = np.array([0,6,12,18])

	df = pd.DataFrame(columns=["stn_id","mu_cape","ml_cape","k_index","s06","Umean800_600",\
			"Umean01","dcape","t_totals"])

	for name, group in groups:

		print(name)

		group.loc[group.dp < -100, "dp"] = np.nan

		group = group.dropna(subset=["ta","dp","rh","ua","va","z"])

		if ((group.shape[0] > min_no_of_points) & (group.p.min()<200) & (group.p.max()>850) ):
			prof = profile.create_profile(pres = group.p, hght = group.z, tmpc = group.ta, \
					dwpc = group.dp, u = group.ua, v=group.va, strictQC=False)
			
			sb_parcel = params.parcelx(prof, flag=1, dp=-10)
			mu_parcel = params.parcelx(prof, flag=3, dp=-10)
			ml_parcel = params.parcelx(prof, flag=4, dp=-10)
			eff_parcel = params.parcelx(prof, flag=6, ecape=100, ecinh=-250, dp=-10)
			p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
			p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
			sfc = prof.pres[prof.sfc]
			s06_u, s06_v = winds.wind_shear(prof, sfc, p6km)
			u01, v01  = winds.mean_wind(prof, sfc, p1km)
			u800_600, v800_600  = winds.mean_wind(prof, 800, 600)
			s06 = utils.KTS2MS( utils.mag(s06_u, s06_v) )
			Umean800_600 = utils.KTS2MS( utils.mag(u800_600, v800_600) )
			Umean01 = utils.KTS2MS( utils.mag(u01, v01) )
			v_totals = params.v_totals(prof)
			c_totals = params.c_totals(prof)
			t_totals = c_totals + v_totals
			dcape = params.dcape(prof)[0]
			if dcape < 0:
				dcape = 0
			ml_el = ml_parcel.elhght
			if np.ma.is_masked(ml_el):
				ml_el = np.nanmax(prof.hght)

			t = an - (group.date.iloc[0].hour + group.date.iloc[0].minute/60.)
			t_idx = np.argmin(abs(t))

			print(ml_parcel.bplus)

			df = pd.concat([ df, \
				pd.DataFrame({"stn_id":group.stn_id.iloc[0],"mu_cape":mu_parcel.bplus, \
						"ml_cape":ml_parcel.bplus, "mu_cin":abs(mu_parcel.bminus),\
						"ml_cin":abs(ml_parcel.bminus),"s06":s06,\
						"Umean800_600":Umean800_600, "k_index":params.k_index(prof) ,\
						"Umean01":Umean01,"t_totals":t_totals, "ml_el":ml_el,\
						"dcape":dcape}, \
						index=[group.date.iloc[0].replace(hour=int(an[t_idx]), minute=0)])],\
					axis=0)

	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/UA_points.pkl")

	return df

def read_aws_daily_sa(loc):

	#Read daily S.A. AWS data which has been downloaded for 1979-2017

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

def read_convective_wind_gusts():

	#Creates a (dataframe of) "convective wind gust" data

	#Read daily AWS data which has been downloaded for 35 stations Australia wide, 1979-2017
	#Remove suspect, wrong or inconsistent quality controlled data
	#Converts to UTC, and adds lightning data, as well as STA reports

	#AWS daily max gusts are moved to the closest analysis time (00, 06, 12, 18 UTC), based on the 
	# "time of max gust" in UTC.
	# If the gust is exactly halfway between analysis times, the earlier time is taken.
	#
	#Lightning data (2005-2015) is then assigned as the maximum stroke count at the AWS station 
	# between the most recent and next analysis time. For example, if a maximum gust is recorded 
	# at 2:20 UTC, then the associated lightning count is taken as the maximum between the 00 and 
	# 06 UTC values. If the gust occurs at an analysis time, the current time and the previous time 
	# are considered. Lightning data has also already been summed over +/- 4 grid points (lat/lon) 
	# relative to each AWS station (around +/- 1 degree).

	#STA reports are considered if the report occurs within 0.5 degrees of lat and lon of an AWS station. 
	# Similar to the AWS data, the reports are assigned to the closest analysis time.

	#In addition, for AWS daily gusts data, a TC_affected flag is raised if a TC was present on the same day
	# within 2 degrees of latitude and longitude, from the BoM best track data 

	#An INCOMPLETE_MONTH flag is raised at a station if for a given month, less than 90% of days are recorded

	#Set csv column names
	names = ["record_id","stn_no","stn_name","locality", "state","lat","lon","district","height","date_str",\
		"wind_gust","quality","wind_dir", "wind_dir_quality", "max_gust_str_lt", \
		"max_gust_time_quality", "eof"]

	#Dict to map station names to
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

	#Set csv read data types
	data_types = dict(record_id=str, stn_no=int, stn_name=str, locality=str, state=str, lat=float, lon=float,\
				district=str, height=str, date_str=str, wind_gust=float, quality=str, \
				wind_dir=str, wind_dir_quality=str, max_gust_str_lt=str, max_gust_time_quality=str,\
				eof=str)

	#Load csv file
	print("LOADING TEXT FILE")
	#f = "/short/eg3/ab4502/ExtremeWind/aws/daily_aus_full/DC02D_Data_999999999643799.txt"
	f = "/short/eg3/ab4502/ExtremeWind/aws/daily_aus_full_2005_2015/DC02D_Data_999999999662395.txt"
	df = pd.read_csv(f, names=names, dtype=data_types, \
		na_values={"wind_gust":'     ', "max_gust_str_lt":"    "})
	df = df.replace({"stn_name":renames})
	df["locality"] = df["locality"].str.strip()
	df["wind_dir"] = df["wind_dir"].str.strip()
	df["stn_name"] = df["stn_name"].str.strip()

	#Get station info
	loc_id = list(df.stn_name.unique())
	lon_list = []
	lat_list = []
	points = []
	for loc in loc_id:
		lon = df[df.stn_name==loc]["lon"].unique()[0]
		lat = df[df.stn_name==loc]["lat"].unique()[0]
		lon_list.append(lon)
		lat_list.append(lat)
		points.append((lon,lat))
	stn_info = pd.DataFrame({"stn_name":loc_id, "lon":lon_list, "lat":lat_list})

	#Split the date column and convert to datetime object
	print("CONVERTING DATES TO DATETIME OBJECTS...")
	df["year"] = df.date_str.str.slice(6,10).astype("int")
	df["month"] = df.date_str.str.slice(3,5).astype("int")
	df["day_lt"] = df.date_str.str.slice(0,2).astype("int")
	df["daily_date_lt"] = [dt.datetime(df["year"][i], df["month"][i], df["day_lt"][i]) \
				for i in np.arange(df.shape[0])]

	#For each month/station, see whether greater than 90% of the month has data. Create a new column with this
	# information
	print("IDENTIFYING INCOMPLETE MONTHS...")
	df["incomplete_month"] = 0
	groups = df.groupby(["stn_name","year","month"])
	cnt = 0
	for name, group in groups:
		days_in_month = group.shape[0]
		comp_days = (~(group["wind_gust"].isna())).sum()
		if (comp_days / float(days_in_month)) < 0.9:
			df.loc[(df.stn_name==group.stn_name.iloc[0]) & (df.year==group.year.iloc[0]) & \
				(df.month==group.month.iloc[0]),"incomplete_month"] = 1

	#Split the time of observed gust column and convert to datetime object. If a gust hasn't been recorded,
	# then assign the gust time the same as the date time for that day
	print("CONVERTING THE TIME OF MAX GUST INTO A DATETIME OBJECT...")
	df.loc[df["max_gust_str_lt"].isna(),"max_gust_str_lt"] = "0000"
	df["hour_lt"] = df.max_gust_str_lt.str.slice(0,2).astype("int")
	df["min_lt"] = df.max_gust_str_lt.str.slice(2,4).astype("int")
	df["gust_time_lt"] = [dt.datetime(df["year"][i], df["month"][i], df["day_lt"][i], \
				df["hour_lt"][i], df["min_lt"][i]) for i in np.arange(df.shape[0])]

	#Remove gusts where the quality flag is "Suspect", "Inconsistent (with other known information" or 
	# "Wrong". These represent 3, 0 and 1 gusts over 25 m/s. Out of the other flags for gusts over 25 m/s,
	# 80 don't have a flag, 72 have "N"ot been quality controlled, 2031 have been qualit"Y" controlled.
	df.loc[np.in1d(df.quality,np.array(["S","W","I"])),"wind_gust"] = np.nan
	
	#Set up TC dataframe
	tc_df = read_bom_shtc("/short/eg3/ab4502/ird_tc_model/tropical_cyclone_prediction/shtc_2604.csv")
	tc_df = tc_df[(tc_df.datetime >= dt.datetime(1979,1,1)) & (tc_df.datetime<dt.datetime(2018,1,1))]
	#Loop through the TCs and assign a stn_name if it is within 2 degree (lat and lon) of a tc
	tc_affected_date = []
	tc_affected_stn = []
	for i in np.arange(tc_df.shape[0]):
		tc_affected_stns = stn_info[(abs(tc_df.iloc[i].lat - stn_info.lat) <= 2) & \
					(abs(tc_df.iloc[i].lon - stn_info.lon) <= 2)]
		if tc_affected_stns.shape[0] > 0:
			for j in np.arange(tc_affected_stns.shape[0]):
				tc_affected_date.append(dt.datetime( tc_df.iloc[i].datetime.year, \
					tc_df.iloc[i].datetime.month, tc_df.iloc[i].datetime.day))
				tc_affected_stn.append(tc_affected_stns.iloc[j].stn_name)
	tc_affected_df = pd.DataFrame({"stn_name":tc_affected_stn, "date":tc_affected_date})
	tc_affected_df = tc_affected_df.drop_duplicates(subset=["date"])

	#For each gust from 2005-2015, get lightning information. As we are interested if the 
	# gust is associated with a thunderstorm, take the number of strokes as the maximum between
	# the previous and next 6-hourly period
	#lightning = load_lightning(domain="aus", daily=False, smoothing=True).set_index(\
	#		["date","loc_id"]).reset_index("loc_id")
	lightning = load_lightning(domain="aus_large", daily=False, smoothing=True).set_index(\
			["date","loc_id"]).reset_index("loc_id")
	locs = np.unique(lightning["loc_id"])
	print("RESAMPLING LIGHTNING DATA WITH A -2 AND 2 ROLLING WINDOW...")
	lightning_resampled = pd.DataFrame()
	for l in locs:
		temp_df = lightning[lightning.loc_id==l]
		temp_df["lightning_prev"] = temp_df["lightning"].sort_index().rolling("6H",\
						closed="both").max()
		temp_df["lightning_next"] = temp_df["lightning"].sort_index().rolling("12H",\
						closed="right").max().shift(-1)
		lightning_resampled = pd.concat([lightning_resampled, temp_df],axis=0)

	#Get tz info
	print("GETTING TZ INFO...")
	tzwhere_mod = tzwhere.tzwhere()
	unique_locs = np.unique(df["stn_name"])
	tz_list = []
	for l in unique_locs:
		tz_str = tzwhere_mod.tzNameAt(df[df.stn_name==l].lat.unique()[0], \
			df[df.stn_name==l].lon.unique()[0]) 
		tz_list.append(pytz.timezone(tz_str))

	#Convert the date-time object to UTC. Needs to be done separately for each station (different time zones)
	print("\nCONVERTING FROM LT TO UTC...\n")
	temp_df2 = pd.DataFrame()
	for l in unique_locs:
		temp_df = df.loc[df.stn_name==l].reset_index()
		for t in np.arange(temp_df.shape[0]):
			try:
				temp_df.loc[t,"gust_time_utc"] = temp_df.loc[t,"gust_time_lt"] - \
					tz_list[np.where(np.array(unique_locs)==l)[0][0]].\
					utcoffset(temp_df["gust_time_lt"][t]) 
			except:
				temp_df.loc[t,"gust_time_utc"] = temp_df.loc[t,"gust_time_lt"] - \
					tz_list[np.where(np.array(unique_locs)==l)[0][0]].\
					utcoffset(temp_df["gust_time_lt"][t], is_dst=False) 

		temp_df2 = pd.concat([temp_df2, temp_df], axis=0)

	df = df.sort_values(["stn_name","daily_date_lt"]).reset_index()
	df["gust_time_utc"] = temp_df2.sort_values(["stn_name","daily_date_lt"]).reset_index()["gust_time_utc"]

	print("CONVERTING UTC GUST TIME BACK TO DAILY DATE")
	df = df.reset_index()
	df["daily_date_utc"] = [dt.datetime(df.loc[i,"gust_time_utc"].year, \
				df.loc[i,"gust_time_utc"].month, \
				df.loc[i,"gust_time_utc"].day) for i in np.arange(df.shape[0])]

	#Get the most recent analysis time (in order to compare to reanalysis, 00, 06, 12, 18 hours),
	# the next analysis time, and the closest analysis time
	print("\nCONVERTING FROM UTC TO CLOSEST (RE)ANALYSIS TIME...")
	an = np.array([0,6,12,18,24])
	an_hour_utc = []
	year_utc = []
	month_utc = []
	day_utc = []
	prev_or_next = []
	for i in np.arange(df.shape[0]):
		t = an - (df["gust_time_utc"][i].hour + df["gust_time_utc"][i].minute/60.)
		t_idx = np.argmin(abs(t))
		if an[t_idx]<24:
			an_hour_utc.append(an[t_idx])
			year_utc.append(df["gust_time_utc"][i].year)
			month_utc.append(df["gust_time_utc"][i].month)
			day_utc.append(df["gust_time_utc"][i].day)
		else:
			temp_time = (df["gust_time_utc"][i] + dt.timedelta(days=1)).replace(hour=0)
			an_hour_utc.append(0)
			year_utc.append(temp_time.year)
			month_utc.append(temp_time.month)
			day_utc.append(temp_time.day)

		if t[t_idx] >= 0:
			prev_or_next.append("prev")
		else:
			prev_or_next.append("next")
		
	df["an_date"] = [dt.datetime(year_utc[i], month_utc[i], day_utc[i], an_hour_utc[i]) \
			for i in np.arange(len(an_hour_utc))]
	df["prev_or_next"] = prev_or_next

	#df["previous_an_date"] = [dt.datetime(year_utc[i], month_utc[i], day_utc[i], \
	#			an_hour_utc[i]) for i in np.arange(df.shape[0])]
	#df["next_an_date"] = df["previous_an_date"] + dt.timedelta(hours=6) 

	#Drop duplicates which are formed by the same gust being recorded at the end of one day (e.g. 23:50 LT)
	# and the start of the next day (e.g. 02:00 LT), which then are given the same time when 
	# converted to UTC and placed at the most recent analysis time (e.g. 12:00 UTC)
	#Keep the entry with the highest wind gust
	df = df.sort_values("wind_gust", ascending=False).drop_duplicates(\
			subset=["an_date","stn_name"]).sort_index() 

	#Insert TC flag from TC dataframe
	df["tc_affected"] = 0
	for i in np.arange(tc_affected_df.shape[0]):
		df.loc[(df["daily_date_utc"]==tc_affected_df.iloc[i].date) & \
			(df["stn_name"]==tc_affected_df.iloc[i].stn_name), "tc_affected"] = 1 

	#Combine lightning and AWS data
	df = pd.concat([df.set_index(["an_date","stn_name"]), \
		lightning_resampled.set_index(["loc_id"],append=True)[["lightning_prev", "lightning_next"]]], \
		axis=1)
	df.loc[df["prev_or_next"]=="prev", "lightning_next"] = np.nan
	df.loc[df["prev_or_next"]=="next", "lightning_prev"] = np.nan
	df["lightning"] = df["lightning_prev"].combine_first(df["lightning_next"])
	df = df.dropna(subset=["wind_gust","lightning"])

	#Add STA
	print("ADDING STA REPORTS...")
	df = df.drop("level_0", axis=1)
	df = verify_sta(df)
	
	#Save as pickle
	df[["state","lat","lon","height","wind_gust","wind_dir","year","month",\
		"day_lt","hour_lt","min_lt","daily_date_utc","gust_time_lt","gust_time_utc",\
		"tc_affected","lightning","incomplete_month","sta_wind","sta_wind_id"]].\
		to_pickle("/short/eg3/ab4502/ExtremeWind/aws/convective_wind_gust_aus_large_2005_2015.pkl")

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

def read_lightning(fname, dmax = False, smoothing=True, loc_id=None, points=None):
	#Read Andrew Dowdy's lightning dataset for a list of points. Either nearest point (smoothing = False) or 
	# sum over +/- 1 degree in lat/lon
	if (loc_id is None) | (points is None):
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
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/"+fname+"_smoothed.pkl")
	else:
		df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/"+fname+".pkl")

	if dmax:

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
	if domain == "aus_large":
		df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_572_stns_smoothed.pkl")
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

def read_bom_shtc(infile):
    """
    Read CSV file downloaded from http://www.bom.gov.au/cyclone/history/#db called shtc_<date downloaded mmdd>.csv
    """

    #print " INFO : Extracting data from site file %s" %(infile)

    #Read infile. Note that lines before header contains all named TCs in the data file. Will change if downloaded again with new TCs
    #Read only columns of Name, datetime, lat, lon, central pressure and max wind speed
    data = pd.read_csv(infile,error_bad_lines=False,header=62,usecols=[0,1,2,7,8,16,49],low_memory = False)

    #Change time axis to datetime objects. Remove data where there is no time
    data_subset = data[data.TM != ' ']
    #data_subset['TM'] = pd.to_datetime(map(str,data_subset['TM'].tolist()),format='%Y-%m-%d %H:%M')    
    data_subset.TM = pd.to_datetime(data_subset.TM,format='%Y-%m-%d %H:%M')

    #Rename to be consistent with read_obs.read_obs
    data_subset = data_subset.rename(columns = {'NAME':'name','DISTURBANCE_ID':'storm_no','TM':'datetime','LAT':'lat','LON':'lon','CENTRAL_PRES':'pressure','MAX_WIND_SPD':'wind_mps'})

    #Change lat, lon to numeric
    data_subset['lat'] = pd.to_numeric(data_subset['lat'])
    data_subset['lon'] = pd.to_numeric(data_subset['lon'])

    #Create a column composed of storm number and storm year to create a unique storm_id
    def storm_id(row):
        return '{}_{}'.format(row['datetime'].year, row['storm_no'])

    data_subset['storm_id'] = data_subset.apply(storm_id, axis=1)


    return data_subset


def read_sta():

	#Read the "cleaned" STA wind report dataset. Cleaned means that some 
	# comments from the original dataset have been removed such that it can actually be
	# read in as a csv

	#Possibly need to screen for non-straight line winds and/or downbursts (e.g. tornadoes
	# but with no squall/burst, tropical cyclones). Also, there are ~6000 reports, so can afford
	# to be strict. E.g. remove reports without comments, reports without a wind speed, etc. 

	sta = pd.read_csv("/short/eg3/ab4502/ExtremeWind/sta_wind_clean.csv",usecols=np.arange(0,8),\
			engine="python")

	#an = np.array([0,6,12,18])
	for i in np.arange(0,sta.shape[0]):
		temp_date = dt.datetime.strptime(sta["Date/Time"].loc[i], "%d/%m/%Y %H:%M")

		#CLOSEST ANALYSIS TIME
		an = np.array([dt.datetime(temp_date.year,temp_date.month,temp_date.day,0), \
			dt.datetime(temp_date.year,temp_date.month,temp_date.day,6),\
			dt.datetime(temp_date.year,temp_date.month,temp_date.day,12),\
			dt.datetime(temp_date.year,temp_date.month,temp_date.day,18)])
		sta.loc[i,"an_date"] = an[np.argmin(abs(temp_date-an))]

	sta["Wind ID"] = sta["Wind ID"].astype(str)
	sta["Comments"] = sta["Comments"].str.lower()
	sta["Max Gust speed"] = sta["Max Gust speed"] * 0.514444

	return sta


def verify_sta(df):

	#Take a dataframe of "events", which contains a date field, and add the Severe Thunderstorm
	# Archive (STA) from the BoM. Output a dataframe with an additional column, "sta_wind"

	#Comments are available in the csv, but are not read in 

	#The input dataframe must be indexed by "daily_date" and "stn_name" (as is the output from analyse_events()

	sta = read_sta()

	df = df.reset_index().rename({"level_0":"daily_dates","level_1":"stn_name"},axis=1)

	#Get station info
	loc_id = list(df.stn_name.unique())
	lon_list = []
	lat_list = []
	points = []
	for loc in loc_id:
		lon = df[df.stn_name==loc]["lon"].unique()[0]
		lat = df[df.stn_name==loc]["lat"].unique()[0]
		lon_list.append(lon)
		lat_list.append(lat)
		points.append((lon,lat))
	stn_info = pd.DataFrame({"stn_name":loc_id, "lon":lon_list, "lat":lat_list})
	
	#For each STA report, find station locations within one degree lat and lon (from the 
	# passed AWS dataframe). Assign the report with that station name. If there is more than 
	# one station, take the one with minimum distance
	sta["stn_name"] = "NULL"
	for i in np.arange(0,sta.shape[0]):
		
		lon_diff = abs(sta.Longitude[i] - stn_info.lon)
		lat_diff = abs(sta.Latitude[i] - stn_info.lat)

		if ( (lon_diff.min() <= 0.5) & (lat_diff.min() <= 0.5) ):
			dist = np.sqrt(np.square( lon_diff) + np.square(lat_diff))
			sta.loc[i, "stn_name"] = stn_info.iloc[dist.idxmin()]["stn_name"]

	sta = sta[~(sta.stn_name=="NULL")]
	sta = sta.drop_duplicates(subset=["stn_name","an_date"]).set_index(["an_date","stn_name"])

	df = df.set_index(["daily_dates","stn_name"])
	df = pd.concat([df,sta[["Max Gust speed","Wind ID"]]], axis=1).dropna(subset=["wind_gust"])
	df = df.rename({"Max Gust speed":"sta_wind","Wind ID":"sta_wind_id"}, axis=1)

	return df
			
			
			


if __name__ == "__main__":

	#df = calc_obs()
	
	#df.to_csv("/home/548/ab4502/working/ExtremeWind/data_obs_"+\
	#	"Nov2012"+".csv",float_format="%.3f")
	
	#read_convective_wind_gusts()

	#df = read_lightning(False)
	#read_aws_daily_aus()
	read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2016,1,1))

