from obs_read import *


def tc_analysis():

	#Load in AWS data as in read_convective_wind_gusts(), and merge TC track data with the AWS data
	#Save a dataframe with TC and AWS TC info
			
	#Define the grid box width centred around each AWS site, within which to include TCs
	lat_lon_box = 2

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
	f = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_Data_999999999643799.txt"
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
	df["daily_date_lt"] = pd.to_datetime({"year":df["year"], "month":df["month"], "day":df["day_lt"]})

	#Split the time of observed gust column and convert to datetime object. If a gust hasn't been recorded,
	# then assign the gust time the same as the date time for that day
	print("CONVERTING THE TIME OF MAX GUST INTO A DATETIME OBJECT...")
	df.loc[df["max_gust_str_lt"].isna(),"max_gust_str_lt"] = "0000"
	df["hour_lt"] = df.max_gust_str_lt.str.slice(0,2).astype("int")
	df["min_lt"] = df.max_gust_str_lt.str.slice(2,4).astype("int")
	df["gust_time_lt"] = pd.to_datetime({"year":df["year"], "month":df["month"], "day":df["day_lt"],\
				"hour":df["hour_lt"], "minute":df["min_lt"]})

	#Remove gusts where the quality flag is "Suspect", "Inconsistent (with other known information" or 
	# "Wrong". These represent 3, 0 and 1 gusts over 25 m/s. Out of the other flags for gusts over 25 m/s,
	# 80 don't have a flag, 72 have "N"ot been quality controlled, 2031 have been qualit"Y" controlled.
	df.loc[np.in1d(df.quality,np.array(["S","W","I"])),"wind_gust"] = np.nan
	
	#Set up TC dataframe
	tc_df = read_bom_shtc("/g/data/eg3/ab4502/ExtremeWind/shtc_2604.csv")
	tc_df = tc_df[(tc_df.datetime >= dt.datetime(1979,1,1)) & (tc_df.datetime<dt.datetime(2018,1,1))]
	#Loop through the TCs and assign a stn_name if it is within 2 degree (lat and lon) of a tc on
	# that day
	tc_affected_date = []
	tc_affected_stn = []
	tc_affected_lat = []
	tc_affected_lon = []
	tc_affected_ws = []
	for i in np.arange(tc_df.shape[0]):
		if tc_df.iloc[i].type == "T":
			tc_affected_stns = stn_info[(abs(tc_df.iloc[i].lat - stn_info.lat) <= lat_lon_box) & \
						(abs(tc_df.iloc[i].lon - stn_info.lon) <= lat_lon_box)]
			if tc_affected_stns.shape[0] > 0:
				for j in np.arange(tc_affected_stns.shape[0]):
					tc_affected_date.append(dt.datetime( tc_df.iloc[i].datetime.year, \
						tc_df.iloc[i].datetime.month, tc_df.iloc[i].datetime.day))
					tc_affected_stn.append(tc_affected_stns.iloc[j].stn_name)
					tc_affected_lat.append(tc_df.iloc[i].lat)
					tc_affected_lon.append(tc_df.iloc[i].lon)
					tc_affected_ws.append(tc_df.iloc[i].gust_mps)
	tc_affected_df = pd.DataFrame({"stn_name":tc_affected_stn, "date":tc_affected_date,\
				"tc_lat":tc_affected_lat, "tc_lon":tc_affected_lon, \
				"tc_gust":tc_affected_ws})
	tc_affected_df = tc_affected_df.drop_duplicates(subset=["date", "stn_name"])

	merged = pd.concat([tc_affected_df.set_index(["stn_name","date"]), \
		df.rename(columns={"daily_date_lt":"date"}).\
		set_index(["stn_name","date"])],axis=1).\
		dropna(subset=["tc_lat"])[["tc_lat","tc_lon","lat","lon","wind_gust","tc_gust"]]

	import matplotlib.colors
	import matplotlib.pyplot as plt
	from mpl_toolkits.basemap import Basemap
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, urcrnrlat=-10,projection="cyl")

	merged = merged.sort_values("wind_gust", ascending=True)
	maxd= 60
	#maxd= 100
	mind=0
	cm = plt.get_cmap("Reds")
	#cm = plt.get_cmap("Greys")
	norm = matplotlib.colors.Normalize(vmin=mind,vmax=maxd)
	sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
	scale=20
	plt.subplot(211)
	[m.plot(merged["tc_lon"].iloc[i], merged["tc_lat"].iloc[i], "o", linestyle="none",\
		color = cm((merged["wind_gust"].iloc[i] - mind)/(maxd-mind) ),\
		markersize = (merged["wind_gust"].iloc[i] - mind)/(maxd-mind) * scale )\
		 for i in np.arange(merged.shape[0])]
	#merged = tc_df.rename(columns={"lon":"tc_lon", "lat":"tc_lat", "gust_mps":"tc_gust"})
	#merged = merged[(merged.sfc_code == 2) | (merged.sfc_code==3) | (merged.sfc_code==4)]
	#merged = merged.sort_values("tc_gust", ascending=True)
	#[m.plot(merged["tc_lon"].iloc[i], merged["tc_lat"].iloc[i], "o", linestyle="none",\
	#	color = cm((merged["tc_gust"].iloc[i] - mind)/(maxd-mind) ),\
	#	markersize = (merged["tc_gust"].iloc[i] - mind)/(maxd-mind) * scale )\
	#	 for i in np.arange(merged.shape[0])]
	[m.plot(merged["lon"].iloc[i], merged["lat"].iloc[i], "bx", linestyle="none",markersize=10) \
		for i in np.arange(merged.shape[0])] 
	m.drawcoastlines()
	plt.colorbar(sm)
	plt.subplot(212)
	plt.hist([merged.wind_gust, merged.tc_gust],log=True, density=True, color=["r","b"],\
			label=["AWS measured", "Best track estimated"])
	#plt.hist(merged.tc_gust, log=True, label="Best track estimated")
	plt.legend()
	plt.show()
	

