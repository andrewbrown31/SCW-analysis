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

def read_aws():
	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality","aws_flag",\
			"#"]
	aws = pd.read_csv("/short/eg3/ab4502/ExtremeWind/HM01X_Data_023034_999999999555666.txt"\
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

if __name__ == "__main__":

	#df = calc_obs()
	
	#df.to_csv("/home/548/ab4502/working/ExtremeWind/data_obs_"+\
	#	"Nov2012"+".csv",float_format="%.3f")
	
	obs=calc_obs()
