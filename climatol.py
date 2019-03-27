
#Prepare daily max wind gust data from AWS for processing by R package "climatol"

import pandas as pd
import numpy as np

aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_sa_1979_2017.pkl")

full_stations = np.array(["Adelaide AP","Mount Gambier","Woomera"])

elevation = np.array([2,63,166.6])


df = pd.DataFrame()

for stn in full_stations:
	temp = aws[aws.stn_name == stn].sort_values("date").set_index("date")
	df = pd.concat([df,temp["wind_gust"]],axis=1)
	df = df.rename(columns={"wind_gust":stn})

df_long = pd.DataFrame()
est = pd.DataFrame()
for stn in full_stations:
	temp = df[stn].reset_index().drop(columns="date").rename(columns={stn:"wind_gust"})
	df_long = pd.concat([df_long,temp],axis=0)

	lat = aws[aws.stn_name==stn].lat.unique()[0]
	lon = aws[aws.stn_name==stn].lon.unique()[0]
	z = elevation[full_stations == stn][0]
	code = aws[aws.stn_name==stn].stn_no.unique()[0]
	est_temp = pd.DataFrame([[lon,lat,z,code,str(stn)]])
	est = pd.concat([est,est_temp],axis=0)

df_long.to_csv("/short/eg3/ab4502/ExtremeWind/aws/wg_1979-2017.dat",na_rep="NA",columns=["wind_gust"],\
	header=False,index=False)
est.to_csv("/short/eg3/ab4502/ExtremeWind/aws/wg_1979-2017.est",sep=" ",\
	header=False,index=False)
