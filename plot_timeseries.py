import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.dates as mdates

param = "mu_cape"
lim = [0,2500]
ltype = "-o"	#-o

#Read csv files which have been saved by calc_*model*.py and plot
df_erai = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/points/erai_points_2010_2015.csv")
#df_erai_wrf = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/adelaideAP/data_erai_wrf_20121101_20121201.csv")
#df_erai_wrf3d = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/adelaideAP/data_erai_wrf3d_20121101_20121201.csv")
#df_erai_wrf_new = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/adelaideAP/data_erai_wrf_new_20121101_20121201.csv")
#df_erai_idl = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/adelaideAP/point_output_AdelaideAP2012.csv",\
	#names=["mu_cape","s06",".","lat","lon","year","month","day","hour"])
#df_erai_idl.index = np.arange(0,df_erai_idl.shape[0])
df_barra = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/points/barra_points_max_2010_2015.csv")
df_obs = pd.read_csv("/g/data/eg3/ab4502//ExtremeWind/adelaideAP/data_obs_points_wrf_20100101_20151231.csv")

df_erai = df_erai[(df_erai.year==2014) & (df_erai.month==10) & (df_erai.loc_id=="Adelaide AP")].reset_index()
df_obs = df_obs[(df_obs.year==2014) & (df_obs.month==10)].reset_index()
df_barra = df_barra[(df_barra.year==2014) & (df_barra.month==10) & (df_barra.loc_id=="Adelaide AP")].reset_index()


data = [df_erai,df_barra,df_obs]
names = ["ERA-Interim","BARRA","Radiosonde"]

plt.figure()

for i in np.arange(0,len(data)):
	t = [dt.datetime(data[i].loc[x].year,data[i].loc[x].month,data[i].loc[x].day\
		,data[i].loc[x].hour) 	for x in np.arange(0,data[i].shape[0])]
	plt.plot(t,data[i][param],ltype,label=names[i])

fmt = mdates.DateFormatter("%d-%m")
ax = plt.gca()
ax.set_ylim(lim)
ax.xaxis.set_major_formatter(fmt)
plt.legend(loc=9)
plt.xticks(rotation=30,size=15)
plt.show()
