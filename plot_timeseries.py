import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.dates as mdates

param = "s06"
lim = [0,45]
ltype = "-"	#-o

#Read csv files which have been saved by calc_*model*.py and plot
df_erai = pd.read_csv("/home/548/ab4502/working/ExtremeWind/data_erai_20121101_20121201.csv")
df_barra = pd.read_csv("/home/548/ab4502/working/ExtremeWind/data_barra_20121101_20121201.csv")
df_obs = pd.read_csv("/home/548/ab4502/working/ExtremeWind/data_obs_Nov2012.csv")

data = [df_erai,df_barra,df_obs]
names = ["ERA-Interim","BARRA","Observed"]

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
