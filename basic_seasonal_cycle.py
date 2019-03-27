
#Basic script to plot the monthly distribution of extreme wind gusts from AWS
#data

#SANITY CHECK

import pandas as pd
import datetime as dt

def basic_seasonal_cycle(fname):
	#Read daily max wind gust file for a given file (fname)
	names = ["none1","id","stn","lat","lon","date_str","wg","none2","none3"]
	df = pd.read_csv("/short/eg3/ab4502/ExtremeWind/aws/daily_1979_2017/"\
			+fname,names=names)

	#Convert date string to date time
	df["date"] = [dt.datetime.strptime(t,"%d/%m/%Y") for t in df["date_str"]]

	#Create "month" column
	df["month"] = [t.month for t in df["date"]]

	#Restrict dataframe to days above 20 m/s
	df = df[df["wg"] >= 20]

	#Plot a histogram of months for days above 20 m/s
	plt.figure();plt.title(df["stn"].values[0].strip())
	plt.hist(df["month"],bins=12,range=[1,13],normed=True)

if __name__ == "__main__":

	#For Adelaide AP
	basic_seasonal_cycle("DC02D_Data_023034_999999999565266.txt")
	#For Port Augusta
	basic_seasonal_cycle("DC02D_Data_018201_999999999565266.txt")
	#For Woomera
	basic_seasonal_cycle("DC02D_Data_016001_999999999565266.txt")

	plt.show()
