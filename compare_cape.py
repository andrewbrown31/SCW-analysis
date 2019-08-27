#Compare cape between my data, Andrew D's data and ERA-Interim forecast data

import datetime as dt
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from barra_read import date_seq
import matplotlib.pyplot as plt
import numpy as np

time = dt.datetime(2016,9,28,12)

start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
domain = [start_lat,end_lat,start_lon,end_lon]
m = Basemap(llcrnrlon = domain[2], llcrnrlat = domain[0], urcrnrlon = domain[3], \
	urcrnrlat = domain[1], projection="cyl", resolution = "i")


f_erai = nc.Dataset("/g/data/ub4/erai/netcdf/3hr/atmos/oper_fc_sfc/v01/cape/cape_3hrs_ERAI_historical_fc-sfc_20160901_20160930.nc")
f_ab = nc.Dataset("/g/data/eg3/ab4502/ExtremeWind/sa_small/erai/erai_20160901_20160930.nc")

lon_ab = f_ab.variables["lon"][:]
lat_ab = f_ab.variables["lat"][:]
x_ab,y_ab = np.meshgrid(lon_ab,lat_ab)

try:
	f_ad = nc.Dataset("/g/data/eg3/ab4502/ExtremeWind/ad_data/mu_cape/MUCAPE_southern_aus_2010.nc")
	lon_ad = f_ad.variables["lon"][:]
	lat_ad = f_ad.variables["lat"][:]
	x_ad,y_ad = np.meshgrid(lon_ad,lat_ad)
	t_ind_ad = np.where( np.array( date_seq([dt.datetime(2010,1,1), dt.datetime(2010,12,31,18)], "hours", 6) ) == \
				np.array(time) )[0][0]
except:
	pass

lon_erai = f_erai.variables["lon"][:]
lat_erai = f_erai.variables["lat"][:]
x_erai,y_erai = np.meshgrid(lon_erai,lat_erai)

t_ind_ab = np.where( np.array(nc.num2date(f_ab.variables["time"][:], f_ab.variables["time"].units)) == \
			np.array(time) )[0][0]
t_ind_erai = np.where( np.array(nc.num2date(f_erai.variables["time"][:], f_erai.variables["time"].units)) == \
			np.array(time) )[0][0]

plt.close();plt.subplot(131);m.drawcoastlines();m.contourf(x_ab,y_ab,f_ab.variables["mu_cape"][t_ind_ab]);plt.colorbar();plt.title("AB")
plt.subplot(132);m.drawcoastlines();m.contourf(x_erai,y_erai,f_erai.variables["cape"][t_ind_erai]);plt.colorbar();plt.title("ERA-I")
try:
	plt.subplot(133);m.drawcoastlines();m.contourf(x_ad,y_ad,f_ad.variables["MUCAPE_ADowdy_version"][t_ind_ad]);plt.colorbar();plt.title("AD")
except:
	pass
plt.show()
