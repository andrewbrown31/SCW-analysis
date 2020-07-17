#Do analysis on convection resolving model data (i.e. BARRA-XX)

import xarray as xr
import wrf
import metpy.units as units
import metpy.calc as mpcalc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime as dt



def get_dp(ta,hur,dp_mask=True):

        dp = np.array(mpcalc.dewpoint_rh(ta * units.units.degC, hur * units.units.percent))

        if dp_mask:
                return dp
        else:
                dp = np.array(dp)
                dp[np.isnan(dp)] = -85.
                return dp


#Set the date (closest 6-hourly UTC time), domain and point of interest
date = "20160928T0000Z"
year = date[0:4]
month = date[4:6]
valid = "2016-09-28 05:00"
model = "BARRA_AD"
wg_lead = 0	#In hours; the lead time to give the wind gust field (which is an hourly maximum) relative to other fields (inst.)
point = [-31.1558, 136.8054]

#Check what the wind gust speed time series looks like for a point
xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/spec/max_wndgust10m/"+year+"/"+month+"/max_wndgust10m-fc-spec-PT1H-"+model+"-v1-"+date+".sub.nc").max_wndgust10m.sel({"latitude":point[0], "longitude":point[1]}, method="nearest").plot();plt.show()

#Load 3d data
da_w = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/prs/vertical_wnd/"+year+"/"+month+"/vertical_wnd-fc-prs-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).vertical_wnd
lon = da_w.longitude.values
lat = da_w.latitude.values
x,y = np.meshgrid(lon,lat)
p = da_w.pressure.values
p_3d =   np.moveaxis(np.tile(p,[da_w.shape[1],da_w.shape[2],1]),[0,1,2],[1,2,0])
da_rh = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/prs/relhum/"+year+"/"+month+"/relhum-fc-prs-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).relhum.values
da_ta = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/prs/air_temp/"+year+"/"+month+"/air_temp-fc-prs-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).air_temp.values - 273.15
da_ua = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/spec/uwnd10m/"+year+"/"+month+"/uwnd10m-fc-spec-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).uwnd10m.coarsen({"latitude":20, "longitude":20}, boundary="trim").mean()
da_va = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/spec/vwnd10m/"+year+"/"+month+"/vwnd10m-fc-spec-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).vwnd10m.coarsen({"latitude":20, "longitude":20}, boundary="trim").mean()
qx,qy = np.meshgrid(da_ua.longitude, da_ua.latitude)
terrain = xr.open_dataset("/g/data/ma05/"+model+"/v1/static/topog-fc-slv-PT0H-"+model+"-v1.nc").topog.values
da_z = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/prs/geop_ht/"+year+"/"+month+"/geop_ht-fc-prs-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":valid}).geop_ht.values - terrain

#Load wind gust data
da_wg = xr.open_dataset("/g/data/ma05/"+model+"/v1/forecast/spec/max_wndgust10m/"+year+"/"+month+"/max_wndgust10m-fc-spec-PT1H-"+model+"-v1-"+date+".sub.nc").sel({"time":dt.datetime.strptime(valid, "%Y-%m-%d %H:%M") + dt.timedelta(hours=wg_lead)}).max_wndgust10m

#Calculate a few thermodynamic variables
dp = get_dp(hur=da_rh, ta=da_ta, dp_mask = False)
ta_unit = units.units.degC*da_ta
dp_unit = units.units.degC*dp
p_unit = units.units.hectopascals*p_3d
hur_unit = mpcalc.relative_humidity_from_dewpoint(ta_unit, dp_unit)*\
          100*units.units.percent
q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
           ta_unit,p_unit)

#plot
m = Basemap(llcrnrlon=lon.min(), llcrnrlat=lat.min(), urcrnrlon=lon.max(), urcrnrlat=lat.max(),projection="cyl",resolution="h")
#omega = wrf.omega(q_unit, ta_unit.to("K"), da_w.values, p_3d*100)
#omega = omega.assign_coords(dim_0=p, dim_2=lon, dim_1=lat)
#omega = xr.where((da_z >= 1000) & (da_z <= 3000), omega, np.nan)
#c = xr.plot.contour(omega.mean("dim_0"), levels=[10,20,50], colors=["#d0d1e6","#3690c0","#034e7b"])
c = xr.plot.contour(da_w.min("pressure").coarsen({"latitude":10, "longitude":10}, boundary="trim").min(), levels=[-15,-10,-5], colors=["#d0d1e6","#3690c0","#034e7b"])
plt.colorbar(c)
#c = xr.plot.contour(da_w.max("pressure").coarsen({"latitude":1, "longitude":1}, boundary="trim").max(), levels=[5,10,15], colors=["#fdd49e","#ef6548","#990000"])
#plt.colorbar(c)
m.drawcoastlines()
#plt.contourf(x,y,da_wg,cmap=plt.get_cmap("Reds"),levels=np.arange(0,45,5),alpha=1)
c=xr.plot.contour(da_wg.coarsen({"latitude":10, "longitude":10}, boundary="trim").max(), levels=[25], colors=["#990000"])
m.contourf(x,y,np.squeeze(da_rh[p==300]), cmap=plt.get_cmap("Greys"), alpha=0.3)
m.plot(point[1], point[0], marker="x", color="g", latlon=True, ms=10, mew=3)
plt.quiver(qx, qy, da_ua.values, da_va.values, color="k")

plt.figure()
(xr.where( (da_w.min("pressure") < -5) & (da_wg >= 20) ,1 ,0)).coarsen({"latitude":20, "longitude":20}, boundary="trim").max().plot();m.drawcoastlines()

plt.show()
