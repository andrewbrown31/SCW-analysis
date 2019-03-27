from plot_param import *

#Set options and plot details
#start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
start_lat = -44; end_lat = -24; start_lon = 112; end_lon = 156
domain = [start_lat,end_lat,start_lon,end_lon]
m = Basemap(llcrnrlon = lon.min(), llcrnrlat = lat.min(), urcrnrlon = lon.max(), \
	urcrnrlat = lat.max(), projection="cyl", resolution = "i")
levels = np.linspace(970,1025,21)
levels_ts = np.linspace(-10+273,20+273,21)
t = dt.datetime(2016,9,28,06)

#Load/draw ERA-Interim
#-----------------------------------------------------------------------------------------------------
#SYSTEM BLACK
#f_ts = nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/"+\
#	"ta_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
#f = nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/psl/"+\
#	"psl_6hrs_ERAI_historical_an-sfc_20160901_20160930.nc")
#plt.subplot(121)
#-----------------------------------------------------------------------------------------------------
#EVENT 1979
t = dt.datetime(1979,11,14,06)
f_ts = nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/"+\
	"ta_6hrs_ERAI_historical_an-pl_19791101_19791130.nc")
f = nc.Dataset("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_sfc/v01/psl/"+\
	"psl_6hrs_ERAI_historical_an-sfc_19791101_19791130.nc")
levels_ts = np.linspace(-10+273,30+273,21)
plt.figure()
#-----------------------------------------------------------------------------------------------------
lon = f.variables["lon"][:]
lat = f.variables["lat"][:]
lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
lon = lon[lon_ind]
lat = lat[lat_ind]
x,y = np.meshgrid(lon,lat)
times = nc.num2date(f.variables["time"][:],f.variables["time"].units)
mslp = np.squeeze(f.variables["psl"][times == t,lat_ind,lon_ind]) / 100.
ts = np.squeeze(f_ts.variables["ta"][times == t,6,lat_ind,lon_ind])
fig = plt.figure()
m.drawcoastlines()
m.contour(x,y,mslp,latlon=True,extend="both",levels=levels,colors="k")
m.contourf(x,y,ts,latlon=True,extend="both",levels=levels_ts)
m.drawmeridians(np.arange(domain[2],domain[3],4),labels=[True,False,False,True])
m.drawparallels(np.arange(domain[0],domain[1],4),labels=[True,False,True,False])

#Load/draw BARRA-R
f_ts = nc.Dataset("/g/data/ma05/BARRA_R/v1/analysis/prs/air_temp/2016/09/"+\
	"air_temp-an-prs-PT0H-BARRA_R-v1-20160928T"+t.strftime("%H")+"00Z.sub.nc")
f = nc.Dataset("/g/data/ma05/BARRA_R/v1/analysis/slv/av_mslp/2016/09/"+\
	"av_mslp-an-slv-PT0H-BARRA_R-v1-20160928T"+t.strftime("%H")+"00Z.nc")
lon = f.variables["longitude"][:]
lat = f.variables["latitude"][:]
lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
lon = lon[lon_ind]
lat = lat[lat_ind]
x,y = np.meshgrid(lon,lat)
times = nc.num2date(f.variables["time"][:],f.variables["time"].units)
mslp = np.squeeze(f.variables["av_mslp"][lat_ind,lon_ind]) / 100.
ts = np.squeeze(f_ts.variables["air_temp"][15,lat_ind,lon_ind])
plt.subplot(122)
m.drawcoastlines()
m.contour(x,y,mslp,latlon=True,extend="both",levels=levels,colors="k")
m.contourf(x,y,ts,latlon=True,extend="both",levels=levels_ts)
m.drawmeridians(np.arange(domain[2],domain[3],4),labels=[True,False,False,True])
m.drawparallels(np.arange(domain[0],domain[1],4),labels=[True,False,True,False])

#Colorbar
cbar_ax = fig.add_axes([0.1,0.1,0.8,0.05])
plt.colorbar(cax=cbar_ax,orientation="horizontal")
plt.show()

