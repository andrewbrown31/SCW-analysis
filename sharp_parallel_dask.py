import dask
import xarray as xr
import datetime as dt
from erai_read import date_seq, format_dates

def sharp_parcel_xr(p,ua,va,hgt,ta,dp):

	#Exact same as sharp parcel, but intended to use xarray/dask
	
	prof = profile.create_profile(pres = p, \
		hght = hgt, \
		tmpc = ta, \
		dwpc = dp, \
		u = ua, \
		v = va, \
		strictqc=False)

	#create parcels
	sb_parcel = params.parcelx(prof, flag=1, dp=-10)
	mu_parcel = params.parcelx(prof, flag=3, dp=-10)
	ml_parcel = params.parcelx(prof, flag=4, dp=-10)
	eff_parcel = params.parcelx(prof, flag=6, ecape=100, ecinh=-250, dp=-10)

	return (prof, mu_parcel ,ml_parcel, sb_parcel, eff_parcel)


if __name__ == "__main__":

	t1 = "2016092800"
	t2 = "2016092818"
	model = "erai"
	start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	domain = [start_lat,end_lat,start_lon,end_lon]
	time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]

	ref = dt.datetime(1900,1,1,0,0,0)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times
	formatted_dates = [format_dates(x) for x in date_list]
	unique_dates = np.unique(formatted_dates)

	ta_file = xr.open_dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ta/ta_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0], decode_times=True, chunks=16)
	z_file = xr.open_dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/z/z_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0], decode_times=True, chunks=16)
	hur_file = xr.open_dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/hur/hur_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0], decode_times=True, chunks=16)
	ua_file = xr.open_dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/ua/ua_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0], decode_times=True, chunks=16)
	va_file = xr.open_dataset(glob.glob("/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/va/va_6hrs_ERAI_historical_an-pl_"+date+"*.nc")[0], decode_times=True, chunks=16)
	
	
