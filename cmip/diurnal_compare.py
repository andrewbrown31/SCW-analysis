from mpl_toolkits.axes_grid1 import inset_locator
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt
import xarray as xr
from percent_mean_change import transform_from_latlon, rasterize
import netCDF4 as nc

def get_lsm():
        #Load the ERA-Interim land-sea mask (land = 1)
        lsm_file = nc.Dataset("/g/data/ub4/era5/netcdf/static_era5.nc")
        lsm = np.squeeze(lsm_file.variables["lsm"][:])
        lsm_lon = np.squeeze(lsm_file.variables["longitude"][:])
        lsm_lat = np.squeeze(lsm_file.variables["latitude"][:])
        lsm_file.close()
        return [lsm,lsm_lon,lsm_lat]

def get_mask(lon,lat,thresh=0.5):
        #Return lsm for a given domain (with lats=lat and lons=lon)
        lsm,nat_lon,nat_lat = get_lsm()
        lon_ind = np.where((nat_lon >= lon[0]) & (nat_lon <= lon[-1]))[0]
        lat_ind = np.where((nat_lat >= lat[-1]) & (nat_lat <= lat[0]))[0]
        lsm_domain = lsm[(lat_ind[0]):(lat_ind[-1]+1),(lon_ind[0]):(lon_ind[-1]+1)]
        lsm_domain = np.where(lsm_domain > thresh, 1, 0)

        return lsm_domain

def get_era5_lsm(regrid=False):

        #load the era5 lsm, subset to Australia, and coarsen as in regrid_era5(). Consider the coarse
        # lsm to be land if greater than 50% of the points are land


        f = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/era5_19790101_19790131.nc")
        lsm = get_mask(f.lon.values,f.lat.values)
        if regrid:
                lsm_coarse = xr.DataArray(lsm, dims=("lat","lon")).coarsen({"lat":6,"lon":6},\
                            boundary="trim", side="left").sum()
                return xr.where(lsm_coarse>=18, 1, 0).values
        else:
                return lsm

def load_logit(model, ensemble, era52=False):

        if era52:
                model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model+"_"+ensemble+"_logit_aws_historical_2006_2018.nc")
        else:
                model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model+"_"+ensemble+"_logit_aws_historical_1979_2005.nc")

        return model_6hr

if __name__ == "__main__":
	access0 = load_logit("ACCESS1-0","r1i1p1").persist()
	access3 = load_logit("ACCESS1-3","r1i1p1").persist()
	bnu = load_logit("BNU-ESM","r1i1p1").persist()
	cnrm = load_logit("CNRM-CM5","r1i1p1").persist()
	gfdl_cm3 = load_logit("GFDL-CM3","r1i1p1").persist()
	gfdl2g = load_logit("GFDL-ESM2G","r1i1p1").persist()
	gfdl2m = load_logit("GFDL-ESM2M","r1i1p1").persist()
	ipsll = load_logit("IPSL-CM5A-LR","r1i1p1").persist()
	ipslm = load_logit("IPSL-CM5A-MR","r1i1p1").persist()
	miroc = load_logit("MIROC5","r1i1p1").persist()
	mri = load_logit("MRI-CGCM3","r1i1p1").persist()
	bcc = load_logit("bcc-csm1-1","r1i1p1").persist()
	era5 = load_logit("ERA5","").persist()

	x = [access0, access3, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5]
	n = ["ACCESS1-0","ACCESS1-3","BNU-ESM","CNRM","GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M"\
		,"IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MRI-CGCM3","bcc-csm1-1","ERA5"]
	[print(n[i], np.unique(x[i]["time.hour"])) for i in np.arange(len(x))]
	df = pd.DataFrame(index=[0,6,12,18])
	for i in np.arange(len(x)):
		print(i)
		tot = np.nansum(x[i]["logit_aws"].values)
		out = []
		if n[i] in ["IPSL-CM5A-LR", "IPSL-CM5A-MR"]:
			times = [3,9,15,21]
		else:
			times = [0,6,12,18]
		for t in times:
			print(t)
			out.append(np.nansum(x[i].sel({"time":x[i]["time.hour"]==t})["logit_aws"].values) / tot)
		df[n[i]] = out



