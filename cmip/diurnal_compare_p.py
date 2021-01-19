from dask.diagnostics import ProgressBar
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

#As in diurnal_compare.py but for different diagnostics

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

def load_logit(model, ensemble, p):

	try:
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model+"_"+ensemble+"_"+p+"_historical_1979_2005.nc")
	except:
		model_6hr = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+\
                    model+"_"+ensemble+"_historical_"+p+"_qm_lsm_1979_2005.nc")

	return model_6hr

def load_resampled_era5(p):
	if p == "dcp":
		var = "dcp"
	elif p == "scp":
		var = "scp_fixed"
	elif p == "cs6":
		var = "mucape*s06"
	elif p == "mu_cape":
		var = "mu_cape"
	da = xr.open_mfdataset("/g/data/eg3/ab4502/ExtremeWind/aus/era5/*.nc", concat_dim="time", combine="by_coords")[var]
	da = da.sel({"time":da["time.year"] <= 2005})
	return da.sel({"time":np.in1d(da["time.hour"], [0,6,12,18])})

if __name__ == "__main__":

	lsm = get_era5_lsm()
	path = "/g/data/eg3/ab4502/ExtremeWind/trends/"
	plt.figure(figsize=[6,6])
	load = True
	cnt=1
	alph = ["","a","b","c"]
	thresh = {"logit_aws":0.72, "dcp":0.128, "scp":0.04, "cs6":30768, "mu_cape":230}

	for p in ["dcp","cs6","mu_cape"]:

		if load:
			df = pd.read_csv(path+"cmip5_IPSL_diurnal_"+p+".csv")
			df2 = pd.read_csv(path+"cmip5_diurnal_"+p+".csv")
			#df3 = pd.read_csv(path+"cmip6_diurnal_"+p+".csv")
			df.index = [3,9,15,21]
			df2.index = [0,6,12,18]
			#df3.index = [0,6,12,18]
		else:
			ProgressBar().register()
			access0 = load_logit("ACCESS1-0","r1i1p1",p).persist()
			access3 = load_logit("ACCESS1-3","r1i1p1",p).persist()
			#access_esm = load_logit("ACCESS-ESM1-5","r1i1p1f1",p).persist()
			#access_cm2 = load_logit("ACCESS-CM2","r1i1p1f1",p).persist()
			bnu = load_logit("BNU-ESM","r1i1p1",p).persist()
			cnrm = load_logit("CNRM-CM5","r1i1p1",p).persist()
			gfdl_cm3 = load_logit("GFDL-CM3","r1i1p1",p).persist()
			gfdl2g = load_logit("GFDL-ESM2G","r1i1p1",p).persist()
			gfdl2m = load_logit("GFDL-ESM2M","r1i1p1",p).persist()
			ipsll = load_logit("IPSL-CM5A-LR","r1i1p1",p).persist()
			ipslm = load_logit("IPSL-CM5A-MR","r1i1p1",p).persist()
			miroc = load_logit("MIROC5","r1i1p1",p).persist()
			mri = load_logit("MRI-CGCM3","r1i1p1",p).persist()
			bcc = load_logit("bcc-csm1-1","r1i1p1",p).persist()
			era5 = load_resampled_era5(p).persist()

			x = [access0, access3, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5]
			n = ["ACCESS1-0","ACCESS1-3","BNU-ESM","CNRM","GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M"\
				,"IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MRI-CGCM3","bcc-csm1-1","ERA5"]
			#x = [access0, access3, access_esm, access_cm2, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5]
			#n = ["ACCESS1-0","ACCESS1-3","ACCESS-ESM1-5","ACCESS-CM2","BNU-ESM","CNRM","GFDL-CM3","GFDL-ESM2G","GFDL-ESM2M"\
			#	,"IPSL-CM5A-LR","IPSL-CM5A-MR","MIROC5","MRI-CGCM3","bcc-csm1-1","ERA5"]
			[print(n[i], np.unique(x[i]["time.hour"])) for i in np.arange(len(x))]
			df2 = pd.DataFrame(index=[0,6,12,18])
			df = pd.DataFrame(index=[3,9,15,21])
			#df3 = pd.DataFrame(index=[0,6,12,18])
			for i in np.arange(len(x)):
				print(i)
				try:
					tot = np.nansum(xr.where(lsm==1, np.nansum(x[i][p].values >= thresh[p], axis=0), np.nan))
				except:
					tot = np.nansum(xr.where(lsm==1, np.nansum(x[i].values >= thresh[p], axis=0), np.nan))
				out = []
				if n[i] in ["IPSL-CM5A-LR", "IPSL-CM5A-MR"]:
					times = [3,9,15,21]
				else:
					times = [0,6,12,18]
				for t in times:
					print(t)
					try:
						out.append(np.nansum(xr.where(lsm==1, np.nansum(x[i][p].sel({"time":x[i]["time.hour"]==t}).values >= thresh[p], axis=0), np.nan)) / tot)
					except:
						out.append(np.nansum(xr.where(lsm==1, np.nansum(x[i].sel({"time":x[i]["time.hour"]==t}).values >= thresh[p], axis=0), np.nan)) / tot)
					#out.append(np.nansum(x[i].sel({"time":x[i]["time.hour"]==t})[p].values) / tot)
				if n[i] in ["IPSL-CM5A-LR", "IPSL-CM5A-MR"]:
					df[n[i]] = out
				else:
					df2[n[i]] = out
			df.to_csv(path+"cmip5_IPSL_diurnal_"+p+".csv", index=False)
			df2.to_csv(path+"cmip5_diurnal_"+p+".csv", index=False)
			#df3.to_csv(path+"cmip6_diurnal_"+p+".csv", index=False)

			#del x, access0, access3, access_esm, access_cm2, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5
			del x, access0, access3, bnu, cnrm, gfdl_cm3, gfdl2g, gfdl2m, ipsll, ipslm, miroc, mri, bcc, era5

		ax = plt.subplot(3,1,cnt)
		boxprop = {"lw":1.5}
		box=plt.boxplot([df2.loc[t,np.in1d(df2.columns, ["ERA5"], invert=True)].values for t in [0,6,12,18]], whis = (0,100), positions=[0,6,12,18], widths=2, boxprops=boxprop, whiskerprops=boxprop, medianprops=boxprop, manage_ticks=False)
		[b.set_color("k") for b in box["medians"]]
		#df2.loc[:,np.in1d(df2.columns, ["ERA5"], invert=True)].median(axis=1).plot(ax=ax, legend=False, color="k", marker="o")
		df2.loc[:,np.in1d(df2.columns, ["ERA5"])].plot(ax=ax, legend=False, color="tab:red", marker="o")
		l=df.plot(ax=ax, legend=False, color=["dimgrey","darkgrey"], marker="o")
		#l=df3.plot(ax=ax, legend=False, color=["tab:green","tab:blue"], marker="o")
		ax.xaxis.set_ticks([0,3,6,9,12,15,18,21])
		if cnt < 3:
			plt.gca().set_xticklabels([''])
		#ax.xaxis.set_ticks([3,9,15,21], minor=True)


		ax.grid(True)
		if p=="mu_cape":
			plt.title("c) MUCAPE")
		elif p=="cs6":
			plt.title("b) MUCS6")
		else:
			plt.title(alph[cnt]+") "+p.upper())
		plt.subplots_adjust(bottom=0.2, hspace=0.3)

		plt.xlim([-1, 22])
		cnt=cnt+1
	plt.xlabel("UTC")
	#plt.legend((l.lines[-6], l.lines[-5],l.lines[-4],l.lines[-3],l.lines[-2],l.lines[-1]), ("CMIP5*","ERA5","IPSL-CM5A-LR","IPSL-CM5A-MR","ACCESS-ESM1-5","ACCESS-CM2"), loc=8, bbox_to_anchor=(0.5,-1), ncol=3)
	plt.legend((l.lines[-3],l.lines[-2],l.lines[-1]), ("ERA5","IPSL-CM5A-LR","IPSL-CM5A-MR"), loc=8, bbox_to_anchor=(0.5,-1), ncol=3)
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/diurnal_variability_indices.png",bbox_inches="tight")
