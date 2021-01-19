import matplotlib.pyplot as plt
from cmip_analysis import load_model_data
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

def save_cmip_mean(models, p, experiment, y1, y2, cmipstr, subset_y1, subset_y2):

	ProgressBar().register()

	era5_data = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_lr36_historical_1979_2005.nc")

	out_hist = load_model_data(models, p, lsm=False,\
		    force_cmip_regrid=True,\
		    experiment=experiment, era5_y1=y1, era5_y2=y2,\
		    y1=y1, y2=y2, save=False, era5_data=era5_data)
	print("Computing mean/median for "+p+"...")
	out_means = []
	for i in np.arange(len(models)):
		sub = out_hist[i].sel({"time":(out_hist[i]["time.year"]>=subset_y1) & (out_hist[i]["time.year"]<=subset_y2)})
		sub = xr.where(np.isinf(sub), np.nan, sub)
		out_means.append(sub.mean("time", skipna=True).values)
	for i in np.arange(len(out_means)):
		plt.contourf(out_means[i]); plt.colorbar()
		plt.savefig("/g/data/eg3/ab4502/figs/CMIP/"+experiment+"/"+models[i][0]+"_"+p+".png")
		plt.close()
	mean = np.mean( np.stack( out_means ), axis=0)
	print("Saving...")
	xr.Dataset(data_vars={p:(("lat","lon"), mean)}, \
		    coords={"lat":out_hist[0].lat.values, "lon":out_hist[0].lon.values}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+p+\
			"_"+str(subset_y1)+"_"+str(subset_y2)+"_ensemble_mean_"+cmipstr+".nc",\
		    engine="h5netcdf")

def save_era5_mean(models, p, y1, y2):

	ProgressBar().register()
	out_hist = load_model_data(models, p, lsm=False,\
		    force_cmip_regrid=True,\
		    experiment="historical", era5_y1=y1, era5_y2=y2,\
		    y1=y1, y2=y2, save=False)
	print("Computing mean/median for "+p+"...")
	mean =	out_hist[0].mean("time").values
	print("Saving...")
	xr.Dataset(data_vars={p:(("lat","lon"), mean)}, \
		    coords={"lat":out_hist[0].lat.values, "lon":out_hist[0].lon.values}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/ERA5__mean_"+p+\
			"_historical_"+str(y1)+"_"+str(y2)+".nc",\
		    engine="h5netcdf")

if __name__ == "__main__":

	models = [ \
                         ["ACCESS1-3","r1i1p1",5,""] ,\
                         ["ACCESS1-0","r1i1p1",5,""] , \
                         ["BNU-ESM","r1i1p1",5,""] , \
                         ["CNRM-CM5","r1i1p1",5,""] ,\
                         ["GFDL-CM3","r1i1p1",5,""] , \
                         ["GFDL-ESM2G","r1i1p1",5,""] , \
                         ["GFDL-ESM2M","r1i1p1",5,""] , \
                         ["IPSL-CM5A-LR","r1i1p1",5,""] ,\
                         ["IPSL-CM5A-MR","r1i1p1",5,""] , \
                         ["MIROC5","r1i1p1",5,""] ,\
                         ["MRI-CGCM3","r1i1p1",5,""], \
                         ["bcc-csm1-1","r1i1p1",5,""], \
                         ]
	models_cmip6 = [ ["ERA5",""], ["ACCESS-ESM1-5", "r1i1p1f1", 6, ""], ["ACCESS-CM2", "r1i1p1f1", 6, ""] ]

	#save_cmip_mean(models, "lr36", "historical", 1979, 2005, "cmip5", 1979, 1998)
	#save_cmip_mean(models, "qmean01", "historical", 1979, 2005, "cmip5", 1979, 1998)
	#save_cmip_mean(models, "mu_cape", "historical", 1979, 2005, "cmip5", 1979, 1998)
	#save_cmip_mean(models, "dcape", "historical", 1979, 2005, "cmip5", 1979, 1998)
	#save_cmip_mean(models, "s06", "historical", 1979, 2005, "cmip5", 1979, 1998)
	#save_cmip_mean(models, "Umean06", "historical", 1979, 2005, "cmip5", 1979, 1998)

	#save_cmip_mean(models, "lr36", "historical", 1979, 2005, "cmip5", 1999, 2018)
	#save_cmip_mean(models, "qmean01", "historical", 1979, 2005, "cmip5", 1999, 2018)
	#save_cmip_mean(models, "mu_cape", "historical", 1979, 2005, "cmip5", 1999, 2018)
	#save_cmip_mean(models, "dcape", "historical", 1979, 2005, "cmip5", 1999, 2018)
	#save_cmip_mean(models, "s06", "historical", 1979, 2005, "cmip5", 1999, 2018)
	#save_cmip_mean(models, "Umean06", "historical", 1979, 2005, "cmip5", 1999, 2018)

	#save_era5_mean([["ERA5",""]], "dcp", 1979, 2005)
	#save_era5_mean([["ERA5",""]], "mucape*s06", 1979, 2005)
	#save_era5_mean([["ERA5",""]], "scp_fixed", 1979, 2005)

	#save_cmip_mean(models, "Umean800_600", "historical", 1979, 2005, "cmip5", 1979, 2005)
	#save_cmip_mean(models, "lr13", "historical", 1979, 2005, "cmip5", 1979, 2005)
	#save_cmip_mean(models, "rhmin13", "historical", 1979, 2005, "cmip5", 1979, 2005)
	save_cmip_mean(models, "q_melting", "historical", 1979, 2005, "cmip5", 1979, 2005)
	#save_cmip_mean(models, "eff_lcl", "historical", 1979, 2005, "cmip5", 1979, 2005)
	#save_cmip_mean(models, "srhe_left", "historical", 1979, 2005, "cmip5", 1979, 2005)

	#save_cmip_mean(models_cmip6, "lr36", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "mhgt", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "ml_el", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "qmean01", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "srhe_left", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "Umean06", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "mu_cape", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "dcape", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "s06", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "srh01_left", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "dcp", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "mucape*s06", "historical", 1979, 2005, "cmip6")
	#save_cmip_mean(models_cmip6, "scp_fixed", "historical", 1979, 2005, "cmip6")
