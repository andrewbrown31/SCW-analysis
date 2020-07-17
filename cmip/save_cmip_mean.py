from cmip_analysis import load_model_data
import numpy as np
import xarray as xr

def save_cmip_mean(models, p, experiment, y1, y2, cmipstr):

	out_hist = load_model_data(models, p, lsm=True,\
		    force_cmip_regrid=True,\
		    experiment=experiment, era5_y1=y1, era5_y2=y2,\
		    y1=y1, y2=y2, save=False)
	print("Computing mean/median for "+p+"...")
	mean = np.median(\
		np.stack([out_hist[i].mean("time").values for i in np.arange(1,len(models))]),\
		axis=0)
	print("Saving...")
	xr.Dataset(data_vars={p:(("lat","lon"), mean)}, \
		    coords={"lat":out_hist[0].lat.values, "lon":out_hist[0].lon.values}).\
		to_netcdf("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+p+\
			"_"+str(y1)+"_"+str(y2)+"_ensemble_mean_"+cmipstr+".nc",\
		    engine="h5netcdf")

if __name__ == "__main__":

	models = [ ["ERA5",""] ,\
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
	models_cmip6 = [ ["ACCESS-ESM1-5", "r1i1p1f1", 6, ""], ["ACCESS-CM2", "r1i1p1f1", 6, ""] ]

	save_cmip_mean(models, "lr36", "historical", 1979, 2005, "cmip5")
	save_cmip_mean(models, "mhgt", "historical", 1979, 2005, "cmip5")
	save_cmip_mean(models, "ml_el", "historical", 1979, 2005, "cmip5")
	save_cmip_mean(models, "qmean01", "historical", 1979, 2005, "cmip5")
	save_cmip_mean(models, "srhe_left", "historical", 1979, 2005, "cmip5")
	save_cmip_mean(models, "Umean06", "historical", 1979, 2005, "cmip5")

	save_cmip_mean(models_cmip6, "lr36", "historical", 1979, 2005, "cmip6")
	save_cmip_mean(models_cmip6, "mhgt", "historical", 1979, 2005, "cmip6")
	save_cmip_mean(models_cmip6, "ml_el", "historical", 1979, 2005, "cmip6")
	save_cmip_mean(models_cmip6, "qmean01", "historical", 1979, 2005, "cmip6")
	save_cmip_mean(models_cmip6, "srhe_left", "historical", 1979, 2005, "cmip6")
	save_cmip_mean(models_cmip6, "Umean06", "historical", 1979, 2005, "cmip6")
