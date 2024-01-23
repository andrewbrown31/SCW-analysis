import metpy.calc as mpcalc
from cmip_scenario_global import *

if __name__ == "__main__":

	#TEST OPTIONS
	lsm=False; hist_y1=2000; hist_y2=2000; scenario_y1=2100; scenario_y2=2100; ensemble="r1i1p1";experiment="rcp85"; force_compute=True; save_hist_qm=True; force_cmip_regrid=True; era5_y1=2000; era5_y2=2000; model="ACCESS1-0"

	parser = argparse.ArgumentParser(description='Post-processing of global CMIP laplacian of geopotential')
	parser.add_argument("-m",help="Model",default="",nargs="+")
	parser.add_argument("-e",help="Experiment (e.g. rcp85). Note historical is always loaded",required=True)
	parser.add_argument("--threshold",help="If considering diagnostic indices, use a threshold",default=0,\
		type=float)
	parser.add_argument("--era5_y1",help="Start year for ERA5",default=1979,\
		type=int)
	parser.add_argument("--era5_y2",help="End year for ERA5",default=2005,\
		type=int)
	parser.add_argument("--hist_y1",help="Start year for CMIP historical period",default=1979,\
		type=int)
	parser.add_argument("--hist_y2",help="End year for CMIP historical period",default=2005,\
		type=int)
	parser.add_argument("--scenario_y1",help="Start year for CMIP scenario period",default=2081,\
		type=int)
	parser.add_argument("--scenario_y2",help="End year for CMIP scenario period",default=2100,\
		type=int)
	parser.add_argument("--loop",help="If True, then loop over each spatial point when QQ-matching",default=True,\
		type=str2bool)
	parser.add_argument("--force_compute",help="Force quantile mapping of CMIP data?",default=False,\
		type=str2bool)
	parser.add_argument("--force_cmip_regrid",help="Force regridding of CMIP data?",default=True,\
		type=str2bool)
	parser.add_argument("--save_hist_qm",help="Save the historical quantile matched CMIP data",default=False,\
		type=str2bool)
	parser.add_argument("--hist_only",help="For QM indices, only resample/save seasonal frequency for historical period",default=False,\
		type=str2bool)
	parser.add_argument("--lsm",help="Mask ocean values using the ERA5 lsm?",default=True,\
		type=str2bool)
	parser.add_argument("--log",help="Make plots with a LogNorm color-scale, and log y-axis?",default=False,\
		type=str2bool)
	parser.add_argument("--mean_only",help="Instead of loading 3d time data, just load in the mean over each period"\
		,default=False, type=str2bool)
	parser.add_argument("--vmin",help="Minimum colour value for spatial distribution plot"\
		,default=None, type=float)
	parser.add_argument("--vmax",help="Maximum colour value for spatial distribution plot"\
		,default=None, type=float)
	parser.add_argument("--rel_vmin",help="Minimum colour value for relative difference plots"\
		,default=None, type=float)
	parser.add_argument("--rel_vmax",help="Maximum colour value for relative difference plots"\
		,default=None, type=float)
	parser.add_argument("--season",help="Season for scenario difference plots"\
		,default="", type=str)

	#Take all the same options as cmip_scenario_global.py
	ProgressBar().register()
	args = parser.parse_args()
	subplots1=[4,4]
	subplots2=[3,4]
	model = args.m
	experiment = args.e
	threshold = args.threshold
	era5_y1 = args.era5_y1
	era5_y2 = args.era5_y2
	hist_y1 = args.hist_y1
	hist_y2 = args.hist_y2
	scenario_y1 = args.scenario_y1
	scenario_y2 = args.scenario_y2
	log = args.log
	save_hist_qm=args.save_hist_qm
	hist_only=args.hist_only
	force_cmip_regrid=args.force_cmip_regrid
	force_compute=args.force_compute
	mean_only=args.mean_only
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
			["ACCESS-ESM1-5", "r1i1p1f1", 6, ""], \
			["ACCESS-CM2", "r1i1p1f1", 6, ""],\
			["BARPA", ""]
                        ]
	if model == "":
		model = ["ERA5", "ACCESS1-3", "ACCESS1-0", "BNU-ESM", "CNRM-CM5", "GFDL-CM3", \
			    "GFDL-ESM2G", "GFDL-ESM2M", "IPSL-CM5A-LR", "IPSL-CM5A-MR", \
			    "MIROC5", "MRI-CGCM3", "bcc-csm1-1"]
	models = list(np.array(models)[np.in1d([m[0] for m in models], model)])
	
	#Add ERA5 to the model list if we are "forcing compute"
	if (force_compute) & (["ERA5",""] not in models):
		if models != [["BARPA",""]]:
			models = [ ["ERA5", ""] ]+models
		print(models)

	#########################################################
	#   LOAD RAW LAPLACIAN DATA FROM GCM AND ERA5   	#			  
	#########################################################
	laplacian_raw_hist = load_model_data(models, "laplacian", lsm=lsm,\
		force_cmip_regrid=True,\
		experiment="historical", era5_y1=era5_y1, era5_y2=era5_y2,\
		y1=hist_y1, y2=hist_y2, save=False, domain="global") 
	print("Loading re-gridded scenario model data...")
	laplacian_raw_scenario = load_model_data(models, "laplacian", lsm=lsm,\
	    force_cmip_regrid=True, \
	    experiment=experiment, y2=scenario_y2, \
	    y1=scenario_y1, era5_data=laplacian_raw_hist[0], save=False, domain="global") 

	#################################
	#   2) QQ-MATCH LAPLACIAN DATA	#			  
	#################################
	print("Quantile mapping to ERA5...")
	load_all_qm_combined(\
		laplacian_raw_hist, laplacian_raw_scenario, models, "laplacian", lsm, \
		False, experiment, \
		hist_y1, hist_y2, scenario_y1, scenario_y2, \
		force_compute=force_compute, save_hist_qm=True)

#	for m in np.arange(1,len(models)):
#		print("Loading re-gridded historical model data...")
#
#		#########################################################
#		#   1) COMPUTE LAPLACIAN ON BIAS CORRECTED Z500 DATA   	#			  
#		#########################################################
#
#		#Load bias corrected z500 data
#		out_hist, out_scenario = load_qm(\
#			    models[m][0], models[m][1], \
#			    "z500", lsm, hist_y1, hist_y2,\
#			    scenario_y1, scenario_y2,\
#			    experiment=experiment)
#
#		#HISTORICAL
#		laplacian_hist = mpcalc.laplacian(\
#			    xr.Dataset({"z500":out_hist}).metpy.parse_cf()["z500"].\
#			    rolling(dim={"lat":4},center=True).mean().rolling(dim={"lon":4},center=True).mean(),\
#			axes=["lat","lon"]) * 1e9
#		laplacian_hist.attrs["bc_notes"] = "For this version of Laplacian, GCM z500 is bias corrected using QQ-matching to ERA5, then Laplanian is computed, and saved"
#		laplacian_hist = xr.Dataset({"laplacian":(laplacian_hist)}).drop('crs')
#		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+models[m][0]+"_"+models[m][1]+"_historical_laplacian1_qm_"+str(hist_y1)+"_"+str(hist_y2)+".nc"
#		laplacian_hist.to_netcdf(fname, mode="w",\
#                                    encoding={"laplacian":{"zlib":True, "complevel":1, "least_significant_digit":3}})
#
#		#SCENARIO
#		laplacian_scenario = mpcalc.laplacian(\
#			    xr.Dataset({"z500":out_scenario}).metpy.parse_cf()["z500"].\
#			    rolling(dim={"lat":4},center=True).mean().rolling(dim={"lon":4},center=True).mean(),\
#			axes=["lat","lon"]) * 1e9
#		laplacian_scenario.attrs["bc_notes"] = "For this version of Laplacian, GCM z500 is bias corrected using QQ-matching to ERA5, then Laplanian is computed, and saved"
#		laplacian_scenario = xr.Dataset({"laplacian":(laplacian_scenario)}).drop('crs')
#		fname = "/g/data/eg3/ab4502/ExtremeWind/global/regrid_1.5/"+models[m][0]+"_"+models[m][1]+"_"+experiment+"_laplacian1_qm_"+str(scenario_y1)+"_"+str(scenario_y2)+".nc"
#		#TODO: ADD BC_NOTES TO LAPLACIAN2 AND LAPLACIAN OUTPUTS, WHICH ARE GENERATED BY LOAD_ALL_QM_COMBINED()
#		laplacian_scenario.to_netcdf(fname, mode="w",\
#                                    encoding={"laplacian":{"zlib":True, "complevel":1, "least_significant_digit":3}})
#
#
#		###################################################################
#		#   3) BIAS CORRECT LAPLACIAN CALCULATED ON BIAS CORRECTED Z500   #			  
#		###################################################################
#		#Replace raw Laplacian with Laplacian calculated on the bias corrected z500
#		laplacian_raw_hist[m] = laplacian_hist["laplacian"]
#		laplacian_raw_scenario[m] = laplacian_scenario["laplacian"]
#
#
#	print("Quantile mapping to ERA5...")
#	load_all_qm_combined(\
#		laplacian_raw_hist, laplacian_raw_scenario, models, "laplacian2", lsm, \
#		False, experiment, \
#		hist_y1, hist_y2, scenario_y1, scenario_y2, \
#		force_compute=force_compute, save_hist_qm=True)
