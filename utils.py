import netCDF4 as nc
import numpy as np
import datetime as dt
import os
import metpy.units as units
import metpy.calc as mpcalc

def get_dp(ta,hur,dp_mask=True):
	#Dew point approximation found at https://gist.github.com/sourceperl/45587ea99ff123745428
	#Same as "Magnus formula" https://en.wikipedia.org/wiki/Dew_point
	#For points where RH is zero, set the dew point temperature to -85 deg C
	#EDIT: Leave points where RH is zero masked as NaNs. Hanlde them after creating a SHARPpy object (by 
	# making the missing dp equal to the mean of the above and below layers)
	#EDIT: Replace with metpy code

	#a = 17.27
	#b = 237.7
	#alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
	#dp = (b*alpha) / (a - alpha)

	dp = np.array(mpcalc.dewpoint_rh(ta * units.units.degC, hur * units.units.percent))

	if dp_mask:
		return dp
	else:
		dp = np.array(dp)
		dp[np.isnan(dp)] = -85.
		return dp

def save_netcdf(out_path,out_name,times,lat,lon,param,param_out,append=False,out_dtype="f8",compress=False):

	fname = out_path+out_name+"_"+\
		dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
		dt.datetime.strftime(times[-1],"%Y%m%d")+".nc"
	if (os.path.isfile(fname)) & (append):
		param_file = nc.Dataset(fname,"a",format="NETCDF4_CLASSIC")
	else:
		if os.path.isfile(fname):
			os.remove(fname)
		param_file = nc.Dataset(fname,"w",format="NETCDF4_CLASSIC")
		time_dim = param_file.createDimension("time",None)
		lat_dim = param_file.createDimension("lat",len(lat))
		lon_dim = param_file.createDimension("lon",len(lon))		
		time_var = param_file.createVariable("time",lat.dtype,("time",))
		time_var.units = "hours since 1970-01-01 00:00:00"
		time_var.long_name = "time"
		lat_var = param_file.createVariable("lat",lat.dtype,("lat",))
		lat_var.units = "degrees_north"
		lat_var.long_name = "latitude"
		lon_var = param_file.createVariable("lon",lat.dtype,("lon",))
		lon_var.units = "degrees_east"
		lon_var.long_name = "longitude"
		time_var[:] = nc.date2num(times,time_var.units)
		lat_var[:] = lat
		lon_var[:] = lon

	for i in np.arange(0,len(param)):
		if append:
			if param[i] not in param_file.variables.keys():
				var_units, var_long_name, least_significant_digit = nc_attributes(param[i])
				if compress:
					temp_var = param_file.createVariable(param[i],out_dtype,\
						("time","lat","lon"),zlib=True,least_significant_digit=\
						least_significant_digit, complevel=1)
				else:
					temp_var = param_file.createVariable(param[i],out_dtype,\
						("time","lat","lon"))
				temp_var[:] = param_out[i].astype(out_dtype)
				temp_var.units = var_units
				temp_var.long_name = var_long_name
			else:
				param_file[param[i]][:] = param_out[i]
		else:
			var_units, var_long_name, least_significant_digit = nc_attributes(param[i])
			if compress:
				temp_var = param_file.createVariable(param[i],out_dtype,\
					("time","lat","lon"),zlib=True,\
					least_significant_digit=least_significant_digit, complevel=1)
			else:
				temp_var = param_file.createVariable(param[i],out_dtype,\
					("time","lat","lon"))
			temp_var[:] = param_out[i].astype(out_dtype)
			temp_var.units = var_units
			temp_var.long_name = var_long_name
	param_file.close()

def nc_attributes(param):
	if param=="ml_cape":
		units = "J/kg"
		long_name = "mixed_layer_cape"
		least_significant_digit = 1
	elif param=="ml_cin":
		units = "J/kg"
		long_name = "mixed_layer_cin"
		least_significant_digit = 1
	elif param=="cape_gb_mu1":
		units = "J/kg"
		long_name = "most_unstable_cape_mu1"
		least_significant_digit = 1
	elif param=="cape_gb_mu4":
		units = "J/kg"
		long_name = "most_unstable_cape_mu4"
		least_significant_digit = 1
	elif param=="mu_cape":
		units = "J/kg"
		long_name = "most_unstable_cape"
		least_significant_digit = 1
	elif param=="mu_cin":
		units = "J/kg"
		long_name = "most_unstable_cin"
		least_significant_digit = 1
	elif param=="ssfc850":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-850hPa"
		least_significant_digit = 2
	elif param=="s010":
		units = "m/s"
		long_name = "bulk_wind_shear_0-10km"
		least_significant_digit = 2
	elif param=="s0500":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc_500hPa"
		least_significant_digit = 2
	elif param=="s06":
		units = "m/s"
		long_name = "bulk_wind_shear_0-6km"
		least_significant_digit = 2
	elif param=="s03":
		units = "m/s"
		long_name = "bulk_wind_shear_0-3km"
		least_significant_digit = 2
	elif param=="s01":
		units = "m/s"
		long_name = "bulk_wind_shear_0-1km"
		least_significant_digit = 2
	elif param=="ssfc500":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-6km"
		least_significant_digit = 2
	elif param=="ssfc6":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-500m"
		least_significant_digit = 2
	elif param=="ssfc3":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-3km"
		least_significant_digit = 2
	elif param=="ssfc1":
		units = "m/s"
		long_name = "bulk_wind_shear_sfc-1km"
		least_significant_digit = 2
	elif param == "srhe_left":
		units = "m^2/s^2"
		long_name = "effective_layer_storm_relative_helicity_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh01_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-1km_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh03_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-3km_left_moving_storm"
		least_significant_digit = 2
	elif param=="srh06_left":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-6 km_left_moving_storm"
		least_significant_digit = 2
	elif param == "srhe":
		units = "m^2/s^2"
		long_name = "effective_layer_storm_relative_helicity"
		least_significant_digit = 2
	elif param=="srh01":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-1km"
		least_significant_digit = 2
	elif param=="srh03":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-3km"
		least_significant_digit = 2
	elif param=="srh06":
		units = "m^2/s^2"
		long_name = "storm_relative_helicity_0-6 km"
		least_significant_digit = 2
	elif param=="scp":
		units = ""
		long_name = "supercell_composite_parameter"
		least_significant_digit = 8
	elif param=="scp_fixed":
		units = ""
		long_name = "supercell_composite_parameter_fixed_layer_srh"
		least_significant_digit = 8
	elif param=="estp":
		units = ""
		long_name = "non_zero_cape_significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="stp":
		units = ""
		long_name = "significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="ship":
		units = ""
		long_name = "significant_hail_parameter"
		least_significant_digit = 8
	elif param=="mmp":
		units = ""
		long_name = "mcs_maintanance_probability"
		least_significant_digit = 8
	elif param=="relhum850-500":
		units = "%"
		long_name = "avg_relative_humidity_850-500hPa"
		least_significant_digit = 3
	elif param=="relhum1000-700":
		units = "%"
		long_name = "avg_relative_humidity_850-500hPa"
		least_significant_digit = 3
	elif param=="crt":
		units = "degrees"
		long_name = "tornado_critical_angle"
		least_significant_digit = 3
	elif param=="non_sc_stp":
		units = ""
		long_name = "non-supercell_significant_tornado_parameter"
		least_significant_digit = 8
	elif param=="vo700":
		units = "s^-1 * 1e5"
		long_name = "relative_vorticity_700hPa"
		least_significant_digit = 5
	elif param=="vo500":
		units = "s^-1 * 1e5"
		long_name = "relative_vorticity_500hPa"
		least_significant_digit = 5
	elif param=="vo10":
		units = "s^-1 * 1e5"
		long_name = "relative_vorticity_10m"
		least_significant_digit = 5
	elif param=="lr1000":
		units = "degC/km"
		long_name = "lapse_rate_sfc-1000m"
		least_significant_digit = 3
	elif param=="lcl":
		units = "m"
		long_name = "most_unstable_lifting_condensation_level"
		least_significant_digit = 1
	elif param == "mod_cape*s06":
		units = ""
		long_name = "model_cape_times_s06167"
		least_significant_digit = 1
	elif param == "cape*s06_2":
		units = ""
		long_name = "erai_cape_times_s06"
		least_significant_digit = 1
	elif param == "cape*ssfc6":
		units = ""
		long_name = "cape*ssfc6"
		least_significant_digit = 1
	elif param == "wg10":
		units = "m/s"
		long_name = "wind_gust_10m"
		least_significant_digit = 2
	elif param == "wg":
		units = "m/s"
		long_name = "max_wind_gust_10m"
		least_significant_digit = 2
	elif param == "dcp":
		units = ""
		long_name = "derecho_composite_parameter"
		least_significant_digit = 8
	elif param == "mod_cin":
		units = "J/kg"
		long_name = "model_cin"
		least_significant_digit = 1
	elif param == "mod_cape":
		units = "J/kg"
		long_name = "model_cape"
		least_significant_digit = 1
	elif param == "conv10":
		units = "s^-1 * 1e5"
		long_name = "convergence_10m"
		least_significant_digit = 5
	elif param == "conv1000-850":
		units = "s^-1"
		long_name = "mean_convergence_1000-850hPa"
		least_significant_digit = 5
	elif param == "conv800-600":
		units = "s^-1"
		long_name = "mean_convergence_1000-850hPa"
		least_significant_digit = 5
	elif param == "conv925":
		units = "s^-1 * 1e-5"
		long_name = "convergence_925hPa"
		least_significant_digit = 5
	elif param == "conv850":
		units = "s^-1 * 1e-5"
		long_name = "convergence_850hPa"
		least_significant_digit = 5
	elif param == "conv700":
		units = "s^-1 * 1e-5"
		long_name = "convergence_700hPa"
		least_significant_digit = 5
	elif param == "td950":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "td800":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "td850":
		units = "decC"
		long_name = "dew_point_depression850hPa"
		least_significant_digit = 3
	elif param == "dcape":
		units = "J/kg"
		long_name = "downwards_maximum_convective_available_potential_energy"
		least_significant_digit = 1
	elif param == "ddraft_temp":
		units = "J/kg"
		long_name = "temperature_defecit_downdraft_parcel_sfc"
		least_significant_digit = 3
	elif param == "dlm":
		units = "m/s"
		long_name = "deep_layer_mean_wind_speed"
		least_significant_digit = 2
	elif param == "sb_cape":
		units = "J/kg"
		long_name = "surface_based_cape"
		least_significant_digit = 1
	elif param == "sb_cin":
		units = "J/kg"
		long_name = "surface_based_cin"
		least_significant_digit = 1
	elif param == "eff_lcl":
		units = "m"
		long_name = "effective_layer_parcel_lcl"
		least_significant_digit = 1
	elif param == "ml_lcl":
		units = "m"
		long_name = "mixed_layer_parcel_lcl"
		least_significant_digit = 1
	elif param == "mu_lcl":
		units = "m"
		long_name = "most_unstable_parcel_lcl"
		least_significant_digit = 1
	elif param == "sb_lcl":
		units = "m"
		long_name = "sfc_based_parcel_lcl"
		least_significant_digit = 1
	elif param == "tp":
		units = "mm"
		long_name = "total_precipitation_since_last_time_step"
		least_significant_digit = 8
	elif param == "cp":
		units = "mm"
		long_name = "convective_precipitation_since_last_time_step"
		least_significant_digit = 8
	elif param == "dcp2":
		units = ""
		long_name = "dcp_erai_cape"
		least_significant_digit = 8
	elif param == "ebwd":
		units = "m/s"
		long_name = "effective_layer_bulk_wind_fifference(shear)"
		least_significant_digit = 2
	elif param == "Umean01":
		units = "m/s"
		long_name = "mean_wind_0km_1km"
		least_significant_digit = 2
	elif param == "Umean03":
		units = "m/s"
		long_name = "mean_wind_0km_3km"
		least_significant_digit = 2
	elif param == "Umean06":
		units = "m/s"
		long_name = "mean_wind_0km_6km"
		least_significant_digit = 2
	elif param == "U500":
		units = "m/s"
		long_name = "wind_speed_500hPa"
		least_significant_digit = 2
	elif param == "U10":
		units = "m/s"
		long_name = "diagnostic_wind_speed_10m"
		least_significant_digit = 2
	elif param == "Uwindinf":
		units = "m/s"
		long_name = "wind_speed_top_of_effective_layer"
		least_significant_digit = 2
	elif param == "Umeanwindinf":
		units = "m/s"
		long_name = "mean_wind_speed_effective_layer"
		least_significant_digit = 2
	elif param == "Umean800_600":
		units = "m/s"
		long_name = "mean_wind_speed800_600hPa"
		least_significant_digit = 2
	elif param == "stp_cin_left":
		units = ""
		long_name = "significant_tornado_parameter_with_cin_left_moving_storm"
		least_significant_digit = 8
	elif param == "stp_fixed_left":
		units = ""
		long_name = "significant_tornado_parameter_with_fixed_layer_left_moving_storm"
		least_significant_digit = 8
	elif param == "stp_cin":
		units = ""
		long_name = "significant_tornado_parameter_with_cin"
		least_significant_digit = 8
	elif param == "stp_fixed":
		units = ""
		long_name = "significant_tornado_parameter_with_fixed_layer"
		least_significant_digit = 8
	elif param == "mlcape*s06":
		units = ""
		long_name = "mixed_layer_cape_times_s06167"
		least_significant_digit = 1
	elif param == "mlcape*s06_2":
		units = ""
		long_name = "mixed_layer_cape_times_s06"
		least_significant_digit = 1
	elif param == "mucape*s06":
		units = ""
		long_name = "most_unstable_cape_times_s06167"
		least_significant_digit = 1
	elif param == "mucape*s06_2":
		units = ""
		long_name = "most_unstable_cape_times_s06"
		least_significant_digit = 1
	elif param == "effcape*s06":
		units = ""
		long_name = "effective_cape_times_s06167"
		least_significant_digit = 1
	elif param == "effcape*s06_2":
		units = ""
		long_name = "effective_cape_times_s06"
		least_significant_digit = 1
	elif param == "sbcape*s06":
		units = ""
		long_name = "surface_based_cape_times_s06167"
		least_significant_digit = 1
	elif param == "sbcape*s06_2":
		units = ""
		long_name = "surface_based_cape_times_s06"
		least_significant_digit = 1
	elif param == "dmgwind":
		units = ""
		long_name = "damaging_wind_kuchera"
		least_significant_digit = 8
	elif param == "dmgwind_fixed":
		units = ""
		long_name = "damaging_wind_kuchera_fixed"
		least_significant_digit = 8
	elif param == "ducs6":
		units = ""
		long_name = "convgust_times_mlcape*s06"
		least_significant_digit = 8
	elif param == "convgust_dry":
		units = ""
		long_name = "convective_gust_dry_mburst_ewd"
		least_significant_digit = 3
	elif param == "convgust_wet":
		units = ""
		long_name = "convective_gust_wet_mburst_ewd"
		least_significant_digit = 3
	elif param == "windex":
		units = "m/s"
		long_name = "wind_index_mccann_microburst"
		least_significant_digit = 3
	elif param == "gustex":
		units = "m/s"
		long_name = "gust_index_geerts_original"
		least_significant_digit = 3
	elif param == "gustex2":
		units = "m/s"
		long_name = "gust_index_geerts_Umean06"
		least_significant_digit = 3
	elif param == "gustex3":
		units = ""
		long_name = "gust_index_geerts_times_mlcape*s06"
		least_significant_digit = 3
	elif param == "lr01":
		units = "deg/km"
		long_name = "lapse_rate_0_1km"
		least_significant_digit = 3
	elif param == "lr03":
		units = "deg/km"
		long_name = "lapse_rate_0_3km"
		least_significant_digit = 3
	elif param == "lr700_500":
		units = "deg/km"
		long_name = "lapse_rate_700_500hPa"
		least_significant_digit = 3
	elif param == "lr24":
		units = "deg/km"
		long_name = "lapse_rate_2_4km"
		least_significant_digit = 3
	elif param == "lr13":
		units = "deg/km"
		long_name = "lapse_rate_1_3km"
		least_significant_digit = 3
	elif param == "lr36":
		units = "deg/km"
		long_name = "lapse_rate_3_6km"
		least_significant_digit = 3
	elif param == "lr_subcloud":
		units = "deg/km"
		long_name = "lapse_rate_sfc_lcl"
		least_significant_digit = 3
	elif param == "lr_freezing":
		units = "deg/km"
		long_name = "lapse_rate_sfc_to_freezing"
		least_significant_digit = 3
	elif param == "muq":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_most_unstable_parcel"
		least_significant_digit = 3
	elif param == "qmean01":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_1km"
		least_significant_digit = 3
	elif param == "qmean03":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_3km"
		least_significant_digit = 3
	elif param == "qmean06":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_0_6km"
		least_significant_digit = 3
	elif param == "qmean13":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_1_3km"
		least_significant_digit = 3
	elif param == "qmean36":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_3_6km"
		least_significant_digit = 3
	elif param == "qmeansubcloud":
		units = "g/kg"
		long_name = "water_vapour_mean_mixing_ratio_sfc_to_mllcl"
		least_significant_digit = 3
	elif param == "q_melting":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_melting_level"
		least_significant_digit = 3
	elif param == "q1":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_1km"
		least_significant_digit = 3
	elif param == "q3":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_3km"
		least_significant_digit = 3
	elif param == "q6":
		units = "g/kg"
		long_name = "water_vapour_mixing_ratio_6km"
		least_significant_digit = 3
	elif param == "rhmin01":
		units = "%"
		long_name = "minimum_rh_0_1km"
		least_significant_digit = 3
	elif param == "rhmin03":
		units = "%"
		long_name = "minimum_rh_0_3km"
		least_significant_digit = 3
	elif param == "rhmin06":
		units = "%"
		long_name = "minimum_rh_0_6km"
		least_significant_digit = 3
	elif param == "rhmin13":
		units = "%"
		long_name = "minimum_rh_1_3km"
		least_significant_digit = 3
	elif param == "rhmin36":
		units = "%"
		long_name = "minimum_rh_3_6km"
		least_significant_digit = 3
	elif param == "rhminsubcloud":
		units = "%"
		long_name = "minimum_rh_sfc_mllcl"
		least_significant_digit = 3
	elif param == "rho925":
		units = "kg/m^-3"
		long_name = "density_at_925hPa"
		least_significant_digit = 3
	elif param == "rho850":
		units = "kg/m^-3"
		long_name = "density_at_825hPa"
		least_significant_digit = 3
	elif param == "rho700":
		units = "kg/m^-3"
		long_name = "density_at_700Pa"
		least_significant_digit = 3
	elif param == "ta925":
		units = "degC"
		long_name = "temperature_at_925hPa"
		least_significant_digit = 3
	elif param == "ta850":
		units = "degC"
		long_name = "temperature_at_850hPa"
		least_significant_digit = 3
	elif param == "t500":
		units = "degC"
		long_name = "temperature_at_500hPa"
		least_significant_digit = 3
	elif param == "mhgt":
		units = "m"
		long_name = "melting_lvl_hgt"
		least_significant_digit = 1
	elif param == "mu_el":
		units = "m"
		long_name = "equilibrium_lvl_mu_parcel"
		least_significant_digit = 1
	elif param == "ml_el":
		units = "m"
		long_name = "equilibrium_lvl_ml_parcel"
		least_significant_digit = 1
	elif param == "sb_el":
		units = "m"
		long_name = "equilibrium_lvl_sb_parcel"
		least_significant_digit = 1
	elif param == "eff_el":
		units = "m"
		long_name = "equilibrium_lvl_eff_parcel"
		least_significant_digit = 1
	elif param == "s13":
		units = "m/s"
		long_name = "wind_shear_1_3km"
		least_significant_digit = 2
	elif param == "s36":
		units = "m/s"
		long_name = "wind_shear_3_6km"
		least_significant_digit = 2
	elif param == "scld":
		units = "m/s"
		long_name = "wind_shear_mllcl_to_half_the_el"
		least_significant_digit = 2
	elif param == "U1":
		units = "m/s"
		long_name = "wind_speed_1km"
		least_significant_digit = 2
	elif param == "U3":
		units = "m/s"
		long_name = "wind_speed_3km"
		least_significant_digit = 2
	elif param == "U6":
		units = "m/s"
		long_name = "wind_speed_6km"
		least_significant_digit = 2
	elif param == "Ust_left":
		units = "m/s"
		long_name = "non_parcel_bunkers_storm_motion_speed_left_moving_storm"
		least_significant_digit = 2
	elif param == "Ust":
		units = "m/s"
		long_name = "non_parcel_bunkers_storm_motion_speed"
		least_significant_digit = 2
	elif param == "Usr01_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_1km"
		least_significant_digit = 2
	elif param == "Usr03_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_3km"
		least_significant_digit = 2
	elif param == "Usr06_left":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_left_0_6km"
		least_significant_digit = 2
	elif param == "Usr01":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_1km"
		least_significant_digit = 2
	elif param == "Usr03":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_3km"
		least_significant_digit = 2
	elif param == "Usr06":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_0_6km"
		least_significant_digit = 2
	elif param == "Usr13":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_1_3km"
		least_significant_digit = 2
	elif param == "Usr36":
		units = "m/s"
		long_name = "storm_relative_mean_wind_speed_using_Ust_3_6km"
		least_significant_digit = 2
	elif param == "mosh":
		units = ""
		long_name = "modified_sherb"
		least_significant_digit = 8
	elif param == "moshe":
		units = ""
		long_name = "modified_sherb_with_effective_shear"
		least_significant_digit = 8
	elif param == "sherb":
		units = ""
		long_name = "sever_hazards_with_reduced_buoyancy_parameter"
		least_significant_digit = 8
	elif param == "eff_sherb":
		units = ""
		long_name = "sever_hazards_with_reduced_buoyancy_parameter_effective_layer_form"
		least_significant_digit = 8
	elif param == "v_totals":
		units = ""
		long_name = "vertical_totals_index"
		least_significant_digit = 3
	elif param == "c_totals":
		units = ""
		long_name = "cross_totals_index"
		least_significant_digit = 3
	elif param == "t_totals":
		units = ""
		long_name = "total_totals_index"
		least_significant_digit = 3
	elif param == "pwat":
		units = "in"
		long_name = "precip_water_sfc_400_hPa"
		least_significant_digit = 3
	elif param == "eff_cape":
		units = "J/kg"
		long_name = "effective_inflow_layer_parcel_define_cape"
		least_significant_digit = 1
	elif param == "eff_cin":
		units = "J/kg"
		long_name = "effective_inflow_layer_parcel_define_cin"
		least_significant_digit = 1
	elif param == "eff_lcl":
		units = "m"
		long_name = "effective_inflow_layer_parcel_define_lcl"
		least_significant_digit = 1
	elif param == "wndg":
		units = ""
		long_name = "wind_damage_parameter_spc"
		least_significant_digit = 8
	elif param == "mburst":
		units = ""
		long_name = "microburst_composite_index_spc"
		least_significant_digit = 1
	elif param == "sweat":
		units = ""
		long_name = "severe_weather_threat_index_spc"
		least_significant_digit = 3
	elif param == "maxtevv":
		units = "K Pa / (km s)"
		long_name = "max_thetae_vertical_velocity_sherburn"
		least_significant_digit = 8
	elif param == "omega01":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-1km"
		least_significant_digit = 8
	elif param == "omega03":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-3km"
		least_significant_digit = 8
	elif param == "omega06":
		units = "Pa/s"
		long_name = "mean_vertical_velocity_0-6km"
		least_significant_digit = 8
	elif param == "w925":
		units = "m/s"
		long_name = "vertical_velocity_925hPa"
		least_significant_digit = 8
	elif param == "w850":
		units = "m/s"
		long_name = "vertical_velocity_925hPa"
		least_significant_digit = 8
	elif param == "w700":
		units = "m/s"
		long_name = "vertical_velocity_925hPa"
		least_significant_digit = 8
	elif param == "dmi":
		units = ""
		long_name = "dry_microburst_index_goes"
		least_significant_digit = 3
	elif param == "hmi":
		units = ""
		long_name = "hybrid_microburst_index_goes"
		least_significant_digit = 3
	elif param == "wmsi_mu":
		units = ""
		long_name = "wet_microburst_severity_index_goes_mucape"
		least_significant_digit = 3
	elif param == "wmsi_ml":
		units = ""
		long_name = "wet_microburst_severity_index_goes_mlcape"
		least_significant_digit = 3
	elif param == "mwpi_mu":
		units = ""
		long_name = "microburst_windspeed_potential_index_goes_mucape"
		least_significant_digit = 3
	elif param == "mwpi_ml":
		units = ""
		long_name = "microburst_windspeed_potential_index_goes_mlcape"
		least_significant_digit = 3
	elif param == "wmpi":
		units = ""
		long_name = "wet_microburst_potential_index_proctor"
		least_significant_digit = 3
	elif param == "sfc_thetae":
		units = "degC"
		long_name = "sfc_equivalent_potential_temperature"
		least_significant_digit = 3
	elif param == "dpd850":
		units = "degC"
		long_name = "dewpoint_depression_850hPa"
		least_significant_digit = 3
	elif param == "dpd700":
		units = "degC"
		long_name = "dewpoint_depression_700hPa"
		least_significant_digit = 3
	elif param == "tei":
		units = "degC"
		long_name = "thetae_index"
		least_significant_digit = 3
	elif param == "te_diff":
		units = "degC"
		long_name = "max_min_thetae_diff_3000m"
		least_significant_digit = 3
	elif param == "ml_lfc":
		units = "m"
		long_name = "mixed_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "mu_lfc":
		units = "m"
		long_name = "most_unstable_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "eff_lfc":
		units = "m"
		long_name = "effective_layer_parcel_lfc"
		least_significant_digit = 1
	elif param == "pbl_top":
		units = "m"
		long_name = "pbl_level"
		least_significant_digit = 3
	elif param == "sb_lfc":
		units = "m"
		long_name = "surface_parcel_lfc"
		least_significant_digit = 1
	elif param == "k_index":
		units = ""
		long_name = "k_index"
		least_significant_digit = 3
	elif param == "esp":
		units = ""
		long_name = "enhanced_stretching_potential"
		least_significant_digit = 8
	elif param == "wbz":
		units = "m"
		long_name = "wet_bulb_zero_height"
		least_significant_digit = 1
	elif param == "Vprime":
		units = "m/s"
		long_name = "miller_1972_wind_speed"
		least_significant_digit = 3
	elif param == "F10":
		units = "thetae/100km/3h"
		long_name = "frontogenesis_function_10m"
		least_significant_digit = 8
	elif param == "Fn10":
		units = "thetae/100km/3h"
		long_name = "frontogenetical_function_10m"
		least_significant_digit = 8
	elif param == "Fs10":
		units = "thetae/100km/3h"
		long_name = "rotational_frontogenesis_10m"
		least_significant_digit = 8
	elif param == "icon10":
		units = "s-1 * 1e5"
		long_name = "instantaneous_contraction_rate_10m"
		least_significant_digit = 8
	elif param == "vgt10":
		units = "s-1 * 1e5"
		long_name = "horizontal_velocity_gradient_tensor_magnitude_10m"
		least_significant_digit = 8
	elif param == "ta1000":
		units = "degC"
		long_name = "air_temp_1000hPa"
		least_significant_digit = 2
	elif param == "ta850":
		units = "degC"
		long_name = "air_temp_850hPa"
		least_significant_digit = 2
	elif param == "ta700":
		units = "degC"
		long_name = "air_temp_700hPa"
		least_significant_digit = 2
	elif param == "ta500":
		units = "degC"
		long_name = "air_temp_500hPa"
		least_significant_digit = 2
	elif param == "dp850":
		units = "degC"
		long_name = "dewpoint_850"
		least_significant_digit = 2
	elif param == "u850":
		units = "m/s"
		long_name = "uwind850hpa"
		least_significant_digit = 2
	elif param == "v850":
		units = "m/s"
		long_name = "vwind850hpa"
		least_significant_digit = 2
	elif param == "u500":
		units = "m/s"
		long_name = "uwind500hpa"
		least_significant_digit = 2
	elif param == "z700":
		units = "m"
		long_name = "geopotential700hpa"
		least_significant_digit = 2
	elif param == "z500":
		units = "m"
		long_name = "geopotential500hpa"
		least_significant_digit = 2
	elif param == "v500":
		units = "m/s"
		long_name = "vwind500hpa"
		least_significant_digit = 2
	elif param == "hus1000":
		units = "g/kg"
		long_name = "specific_humidity1000hPa"
		least_significant_digit = 7
	elif param == "hus850":
		units = "g/kg"
		long_name = "specific_humidity850hPa"
		least_significant_digit = 7
	elif param == "hus700":
		units = "g/kg"
		long_name = "specific_humidity700hPa"
		least_significant_digit = 7
	elif param == "hus500":
		units = "g/kg"
		long_name = "specific_humidity500hPa"
		least_significant_digit = 7
	elif param == "mag_wb":
		units = ""
		long_name = "magnitude_of_the_gradient_of_850hPa_wetbulb"
		least_significant_digit = 3
	elif param == "mag_tfp":
		units = "100 km^-3"
		long_name = "magnitude_of_the_gradient_of_the_thermal_front_parameter"
		least_significant_digit = 12
	elif param == "tfp":
		units = "100 km^-2"
		long_name = "thermal_front_parameter"
		least_significant_digit = 12
	elif param == "mslp":
		units = "hPa"
		long_name = "mean_sea_level_pressure"
		least_significant_digit = 2
	elif param == "laplacian":
		units = ""
		long_name = "laplacian_operator_applied_to_z500"
		least_significant_digit = 2
	elif param == "sst":
		units = "degC"
		long_name = "sea_sfc_temp"
		least_significant_digit = 2
	elif param == "bdsd":
		units = ""
		long_name = "brown_dowdy_stat_diag"
		least_significant_digit = 2
	elif param == "ncape":
		units = "m s^-2"
		long_name = "mu_cape_normalised_by_depth"
		least_significant_digit = 3
	else:
		units = ""
		long_name = ""
		least_significant_digit = 8
		print("WARNING: "+param+" HAS NO UNITS OR LONG NAME CODED IN NC_ATTRIBUTES()")

	return [units,long_name,least_significant_digit]
