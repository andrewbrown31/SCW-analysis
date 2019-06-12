#NOTE THAT TO CALCULATE DCAPE, NUMPY HAS BEEN UPGRADED TO 1.16.0. SO, TO RUN, SWAP TO ENVIRONMENT "PYCAT"
#source activate pycat

import pp
from SkewT import get_dcape
try:
	import metpy.units.units as units
	import metpy.calc as mpcalc
except:
	pass
from metpy.units import units as units
import metpy.calc as mpcalc
import wrf
#from calc_param import get_ship
from calc_param import *

import numpy as np
#from calc_param import calc_param_wrf
from erai_read import read_erai, read_erai_points, read_erai_fc
from barra_read import read_barra, read_barra_points, get_mask
from barra_ad_read import read_barra_ad
from barra_r_fc_read import read_barra_r_fc
from event_analysis import load_array_points
import datetime as dt
import itertools
import multiprocessing
import sharppy.sharptab.utils as utils
import os
#from calc_param import save_netcdf

#Functions to drive *model*_read.py (to extract variables from reanalysis) and calc_param.py (to calculate 
# convective parameters)

def calc_model(model,out_name,method,time,param,issave,region,cape_method):

   global var_dict
   var_dict = {}

# - Model is either "erai" or "barra"
# - "out_name" specifies the "model" part of the netcdf file saved if issave = True
# - Method is either "domain" or "points"
# - time is a list of the form [start,end] where start and end are both datetime objects
# - param is a list of strings corresponding to thunderstorm parameters to be calculated. 
#	See calc_params.py for a list
# - issave (boolean) - is the output to be saved to a netcdf file?
# - region is either "aus","sa_small", "sa_large" or "adelaideAP".
# - cape_method refers to the method of deriving CAPE from the reanalysis, and subsequently, the
# 	method for deriving other parameters. If "wrf", then wrf.cape3d is used, and the rest of
#	the parameters have been manually coded. If "SHARPpy", then the SHARPpy package is used
#	to create profile/parcel objects, which contain the parameters required. "wrf" is greatly
#	preferred for efficiency purposes (O(1 sec) to derive parameters for one time step
#	versus O(10 minutes)). However, "SHARPpy" can be used for verification of manually coded
#	parameters.

   if method == "domain":
        if region == "aus":
       	    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
        elif region == "sa_small":
       	    start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
       	elif region == "sa_large":
       	    start_lat = -40; end_lat = -24; start_lon = 112; end_lon = 156
       	else:
       	    raise NameError("Region must be one of ""aus"", ""sa_small"" or ""sa_large""")
       	domain = [start_lat,end_lat,start_lon,end_lon]
       
       	#Extract variables for given domain and time period
       	if model=="barra":
       		print("\n	INFO: READING IN BARRA DATA ON " + region +"...\n")
       		s = dt.datetime.now()
       		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
       			read_barra(domain,time)
       		lsm = get_mask(lon,lat)
       		print("Time to load in data:")
       		print(dt.datetime.now() - s)
       	elif model=="erai":
       		print("\n	INFO: READING IN ERA-Interim DATA...\n")
       		ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
       			read_erai(domain,time)
       		lsm = np.ones((len(lat),len(lon)))
       	elif model=="erai_fc":
       		print("\n	INFO: READING IN ERA-Interim DATA...\n")
       		wg10,cape,lon,lat,date_list = \
       			read_erai_fc(domain,time)
       		param = ["wg10","cape"]
       		param_out = [wg10,cape]
       		save_netcdf(region,model,date_list,lat,lon,param,param_out)	
       	elif model=="barra_ad":
       		print("\n	INFO: READING IN BARRA-AD DATA...\n")
       		max_max_wg,wg,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
       			read_barra_ad(domain,time,wg_only=False)
       	elif model=="barra_r_fc":
       		print("\n	INFO: READING IN BARRA-R FORECAST DATA...\n")
       		wg,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
       			read_barra_r_fc(domain,time,wg_only=False)
       	else:
       		raise NameError("""model"" must be ""erai"", ""erai_fc"" or ""barra""")
       
       	#Calculate parameters
       	print("\n	INFO: CALCULATING PARAMETERS\n")
       	if model != "erai_fc":
       		if cape_method == "SHARPpy":
       			s = dt.datetime.now()
       			param_out = calc_param_sharppy(ta,dp,hur,hgt,ua,va,uas,vas,ps,lsm,terrain,p,lon,lat,\
       				date_list,param,issave,out_name,region,model)
       			print("Time to run SHARPpy and save netcdf:")
       			print(dt.datetime.now() - s)
       		elif cape_method == "wrf":
       		    if (model == "barra_r_fc") or (model == "barra_ad"):
       			#IF BARRA-R, include wind gust in the netcdf save function. Also, use every 6 hrs
       		        date_list_hrs = [t.hour for t in date_list]
       		        date_inds = np.in1d(np.array(date_list_hrs),np.array([0,6,12,18]))
       		        param_out = calc_param_wrf(date_list[date_inds],ta[date_inds],dp[date_inds],hur[date_inds],\
       				hgt[date_inds],terrain,p,ps[date_inds],ua[date_inds],va[date_inds],\
       				uas[date_inds],vas[date_inds],lon,lat,param,model,out_name,issave,region,\
       				wg=wg[date_inds])
       		    else:
       		        param_out = calc_param_wrf(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,\
       				uas,vas,lon,lat,param,model,out_name,issave,region)
       		elif cape_method == "wrf_par":
       			#USES WRF METHOD BUT PASSES DATES IN PARALLEL (USUALLY ONE MONTHS WORTH)
       			
       		        pool = multiprocessing.Pool()
       		        param_temp = pool.map(calc_param_wrf_par,itertools.product(np.arange(0,len(date_list)),[param]))
       			#Now have the param output. Need to check that it is in the right time order, and save
       		        pool.close()
       		        pool.join()
       		        param_out = []
       		        for p in np.arange(0,len(param)):
       		                temp = []
       		                [temp.append(param_temp[t][p]) for t in np.arange(0,len(date_list))]
       		                param_out.append(np.stack(temp))
       		        if issave:
       		                save_netcdf(region,model,out_name,date_list,lat,lon,param,param_out)
       
       		else:
       		        raise NameError("""cape_method"" must be ""SHARPpy"" or ""wrf""")

   elif method == "points":
        if region == "adelaideAP":
       	        points = [(138.5204, -34.5924)]
       	        loc_id = ["Adelaide AP"]
       	elif (model == "barra_ad") or (model == "barra_r_fc"):
       	        points = []
       	        loc_id = []
       	else:
       	    raise NameError("Region must be ""adelaideAP"", or model must be BARRA-AD or BARRA-R")
       
       	if model=="barra":
       	        print("\n	INFO: READING IN BARRA DATA...\n")
       	        ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,date_list= \
       	                read_barra_points(points,times)
       	        df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
       	            lon_used,lat_used,param,loc_id,cape_method)
       	elif model=="erai":
       	        print("\n	INFO: READING IN ERA-Interim DATA...\n")
       	        ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,lon_used,lat_used,\
       	        date_list = read_erai_points(points,times)
       	        df = calc_param_points(date_list,ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,\
       				lon_used,lat_used,param,loc_id,cape_method)
       	elif model=="barra_ad":
       	        print("\n	INFO: READING IN BARRA-AD DATA...\n")
       	        start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
       	        domain = [start_lat,end_lat,start_lon,end_lon]
       	        max_max_wg10,max_wg10,lon,lat,date_list = \
       	        read_barra_ad(domain,time,wg_only=True)
       	        param = ["max_max_wg10","max_wg10"]
       	        param_out = [max_max_wg10,max_wg10]
       	elif model=="barra_r_fc":
       	        print("\n	INFO: READING IN BARRA-R FORECAST DATA...\n")
       	        start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
       	        domain = [start_lat,end_lat,start_lon,end_lon]
       	        max_wg10,lon,lat,date_list = \
       	        read_barra_r_fc(domain,time,wg_only=True)
       	        param = ["max_wg10"]
       	        param_out = [max_wg10]
       	else:
       	        raise NameError("""model"" must be ""erai"" or ""barra""")
       
       	if issave:
       	        df.to_csv("/g/data/eg3/ab4502/ExtremeWind/"+region+"/data_"+out_name+"_"+\
       			dt.datetime.strftime(times[0],"%Y%m%d")+"_"+\
       			dt.datetime.strftime(times[1],"%Y%m%d")+".csv",float_format="%.3f")	
       
   return [param,param_out,lon,lat,date_list]

def calc_param_wrf_par(it):

	#Have copied this function here from calc_param_wrf_par, to use global arrays

	t,param = it
	wg=False

	p_3d = np.moveaxis(np.tile(p,[ta.shape[0],ta.shape[2],ta.shape[3],1]),[0,1,2,3],[0,2,3,1])
	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(lat),len(lon)))
	if len(param) != len(np.unique(param)):
		ValueError("Each parameter can only appear once in parameter list")
	print(date_list[t])

	start = dt.datetime.now()
	hur_unit = units.percent*hur[t,:,:,:]
	ta_unit = units.degC*ta[t,:,:,:]
	dp_unit = units.degC*dp[t,:,:,:]
	p_unit = units.hectopascals*p_3d[t,:,:,:]
	q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
		ta_unit,p_unit)
	theta_unit = mpcalc.potential_temperature(p_unit,ta_unit)
	q = np.array(q_unit)

	ml_inds = ((p_3d[t] <= ps[t]) & (p_3d[t] >= (ps[t] - 100)))
	ml_ta_avg = np.ma.masked_where(~ml_inds, ta[t]).mean(axis=0).data
	ml_q_avg = np.ma.masked_where(~ml_inds, q).mean(axis=0).data
	ml_hgt_avg = np.ma.masked_where(~ml_inds, hgt[t]).mean(axis=0).data
	ml_p3d_avg = np.ma.masked_where(~ml_inds, p_3d[t]).mean(axis=0).data
	ml_ta_arr = np.insert(ta[t],0,ml_ta_avg,axis=0)
	ml_q_arr = np.insert(q,0,ml_q_avg,axis=0)
	ml_hgt_arr = np.insert(hgt[t],0,ml_hgt_avg,axis=0)
	ml_p3d_arr = np.insert(p_3d[t],0,ml_p3d_avg,axis=0)
	a,temp1,temp2 = np.meshgrid(np.arange(ml_p3d_arr.shape[0]) ,\
		 np.arange(ml_p3d_arr.shape[1]), np.arange(ml_p3d_arr.shape[2]))
	sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),ml_p3d_arr],axis=0))
	ml_ta_arr = np.take_along_axis(ml_ta_arr, sort_inds, axis=0)
	ml_p3d_arr = np.take_along_axis(ml_p3d_arr, sort_inds, axis=0)
	ml_hgt_arr = np.take_along_axis(ml_hgt_arr, sort_inds, axis=0)
	ml_q_arr = np.take_along_axis(ml_q_arr, sort_inds, axis=0)
	cape3d_mlavg = wrf.cape_3d(ml_p3d_arr,ml_ta_arr + 273.15,\
		ml_q_arr,ml_hgt_arr,terrain,ps[t,:,:],False,meta=False,missing=0)
	ml_cape = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
		cape3d_mlavg.data[0]).max(axis=0).filled(0)
	ml_cin = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) & (ml_p3d_arr==ml_p3d_avg)),\
		cape3d_mlavg.data[1]).max(axis=0).filled(0)

	cape3d = wrf.cape_3d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q,hgt[t,:,:,:],terrain,ps[t,:,:],\
		True,meta=False,missing=0)
	cape = cape3d.data[0]
	cin = cape3d.data[1]
	cape[p_3d[t] > ps[t]-25] = np.nan
	cin[p_3d[t] > ps[t]-25] = np.nan
	mu_cape_inds = np.nanargmax(cape,axis=0)
	mu_cape = mu_cape_inds.choose(cape)
	mu_cin = mu_cape_inds.choose(cin)
	cape_2d = wrf.cape_2d(p_3d[t,:,:,:],ta[t,:,:,:]+273.15,q\
		,hgt[t,:,:,:],terrain,ps[t,:,:],True,meta=False,missing=0)
	lcl = cape_2d[2].data
	lfc = cape_2d[3].data

	del hur_unit, dp_unit, theta_unit, ml_inds, ml_ta_avg, ml_q_avg, \
		ml_hgt_avg, ml_p3d_avg, ml_ta_arr, ml_q_arr, ml_hgt_arr, ml_p3d_arr, a, temp1, temp2,\
		sort_inds, cape3d_mlavg, cape3d, cape, cin, cape_2d

	if "relhum850-500" in param:
		param_ind = np.where(param=="relhum850-500")[0][0]
		param_out[param_ind] = get_mean_var_p(hur[t],p,850,500)
	if "relhum1000-700" in param:
		param_ind = np.where(param=="relhum1000-700")[0][0]
		param_out[param_ind] = get_mean_var_p(hur[t],p,1000,700)
	if "mu_cape" in param:
		param_ind = np.where(param=="mu_cape")[0][0]
		param_out[param_ind] = mu_cape
	if "ml_cape" in param:
		param_ind = np.where(param=="ml_cape")[0][0]
		param_out[param_ind] = ml_cape
	if "s06" in param:
		param_ind = np.where(param=="s06")[0][0]
		s06 = get_shear_hgt(ua[t],va[t],hgt[t],0,6000,\
			uas[t],vas[t])
		param_out[param_ind] = s06
	if "s03" in param:
		param_ind = np.where(param=="s03")[0][0]
		s03 = get_shear_hgt(ua[t],va[t],hgt[t],0,3000,\
			uas[t],vas[t])
		param_out[param_ind] = s03
	if "s01" in param:
		param_ind = np.where(param=="s01")[0][0]
		s01 = get_shear_hgt(ua[t],va[t],hgt[t],0,1000,\
			uas[t],vas[t])
		param_out[param_ind] = s01
	if "s0500" in param:
		param_ind = np.where(param=="s0500")[0][0]
		param_out[param_ind] = get_shear_hgt(ua[t],va[t],hgt[t],0,500,\
			uas[t],vas[t])
	if "lr1000" in param:
		param_ind = np.where(param=="lr1000")[0][0]
		lr1000 = get_lr_hgt(ta[t],hgt[t],0,1000)
		param_out[param_ind] = lr1000
	if "mu_cin" in param:
		param_ind = np.where(param=="mu_cin")[0][0]
		param_out[param_ind] = mu_cin
	if "lcl" in param:
		param_ind = np.where(param=="lcl")[0][0]
		temp_lcl = np.copy(lcl)
		temp_lcl[temp_lcl<=0] = np.nan
		param_out[param_ind] = temp_lcl
	if "ml_cin" in param:
		param_ind = np.where(param=="ml_cin")[0][0]
		param_out[param_ind] = ml_cin
	if "srh01" in param:
		param_ind = np.where(param=="srh01")[0][0]
		srh01 = get_srh(ua[t],va[t],hgt[t],1000,True,850,700,p)
		param_out[param_ind] = srh01
	if "srh03" in param:
		srh03 = get_srh(ua[t],va[t],hgt[t],3000,True,850,700,p)
		param_ind = np.where(param=="srh03")[0][0]
		param_out[param_ind] = srh03
	if "srh06" in param:
		param_ind = np.where(param=="srh06")[0][0]
		srh06 = get_srh(ua[t],va[t],hgt[t],6000,True,850,700,p)
		param_out[param_ind] = srh06
	if "ship" in param:
		if "s06" not in param:
			raise NameError("To calculate ship, s06 must be included")
		param_ind = np.where(param=="ship")[0][0]
		muq = mu_cape_inds.choose(q)
		ship = get_ship(mu_cape,np.copy(muq),ta[t],ua[t],va[t],hgt[t],p,np.copy(s06))
		param_out[param_ind] = ship
	if "mmp" in param:
		param_ind = np.where(param=="mmp")[0][0]
		param_out[param_ind] = get_mmp(ua[t],va[t],uas[t],vas[t],\
			mu_cape,ta[t],hgt[t])
	if "scp" in param:
		if "srh03" not in param:
			raise NameError("To calculate ship, srh03 must be included")
		param_ind = np.where(param=="scp")[0][0]
		scell_pot = get_supercell_pot(mu_cape,ua[t],va[t],hgt[t],ta_unit,p_unit,\
				q_unit,srh03)
		param_out[param_ind] = scell_pot
	if "stp" in param:
		if "srh01" not in param:
			raise NameError("To calculate stp, srh01 must be included")
		param_ind = np.where(param=="stp")[0][0]
		stp = get_tornado_pot(ml_cape,np.copy(lcl),np.copy(ml_cin),ua[t],va[t],p_3d[t],hgt[t],p,\
			np.copy(srh01))
		param_out[param_ind] = stp
	if "vo10" in param:
		param_ind = np.where(param=="vo10")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		vo10 = get_vo(uas[t],vas[t],dx,dy)
		param_out[param_ind] = vo10
	if "conv10" in param:
		param_ind = np.where(param=="conv10")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = get_conv(uas[t],vas[t],dx,dy)
	if "conv1000-850" in param:
		levs = np.where((p<=1001)&(p>=849))[0]
		param_ind = np.where(param=="conv1000-850")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = \
			np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
	if "conv800-600" in param:
		levs = np.where((p<=801)&(p>=599))[0]
		param_ind = np.where(param=="conv800-600")[0][0]
		x,y = np.meshgrid(lon,lat)
		dx,dy = mpcalc.lat_lon_grid_deltas(x,y)
		param_out[param_ind] = \
			np.mean(np.stack([get_conv(ua[t,i],va[t,i],dx,dy) for i in levs]),axis=0)
	if "non_sc_stp" in param:
		if "vo10" not in param:
			raise NameError("To calculate non_sc_stp, vo must be included")
		if "lr1000" not in param:
			raise NameError("To calculate non_sc_stp, lr1000 must be included")
		param_ind = np.where(param=="non_sc_stp")[0][0]
		non_sc_stp = get_non_sc_tornado_pot(ml_cape,ml_cin,np.copy(lcl),ua[t],va[t],\
			uas[t],vas[t],p_3d[t],ta[t],hgt[t],p,vo10,lr1000)
		param_out[param_ind] = non_sc_stp
	if "cape*s06" in param:
		param_ind = np.where(param=="cape*s06")[0][0]
		cs6 = ml_cape * np.power(s06,1.67)
		param_out[param_ind] = cs6
	if "td850" in param:
		param_ind = np.where(param=="td850")[0][0]
		td850 = get_td_diff(ta[t],dp[t],p_3d[t],850)
		param_out[param_ind] = td850
	if "td800" in param:
		param_ind = np.where(param=="td800")[0][0]
		param_out[param_ind] = get_td_diff(ta[t],dp[t],p_3d[t],800)
	if "td950" in param:
		param_ind = np.where(param=="td950")[0][0]
		param_out[param_ind] = get_td_diff(ta[t],dp[t],p_3d[t],950)
	if "wg" in param:
		try:
			param_ind = np.where(param=="wg")[0][0]
			param_out[param_ind] = wg[t]
		except ValueError:
			print("wg field expected, but not parsed")
	if "dcape" in param:
		param_ind = np.where(param=="dcape")[0][0]
		dcape = np.nanmax(get_dcape(p_3d[t],ta[t],hgt[t],p,ps[t]),axis=0)
		param_out[param_ind] = dcape
	if "mlm" in param:
		param_ind = np.where(param=="mlm")[0][0]
		mlm_u, mlm_v = get_mean_wind(ua[t],va[t],hgt[t],800,600,False,None,"plevels",p)
		mlm = np.sqrt(np.square(mlm_u) + np.square(mlm_v))
		param_out[param_ind] = mlm
	if "dlm" in param:
		param_ind = np.where(param=="dlm")[0][0]
		dlm_u, dlm_v = get_mean_wind(ua[t],va[t],hgt[t],1000,500,False,None,"plevels",p)
		dlm = np.sqrt(np.square(dlm_u) + np.square(dlm_v))
		param_out[param_ind] = dlm
	if "dlm+dcape" in param:
		param_ind = np.where(param=="dlm+dcape")[0][0]
		dlm_dcape = dlm + np.sqrt(2*dcape)
		param_out[param_ind] = dlm_dcape
	if "mlm+dcape" in param:
		param_ind = np.where(param=="mlm+dcape")[0][0]
		mlm_dcape = mlm + np.sqrt(2*dcape)
		param_out[param_ind] = mlm_dcape
	if "dcape*cs6" in param:
		param_ind = np.where(param=="dcape*cs6")[0][0]
		param_out[param_ind] = (dcape/980.) * (cs6/20000)
	if "dlm*dcape*cs6" in param:
		param_ind = np.where(param=="dlm*dcape*cs6")[0][0]
		param_out[param_ind] = (dlm_dcape/30.) * (cs6/20000)
	if "mlm*dcape*cs6" in param:
		param_ind = np.where(param=="mlm*dcape*cs6")[0][0]
		param_out[param_ind] = (mlm_dcape/30.) * (cs6/20000)
	if "dcp" in param:
		param_ind = np.where(param=="dcp")[0][0]
		param_out[param_ind] = (dcape/980)*(mu_cape/2000)*(s06/10)*(dlm/8)
	if "mf" in param:
		param_ind = np.where(param=="mf")[0][0]
		mf = ((ml_cape>120) & (dcape>350) & (mlm<26) )
		mf = mf * 1.0
		param_out[param_ind] = mf
	if "sf" in param:
		param_ind = np.where(param=="sf")[0][0]
		sf = ((s06>=30) & (dcape<500) & (mlm>=26) )                
		sf = sf * 1.0
		param_out[param_ind] = sf
	if "cond" in param:
		param_ind = np.where(param=="cond")[0][0]
		cond = (sf==1.0) | (mf==1.0)
		cond = cond * 1.0
		param_out[param_ind] = cond
	
	return param_out

def calc_param_sharppy(ta,dp,hur,hgt,ua,va,uas,vas,ps,lsm,terrain,p,lon,lat,times,param,save,out_name,region,model):
	#Calculate parameters based on the SHARPpy package for creating profiles

	#For each time in "times", loop over lat/lon points in domain and calculate:
	# 1) profile 2) parcel (if create_parcel is set) 3) parameter
	#NOTE the choice of parameter may affect both steps 2) and 3)
	#Input vars are of shape [time, levels, lat, lon]
	#Output is a list of numpy arrays of length=len(params) with dimensions [time,lat,lon]
	#Option to save as a netcdf file

	param = np.array(param)
	param_out = [0] * (len(param))
	for i in np.arange(0,len(param)):
		param_out[i] = np.empty((len(times),len(lat),len(lon)))

	np.warnings.filterwarnings('ignore')

########################################################################################################
	#This is a block of code intended for use with the multiprocessing module. However, it's become
	#apparant that this module does not have multi-node support for use on raijin.

	#Try making a 4d variable (e.g. ta[t]) a shared multiprocessing array before 
	# starting process
	# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
########################################################################################################
#	#CREATE SHARED VAR
#	shape_4d = (ta.shape[0], ta.shape[1], ta.shape[2], ta.shape[3])
#	shape_2d = (lsm.shape[0], lsm.shape[1])
#	#TA
#	ta_shared = multiprocessing.RawArray("d",ta.shape[0]*ta.shape[1]*ta.shape[2]*ta.shape[3])
#	ta_shared_np = np.frombuffer(ta_shared,dtype=np.float64).reshape(shape_4d)
#	np.copyto(ta_shared_np,ta)
#	#UA
#	ua_shared = multiprocessing.RawArray("d",ua.shape[0]*ua.shape[1]*ua.shape[2]*ua.shape[3])
#	ua_shared_np = np.frombuffer(ua_shared,dtype=np.float64).reshape(shape_4d)
#	np.copyto(ua_shared_np,ua)
#	#VA
#	va_shared = multiprocessing.RawArray("d",va.shape[0]*va.shape[1]*va.shape[2]*va.shape[3])
#	va_shared_np = np.frombuffer(va_shared,dtype=np.float64).reshape(shape_4d)
#	np.copyto(va_shared_np,va)
#	#HGT
#	hgt_shared = multiprocessing.RawArray("d",hgt.shape[0]*hgt.shape[1]*hgt.shape[2]*hgt.shape[3])
#	hgt_shared_np = np.frombuffer(hgt_shared,dtype=np.float64).reshape(shape_4d)
#	np.copyto(hgt_shared_np,hgt)
#	#DP
#	dp_shared = multiprocessing.RawArray("d",dp.shape[0]*dp.shape[1]*dp.shape[2]*dp.shape[3])
#	dp_shared_np = np.frombuffer(dp_shared,dtype=np.float64).reshape(shape_4d)
#	np.copyto(dp_shared_np,dp)
#	#LSM
#	lsm_shared = multiprocessing.RawArray("d",lsm.shape[0]*lsm.shape[1])
#	lsm_shared_np = np.frombuffer(lsm_shared,dtype=np.float64).reshape(shape_2d)
#	np.copyto(lsm_shared_np,lsm)
#	#CREATE "INITIALISER"
#	def init_worker(ta_shared, ua_shared, va_shared, hgt_shared, dp_shared, lsm_shared, p, shape_4d, shape_2d):
#		var_dict['ta_shared'] = ta_shared
#		var_dict['ua_shared'] = ua_shared
#		var_dict['va_shared'] = va_shared
#		var_dict['hgt_shared'] = hgt_shared
#		var_dict['dp_shared'] = dp_shared
#		var_dict['lsm_shared'] = lsm_shared
#		var_dict['p'] = p
#		var_dict['shape_4d'] = shape_4d
#		var_dict['shape_2d'] = shape_2d
########################################################################################################

	job_server = pp.Server()

	for t in np.arange(0,len(times)):
		print(times[t])

########################################################################################################
	#This is a block of code intended for use with the multiprocessing module. However, it's become
	#apparant that this module does not have multi-node support for use on raijin.
########################################################################################################
	#	xyt = itertools.product(np.arange(0,len(lat)),np.arange(0,len(lon)),[t])
	#	#ncpu = multiprocessing.cpu_count()
	#	ncpu = int(os.environ["PBS_NCPUS"])
	#	ncpu = 64
	#	print("Using "+str(ncpu)+" cpus")
	#	pool=multiprocessing.Pool(processes=ncpu, \
	#		initializer=init_worker,initargs=(ta_shared, ua_shared, va_shared, hgt_shared, dp_shared, lsm_shared, p, shape_4d, shape_2d))
	#	result = list(pool.imap(sharp_parcel,xyt,chunksize=ncpu))

	#	param_out[np.where(param=="mu_cape")[0][0]][t] = np.array([pcl[0] for pcl in result]).\
	#		reshape((len(lat),len(lon)))
	#	param_out[np.where(param=="ml_cape")[0][0]][t] = np.array([pcl[1] for pcl in result]).\
	#		reshape((len(lat),len(lon)))
	#	pool.close()
	#	pool.join()
########################################################################################################
		args = ( np.arange(0,len(lat)), np.arange(0,len(lon)), [t], lsm, [ua[t]], [va[t]], \
				[p], [hgt[t]], [ta[t]], [dp[t]] )
		modules = ("utils","profile","params")
		depfuncs = (utils.MS2KTS,)
		r = job_server.submit(sharp_parcel_pp, args, depfuncs=depfuncs, modules=modules, globals=globals())
		r()

	if save:
		save_netcdf(region,model,out_name,times,lat,lon,param,param_out)

	return r

def sharp_parcel_pp(y,x,t,lsm,uat,vat,p,hgtt,tat,dpt):

	#Exact same as sharp parcel, but intends to use the "pp" module (parallel python)
	
	for i in y:
		for j in x:
	
			if lsm[i,j] == 1:

				#convert u and v to kts for use in profile
				ua_p_kts = utils.MS2KTS(ua[t,:,i,j])
				va_p_kts = utils.ms2kts(va[t,:,i,j])

				#create profile
				prof = profile.create_profile(pres=p, hght=hgt[t,:,i,j], \
						tmpc=ta[t,:,i,j], \
						dwpc=dp[t,:,i,j], u=ua_p_kts, v=va_p_kts,\
						 missing=np.nan,\
						strictqc=false)

				#create most unstable parcel
				mu_parcel = params.parcelx(prof, flag=3, dp=-10) #3 = mu
				ml_parcel = params.parcelx(prof, flag=4, dp=-10) #4 = ml
				return (mu_parcel.bplus,ml_parcel.bplus)
			else:
				return (np.nan,np.nan)

def sharp_parcel(i):

	#function which produces an array of parcel and/or wind objects from sharppy in parallel
	
			#y,x,lon,lat,temp_ta,temp_dp,temp_hur,temp_hgt,p,temp_ua,temp_va,temp_uas,temp_vas = i
			y,x,t = i
	
			lsm_shared_np = np.frombuffer(var_dict["lsm_shared"]).reshape(var_dict["shape_2d"])

			if lsm_shared_np[y,x] == 1:

				ta_shared_np = np.frombuffer(var_dict["ta_shared"]).reshape(var_dict["shape_4d"])
				ua_shared_np = np.frombuffer(var_dict["ua_shared"]).reshape(var_dict["shape_4d"])
				va_shared_np = np.frombuffer(var_dict["va_shared"]).reshape(var_dict["shape_4d"])
				hgt_shared_np = np.frombuffer(var_dict["hgt_shared"]).reshape(var_dict["shape_4d"])
				dp_shared_np = np.frombuffer(var_dict["dp_shared"]).reshape(var_dict["shape_4d"])
				#convert u and v to kts for use in profile
				ua_p_kts = utils.ms2kts(ua_shared_np[t,:,y,x])
				va_p_kts = utils.ms2kts(va_shared_np[t,:,y,x])

				#create profile
				prof = profile.create_profile(pres=var_dict["p"], hght=hgt_shared_np[t,:,y,x], \
						tmpc=ta_shared_np[t,:,y,x], \
						dwpc=dp_shared_np[t,:,y,x], u=ua_p_kts, v=va_p_kts,\
						 missing=np.nan,\
						strictqc=false)

				#create most unstable parcel
				mu_parcel = params.parcelx(prof, flag=3, dp=-10) #3 = mu
				ml_parcel = params.parcelx(prof, flag=4, dp=-10) #4 = ml
				return (mu_parcel.bplus,ml_parcel.bplus)
			else:
				return (np.nan,np.nan)

def barra_ad_driver(points,loc_id):
	#Drive calc_model() for each month available in the BARRA dataset, for the sa_small domain

	model = "barra_ad"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	dates = []
	for y in np.arange(2006,2017):
		for m in np.arange(1,13):
			if (m != 12):
				dates.append([dt.datetime(y,m,1,0,0,0),\
					dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
			else:
				dates.append([dt.datetime(y,m,1,0,0,0),\
					dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	df = pd.DataFrame()
	if dates[0][0] == dt.datetime(2006,1,1,0):
		dates[0][0] = dt.datetime(2006,1,1,6)
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		param,param_out,lon,lat,times = calc_model(model,model,"points",dates[t],"",False,\
			region,cape_method)	
		temp_df = load_array_points(param,param_out,lon,lat,times,points,loc_id,\
				model,smooth=False)
		temp_df = temp_df[np.in1d(temp_df.hour,np.array([0,6,12,18]))]
		df = df.append(temp_df)
		temp_fname_pkl = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
			"barra_ad_points_"+dates[t][0].strftime("%Y%m")+".pkl"
		#if os.path.isfile(temp_fname_pkl):
		#	temp_exist = pd.read_pickle(temp_fname_pkl)
		#	temp_df = pd.concat([temp_exist,temp_df])
		temp_df.to_pickle(temp_fname_pkl)
	outname_pkl = "/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_"+\
		"2006_2016.pkl"
	#if os.path.isfile(outname_pkl):
	#	exist_df = pd.read_pickle(outname_pkl)
	#	df = pd.concat([exist_df,df])
	df.to_pickle(outname_pkl)

def barra_r_fc_driver(points,loc_id):
	#Drive calc_model() for each month available in the BARRA dataset, for the sa_small domain

	model = "barra_r_fc"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	smooth = False
	dates = []
	param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
		"stp","ship","mmp","relhum850-500","crt","non_sc_stp","vo","lr1000","lcl",\
		"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
		"cape*s06","cape*ssfc6","td950","td850","td800","cape700","wg"]
	for y in np.arange(2003,2017,1):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
			if (m != 12):
				dates.append([dt.datetime(y,m,1,0,0,0),\
					dt.datetime(y,m+1,1,0,0,0)-dt.timedelta(hours = 6)])
			else:
				dates.append([dt.datetime(y,m,1,0,0,0),\
					dt.datetime(y+1,1,1,0,0,0)-dt.timedelta(hours = 6)])
	df = pd.DataFrame()
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		#To run convective parameters
		#param,param_out,lon,lat,times = calc_model(model,model,"domain",dates[t],param,True,\
		#	region,cape_method)	
		#To run wind gusts point extraction
		param,param_out,lon,lat,times = calc_model(model,model,"points",dates[t],param,False,\
			region,cape_method)
		temp_df = load_array_points(param,param_out,lon,lat,times,points,loc_id,\
				model,smooth=smooth)
		temp_df = temp_df[np.in1d(temp_df.hour,np.array([0,6,12,18]))]
		df = df.append(temp_df)
		temp_df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
			"barra_r_fc_points_"+dates[t][0].strftime("%Y%m")+".pkl")
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/barra_r_fc_points_"+\
		"2006_2016.pkl")

def erai_fc_driver():
	#Drive calc_model() for 2010-2015 forcaset data in ERA-Interim, for the sa_small domain

	model = "erai_fc"
	cape_method = "wrf"
	method = "domain"
	region = "sa_small"
	param = ["wg10","cape"]
	dates = []
	for y in np.arange(1979,2010):
		for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
			if (m != 12):
				dates.append([dt.datetime(y,m,1,6,0,0),\
					dt.datetime(y,m+1,1,0,0,0)])
			else:
				dates.append([dt.datetime(y,m,1,0,0,0),\
					dt.datetime(y+1,1,1,0,0,0)])
	for t in np.arange(0,len(dates)):
		print(str(dates[t][0])+" - "+str(dates[t][1]))
		calc_model(model,model,method,dates[t],param,True,region,cape_method)	

if __name__ == "__main__":

#--------------------------------------------------------------------------------------------------
# 	SETTINGS
#--------------------------------------------------------------------------------------------------

	model = "barra"
	cape_method = "wrf_par"

	method = "domain"
	region = "sa_small"

	experiment = "test"

	#ADELAIDE = UTC + 10:30
	time = [dt.datetime(2016,9,1,0,0,0),dt.datetime(2016,9,30,18,0,0)]
	issave = True

	loc_id = ["Port Augusta","Marree","Munkora","Woomera","Robe","Loxton","Coonawarra",\
			"Renmark","Clare HS","Adelaide AP","Coober Pedy AP","Whyalla",\
			"Padthaway South","Nuriootpa","Rayville Park","Mount Gambier",\
			"Naracoorte","The Limestone","Parafield","Austin Plains","Roseworthy",\
			"Tarcoola","Edinburgh"]
	points = [(137.78,-32.54),(138.0684,-29.6587),(140.3273,-36.1058),(136.82,-31.15),\
			(139.8054,-37.1776),(140.5978,-34.439),(140.8254,-37.2906),\
			(140.6766,-34.1983),(138.5933,-33.8226),(138.53,-34.96),\
			(134.7222,-29.0347),(137.5206,-33.0539),(140.5212,-36.6539),\
			(139.0056,-34.4761),(138.2182,-33.7676),(140.7739,-37.7473),\
			(140.7270,-36.9813),(139.7164,-36.9655),(138.6281,-34.7977),\
			(140.5378,-35.3778),(138.6763,-34.5106),(134.5786,-30.7051),\
			(138.6222,-34.7111)]

#--------------------------------------------------------------------------------------------------
#	RUN CODE
#--------------------------------------------------------------------------------------------------

	if (cape_method == "wrf") | (cape_method == "wrf_par"):
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","vo10","lr1000","lcl",\
			"relhum1000-700","s06","s0500","s01","s03",\
			"cape*s06","dcp","td850","td800","td950","dcape","mlm","dlm",\
			"dcape*cs6","mlm+dcape","mlm*dcape*cs6","mf","sf","cond"]
	elif cape_method == "SHARPpy":
		param = ["ml_cape","mu_cape"]
	elif cape_method == "points_wrf":
		param = ["ml_cape","ml_cin","mu_cin","mu_cape","srh01","srh03","srh06","scp",\
			"stp","ship","mmp","relhum850-500","crt","lr1000","lcl",\
			"relhum1000-700","ssfc6","ssfc3","ssfc1","s06","ssfc850","ssfc500",\
			"cape*s06","cape*ssfc6"]
	elif cape_method == "points_SHARPpy":
		param = ["mu_cape","s06","cape*s06"]	

	#for time in times:
	#[param,param_out,lon,lat,date_list] = calc_model(model,experiment,method,time,\
	#		param,issave,region,cape_method)	
	#print(param_out[0][0])
	#erai_driver(param)
	#barra_driver(param)
	#barra_ad_driver(points,loc_id)
	#barra_r_fc_driver(points,loc_id)

	region="sa_small"
	model="erai";
	out_name="test";
	method="domain";
	time=[dt.datetime(2016,9,28,0,0,0),dt.datetime(2016,9,28,0,0,0)]
	param=["ml_cape","mu_cape"];\
	issave=False;
	cape_method="SHARPpy"
	param,param_out,lon,lat,date_list = calc_model(model,out_name,method,\
			time,param,issave,region,cape_method)

