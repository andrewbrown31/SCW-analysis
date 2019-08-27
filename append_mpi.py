import netCDF4 as nc
import pandas as pd
from erai_read import read_erai
from erai_read import get_mask as get_erai_mask
from barra_read import read_barra
from barra_read import get_mask as get_barra_mask
from calc_param import save_netcdf, get_dp
import datetime as dt
from mpi4py import MPI
import numpy as np
import sharppy.sharptab.profile as profile
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.thermo as thermo
import warnings
import sys

#Script which appends to already-made netcdf files, using MPI4Py/SHARPpy


def sharp_parcel_mpi(p,wap,ua,va,hgt,ta,dp,ps,uas,vas,tas,ta2d,terrain,model):

	agl_idx = (p <= ps)

	dp = replace_dp(dp)

	try:
		prof = profile.create_profile(pres = np.insert(p[agl_idx],0,ps), \
			hght = np.insert(hgt[agl_idx],0,terrain), \
			tmpc = np.insert(ta[agl_idx],0,tas), \
			dwpc = np.insert(dp[agl_idx],0,ta2d), \
			u = np.insert(ua[agl_idx],0,uas), \
			v = np.insert(va[agl_idx],0,vas), \
			strictqc=False, omeg=np.insert(wap[agl_idx],0,wap[agl_idx][0]) )
	except:
		p = p[agl_idx]; ua = ua[agl_idx]; va = va[agl_idx]; hgt = hgt[agl_idx]; ta = ta[agl_idx]; \
		dp = dp[agl_idx]
		p[0] = ps
		ua[0] = uas
		va[0] = vas
		hgt[0] = terrain
		ta[0] = tas
		dp[0] = ta2d
		prof = profile.create_profile(pres = p, \
			hght = hgt, \
			tmpc = ta, \
			dwpc = dp, \
			u = ua, \
			v = va, \
			strictqc=False, omeg=wap[agl_idx])

	return prof

def replace_dp(dp_1d):

	if np.isnan(dp_1d).any():
		dp_nan_idx = np.where(np.isnan(dp_1d))[0]
		for i in dp_nan_idx:
			upper_inds = np.arange(i, len(dp_1d))
			dp_up = np.nan
			for j in upper_inds:
				if ~np.isnan(dp_1d[j]):
					dp_up = dp_1d[j]
					break
			lower_inds = np.flip(np.arange(0, i+1))
			dp_low = np.nan
			for j in lower_inds:
				if ~np.isnan(dp_1d[j]):
					dp_low = dp_1d[j]
					break
			dp_1d[i] = np.nanmean([dp_low, dp_up])
	return dp_1d

if __name__ == "__main__":

	#Ignore warnings from SHARPpy
	warnings.simplefilter("ignore")

	#Get MPI communicator info
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	#Load data into first processer (can be thought of as "local", processer)
	if rank == 0:

		region = "aus"
		if region == "sa_small":
			start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
		elif region == "aus":
       	    		start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275

		domain = [start_lat,end_lat,start_lon,end_lon]
		model = sys.argv[1]
		t1 = sys.argv[2]
		t2 = sys.argv[3]
		time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]

		if model == "erai":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,cp,wg10,cape,lon,lat,date_list = \
				read_erai(domain,time)
			dp = get_dp(hur=hur, ta=ta)
			lsm = np.repeat(get_erai_mask(lon,lat)[np.newaxis],ta.shape[0],0)
			terrain = np.repeat(terrain[np.newaxis],ta.shape[0],0)
		elif model == "barra":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
				read_barra(domain,time)
			dp = get_dp(hur=hur, ta=ta)
			lsm = np.repeat(get_barra_mask(lon,lat)[np.newaxis],ta.shape[0],0)
			terrain = np.repeat(terrain[np.newaxis],ta.shape[0],0)
		else:
			raise ValueError("INVALID MODEL NAME\n")


		orig_shape = ta.shape
		ta = np.moveaxis(ta,[0,1,2,3],[0,3,1,2]).\
			reshape((ta.shape[0]*ta.shape[2]*ta.shape[3],ta.shape[1])).astype("double",order="C")
		dp = np.moveaxis(dp,[0,1,2,3],[0,3,1,2]).\
			reshape((dp.shape[0]*dp.shape[2]*dp.shape[3],dp.shape[1])).astype("double",order="C")
		hur = np.moveaxis(hur,[0,1,2,3],[0,3,1,2]).\
			reshape((hur.shape[0]*hur.shape[2]*hur.shape[3],hur.shape[1])).astype("double",order="C")
		hgt = np.moveaxis(hgt,[0,1,2,3],[0,3,1,2]).\
			reshape((hgt.shape[0]*hgt.shape[2]*hgt.shape[3],hgt.shape[1])).astype("double",order="C")
		wap = np.moveaxis(wap,[0,1,2,3],[0,3,1,2]).\
			reshape((wap.shape[0]*wap.shape[2]*wap.shape[3],wap.shape[1])).astype("double",order="C")
		ua = utils.MS2KTS(np.moveaxis(ua,[0,1,2,3],[0,3,1,2]).\
			reshape((ua.shape[0]*ua.shape[2]*ua.shape[3],ua.shape[1]))).astype("double",order="C")
		va = utils.MS2KTS(np.moveaxis(va,[0,1,2,3],[0,3,1,2]).\
			reshape((va.shape[0]*va.shape[2]*va.shape[3],va.shape[1]))).astype("double",order="C")
		uas = utils.MS2KTS(uas.reshape((uas.shape[0]*uas.shape[1]*uas.shape[2]))).astype("double",order="C")
		vas = utils.MS2KTS(vas.reshape((vas.shape[0]*vas.shape[1]*vas.shape[2]))).astype("double",order="C")
		tas = (tas.reshape((tas.shape[0]*tas.shape[1]*tas.shape[2]))).astype("double",order="C")
		ta2d = (ta2d.reshape((ta2d.shape[0]*ta2d.shape[1]*ta2d.shape[2]))).astype("double",order="C")
		terrain = np.array(terrain.reshape((terrain.shape[0]*terrain.shape[1]*terrain.shape[2]))).\
			astype("double",order="C")
		ps = ps.reshape((ps.shape[0]*ps.shape[1]*ps.shape[2])).astype("double",order="C")
		wg10 = wg10.reshape((wg10.shape[0]*wg10.shape[1]*wg10.shape[2])).astype("double",order="C")
		lsm = np.array(lsm.reshape((lsm.shape[0]*lsm.shape[1]*lsm.shape[2]))).astype("double",order="C")
		
		#Restricting base data to land points. Keep original shape and land sea mask to put data back in
		# to at the end for saving
		orig_length = ta.shape[0]
		lsm_orig = lsm
		ta = ta[lsm==1]
		dp = dp[lsm==1]
		wap = wap[lsm==1]
		hur = hur[lsm==1]
		hgt = hgt[lsm==1]
		ua = ua[lsm==1]
		va = va[lsm==1]
		uas = uas[lsm==1]
		vas = vas[lsm==1]
		tas = tas[lsm==1]
		ta2d = ta2d[lsm==1]
		ps = ps[lsm==1]
		terrain = terrain[lsm==1]
		wg10 = wg10[lsm==1]
		lsm = lsm[lsm==1]

		#---------------------------------------------------------------------------------------------------
		# NOTE
		# PUT HERE THE VARIABLES TO BE APPENDED
		#
		#---------------------------------------------------------------------------------------------------
		param = np.array(["wbz", "Vprime"])
		output_data = np.zeros((ta.shape[0], len(param)))

		#Split/chunk the base arrays on the spatial-temporal grid point dimension, for parallel processing
		ta_split = np.array_split(ta, size, axis = 0)
		dp_split = np.array_split(dp, size, axis = 0)
		wap_split = np.array_split(wap, size, axis = 0)
		hur_split = np.array_split(hur, size, axis = 0)
		hgt_split = np.array_split(hgt, size, axis = 0)
		ua_split = np.array_split(ua, size, axis = 0)
		va_split = np.array_split(va, size, axis = 0)
		uas_split = np.array_split(uas, size, axis = 0)
		vas_split = np.array_split(vas, size, axis = 0)
		tas_split = np.array_split(tas, size, axis = 0)
		ta2d_split = np.array_split(ta2d, size, axis = 0)
		ps_split = np.array_split(ps, size, axis = 0)
		terrain_split = np.array_split(terrain, size, axis = 0)
		wg10_split = np.array_split(wg10, size, axis = 0)

		lsm_split = np.array_split(lsm, size, axis = 0)
		split_sizes = []
		for i in range(0,len(ta_split),1):
			split_sizes = np.append(split_sizes, ta_split[i].shape[0])

		#Remember the points at which splits occur on (noting that Gatherv and Scatterv act on a 
		# "C" style (row-major) flattened array). This will be different for pressure-level and sfc-level
		# variables
		split_sizes_input = split_sizes*ta.shape[1]
		displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
		split_sizes_output = split_sizes*len(param)
		displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
		split_sizes_input_2d = split_sizes
		displacements_input_2d = np.insert(np.cumsum(split_sizes_input_2d),0,0)[0:-1]


	else:
		#Initialise variables on other cores (can be thought of as "remote"), including the name of the 
		# model (as each processer needs to know whether ERA-Interim specific parameters are being included
		model = sys.argv[0]
		split_sizes_input = None; displacements_input = None; split_sizes_output = None;\
			displacements_output = None; split_sizes_input_2d = None; displacements_input_2d = None
		ta_split = None; dp_split = None; hur_split = None; hgt_split = None; ua_split = None;\
			va_split = None; uas_split = None; vas_split = None; lsm_split = None;\
			wg10_split = None; ps_split = None; tas_split = None; ta2d_split = None;\
			terrain_split = None; wap_split = None
		ta = None; dp = None; hur = None; hgt = None; ua = None; wap = None;\
			va = None; uas = None; vas = None; lsm = None; wg10 = None; ps = None; tas = None; \
			ta2d = None; terrain = None
		p = None
		output_data = None
		param = None

	#Broadcast split arrays to other cores
	ta_split = comm.bcast(ta_split, root=0)
	dp_split = comm.bcast(dp_split, root=0)
	wap_split = comm.bcast(wap_split, root=0)
	hur_split = comm.bcast(hur_split, root=0)
	hgt_split = comm.bcast(hgt_split, root=0)
	ua_split = comm.bcast(ua_split, root=0)
	va_split = comm.bcast(va_split, root=0)
	uas_split = comm.bcast(uas_split, root=0)
	vas_split = comm.bcast(vas_split, root=0)
	tas_split = comm.bcast(tas_split, root=0)
	ta2d_split = comm.bcast(ta2d_split, root=0)
	ps_split = comm.bcast(ps_split, root=0)
	terrain_split = comm.bcast(terrain_split, root=0)
	wg10_split = comm.bcast(wg10_split, root=0)
	lsm_split = comm.bcast(lsm_split, root=0)
	p = comm.bcast(p, root=0)
	param = comm.bcast(param, root=0)
	split_sizes_input = comm.bcast(split_sizes_input, root = 0)
	displacements_input = comm.bcast(displacements_input, root = 0)
	split_sizes_output = comm.bcast(split_sizes_output, root = 0)
	displacements_output = comm.bcast(displacements_output, root = 0)

	#Create arrays to receive chunked/split data on each core, where rank specifies the core
	ta_chunk = np.zeros(np.shape(ta_split[rank]))
	dp_chunk = np.zeros(np.shape(dp_split[rank]))
	wap_chunk = np.zeros(np.shape(wap_split[rank]))
	hur_chunk = np.zeros(np.shape(hur_split[rank]))
	hgt_chunk = np.zeros(np.shape(hgt_split[rank]))
	ua_chunk = np.zeros(np.shape(ua_split[rank]))
	va_chunk = np.zeros(np.shape(va_split[rank]))
	uas_chunk = np.zeros(np.shape(uas_split[rank]))
	vas_chunk = np.zeros(np.shape(vas_split[rank]))
	tas_chunk = np.zeros(np.shape(tas_split[rank]))
	ta2d_chunk = np.zeros(np.shape(ta2d_split[rank]))
	ps_chunk = np.zeros(np.shape(ps_split[rank]))
	terrain_chunk = np.zeros(np.shape(terrain_split[rank]))
	wg10_chunk = np.zeros(np.shape(wg10_split[rank]))
	lsm_chunk = np.zeros(np.shape(lsm_split[rank]))
	comm.Scatterv([ta,split_sizes_input, displacements_input, MPI.DOUBLE],ta_chunk,root=0)
	comm.Scatterv([dp,split_sizes_input, displacements_input, MPI.DOUBLE],dp_chunk,root=0)
	comm.Scatterv([wap,split_sizes_input, displacements_input, MPI.DOUBLE],wap_chunk,root=0)
	comm.Scatterv([hur,split_sizes_input, displacements_input, MPI.DOUBLE],hur_chunk,root=0)
	comm.Scatterv([hgt,split_sizes_input, displacements_input, MPI.DOUBLE],hgt_chunk,root=0)
	comm.Scatterv([ua,split_sizes_input, displacements_input, MPI.DOUBLE],ua_chunk,root=0)
	comm.Scatterv([va,split_sizes_input, displacements_input, MPI.DOUBLE],va_chunk,root=0)
	comm.Scatterv([uas,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],uas_chunk,root=0)
	comm.Scatterv([vas,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],vas_chunk,root=0)
	comm.Scatterv([tas,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],tas_chunk,root=0)
	comm.Scatterv([ta2d,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],ta2d_chunk,root=0)
	comm.Scatterv([ps,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],ps_chunk,root=0)
	comm.Scatterv([terrain,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],terrain_chunk,root=0)
	comm.Scatterv([wg10,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],wg10_chunk,root=0)
	comm.Scatterv([lsm,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],lsm_chunk,root=0)
	comm.Barrier()

	#Print diagnostics
	if rank == 0:
		print("TOTAL (LAND) POINTS: %s" %(ta.shape[0]))
		print("CHUNKSIZE: %s" %(ua_chunk.shape,))


#----------------------------------------------------------------------------------------------------------------
	#Run SHARPpy
	start = dt.datetime.now()
	output = np.zeros((ta_chunk.shape[0],len(param)))
	for i in np.arange(0,ta_chunk.shape[0]):
		prof= sharp_parcel_mpi(p, \
			wap_chunk[i],\
			ua_chunk[i],\
			va_chunk[i],\
			hgt_chunk[i],\
			ta_chunk[i],\
			dp_chunk[i],\
			ps_chunk[i],\
			uas_chunk[i],\
			vas_chunk[i],\
			tas_chunk[i],\
			ta2d_chunk[i],\
			terrain_chunk[i],\
			model)

#----------------------------------------------------------------------------------------------------------------
#NOTE
#EDIT BELOW TO APPEND VARS
#----------------------------------------------------------------------------------------------------------------
		sfc = prof.pres[prof.sfc]
		p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
		umean01 , vmean01 = winds.mean_wind(prof, pbot = sfc, ptop = p1km)
		Umean01 = utils.mag(umean01, vmean01)
		pwb0 = params.temp_lvl(prof, 0, wetbulb=True)
		hwb0 = interp.to_agl(prof, interp.hght(prof, pwb0) )
		T1 = abs( thermo.wetlift(prof.pres[0], prof.tmpc[0], 600) - interp.temp(prof, 600) )
		T2 = abs( thermo.wetlift(pwb0, interp.temp(prof, pwb0), sfc) - prof.tmpc[0] )
		Vprime = utils.KTS2MS( 13 * np.sqrt( (T1 + T2) / 2) + (1/3 * (Umean01) ) )

		output[i,np.where(param=="Vprime")[0][0]] = Vprime
		output[i,np.where(param=="wbz")[0][0]] = hwb0
#----------------------------------------------------------------------------------------------------------------

	#Print diagnostics
	if rank == 0:
		print("Time taken for SHARPPy on processor 1: %s" %(dt.datetime.now() - start), )
		print("Time taken for each element on processor 1: %s" \
			%((dt.datetime.now() - start)/float(ua_chunk.shape[0])), )

	#Gather output data together to root node
	comm.Gatherv(output, \
		[output_data, split_sizes_output, displacements_output, MPI.DOUBLE], \
		root=0)

	#Reshape data and save. For effective layer variables (given by effective_layer_params, see line ~143), 
	# which are undefined when there is no surface cape, replace masked values with zeros
	if rank == 0:
		param_out = []
		for param_name in param:
			#Extract data for land points (which is 1d) and replace masked elements with zeros for 
			# effective layer parameters
			temp_data = output_data[:,np.where(param==param_name)[0][0]]
			output_reshaped = np.zeros((orig_length))
			output_reshaped[:] = np.nan
			output_reshaped[lsm_orig==1] =  temp_data
			output_reshaped = output_reshaped.reshape((orig_shape[0],orig_shape[2],orig_shape[3]))
			param_out.append(output_reshaped)

		fname = "/g/data/eg3/ab4502/ExtremeWind/"+region+"/"+model+"/"+model+"_"+\
			dt.datetime.strftime(date_list[0],"%Y%m%d")+"_"+\
			dt.datetime.strftime(date_list[-1],"%Y%m%d")+".nc"
		param_file = nc.Dataset(fname,"a")
#----------------------------------------------------------------------------------------------------------------
#NOTE
#EDIT BELOW TO APPEND VARS
#----------------------------------------------------------------------------------------------------------------
		try:
			wbz_var = param_file.createVariable("wbz",float,\
				("time","lat","lon"))
		except:
			wbz_var = param_file.variables["wbz"]
		wbz_var.units = "m"
		wbz_var.long_name = "wet_bulb_zero_height"
		wbz_var[:] = param_out[np.where(np.array(param)=="wbz")[0][0]]

		try:
			Vprime_var = param_file.createVariable("Vprime",float,\
				("time","lat","lon"))
		except:
			Vprime_var = param_file.variables["Vprime"]
		Vprime_var.units = "m/s"
		Vprime_var.long_name = "miller_1972_wind_speed"
		Vprime_var[:] = param_out[np.where(np.array(param)=="Vprime")[0][0]]

#----------------------------------------------------------------------------------------------------------------
		param_file.close()

