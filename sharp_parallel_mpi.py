#Parallel sharppy using mpi

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
import warnings
import sys

#---------------------------------------------------------------------------------------------------
#TO RUN:
# > mpiexec python -m mpi4py sharp_parallel_mpi.py model region t1 t2 (save) (out_name)
#
#	- model 
#		Is either "barra" or "erai"
#	- region
#		Is either "aus" or "sa_small"
#	- t1
#		Is the start time, specified by "YYYYMMDDHH"
#	- t2
#		Is the end time, specified by "YYYYMMDDHH"
#
#	- save
#		Whether the output is to be saved (default is True)
#
#	- out_name
#		Specifies the prefix of the netcdf output.
#
#Before running
# > source activate sharppy
# > module load mpi4py/3.0.0-py3
# > module unload python3/3.6.2
#
#
#See line ~200 for a list of parameters to get
#---------------------------------------------------------------------------------------------------

def sharp_parcel_mpi(p,wap,ua,va,hgt,ta,dp,ps,uas,vas,tas,ta2d,terrain):

	#Exact same as sharp parcel, but intended to use the "mpi4py" module
	
	#Only use the part of the profile which is above the surface (i.e. where the pressure levels are less than
	# the surface pressure
	agl_idx = (p <= ps)

	#Replace masked dp values
	dp = replace_dp(dp)

	#create profile, inserting surface values at the bottom of each profile
	#It may be the case that the surface level data does not mesh with the pressure level data, such that
	# the surface height is higher than the bottom pressure level height, even though the surface pressure
	# is higher than the bottom pressure level. If this is the case, replace the bottom pressure level with the
	# surface level data
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

	#create parcels
	sb_parcel = params.parcelx(prof, flag=1, dp=-10)
	mu_parcel = params.parcelx(prof, flag=3, dp=-10)
	ml_parcel = params.parcelx(prof, flag=4, dp=-10)
	eff_parcel = params.parcelx(prof, flag=6, ecape=100, ecinh=-250, dp=-10)
	return (prof, mu_parcel ,ml_parcel, sb_parcel, eff_parcel)

def replace_dp(dp_1d):
	
	#Takes a one-dimensional dp array (dp on pressure levels), and replace masked values with an average 
	# between the first layer above and below which are defined. If the masked value is the first or last layer
	# , then just the above/below layers is used (e.g. if the top pressure level is masked, then the masked
	# value is replaced with the second-top pressure level value)

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

def maxtevv_fn(prof):
	
	#Calculate the maximum d(theta-e)/d(z) * omega product calculated from the 0-2 km layer through the 0-6 km
	# layer at 0.5 km intervals (Sherburn 2016)

	sfc = prof.pres[prof.sfc]
	p2km = interp.pres(prof, interp.to_msl(prof, 2000.))
	p6km = interp.pres(prof, interp.to_msl(prof, 6000.))

	idx1 = np.where((prof.pres >= p2km) & (prof.pres <= sfc))[0]
	idx2 = np.where((prof.pres >= p6km) & (prof.pres <= sfc))[0]

	maxtevv = 0
	for i in idx1:
		for j in idx2[~(idx2==i)]:
			dz = (prof.hght[j] - prof.hght[i]) / 1000.
			if dz >= 0.25:
				dthetae = (prof.thetae[j] - prof.thetae[i])
				res = (dthetae / dz) * prof.omeg[j]
				if res > maxtevv:
					maxtevv = res

	return maxtevv

if __name__ == "__main__":

	#Ignore warnings from SHARPpy
	warnings.simplefilter("ignore")

	#Get MPI communicator info
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	#Load data into first processer (can be thought of as "local", processer)
	if rank == 0:

		#Parse arguments from cmd line and set up inputs (date region model)
		model = sys.argv[1]
		region = sys.argv[2]
		t1 = sys.argv[3]
		t2 = sys.argv[4]
		issave = sys.argv[5]
		out_name = sys.argv[6]
		if region == "sa_small":
			start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
		elif region == "aus":
       	    		start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
		else:
			raise ValueError("INVALID REGION\n")
		domain = [start_lat,end_lat,start_lon,end_lon]
		try:
			time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]
		except:
			raise ValueError("INVALID START OR END TIME. SHOULD BE YYYYMMDDHH\n")
		if not ((issave=="True") | (issave=="False")):
			raise ValueError("\n INVALID ISSAVE...SHOULD BE True OR False")

		#Load data and setup base array, which is reformed from a 4d (or 3d for surface data) array to 
		# a 2d array, with rows as spatial-temporal coordinates and columns as vertical levels
		start = dt.datetime.now()
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
		if model == "erai":
			cape = cape.reshape((cape.shape[0]*cape.shape[1]*cape.shape[2])).astype("double",order="C")
			cp = cp.reshape((cp.shape[0]*cp.shape[1]*cp.shape[2])).astype("double",order="C")
		
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
		if model == "erai":
			cp = cp[lsm==1]
			cape = cape[lsm==1]
		lsm = lsm[lsm==1]

		#Set the ouput array (in this case, a matrix with rows given by the number of spatial-temporal 
		# points, and columns given by the length of "param")
		param = np.array(["ml_cape", "mu_cape", "sb_cape", "ml_cin", "sb_cin", "mu_cin",\
					"ml_lcl", "mu_lcl", "sb_lcl", "eff_cape", "eff_cin", "eff_lcl", "dcape", \
					"ml_lfc", "mu_lfc", "eff_lfc", "sb_lfc",\
					"lr01", "lr03", "lr13", "lr36", "lr24", "lr_freezing",\
					"qmean01", "qmean03", "qmean06", "qmean13", "qmean36",\
					"qmeansubcloud", "q_melting", "q1", "q3", "q6",\
					"rhmin01", "rhmin03", "rhmin06", "rhmin13", "rhmin36", \
					"rhminsubcloud", "tei", \
					"mhgt", "mu_el", "ml_el", "sb_el", "eff_el", \
					"pwat", "v_totals", "c_totals", "t_totals", \
					"maxtevv", "te_diff", "dpd850", "dpd700", "pbl_top",\
					\
					"cp", "cape", "dcp2", "cape*s06",\
					\
					"srhe", "srh01", "srh03", "srh06", \
					"srhe_left", "srh01_left", "srh03_left", "srh06_left", \
					"ebwd", "s010", "s06", "s03", "s01", "s13", "s36", "scld", \
					"omega01", "omega03", "omega06", \
					"U500", "U10", "U1", "U3", "U6", \
					"Ust", "Ust_left", "Usr01", "Usr03", "Usr06", "Usr01_left",\
					"Usr03_left", "Usr06_left", \
					"Uwindinf", "Umeanwindinf", "Umean800_600", "Umean06", \
					"Umean01", "Umean03", "wg10",\
					\
					"dcp", "stp_cin", "stp_cin_left", "stp_fixed", "stp_fixed_left",\
					"scp", "ship",\
					"mlcape*s06", "mucape*s06", "sbcape*s06", "effcape*s06", \
					"mlcape*s06_2", "mucape*s06_2", "sbcape*s06_2", \
					"effcape*s06_2", "dmgwind", "hmi", "wmsi_mu", "wmsi_ml",\
					"dmi", "mwpi_mu", "mwpi_ml", "ducs6", "convgust", "windex",\
					"gustex", "gustex2","gustex3", "eff_sherb", "sherb", \
					"moshe", "mosh", "wndg","mburst","sweat","k_index", "esp"])
		output_data = np.zeros((ta.shape[0], len(param)))
	
		#Set effective layer params, as the missing values (over land) need to be set to zeros later
		#(more generally, parameters which are masked over land, incl LFC)
		effective_layer_params = ["srhe", "srhe_left", "ebwd", "Uwindinf", "stp_cin", "dcp", \
						"Umeanwindinf", "scp", "dmgwind", "scld", \
						"eff_sherb", "moshe","ml_lfc","mu_lfc",\
						"eff_lfc","sb_lfc", "eff_cape", "eff_cin",\
						"eff_lcl", "eff_el", "stp_cin_left"]

		#Check there are no double-ups in the param string vector
		if len(param) != len(np.unique(param)):
			unique_params, name_cts = np.unique(param,return_counts=True)
			raise ValueError("THE FOLLOWING PARAMS HAVE BEEN ENTERED TWICE IN THE PARAM LIST %s"\
				 %(unique_params[name_cts>1],))

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
		if model == "erai":
			cape_split = np.array_split(cape, size, axis = 0)
			cp_split = np.array_split(cp, size, axis = 0)
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

		print("Time taken to load in data on processor 1: %s" %(dt.datetime.now() - start), )

	else:
		#Initialise variables on other cores (can be thought of as "remote"), including the name of the 
		# model (as each processer needs to know whether ERA-Interim specific parameters are being included
		model = sys.argv[1]
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
		if model == "erai":
			cape_split = None; cp_split = None; cape = None; cp = None

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
	if model == "erai":
		cp_split = comm.bcast(cp_split, root=0)
		cape_split = comm.bcast(cape_split, root=0)
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
	if model == "erai":
		cp_chunk = np.zeros(np.shape(cp_split[rank]))
		cape_chunk = np.zeros(np.shape(cape_split[rank]))
		comm.Scatterv([cp,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],cp_chunk,root=0)
		comm.Scatterv([cape,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],cape_chunk,root=0)

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
		#Get profile and parcels
		prof, mu_pcl ,ml_pcl, sb_pcl, eff_pcl= sharp_parcel_mpi(p, \
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
			terrain_chunk[i])

		#Extract varibales relevant for output
		#Levels (note that all are agl. However, profile heights are stored as msl, hence they need to be 
		# converted during calls to interp.pres)
		sfc = prof.pres[prof.sfc]
		p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
		p2km = interp.pres(prof, interp.to_msl(prof, 2000.))
		p3km = interp.pres(prof, interp.to_msl(prof, 3000.))
		p4km = interp.pres(prof, interp.to_msl(prof, 4000.))
		p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
		p10km = interp.pres(prof, interp.to_msl(prof, 10000.))
		melting_hgt = ml_pcl.hght0c
		pmelting_hgt = interp.pres(prof, interp.to_msl(prof, melting_hgt))
		pcld = interp.pres(prof, interp.to_msl(prof, 0.5 * mu_pcl.elhght))
		pmllcl = interp.pres(prof, interp.to_msl(prof, ml_pcl.lclhght))
		#Effective (inflow) layer
		ebotp, etopp = params.effective_inflow_layer(prof, mupcl=mu_pcl, ecape=100,\
				 ecinh=-250)
		ebot_hgt = interp.to_agl(prof, interp.hght(prof,ebotp))
		etop_hgt = interp.to_agl(prof, interp.hght(prof,etopp))
		#Winds. Note that mean winds through layers are pressure-weighted
		u01, v01 = interp.components(prof, p1km)
		u03, v03 = interp.components(prof, p3km)
		u06, v06 = interp.components(prof, p6km)
		u10, v10 = interp.components(prof, p10km)
		u500, v500 = interp.components(prof, 500)
		umllcl, vmllcl = interp.components(prof, pmllcl)
		ucld, vcld = interp.components(prof, pcld)
		s01 = np.array(np.sqrt(np.square(u01-uas_chunk[i])+np.square(v01-vas_chunk[i])))
		s03 = np.array(np.sqrt(np.square(u03-uas_chunk[i])+np.square(v03-vas_chunk[i])))
		s06 = np.array(np.sqrt(np.square(u06-uas_chunk[i])+np.square(v06-vas_chunk[i])))
		s010 = np.array(np.sqrt(np.square(u10-uas_chunk[i])+np.square(v10-vas_chunk[i])))
		s13 = np.array(np.sqrt(np.square(u03-u01)+np.square(v03-v01)))
		s36 = np.array(np.sqrt(np.square(u06-u03)+np.square(v06-v03)))
		scld = np.array(np.sqrt(np.square(ucld-umllcl)+np.square(vcld-vmllcl)))
		umean01 , vmean01 = winds.mean_wind(prof, pbot = sfc, ptop = p1km)
		umean03 , vmean03 = winds.mean_wind(prof, pbot = sfc, ptop = p3km)
		umean06 , vmean06 = winds.mean_wind(prof, pbot = sfc, ptop = p6km)
		umean800_600 , vmean800_600 = winds.mean_wind(prof, pbot = 800, ptop = 600)
		Umean01 = utils.mag(umean01, vmean01)
		Umean03 = utils.mag(umean03, vmean03)
		Umean06 = utils.mag(umean06, vmean06)
		Umean800_600 = utils.mag(umean800_600, vmean800_600)
		U500 = utils.mag(u500, v500)
		U1 = utils.mag(u01, v01)
		U3 = utils.mag(u03, v03)
		U6 = utils.mag(u06, v06)

		#Storm relative winds/effective inflow layer winds
		stu, stv, stu_left, stv_left = winds.non_parcel_bunkers_motion(prof)
		Ust = utils.mag(stu,stv)
		Ust_left = utils.mag(stu_left,stv_left)
		uwindinf, vwindinf = interp.components(prof, etopp)
		Uwindinf = utils.mag(uwindinf, vwindinf)
		umeanwindinf , vmeanwindinf = winds.mean_wind(prof, pbot = ebotp, ptop = etopp)
		Umeanwindinf = utils.mag(umeanwindinf, vmeanwindinf)
		usr01, vsr01 = winds.sr_wind(prof, pbot=sfc, ptop=p1km, stu=stu, stv=stv)
		usr03, vsr03 = winds.sr_wind(prof, pbot=sfc, ptop=p3km, stu=stu, stv=stv)
		usr06, vsr06 = winds.sr_wind(prof, pbot=sfc, ptop=p6km, stu=stu, stv=stv)
		usr01_left, vsr01_left = winds.sr_wind(prof, pbot=sfc, ptop=p1km, stu=stu_left, stv=stv_left)
		usr03_left, vsr03_left = winds.sr_wind(prof, pbot=sfc, ptop=p3km, stu=stu_left, stv=stv_left)
		usr06_left, vsr06_left = winds.sr_wind(prof, pbot=sfc, ptop=p6km, stu=stu_left, stv=stv_left)
		Usr01 = utils.mag(usr01, vsr01)
		Usr03 = utils.mag(usr03, vsr03)
		Usr06 = utils.mag(usr06, vsr06)
		Usr01_left = utils.mag(usr01_left, vsr01_left)
		Usr03_left = utils.mag(usr03_left, vsr03_left)
		Usr06_left = utils.mag(usr06_left, vsr06_left)
		#Helicity
		srhe = abs(winds.helicity(prof, ebot_hgt, etop_hgt, stu=stu, stv=stv)[0])
		srh01 = abs(winds.helicity(prof, 0, 1000, stu=stu, stv=stv)[0])
		srh03 = abs(winds.helicity(prof, 0, 3000, stu=stu, stv=stv)[0])
		srh06 = abs(winds.helicity(prof, 0, 6000, stu=stu, stv=stv)[0])
		srhe_left = abs(winds.helicity(prof, ebot_hgt, etop_hgt, stu=stu_left, stv=stv_left)[0])
		srh01_left = abs(winds.helicity(prof, 0, 1000, stu=stu_left, stv=stv_left)[0])
		srh03_left = abs(winds.helicity(prof, 0, 3000, stu=stu_left, stv=stv_left)[0])
		srh06_left = abs(winds.helicity(prof, 0, 6000, stu=stu_left, stv=stv_left)[0])
		#Effective bulk wind shear (diff)
		ebwd = winds.wind_shear(prof, pbot=ebotp, ptop=etopp) 
		prof.ebwd = ebwd
		ebwd = utils.mag(ebwd[0],ebwd[1])
		#Thermodynamic
		rhmin01 = prof.relh[(prof.pres <= sfc) & (prof.pres >= p1km)].min()
		rhmin03 = prof.relh[(prof.pres <= sfc) & (prof.pres >= p3km)].min()
		rhmin06 = prof.relh[(prof.pres <= sfc) & (prof.pres >= p6km)].min()
		rhmin13 = prof.relh[(prof.pres <= p1km) & (prof.pres >= p3km)].min()
		rhmin36 = prof.relh[(prof.pres <= p3km) & (prof.pres >= p6km)].min()
			#Subcloud layer is below the mixed layer parcel lcl
		rhminsubcloud = prof.relh[(prof.pres <= sfc) & (prof.pres >= pmllcl)].min()
		qmean01 = params.mean_mixratio(prof, pbot = sfc, ptop = p1km)
		qmean03 = params.mean_mixratio(prof, pbot = sfc, ptop = p3km)
		qmean06 = params.mean_mixratio(prof, pbot = sfc, ptop = p6km)
		qmean13 = params.mean_mixratio(prof, pbot = p1km, ptop = p3km)
		qmean36 = params.mean_mixratio(prof, pbot = p3km, ptop = p6km)
		qmeansubcloud = params.mean_mixratio(prof, pbot = sfc, ptop = pmllcl)
		q_melting = params.mean_mixratio(prof, pbot=pmelting_hgt, ptop=pmelting_hgt)
		q1 = params.mean_mixratio(prof, pbot=p1km, ptop=p1km)
		q3 = params.mean_mixratio(prof, pbot=p3km, ptop=p3km)
		q6 = params.mean_mixratio(prof, pbot=p6km, ptop=p6km)
		lr_freezing = params.lapse_rate(prof, lower = sfc, upper = pmelting_hgt)
		lr01 = params.lapse_rate(prof, lower = sfc, upper = p1km)
		lr03 = params.lapse_rate(prof, lower = sfc, upper = p3km)
		lr13 = params.lapse_rate(prof, lower = p1km, upper = p3km)
		lr24 = params.lapse_rate(prof, lower = p2km, upper = p4km)
		lr36 = params.lapse_rate(prof, lower = p3km, upper = p6km)
		lr850_670 = params.lapse_rate(prof, lower = 850, upper = 670)
		lr700_500 = params.lapse_rate(prof, lower = 700, upper = 500)
		maxtevv = maxtevv_fn(prof)
		pwat = params.precip_water(prof)
		v_totals = params.v_totals(prof)
		c_totals = params.c_totals(prof)
		t_totals = c_totals + v_totals
		dpd850 = interp.temp(prof,850) - interp.dwpt(prof,850)
		dpd670 = interp.temp(prof,670) - interp.dwpt(prof,670)
		dpd700 = interp.temp(prof,700) - interp.dwpt(prof,700)
		dpd500 = interp.temp(prof,500) - interp.dwpt(prof,500)
		te_diff = params.thetae_diff(prof)
		Rq = qmean01 / 12.
		if Rq > 1:
			Rq = 1
		dcape = params.dcape(prof)[0]
		if dcape < 0:
			dcape = 0
		mu_el = mu_pcl.elhght
		ml_el = ml_pcl.elhght
		eff_el = eff_pcl.elhght
		sb_el = sb_pcl.elhght
		if np.ma.is_masked(ml_el):
			ml_el = np.nanmax(prof.hght)
		if np.ma.is_masked(eff_el):
			eff_el = np.nanmax(prof.hght)
		if np.ma.is_masked(sb_el):
			sb_el = np.nanmax(prof.hght)
		if np.ma.is_masked(mu_el):
			mu_el = np.nanmax(prof.hght)
		#Composite
		stp_fixed = params.stp_fixed(sb_pcl.bplus, sb_pcl.lclhght, srh01, s06)
		stp_fixed_left = params.stp_fixed(sb_pcl.bplus, sb_pcl.lclhght, srh01_left, s06)
		mosh = ((lr03 - 4.)/4.) * ((utils.KTS2MS(s01) - 8)/10.) * ((maxtevv + 10.)/9.)
		moshe = ((lr03 - 4.)/4.) * ((utils.KTS2MS(s01) - 8)/10.) * \
				((utils.KTS2MS(ebwd) - 8)/10.) * ((maxtevv + 10.)/9.)
		if mosh < 0:
			mosh = 0
		if moshe < 0:
			moshe = 0
		windex = 5. * np.power((melting_hgt/1000.)*Rq*(np.power(lr_freezing,2)-30.+\
				qmean01-2.*q_melting),0.5) 
			#WINDEX UNDEFINED FOR HIGHLY STABLE CONDITIONS
		if np.isnan(windex):
			windex = 0
		mwpi_mu = (mu_pcl.bplus / 100.) + (lr850_670 + dpd850 - dpd670)
		if mwpi_mu < 0:
			mwpi_mu = 0
		mwpi_ml = (ml_pcl.bplus / 100.) + (lr850_670 + dpd850 - dpd670)
		if mwpi_ml < 0:
			mwpi_ml = 0
		dmi = lr700_500 + dpd700 - dpd500
		if dmi<0:
			dmi=0
		wmsi_mu = (mu_pcl.bplus * te_diff) / 1000.
		if wmsi_mu<0:
			wmsi_mu = 0
		wmsi_ml = (ml_pcl.bplus * te_diff) / 1000.
		if wmsi_ml<0:
			wmsi_ml = 0
		hmi = lr850_670 + dpd850 - dpd670
		if hmi < 0:
			hmi = 0
		k_index = params.k_index(prof)
		if k_index < 0:
			k_index = 0

		#Fill output
		try:
			#Thermodynamic
			output[i,np.where(param=="ml_cape")[0][0]] = ml_pcl.bplus
			output[i,np.where(param=="mu_cape")[0][0]] = mu_pcl.bplus
			output[i,np.where(param=="sb_cape")[0][0]] = sb_pcl.bplus
			output[i,np.where(param=="eff_cape")[0][0]] = eff_pcl.bplus
			output[i,np.where(param=="ml_cin")[0][0]] = abs(ml_pcl.bminus)
			output[i,np.where(param=="mu_cin")[0][0]] = abs(mu_pcl.bminus)
			output[i,np.where(param=="sb_cin")[0][0]] = abs(sb_pcl.bminus)
			output[i,np.where(param=="eff_cin")[0][0]] = abs(eff_pcl.bminus)
			output[i,np.where(param=="ml_lcl")[0][0]] = ml_pcl.lclhght
			output[i,np.where(param=="mu_lcl")[0][0]] = mu_pcl.lclhght
			output[i,np.where(param=="sb_lcl")[0][0]] = sb_pcl.lclhght
			output[i,np.where(param=="eff_lcl")[0][0]] = eff_pcl.lclhght
			output[i,np.where(param=="ml_lfc")[0][0]] = ml_pcl.lfchght
			output[i,np.where(param=="mu_lfc")[0][0]] = mu_pcl.lfchght
			output[i,np.where(param=="sb_lfc")[0][0]] = sb_pcl.lfchght
			output[i,np.where(param=="eff_lfc")[0][0]] = eff_pcl.lfchght
			output[i,np.where(param=="dcape")[0][0]] = dcape
			output[i,np.where(param=="lr01")[0][0]] = lr01
			output[i,np.where(param=="lr03")[0][0]] = lr03
			output[i,np.where(param=="lr13")[0][0]] = lr13
			output[i,np.where(param=="lr24")[0][0]] = lr24
			output[i,np.where(param=="lr36")[0][0]] = lr36
			output[i,np.where(param=="lr_freezing")[0][0]] = lr_freezing
			output[i,np.where(param=="mhgt")[0][0]] = melting_hgt
			output[i,np.where(param=="mu_el")[0][0]] = mu_el
			output[i,np.where(param=="ml_el")[0][0]] = ml_el
			output[i,np.where(param=="eff_el")[0][0]] = eff_el
			output[i,np.where(param=="sb_el")[0][0]] = sb_el
			output[i,np.where(param=="qmean01")[0][0]] = qmean01
			output[i,np.where(param=="qmean03")[0][0]] = qmean03
			output[i,np.where(param=="qmean06")[0][0]] = qmean06
			output[i,np.where(param=="qmean13")[0][0]] = qmean13
			output[i,np.where(param=="qmean36")[0][0]] = qmean36
			output[i,np.where(param=="qmeansubcloud")[0][0]] = qmeansubcloud
			output[i,np.where(param=="q_melting")[0][0]] = q_melting
			output[i,np.where(param=="q1")[0][0]] = q1
			output[i,np.where(param=="q3")[0][0]] = q3
			output[i,np.where(param=="q6")[0][0]] = q6
			output[i,np.where(param=="rhmin01")[0][0]] = rhmin01
			output[i,np.where(param=="rhmin03")[0][0]] = rhmin03
			output[i,np.where(param=="rhmin06")[0][0]] = rhmin06
			output[i,np.where(param=="rhmin13")[0][0]] = rhmin13
			output[i,np.where(param=="rhmin36")[0][0]] = rhmin36
			output[i,np.where(param=="rhminsubcloud")[0][0]] = rhminsubcloud
			output[i,np.where(param=="v_totals")[0][0]] = v_totals
			output[i,np.where(param=="c_totals")[0][0]] = c_totals
			output[i,np.where(param=="t_totals")[0][0]] = t_totals
			output[i,np.where(param=="pwat")[0][0]] = pwat
			output[i,np.where(param=="maxtevv")[0][0]] = maxtevv
			output[i,np.where(param=="te_diff")[0][0]] = te_diff
			output[i,np.where(param=="tei")[0][0]] = params.tei(prof)
			output[i,np.where(param=="dpd850")[0][0]] = dpd850
			output[i,np.where(param=="dpd700")[0][0]] = dpd700
			output[i,np.where(param=="pbl_top")[0][0]] = params.pbl_top(prof)
			#From model convection scheme (available for ERA-Interim only)
			if model == "erai":
				output[i,np.where(param=="cp")[0][0]] = cp_chunk[i]
				output[i,np.where(param=="cape")[0][0]] = cape_chunk[i]
				output[i,np.where(param=="cape*s06")[0][0]] = \
					cape_chunk[i] * np.power(utils.KTS2MS(s06), 1.67)
				output[i,np.where(param=="dcp2")[0][0]] = \
					(dcape/980.) * (cape_chunk[i]/2000.) * (utils.KTS2MS(s06) / 20.) * \
					(utils.KTS2MS(Umean06) / 16.)
			#Winds
			output[i,np.where(param=="srhe")[0][0]] = \
				srhe  
			output[i,np.where(param=="srh01")[0][0]] = \
				srh01  
			output[i,np.where(param=="srh03")[0][0]] = \
				srh03 
			output[i,np.where(param=="srh06")[0][0]] = \
				srh06 
			output[i,np.where(param=="srhe_left")[0][0]] = \
				srhe_left  
			output[i,np.where(param=="srh01_left")[0][0]] = \
				srh01_left  
			output[i,np.where(param=="srh03_left")[0][0]] = \
				srh03_left 
			output[i,np.where(param=="srh06_left")[0][0]] = \
				srh06_left 
			output[i,np.where(param=="ebwd")[0][0]] = \
				utils.KTS2MS(ebwd)
			output[i,np.where(param=="s01")[0][0]] = \
				utils.KTS2MS(s01)
			output[i,np.where(param=="s03")[0][0]] = \
				utils.KTS2MS(s03)
			output[i,np.where(param=="s06")[0][0]] = \
				utils.KTS2MS(s06)
			output[i,np.where(param=="s010")[0][0]] = \
				utils.KTS2MS(s010)
			output[i,np.where(param=="s13")[0][0]] = \
				utils.KTS2MS(s13)
			output[i,np.where(param=="s36")[0][0]] = \
				utils.KTS2MS(s36)
			output[i,np.where(param=="scld")[0][0]] = \
				utils.KTS2MS(scld)
			output[i,np.where(param=="Umean01")[0][0]] = \
				utils.KTS2MS(Umean01)  
			output[i,np.where(param=="Umean03")[0][0]] = \
				utils.KTS2MS(Umean03)  
			output[i,np.where(param=="Umean06")[0][0]] = \
				utils.KTS2MS(Umean06)  
			output[i,np.where(param=="Umean800_600")[0][0]] = \
				utils.KTS2MS(Umean800_600)  
			output[i,np.where(param=="U500")[0][0]] = \
				utils.KTS2MS(U500)  
			output[i,np.where(param=="U1")[0][0]] = \
				utils.KTS2MS(U1)  
			output[i,np.where(param=="U3")[0][0]] = \
				utils.KTS2MS(U3)  
			output[i,np.where(param=="U6")[0][0]] = \
				utils.KTS2MS(U6)  
			output[i,np.where(param=="Ust_left")[0][0]] = \
				utils.KTS2MS(Ust_left)  
			output[i,np.where(param=="Ust")[0][0]] = \
				utils.KTS2MS(Ust)  
			output[i,np.where(param=="Usr01_left")[0][0]] = \
				utils.KTS2MS(Usr01_left)  
			output[i,np.where(param=="Usr03_left")[0][0]] = \
				utils.KTS2MS(Usr03_left)  
			output[i,np.where(param=="Usr06_left")[0][0]] = \
				utils.KTS2MS(Usr06_left)  
			output[i,np.where(param=="Usr01")[0][0]] = \
				utils.KTS2MS(Usr01)  
			output[i,np.where(param=="Usr03")[0][0]] = \
				utils.KTS2MS(Usr03)  
			output[i,np.where(param=="Usr06")[0][0]] = \
				utils.KTS2MS(Usr06)  
			output[i,np.where(param=="U10")[0][0]] = \
				utils.KTS2MS(utils.mag(uas_chunk[i],vas_chunk[i]))
			output[i,np.where(param=="Uwindinf")[0][0]] = \
				utils.KTS2MS(Uwindinf)
			output[i,np.where(param=="Umeanwindinf")[0][0]] = \
				utils.KTS2MS(Umeanwindinf)
			output[i,np.where(param=="wg10")[0][0]] = wg10_chunk[i]
			output[i,np.where(param=="omega01")[0][0]] = \
				params.mean_omega(prof, pbot=sfc, ptop=p1km)
			output[i,np.where(param=="omega03")[0][0]] = \
				params.mean_omega(prof, pbot=sfc, ptop=p3km)
			output[i,np.where(param=="omega06")[0][0]] = \
				params.mean_omega(prof, pbot=sfc, ptop=p6km)
			#Wind Composite parameters
			output[i,np.where(param=="dmgwind")[0][0]] = \
				(dcape/800.)* (utils.KTS2MS(Uwindinf) / 8.)
			output[i,np.where(param=="convgust")[0][0]] = \
				utils.KTS2MS(Umean800_600) + 2 * np.sqrt(2 * dcape)
			output[i,np.where(param=="ducs6")[0][0]] = \
				(utils.KTS2MS(Umean800_600) + 2 * np.sqrt(2 * dcape)) / 30. \
				* ((ml_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)) / 20000.)
			output[i,np.where(param=="dcp")[0][0]] = \
				(dcape/980.) * (ml_pcl.bplus/2000.) * (utils.KTS2MS(s06) / 20.) * \
				(utils.KTS2MS(Umean06) / 16.)
			output[i,np.where(param=="windex")[0][0]] = \
				windex
			output[i,np.where(param=="gustex")[0][0]] = \
				(0.5 * windex) + (0.5 * utils.KTS2MS(U500))
			output[i,np.where(param=="gustex2")[0][0]] = \
				(0.5 * windex) + (0.5 * utils.KTS2MS(Umean06))
			output[i,np.where(param=="gustex3")[0][0]] = \
				(((0.5 * windex) + (0.5 * utils.KTS2MS(Umean06))) / 30.) * \
				((ml_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)) / 20000.)
			output[i,np.where(param=="hmi")[0][0]] = \
				hmi
			output[i,np.where(param=="wmsi_mu")[0][0]] = \
				wmsi_mu
			output[i,np.where(param=="wmsi_ml")[0][0]] = \
				wmsi_ml
			output[i,np.where(param=="dmi")[0][0]] = \
				dmi
			output[i,np.where(param=="mwpi_mu")[0][0]] = \
				mwpi_mu
			output[i,np.where(param=="mwpi_ml")[0][0]] = \
				mwpi_ml
			#Other composite parameters
			output[i,np.where(param=="stp_cin_left")[0][0]] = params.stp_cin( \
				ml_pcl.bplus, srhe_left, utils.KTS2MS(ebwd), ml_pcl.lclhght, ml_pcl.bminus)
			output[i,np.where(param=="stp_cin")[0][0]] = params.stp_cin( \
				ml_pcl.bplus, srhe, utils.KTS2MS(ebwd), ml_pcl.lclhght, ml_pcl.bminus)
			output[i,np.where(param=="stp_fixed_left")[0][0]] = \
				stp_fixed_left
			output[i,np.where(param=="stp_fixed")[0][0]] = \
				stp_fixed
			output[i,np.where(param=="scp")[0][0]] = params.scp( \
				mu_pcl.bplus, srhe, utils.KTS2MS(ebwd))
			output[i,np.where(param=="ship")[0][0]] = params.ship( \
				prof, mupcl=mu_pcl)
			output[i,np.where(param=="k_index")[0][0]] = \
				k_index
			output[i,np.where(param=="mlcape*s06")[0][0]] = \
				ml_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)
			output[i,np.where(param=="mucape*s06")[0][0]] = \
				mu_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)
			output[i,np.where(param=="sbcape*s06")[0][0]] = \
				sb_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)
			output[i,np.where(param=="effcape*s06")[0][0]] = \
				eff_pcl.bplus * np.power(utils.KTS2MS(s06), 1.67)
			output[i,np.where(param=="mlcape*s06_2")[0][0]] = \
				ml_pcl.bplus * utils.KTS2MS(s06)
			output[i,np.where(param=="mucape*s06_2")[0][0]] = \
				mu_pcl.bplus * utils.KTS2MS(s06)
			output[i,np.where(param=="sbcape*s06_2")[0][0]] = \
				sb_pcl.bplus * utils.KTS2MS(s06)
			output[i,np.where(param=="effcape*s06_2")[0][0]] = \
				eff_pcl.bplus * utils.KTS2MS(s06)
			output[i,np.where(param=="eff_sherb")[0][0]] = \
				params.sherb(prof, effective=True, ebottom=ebotp, etop=etopp, mupcl=mu_pcl)
			output[i,np.where(param=="sherb")[0][0]] = \
				params.sherb(prof, effective=False)
			output[i,np.where(param=="moshe")[0][0]] = \
				moshe
			output[i,np.where(param=="mosh")[0][0]] = \
				mosh
			output[i,np.where(param=="wndg")[0][0]] = \
				params.wndg(prof, mlpcl=ml_pcl)
			output[i,np.where(param=="mburst")[0][0]] = \
				params.mburst(prof, sb_pcl, lr03, dcape, v_totals, pwat)
			output[i,np.where(param=="sweat")[0][0]] = \
				params.sweat(prof)
			prof.lapserate_3km = lr03
			output[i,np.where(param=="esp")[0][0]] = \
				params.esp(prof, mlpcl = ml_pcl)
				
		except:
			raise ValueError("\nMAKE SURE THAT OUTPUT PARAMETERS MATCH PARAMETER LIST\n")

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
			if param_name in effective_layer_params:
				temp_data[np.isnan(temp_data)] = 0.
			#Reshape back into a 3d array (time x lon x lat) on full grid (land and ocean points)
			output_reshaped = np.zeros((orig_length))
			output_reshaped[:] = np.nan
			output_reshaped[lsm_orig==1] =  temp_data
			output_reshaped = output_reshaped.reshape((orig_shape[0],orig_shape[2],orig_shape[3]))
			param_out.append(output_reshaped)

		if issave == "True":
			save_netcdf(region, model, out_name, date_list, lat, lon, param, param_out)

