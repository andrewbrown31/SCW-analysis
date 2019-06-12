#Parallel sharppy using mpi

import matplotlib.pyplot as plt
from erai_read import read_erai
from erai_read import get_mask as get_erai_mask
from barra_read import read_barra
from barra_read import get_mask as get_barra_mask
import datetime as dt
from mpi4py import MPI
import numpy as np
import sharppy.sharptab.profile as profile
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import warnings
import sys

#---------------------------------------------------------------------------------------------------
# mpiexec python sharp_parallel_mpi.py model region t1 t2
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
#Before running
# > source activate py3
# > module load mpi4py/3.0.0-py3
# > module unload python3/3.6.2
#---------------------------------------------------------------------------------------------------


def sharp_parcel_mpi(p,ua,va,hgt,ta,dp):

	#Exact same as sharp parcel, but intends to use the "pp" module (parallel python)
	
				#convert u and v to kts for use in profile
				ua_p_kts = utils.MS2KTS(ua)
				va_p_kts = utils.MS2KTS(va)

				#create profile
				prof = profile.create_profile(pres=p, hght=hgt, \
						tmpc=ta, \
						dwpc=dp, \
						 u=ua_p_kts, v=va_p_kts,\
						 missing=np.nan,\
						strictqc=False)

				#create most unstable parcel
				mu_parcel = params.parcelx(prof, flag=3, dp=-10) #3 = mu
				ml_parcel = params.parcelx(prof, flag=4, dp=-10) #4 = ml
				return (mu_parcel.bplus,ml_parcel.bplus)

if __name__ == "__main__":

	#Ignore warnings from SHARPpy
	warnings.simplefilter("ignore")

	#Get MPI communicator info
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	#Load data into first processer
	if rank == 0:
		#Parse arguments
		model = sys.argv[1]
		region = sys.argv[2]
		t1 = sys.argv[3]
		t2 = sys.argv[4]
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

		#Load data and setup "base" array, which has been reformed into a 2d array (vertical levels x 
		#	(horizontal spatial points x times) )
		if model == "erai":
			ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
				read_erai(domain,time)
			lsm = np.repeat(get_erai_mask(lon,lat)[np.newaxis],ta.shape[0],0)
		elif model == "barra":
			ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,lon,lat,date_list = \
				read_barra(domain,time)
			lsm = np.repeat(get_barra_mask(lon,lat)[np.newaxis],ta.shape[0],0)
		else:
			raise ValueError("INVALID MODEL NAME\n")
		
		ta_base = np.moveaxis(ta,[0,1,2,3],[0,3,1,2]).\
			reshape((ta.shape[0]*ta.shape[2]*ta.shape[3],ta.shape[1])).astype("double",order="C")
		dp_base = np.moveaxis(dp,[0,1,2,3],[0,3,1,2]).\
			reshape((dp.shape[0]*dp.shape[2]*dp.shape[3],dp.shape[1])).astype("double",order="C")
		hur_base = np.moveaxis(hur,[0,1,2,3],[0,3,1,2]).\
			reshape((hur.shape[0]*hur.shape[2]*hur.shape[3],hur.shape[1])).astype("double",order="C")
		hgt_base = np.moveaxis(hgt,[0,1,2,3],[0,3,1,2]).\
			reshape((hgt.shape[0]*hgt.shape[2]*hgt.shape[3],hgt.shape[1])).astype("double",order="C")
		ua_base = np.moveaxis(ua,[0,1,2,3],[0,3,1,2]).\
			reshape((ua.shape[0]*ua.shape[2]*ua.shape[3],ua.shape[1])).astype("double",order="C")
		va_base = np.moveaxis(va,[0,1,2,3],[0,3,1,2]).\
			reshape((va.shape[0]*va.shape[2]*va.shape[3],va.shape[1])).astype("double",order="C")
		lsm_base = np.array(lsm.reshape((lsm.shape[0]*lsm.shape[1]*lsm.shape[2]))).astype("double",order="C")
		
		#Set the ouput array (in this case, a vector of length (horizontal spatial points x times) )
		output_data = np.zeros(ta_base.shape[0])

		#Split up the base array for parallel processing (by horizonal spatial points x times)
		ta_split = np.array_split(ta_base, size, axis = 0)
		dp_split = np.array_split(dp_base, size, axis = 0)
		hur_split = np.array_split(hgt_base, size, axis = 0)
		hgt_split = np.array_split(hgt_base, size, axis = 0)
		ua_split = np.array_split(ua_base, size, axis = 0)
		va_split = np.array_split(va_base, size, axis = 0)
		lsm_split = np.array_split(lsm_base, size, axis = 0)
		split_sizes = []
		for i in range(0,len(ta_split),1):
			split_sizes = np.append(split_sizes, ta_split[i].shape[0])

		#Remember the points at which splits occur on (noting that Gatherv and Scatterv expect a 
		# flattened array)
		split_sizes_input = split_sizes*ta_base.shape[1]
		displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
		split_sizes_output = split_sizes*1
		displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
		split_sizes_input_2d = split_sizes
		displacements_input_2d = np.insert(np.cumsum(split_sizes_input_2d),0,0)[0:-1]

	else:
		#Create variables on other cores
		split_sizes_input = None; displacements_input = None; split_sizes_output = None;\
			displacements_output = None; split_sizes_input_2d = None; displacements_input_2d = None
		ta_split = None; dp_split = None; hur_split = None; hgt_split = None; ua_split = None;\
			va_split = None; lsm_split = None
		ta_base = None; dp_base = None; hur_base = None; hgt_base = None; ua_base = None;\
			va_base = None; lsm_base = None
		p = None
		output_data = None

	#Broadcast split array to other cores
	ta_split = comm.bcast(ta_split, root=0)
	dp_split = comm.bcast(dp_split, root=0)
	hur_split = comm.bcast(hur_split, root=0)
	hgt_split = comm.bcast(hgt_split, root=0)
	ua_split = comm.bcast(ua_split, root=0)
	va_split = comm.bcast(va_split, root=0)
	lsm_split = comm.bcast(lsm_split, root=0)
	p = comm.bcast(p, root=0)
	split_sizes_input = comm.bcast(split_sizes_input, root = 0)
	displacements_input = comm.bcast(displacements_input, root = 0)
	split_sizes_output = comm.bcast(split_sizes_output, root = 0)
	displacements_output = comm.bcast(displacements_output, root = 0)

	#Create array to receive subset of data on each core, where rank specifies the core
	ta_chunk = np.zeros(np.shape(ta_split[rank]))
	dp_chunk = np.zeros(np.shape(dp_split[rank]))
	hur_chunk = np.zeros(np.shape(hur_split[rank]))
	hgt_chunk = np.zeros(np.shape(hgt_split[rank]))
	ua_chunk = np.zeros(np.shape(ua_split[rank]))
	va_chunk = np.zeros(np.shape(va_split[rank]))
	lsm_chunk = np.zeros(np.shape(lsm_split[rank]))
	comm.Scatterv([ta_base,split_sizes_input, displacements_input, MPI.DOUBLE],ta_chunk,root=0)
	comm.Scatterv([dp_base,split_sizes_input, displacements_input, MPI.DOUBLE],dp_chunk,root=0)
	comm.Scatterv([hur_base,split_sizes_input, displacements_input, MPI.DOUBLE],hur_chunk,root=0)
	comm.Scatterv([hgt_base,split_sizes_input, displacements_input, MPI.DOUBLE],hgt_chunk,root=0)
	comm.Scatterv([ua_base,split_sizes_input, displacements_input, MPI.DOUBLE],ua_chunk,root=0)
	comm.Scatterv([va_base,split_sizes_input, displacements_input, MPI.DOUBLE],va_chunk,root=0)
	comm.Scatterv([lsm_base,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],lsm_chunk,root=0)

	comm.Barrier()

	#Get CAPE
	start = dt.datetime.now()
	output = np.zeros(ta_chunk.shape[0])
	for i in np.arange(0,ta_chunk.shape[0]):
		if lsm_chunk[i] == 1:
			mu_cape, ml_cape = sharp_parcel_mpi(p, \
				ua_chunk[i],\
				va_chunk[i],\
				hgt_chunk[i],\
				ta_chunk[i],\
				dp_chunk[i])
		else:
			mu_cape = np.nan
			ml_cape = np.nan
		output[i] = ml_cape

	if rank == 0:
		print("CHUNKSIZE %s" %(ua_chunk.shape,))
		print("Time taken for SHARPPy on processor 1: %s" %(dt.datetime.now() - start), )

	#Gather output data together
	comm.Gatherv(output, [output_data, split_sizes_output, displacements_output, MPI.DOUBLE], root=0)


	#Do some plotting
	if rank == 0:
		output_reshaped = output_data.reshape((ta.shape[0],ta.shape[2],ta.shape[3]))

		plt.subplot(211)
		x,y = np.meshgrid(lon,lat)
		plt.contourf(x,y,output_reshaped[0]);plt.colorbar()
		try:
			plt.subplot(212)
			plt.contourf(x,y,output_reshaped[1]);plt.colorbar()
		except:
			pass
		plt.savefig("/home/548/ab4502/working/new.png");plt.close()
		plt.subplot(211)
		plt.contourf(x,y,np.max(ta[0],axis=(0)));plt.colorbar()
		try:
			plt.subplot(212)
			plt.contourf(x,y,np.max(ta[1],axis=(0)));plt.colorbar()
		except:
			pass
		plt.savefig("/home/548/ab4502/working/orig.png");plt.close()

