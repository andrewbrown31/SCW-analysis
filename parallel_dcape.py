import sys
import pandas as pd
import datetime as dt
from mpi4py import MPI
import numpy as np
from SkewT import get_dcape
from erai_read import read_erai
from erai_read import get_mask as get_erai_mask
from barra_read import read_barra
from barra_read import get_mask as get_barra_mask
from calc_param import save_netcdf

#Using SkewT, calculate DCAPE in parallel

if __name__ == "__main__":

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
			time = [dt.datetime.strptime(t1,"%Y%m%d%H"),\
				dt.datetime.strptime(t2,"%Y%m%d%H")]
		except:
			raise ValueError("INVALID START OR END TIME. SHOULD BE YYYYMMDDHH\n")
		if not ((issave=="True") | (issave=="False")):
			raise ValueError("\n INVALID ISSAVE...SHOULD BE True OR False")

		start = dt.datetime.now()
		if model == "erai":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,cp,\
			wg10,cape,lon,lat,date_list = \
				read_erai(domain,time)
			terrain = np.repeat(terrain[np.newaxis],ta.shape[0],0)
		elif model == "barra":
			ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,\
			lon,lat,date_list = \
				read_barra(domain,time)
			terrain = np.repeat(terrain[np.newaxis],ta.shape[0],0)
		else:
			raise ValueError("INVALID MODEL NAME\n")

		#Make a 4d pressure array from the 1d coordinate vector
		p_3d = np.moveaxis(np.tile(p,[ta.shape[0],ta.shape[2],ta.shape[3],1]),[0,1,2,3],[0,2,3,1])

		#TAKE ONLY UNDER 500 hPA
		p500 = p >= 500
		p_3d = p_3d[:,p500,:,:]
		ta = ta[:,p500,:,:]
		hgt = hgt[:,p500,:,:]
		p = p[p500]

		#Insert sfc values
		ta = np.insert(ta, 0, tas, axis=1) 
		p_3d = np.insert(p_3d, 0, ps, axis=1) 
		hgt = np.insert(hgt, 0, terrain, axis=1) 
		p = np.array(p)

		#Sort by ascending p
		temp0,a,temp1,temp2 = np.meshgrid(np.arange(p_3d.shape[0]) , np.arange(p_3d.shape[1]),\
			 np.arange(p_3d.shape[2]), np.arange(p_3d.shape[3]))
		sort_inds = np.flip(np.lexsort([np.swapaxes(a,1,0),p_3d],axis=1), axis=1)
		hgt = np.take_along_axis(hgt, sort_inds, axis=1)
		p_3d = np.take_along_axis(p_3d, sort_inds, axis=1)
		ta = np.take_along_axis(ta, sort_inds, axis=1)


		orig_shape = ta.shape
		#SPLIT ON SPATIAL-TEMPORAL DIM
		#ta = np.moveaxis(ta,[0,1,2,3],[0,3,1,2]).\
		#	reshape((ta.shape[0]*ta.shape[2]*ta.shape[3],ta.shape[1])).astype("double",order="C")
		#p_3d = np.moveaxis(p_3d,[0,1,2,3],[0,3,1,2]).\
		#	reshape((p_3d.shape[0]*p_3d.shape[2]*p_3d.shape[3],p_3d.shape[1])).astype("double",order="C")
		#hgt = np.moveaxis(hgt,[0,1,2,3],[0,3,1,2]).\
		#	reshape((hgt.shape[0]*hgt.shape[2]*hgt.shape[3],hgt.shape[1])).astype("double",order="C")
		#SPLIT ON TEMPORAL DIM
		ta = ta.\
			reshape((ta.shape[0],ta.shape[1]*ta.shape[2]*ta.shape[3])).astype("double",order="C")
		p_3d = p_3d.\
			reshape((p_3d.shape[0],p_3d.shape[1]*p_3d.shape[2]*p_3d.shape[3])).astype("double",order="C")
		hgt = hgt.\
			reshape((hgt.shape[0],hgt.shape[1]*hgt.shape[2]*hgt.shape[3])).astype("double",order="C")
		ps = ps.\
			reshape((ps.shape[0],ps.shape[1]*ps.shape[2])).astype("double",order="C")

		orig_length = ta.shape[0]

		output_data_dcape = np.zeros((ta.shape[0], ps.shape[1]))
		output_data_ddraft_temp = np.zeros((ta.shape[0], ps.shape[1]))
	
		#Split/chunk the base arrays on the spatial-temporal grid point dimension, for parallel processing
		ta_split = np.array_split(ta, size, axis = 0)
		p_3d_split = np.array_split(p_3d, size, axis = 0)
		hgt_split = np.array_split(hgt, size, axis = 0)
		ps_split = np.array_split(ps, size, axis = 0)
		split_sizes = []
		for i in range(0,len(ta_split),1):
			split_sizes = np.append(split_sizes, ta_split[i].shape[0])

		#Remember the points at which splits occur on (noting that Gatherv and Scatterv act on a 
		# "C" style (row-major) flattened array). This will be different for pressure-level and sfc-level
		# variables
		split_sizes_input = split_sizes*ta.shape[1]
		displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
		split_sizes_input_2d = split_sizes*ps.shape[1]
		displacements_input_2d = np.insert(np.cumsum(split_sizes_input_2d),0,0)[0:-1]
		split_sizes_output = split_sizes*ps.shape[1]
		displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]

		print("Time taken to load in data on processor 1: %s" %(dt.datetime.now() - start), )

	else:
		#Initialise variables on other cores (can be thought of as "remote"), including the name of the 
		# model (as each processer needs to know whether ERA-Interim specific parameters are being included
		model = sys.argv[1]
		split_sizes_input = None; displacements_input = None; split_sizes_output = None;\
			displacements_output = None; split_sizes_input_2d = None; \
			displacements_input_2d = None
		ta_split = None; p_3d_split = None; hgt_split = None
		ta = None; p_3d = None; hgt = None; p = None; ps=None
		output_data_dcape = None; output_data_ddraft_temp = None
		ps_split = None
		orig_shape = None

	#Broadcast split arrays to other cores
	ta_split = comm.bcast(ta_split, root=0)
	p_3d_split = comm.bcast(p_3d_split, root=0)
	hgt_split = comm.bcast(hgt_split, root=0)
	ps_split = comm.bcast(ps_split, root=0)
	p = comm.bcast(p, root=0)
	split_sizes_input = comm.bcast(split_sizes_input, root = 0)
	displacements_input = comm.bcast(displacements_input, root = 0)
	split_sizes_input_2d = comm.bcast(split_sizes_input_2d, root = 0)
	displacements_input_2d = comm.bcast(displacements_input_2d, root = 0)
	split_sizes_output = comm.bcast(split_sizes_output, root = 0)
	displacements_output = comm.bcast(displacements_output, root = 0)
	orig_shape = comm.bcast(orig_shape, root = 0)

	#Create arrays to receive chunked/split data on each core, where rank specifies the core
	ta_chunk = np.zeros(np.shape(ta_split[rank]))
	p_3d_chunk = np.zeros(np.shape(p_3d_split[rank]))
	hgt_chunk = np.zeros(np.shape(hgt_split[rank]))
	ps_chunk = np.zeros(np.shape(ps_split[rank]))
	comm.Scatterv([ta,split_sizes_input, displacements_input, MPI.DOUBLE],ta_chunk,root=0)
	comm.Scatterv([p_3d,split_sizes_input, displacements_input, MPI.DOUBLE],p_3d_chunk,root=0)
	comm.Scatterv([hgt,split_sizes_input, displacements_input, MPI.DOUBLE],hgt_chunk,root=0)
	comm.Scatterv([ps,split_sizes_input_2d, displacements_input_2d, MPI.DOUBLE],ps_chunk,root=0)

	comm.Barrier()

	#Print diagnostics
	if rank == 0:
		print("TOTAL TIMES: %s" %(ta.shape[0]))
		print("TIME CHUNKSIZE: %s" %(ta_chunk.shape,))

#----------------------------------------------------------------------------------------------------------------
#
#		CALCULATE DCAPE
#
#---------------------------------------------------------------------------------------------------------------

	output_dcape = np.zeros((ta_chunk.shape[0], orig_shape[2]*orig_shape[3]))
	output_ddraft_temp = np.zeros((ta_chunk.shape[0], orig_shape[2]*orig_shape[3]))
	for i in np.arange(0,ta_chunk.shape[0]):
		ta_3d = ta_chunk[i].reshape((orig_shape[1:]))
		p_3d = p_3d_chunk[i].reshape((orig_shape[1:]))
		hgt_3d = hgt_chunk[i].reshape((orig_shape[1:]))
		ps = ps_chunk[i].reshape((orig_shape[2], orig_shape[3]))

		#NOTE THAT A LAT LON LOOP IS USED INSIDE OF GET_DCAPE. IF TAKING TOO LONG FOR BARRA, THEN
		# CONSIDER REMOVING
		dcape, ddraft_temp = get_dcape(p_3d, ta_3d, hgt_3d, p, ps)

		ddraft_temp[ddraft_temp<0] = 0

		output_dcape[i] = dcape.reshape((orig_shape[2]*orig_shape[3])).astype("double")
		output_ddraft_temp[i] = ddraft_temp.\
			reshape((orig_shape[2]*orig_shape[3])).astype("double")


#----------------------------------------------------------------------------------------------------------------

	#Print diagnostics
	if rank == 0:
		print("Time taken for SHARPPy on processor 1: %s" %(dt.datetime.now() - start), )
		print("Time taken for each element on processor 1: %s" \
			%((dt.datetime.now() - start)/float(ta_chunk.shape[0])), )

	#Gather output data together to root node
	comm.Gatherv(output_dcape, \
		[output_data_dcape, split_sizes_output, displacements_output, MPI.DOUBLE], \
		root=0)
	comm.Gatherv(output_ddraft_temp, \
		[output_data_ddraft_temp, split_sizes_output, displacements_output, MPI.DOUBLE], \
		root=0)

	if rank == 0:
		output_reshaped = output_data_dcape.reshape((orig_shape[0],\
			orig_shape[2],orig_shape[3]))
		#output_dcape = np.max(output_reshaped,axis=1)
		ddraft_temp_reshaped = output_data_ddraft_temp.reshape((orig_shape[0],\
			orig_shape[2],orig_shape[3]))

#----------------------------------------------------------------------------------------------------------------
		#CAN PUT EXTRA DCAPE CALCULATIONS HERE. E.G. CALCULATING THE DCP. WILL HAVE TO LOAD DATA FROM
		# THE CONVECTIVE PARAMETER NETCDF

		#dmgwind = dcape/800. * Uwindinf / 8.
		#Mburst composite parameter
		#convgust
		#ducs6
		#dcp
		#dcp2


#----------------------------------------------------------------------------------------------------------------

		save_netcdf(region, model, out_name, date_list, lat, lon,\
		["dcape", "ddraft_temp"], \
		[output_dcape, ddraft_temp_reshaped],\
		append=True)

		
