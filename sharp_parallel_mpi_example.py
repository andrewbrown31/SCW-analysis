#Example script to load a time slice of ERA-Interim data, create atmospheric soundings at each
# grid point using SHARPpy, and plot CAPE

#Code adapted from https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array/36082684#36082684

#Sounding calculations are chunked in the spatial dimension and made parallel using mpi4py

#Note that this is a simple CAPE calculation which does not do necessary quality checks on data

#Install sharpy with: conda install -c sharppy sharppy
# (I would clone my conda environment before doing this, as sharppy can conflict with matplotlib)
#Before running, load the mpi4py module with:  > module load mpi4py 

#To run:
# > mpiexec python -m mpi4py sharp_parallel_mpi_example.py
#Or interactively
# > mpiexec python -i sharp_parallel_mpi_example.py

import sharppy.sharptab.profile as profile
import sharppy.sharptab.params as params
import sharppy.sharptab.utils as utils
import netCDF4 as nc
import datetime as dt
from mpi4py import MPI 		#module load mpi4py
import numpy as np
import warnings
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#Loading matplotlib and Basemap while using MPI is not reccomended and throws a bunch of warnings - 
# however for this example it is okay

def sharp_parcel_mpi(p,ua,va,hgt,ta,dp):

	#Use SHARPpy to create a profile object and then lift a parcel

	prof = profile.create_profile(pres = p, \
		hght = hgt, \
		tmpc = ta, \
		dwpc = dp, \
		u = ua, \
		v = va, \
		strictqc=False)
		
	sb_parcel = params.parcelx(prof, flag=1, dp=-10)

	return sb_parcel

def get_dp(ta,hur):
	#Calculate dew point temperature
	a = 17.27
	b = 237.7
	alpha = ((a * ta) / (b + ta)) + np.log(hur/100.0)
	dp = (b*alpha) / (a - alpha)
	dp[np.isnan(dp)] = -85.
	return dp


if __name__ == "__main__":

	#Ignore warnings from SHARPpy
	warnings.simplefilter("ignore")

	#First, get MPI (message passing interface) communicator info. 
	#This allows different nodes/processers to communicate with eachother.
	comm = MPI.COMM_WORLD		#Communicator object
	size = comm.Get_size()		#Number of total processers (ncpus)
	rank = comm.Get_rank()		#Int: the current processer (when submitted in parallel)



	#Load data into first processer, rank=0 (can be thought of as a "local", processer)
	if rank == 0:

		#Load ERA-Interim data
		path_pl = "/g/data/ub4/erai/netcdf/6hr/atmos/oper_an_pl/v01/"
		ta_file = nc.Dataset(path_pl+"ta/ta_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
		z_file = nc.Dataset(path_pl+"z/z_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
		ua_file = nc.Dataset(path_pl+"ua/ua_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
		va_file = nc.Dataset(path_pl+"va/va_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
		hur_file = nc.Dataset(path_pl+"hur/hur_6hrs_ERAI_historical_an-pl_20160901_20160930.nc")
		p = ta_file.variables["lev"][:] / 100.
		#(Restrict spatial domain to Australia)
		start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
		domain = [start_lat,end_lat,start_lon,end_lon]
		lon = ta_file.variables["lon"][:]
		lat = ta_file.variables["lat"][:]
		lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
		lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
		ta = np.squeeze(ta_file.variables["ta"][109, p>= 100, lat_ind, lon_ind]) - 273.15
		hgt = np.squeeze(z_file.variables["z"][109, p>= 100, lat_ind, lon_ind]) / 9.8
		ua = np.squeeze(ua_file.variables["ua"][109, p>= 100, lat_ind, lon_ind])
		va = np.squeeze(va_file.variables["va"][100, p>= 100, lat_ind, lon_ind])
		hur = np.squeeze(hur_file.variables["hur"][100, p>= 100, lat_ind, lon_ind])
		hur[hur<0] = 0
		hur[hur>100] = 100
		dp = get_dp(ta, hur)
		p = p[p>=100]
		lon = lon[lon_ind]
		lat = lat[lat_ind]

		#Reshape 3-dimensional data (pressure x lat x lon) into a 
		# 2d array [ (lat x lon) x pressure ] so the data is able to be passed from 
		#this processer to other processers. Note that mpi expects a C-style row-major
		#array 
		orig_shape = ta.shape
		ta = np.moveaxis(ta,0,2).\
			reshape((ta.shape[1]*ta.shape[2],ta.shape[0])).\
			astype("double",order="C")
		ua = np.moveaxis(ua,0,2).\
			reshape((ua.shape[1]*ua.shape[2],ua.shape[0])).\
			astype("double",order="C")
		va = np.moveaxis(va,0,2).\
			reshape((va.shape[1]*va.shape[2],va.shape[0])).\
			astype("double",order="C")
		hgt = np.moveaxis(hgt,0,2).\
			reshape((hgt.shape[1]*hgt.shape[2],hgt.shape[0])).\
			astype("double",order="C")
		hur = np.moveaxis(hur,0,2).\
			reshape((hur.shape[1]*hur.shape[2],hur.shape[0])).\
			astype("double",order="C")
		dp = np.moveaxis(dp,0,2).\
			reshape((dp.shape[1]*dp.shape[2],dp.shape[0])).\
			astype("double",order="C")

		#Intialise the final output array, for each processer to dump its output into
		output_data = np.zeros(ta.shape[0])

		#Split/chunk the base arrays on the spatial dimension, for parallel processing. 
		#Keep track of the size of each split/chunk for later
		ta_split = np.array_split(ta, size, axis = 0)
		dp_split = np.array_split(dp, size, axis = 0)
		hur_split = np.array_split(hur, size, axis = 0)
		hgt_split = np.array_split(hgt, size, axis = 0)
		ua_split = np.array_split(ua, size, axis = 0)
		va_split = np.array_split(va, size, axis = 0)
		split_sizes = []
		for i in range(0,len(ta_split),1):
			split_sizes = np.append(split_sizes, ta_split[i].shape[0])

		#Remember the points in the base array at which splits occur on, for when we send
		# and recieve data from other processers
		split_sizes_input = split_sizes*ta.shape[1]
		displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
		split_sizes_output = split_sizes
		displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]


	else:		#If this processer isn't the "local" processer which we are loading data on..


		split_sizes_input = None; displacements_input = None; split_sizes_output = None;\
		displacements_output = None; ta_split = None; dp_split = None; hur_split = None; \
		hgt_split = None; ua_split = None; va_split = None; ta = None; dp = None; \
		hur = None; hgt = None; ua = None; va = None; p = None; output_data = None



	#Broadcast split/chunked arrays/information to other cores from the "local" core (root = 0)
	ta_split = comm.bcast(ta_split, root=0)
	dp_split = comm.bcast(dp_split, root=0)
	hur_split = comm.bcast(hur_split, root=0)
	hgt_split = comm.bcast(hgt_split, root=0)
	ua_split = comm.bcast(ua_split, root=0)
	va_split = comm.bcast(va_split, root=0)
	p = comm.bcast(p, root=0)
	split_sizes_input = comm.bcast(split_sizes_input, root = 0)
	displacements_input = comm.bcast(displacements_input, root = 0)
	split_sizes_output = comm.bcast(split_sizes_output, root = 0)
	displacements_output = comm.bcast(displacements_output, root = 0)

	#Create arrays to receive chunked/split data on each processer, where rank specifies the
	# processer. User Scatterv to send the data. Barrier syncs the processers
	ta_chunk = np.zeros(np.shape(ta_split[rank]))
	dp_chunk = np.zeros(np.shape(dp_split[rank]))
	hur_chunk = np.zeros(np.shape(hur_split[rank]))
	hgt_chunk = np.zeros(np.shape(hgt_split[rank]))
	ua_chunk = np.zeros(np.shape(ua_split[rank]))
	va_chunk = np.zeros(np.shape(va_split[rank]))
	comm.Scatterv([ta,split_sizes_input, displacements_input, MPI.DOUBLE],ta_chunk,root=0)
	comm.Scatterv([dp,split_sizes_input, displacements_input, MPI.DOUBLE],dp_chunk,root=0)
	comm.Scatterv([hur,split_sizes_input, displacements_input, MPI.DOUBLE],hur_chunk,root=0)
	comm.Scatterv([hgt,split_sizes_input, displacements_input, MPI.DOUBLE],hgt_chunk,root=0)
	comm.Scatterv([ua,split_sizes_input, displacements_input, MPI.DOUBLE],ua_chunk,root=0)
	comm.Scatterv([va,split_sizes_input, displacements_input, MPI.DOUBLE],va_chunk,root=0)
	comm.Barrier()

	#Print diagnostics
	if rank == 0:
		print("TOTAL (LAND) POINTS: %s" %(ta.shape[0]))
		print("CHUNKSIZE: %s" %(ua_chunk.shape,))


#---------------------------------------------------------------------------------------------------
	#Run SHARPpy
	start = dt.datetime.now()
	output = np.zeros((ta_chunk.shape[0]))			#Output for this processer
	for i in np.arange(0,ta_chunk.shape[0]):		#Loop over grid points in this chunk
		#Run SHARPpy
		sb_pcl = sharp_parcel_mpi(p,\
			ua_chunk[i],\
			va_chunk[i],\
			hgt_chunk[i],\
			ta_chunk[i],\
			dp_chunk[i])
		output[i] = sb_pcl.bplus			#CAPE
	
#---------------------------------------------------------------------------------------------------

	#Print diagnostics
	if rank == 0:
		print("Time taken for SHARPPy on processor 1: %s" %(dt.datetime.now() - start), )
		print("Time taken for each element on processor 1: %s" \
			%((dt.datetime.now() - start)/float(ua_chunk.shape[0])), )
		print("Time taken if we were just using one processer in a loop: %s" \
			%( ( (dt.datetime.now() - start) / float(ua_chunk.shape[0])) * ta.shape[0], ))

	#Gather output data from each processer together to root/local node (Note that all 
	# processes must finish before the program finishes)
	comm.Gatherv(output, \
		[output_data, split_sizes_output, displacements_output, MPI.DOUBLE], \
		root=0)

	#Reshape data and plot
	if rank == 0:
		#Reshape back to a 2d grid
		output_reshaped = output_data.reshape((orig_shape[1],orig_shape[2]))
		x,y = np.meshgrid(lon,lat)
		m = Basemap(llcrnrlon=start_lon, llcrnrlat=start_lat, urcrnrlon=end_lon, \
			urcrnrlat=end_lat, projection="cyl")
		m.contourf(x,y,output_reshaped,levels=np.arange(0,11050,50), extend="max")
		m.drawcoastlines()
		plt.colorbar()
		plt.title("ERA-Interim Surface Based CAPE\n2016-09-28 0600 UTC")
		plt.show()

