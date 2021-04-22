import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime as dt
import glob
import os
import pyart
import zipfile
import warnings

def load_radar(rid, time, radar_t_delta=0):

    #time: Time of interest (in this context, the time of maximum BARPAC gust). datetime object
    #rid: Radar ID (https://www.openradar.io/operational-network). two-digit string
    #radar_t_delta: Add some interval (in minutes) to look at a time relative to the time of maximum model gust. integer
    
    time = time + dt.timedelta(minutes=radar_t_delta)
    
    isfile = len(glob.glob("/g/data/eg3/ab4502/radar/"+rid+"_"+time.strftime("%Y%m%d")+"*")) > 0
    
    if not isfile:
        print("INFO: UNPACKING RADAR FILE FOR ID "+rid+" AND TIME "+time.strftime("%Y%m%d"))
        path_to_zip_file = "/g/data/rq0/level_1/odim_pvol/"+rid+"/"+str(time.year)+"/vol/"+rid+"_"+time.strftime("%Y%m%d")+".pvol.zip"
        directory_to_extract_to = "/g/data/eg3/ab4502/radar/"
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        
    else:
        print("INFO: LOADING RADAR FILE FOR ID "+rid+" AND TIME "+time.strftime("%Y%m%d"))
        
    files = glob.glob("/g/data/eg3/ab4502/radar/"+rid+"_"+time.strftime("%Y%m%d")+"*")
    f_times = [fname.split("/")[-1].split("_")[2].split(".")[0] for fname in files]
    radar_file=pyart.aux_io.read_odim_h5(\
                              files[np.argmin(np.array([abs(time-dt.datetime(time.year,time.month,time.day,int(f[0:2]),int(f[2:4]),int(f[4:6]))) for f in f_times]))],\
                              file_field_names=True)
    return radar_file

