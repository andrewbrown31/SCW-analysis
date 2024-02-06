from read_model_data.era5_read import read_era5_rt52
from read_model_data.barra_read import read_barra
import datetime as dt
from diagnostic_driver import run_diagnostics

if __name__ == "__main__":

    #This code loads ERA5 data from Gadi, using the read_era5_rt52() function
    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
    t1 = "2016092800"; t2="2016092806"

    domain = [start_lat,end_lat,start_lon,end_lon]
    time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]

    print("LOADING DATA...")
    ta,dp,hur,hgt,terrain,p,ps,msl,ua,va,uas,vas,tas,ta2d,\
        cp,tp,wg10,mod_cape,sst,lon,lat,date_list = \
        read_era5_rt52(domain,time,delta_t=6)

    #Run the diagnostic suite and save the output
    run_diagnostics(ta,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list,
					params="full",mdl_lvl=False,is_dcape=True,
					issave=True,out_name="era5",out_path="/g/data/w40/ab4502/")