from read_model_data.era5_read import read_era5_rt52, read_era5_cds
from read_model_data.barra_read import read_barra_fc
import datetime as dt
from diagnostic_driver import run_diagnostics

if __name__ == "__main__":

    start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
    t1 = "2016092800"; t2="2016092806"
    domain = [start_lat,end_lat,start_lon,end_lon]
    time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]

    gadi = True #Can be set to True if working on Gadi
    model = "barra_r"

    if gadi:
        #This code loads ERA5 data from Gadi, using the read_era5_rt52() function

        print("LOADING DATA...")
        if model == "era5":
            ta,dp,hur,hgt,terrain,p,ps,msl,ua,va,uas,vas,tas,ta2d,\
               cp,tp,wg10,mod_cape,sst,lon,lat,date_list = \
               read_era5_rt52(domain,time,delta_t=6)
        elif model == "barra_r":
            ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
                read_barra_fc(domain,time)
    else:
        #This code loads ERA5 data downloaded from the CDS, from read_model_data/download*.py

        if model == "era5":
            ta,dp,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,cp,tp,wg10,cape,lon,lat,date_list = read_era5_cds(
                "read_model_data/data/era5_pl.nc", 
                "read_model_data/data/era5_sfc.nc", 
                domain,time,delta_t=6)
        elif model in ["barra","barra_sy"]:
            print("For barra, set Gadi=True")

    #Run the diagnostic suite and save the output
    run_diagnostics(ta,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list,
					params="full",mdl_lvl=False,is_dcape=True,
					issave=True,out_name=model,out_path="read_model_data/data/")
