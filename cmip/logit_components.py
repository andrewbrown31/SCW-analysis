from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def compute_diff(hist, scenario, months):
	djf = []
	djf_abs = []
	for i in np.arange(len(hist)):
		t2 = np.stack([scenario[i][months[0]].values, scenario[i][months[1]].values, scenario[i][months[2]].values]).mean(axis=0)
		t1 = np.stack([hist[i][months[0]].values, hist[i][months[1]].values, hist[i][months[2]].values]).mean(axis=0)
		djf.append( (t2-t1)/t1 * 100)
		djf_abs.append( (t2-t1) )
	djf_pos = (np.stack(djf) > 0).sum(axis=0)
	djf_neg = (np.stack(djf) <= 0).sum(axis=0)
	djf = np.median(np.stack(djf), axis=0)
	djf_abs = np.median(np.stack(djf_abs), axis=0)
	djf_sig = np.where( ( (djf > 0) & (djf_pos >= 10) ) | ( (djf<=0) & (djf_neg >= 10) ), 1, 0)
	return djf, djf_abs, djf_sig

def load_mean(model, v):
	hist = [xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+m+"_r1i1p1_mean_"+v+"_historical_1979_2005.nc") for m in model]
	scenario = [xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+m+"_r1i1p1_mean_"+v+"_rcp85_2081_2100.nc") for m in model]
	djf, djf_abs, djf_sig = compute_diff(hist, scenario, ["Dec","Jan","Feb"])
	mam, mam_abs, mam_sig = compute_diff(hist, scenario, ["Mar","Apr","May"])
	jja, jja_abs, jja_sig = compute_diff(hist, scenario, ["Jun","Jul","Aug"])
	son, son_abs, son_sig = compute_diff(hist, scenario, ["Sep","Oct","Nov"])
	return [djf, mam, jja, son], [djf_abs, mam_abs, jja_abs, son_abs], [djf_sig, mam_sig, jja_sig, son_sig]

def get_lon_lat(model,v):
	temp = xr.open_dataset("/g/data/eg3/ab4502/ExtremeWind/aus/regrid_1.5/"+model[0]+"_r1i1p1_mean_"+v+"_historical_1979_2005.nc")
	return temp.lon.values, temp.lat.values
		

if __name__ == "__main__":

	model = ["ACCESS1-3", "ACCESS1-0", "BNU-ESM", "CNRM-CM5", "GFDL-CM3", \
			    "GFDL-ESM2G", "GFDL-ESM2M", "IPSL-CM5A-LR", "IPSL-CM5A-MR", \
			    "MIROC5", "MRI-CGCM3", "bcc-csm1-1"]
	lon, lat = get_lon_lat(model, "ta500")

	_, diff_Umean800_600, sig_Umean800_600 = load_mean(model, "Umean800_600")
	_, diff_lr13, sig_lr13 = load_mean(model, "lr13")
	_, diff_rhmin13, sig_rhmin13 = load_mean(model, "rhmin13")
	_, diff_srhe_left, sig_srhe_left = load_mean(model, "srhe_left")
	_, diff_q_melting, sig_q_melting = load_mean(model, "q_melting")
	_, diff_eff_lcl, sig_eff_lcl = load_mean(model, "eff_lcl")
	_, diff_ebwd, sig_ebwd = load_mean(model, "ebwd")

	cnt = 0
	plt.figure(figsize=[8,14])
	m = Basemap(llcrnrlon=110, llcrnrlat=-45, urcrnrlon=160, \
		urcrnrlat=-10,projection="cyl")
	s_title = ["DJF","MAM","JJA","SON"]
	a=ord("a"); alph=[chr(i) for i in range(a,a+26)];
	alph = alph + ["aa","ab"]
	alph = [alph[i]+")" for i in np.arange(len(alph))]
	vmin = {"mu_cape":-200,"dcape":-100, "ebwd":-1, "lr03":-0.5, "lr700_500":-0.5, "dp850":-2, "ta850":-1.5, "ta500":-1.5, "Umean06":-2, "s06":-2,\
			"Umean800_600":-2,"lr13":-1,"rhmin13":-10,"srhe_left":-5,"q_melting":-0.8,"eff_lcl":-250}
	units = {"lr03":"deg km$^{-1}$","mu_cape":"J kg$^{-1}$",
                    "ebwd":"m s$^{-1}$","Umean06":"m s$^{-1}$","s06":"m s$^{-1}$",\
                    "dp850":"deg C","ta500":"deg C","ta850":"deg C","srhe_left":"m$^{-2}$ s$^{-2}$",\
                    "lr700_500":"deg km$^{-1}$", "dcape":"J kg$^{-1}$", "Umean800_600":"m s$^{-1}$",\
		    "lr13":"deg km$^{-1}$", "rhmin13":"%", "q_melting":"g kg$^{-1}$", "eff_lcl":"m"}
	ax=plt.subplot(7,4,cnt+1)
	plt.subplots_adjust(top=0.9, bottom=0.3, wspace=0.1,left=0.1)
	for i in np.arange(4):
		ax=plt.subplot(7,4,cnt+1)
		c=plt.contourf(lon, lat, diff_Umean800_600[i], np.linspace(vmin["Umean800_600"],-vmin["Umean800_600"],11),cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_Umean800_600[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		plt.title(s_title[i])
		if cnt==0:
			plt.ylabel("Umean800-600")
		plt.annotate(alph[cnt], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["Umean800_600"])
		
		ax=plt.subplot(7,4,cnt+5)
		c=plt.contourf(lon, lat, diff_lr13[i], np.linspace(vmin["lr13"],-vmin["lr13"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_lr13[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("LR13")
		plt.annotate(alph[cnt+4], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["lr13"])
		
		ax=plt.subplot(7,4,cnt+9)
		c=plt.contourf(lon, lat, diff_rhmin13[i], np.linspace(vmin["rhmin13"],-vmin["rhmin13"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_rhmin13[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("RHMin13")
		plt.annotate(alph[cnt+8], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["rhmin13"])

		ax=plt.subplot(7,4,cnt+13)
		c=plt.contourf(lon, lat, diff_srhe_left[i], np.linspace(vmin["srhe_left"],-vmin["srhe_left"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_srhe_left[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("SRHE")
		plt.annotate(alph[cnt+13], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["srhe_left"])

		ax=plt.subplot(7,4,cnt+17)
		c=plt.contourf(lon, lat, diff_q_melting[i], np.linspace(vmin["q_melting"],-vmin["q_melting"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_q_melting[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("Q-Melting")
		plt.annotate(alph[cnt+17], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["q_melting"])

		ax=plt.subplot(7,4,cnt+21)
		c=plt.contourf(lon, lat, diff_eff_lcl[i], np.linspace(vmin["eff_lcl"],-vmin["eff_lcl"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_eff_lcl[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("Eff-LCL")
		plt.annotate(alph[cnt+21], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["eff_lcl"])

		ax=plt.subplot(7,4,cnt+25)
		c=plt.contourf(lon, lat, diff_ebwd[i], np.linspace(vmin["ebwd"],-vmin["ebwd"],11), cmap=plt.get_cmap("RdBu_r"), extend="both")
		plt.contourf(lon,lat,sig_ebwd[i], levels=[0.5, 1.5], colors="none", hatches=["//"]) 
		m.drawcoastlines()
		if cnt==0:
			plt.ylabel("EBWD")
		plt.annotate(alph[cnt+24], xy=(0.05, 0.05), xycoords='axes fraction') 
		if i==3:
			pos1 = ax.get_position() # get the original position 
			cax = plt.axes([pos1.x0 + pos1.width + 0.02, pos1.y0,  0.01 , pos1.height] )
			c=plt.colorbar(c,ax=ax,cax=cax,orientation="vertical")
			c.set_label(units["ebwd"])

		cnt=cnt+1
	#cax = plt.axes([0.92,0.58,0.02,0.2])
	#cb=plt.colorbar(c, cax=cax, orientation="vertical", extend="max")
	#cb.set_label("Deg C")
	#cax2 = plt.axes([0.92,0.3,0.02,0.12])
	#cb2=plt.colorbar(c2, cax=cax2, orientation="vertical", extend="both" )
	#cb2.set_label("%")
	plt.savefig("/g/data/eg3/ab4502/figs/CMIP/logit_components.png", bbox_inches="tight")
