from event_analysis import *
import matplotlib.colors as colors

def environment_scatter(df,title,event):
	#CAPE/S06/DCAPE/DLM

	for x in [15,30]:
	   cnt=1
	   fig,ax = plt.subplots(figsize=[15,11])
	   for c in [0,100,200,300,400,600]:
		plt.subplot(3,2,cnt)
		h = plt.hist2d(df[(df[event]==0) & (df["ml_cape"]<=c)]["dlm"],\
			df[(df[event]==0) & (df["ml_cape"]<=c)]["dcape"],bins=20,\
			norm=colors.LogNorm(1,10000),cmap=plt.get_cmap("Greys",8))
		#High S06 high CAPE
		plt.scatter(df[(df[event]==1) & (df.s06>=x) & (df.ml_cape>c)]["dlm"],df[(df[event]==1) & \
			(df.s06>x) & (df.ml_cape>c)]["dcape"],color="b",\
			label="S06 >= "+str(x)+", MLCAPE > X")
		#High S06 low CAPE
		plt.scatter(df[(df[event]==1) & (df.s06>=x) & (df.ml_cape<=c)]["dlm"],df[(df[event]==1) & \
			(df.s06>x) & (df.ml_cape<=c)]["dcape"],color="b",marker="^",\
			label="S06 >= "+str(x)+", MLCAPE <= X") 
		#Low S06 high CAPE
		plt.scatter(df[(df[event]==1) & (df.s06<x) & (df.ml_cape>c)]["dlm"],df[(df[event]==1) & \
			(df.s06<x) & (df.ml_cape>c)]["dcape"],color="r",\
			label="S06 < "+str(x)+", MLCAPE > X")
		#Low S06 low CAPE
		plt.scatter(df[(df[event]==1) & (df.s06<x) & (df.ml_cape<=c)]["dlm"],df[(df[event]==1) & \
			(df.s06<x) & (df.ml_cape<=c)]["dcape"],color="r",marker="^",\
			label="S06 < "+str(x)+", MLCAPE <= X")
		plt.title("X = " + str(c) + " J/kg")
		if cnt in [5,6]:
			plt.xlabel("MLM (m/s)")
		if cnt in [1,3,5]:
			plt.ylabel("DCAPE (J/kg)")
		plt.ylim([-10,2000])
		plt.xlim([-1,40])
		#DRAW CONDS
		plt.plot([26,26],[0,500],color="b",linestyle="--")
		plt.plot([26,26],[350,2000],color="r",linestyle="--")
		plt.plot([26,40],[500,500],"b--")
		plt.plot([26,0],[350,350],"r--")
		cnt=cnt+1
	   plt.subplot(3,2,cnt-1)
	   plt.legend(bbox_to_anchor=(2,2))
	   plt.subplots_adjust(right=0.7)
	   plt.suptitle(title)
	   cax = fig.add_axes([0.1,0.025,0.6,0.01])
	   plt.colorbar(h[-1],cax,orientation="horizontal",extend="max")
	   plt.show()


if __name__ == "__main__":

	df,jdh_df,non_jdh_df = analyse_jdh_events()
	df["strong_gust"] = 0;df["extreme_gust"] = 0
	df.loc[(df.wind_gust >= 25) & (df.wind_gust < 30),"strong_gust"] = 1
	df.loc[(df.wind_gust >= 30),"extreme_gust"] = 1
	#Total
	#environment_scatter(df,"All JDH Events","jdh")
	#Seasonal
	#environment_scatter(df[np.in1d(df.month,np.array([11,12,1,2,3,4]))],"November - April","jdh")
	#environment_scatter(df[np.in1d(df.month,np.array([5,6,7,8,9,10]))],"May - October","jdh")
	#Locations
	#environment_scatter(df[np.in1d(df.index.get_level_values(1),\
	#	np.array(["Adelaide AP","Mount Gambier","Edinburgh"]))],"Adelaide, Mt. Gambier, Edinburgh","jdh")
	#environment_scatter(df[np.in1d(df.index.get_level_values(1),\
		#np.array(["Woomera"]))],"Woomera","jdh")

	#These are pretty good
	u1 = 26
	u2 = 26
	c_u = 400
	c_l = 120
	s1 = 30
	s2 = 35
	d = 350
	d_u = 820

	#1) 
	#Condition 1: 1st type of convective wind event
	#High S06/DLM environment, with low CAPE and DCAPE
	#cond1 = ( (df["ml_cape"]<400) & (df["s06"]>30) & (df["dcape"]<820) & \
	#	(df["dlm"]>=26) & (df["dcape"] > 0) )
	#Condition 2: 2nd type of convective wind event
	#Increased CAPE with high DCAPE and low S06/DLM
	#cond2 = ( (df["ml_cape"]>120) & (df["s06"]<35) & (df["dcape"]>350) & (df["dlm"]<26) & (df["s06"]>8) )

	#2)
	#Try without low cape restrctions on SF case and without low S06 restrictions on WF case
	#cond1 = ( (df["s06"]>30) & (df["dcape"]<820) & (df["dlm"]>=26) & (df["dcape"] > 0) )
	#cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>350) & (df["dlm"]<26) )

	#3)
	#As above but with stronger DCAPE restriction for SF case and relaxed DCAPE threshold for WF case
	#cond1 = ( (df["s06"]>=30) & (df["dcape"]<500) & (df["dlm"]>=26) & (df["dcape"] > 0) )
	#cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>250) & (df["dlm"]<26) )

	#4)
	#As above but with a non-zero MUCAPE for the SF case
	#cond1 = ( (df["mu_cape"]>0) & (df["s06"]>=30) & (df["dcape"]<500) & (df["dlm"]>=26) & (df["dcape"] > 0) )
	#cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>250) & (df["dlm"]<26) )

	#4)
	#As in 3) but with a third, hybrid condition
	cond1 = ( (df["s06"]>=30) & (df["dcape"]<500) & (df["dlm"]>=26) & (df["dcape"] > 0) )	#SF
	cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>250) & (df["dlm"]<=15) )	#MF
	cond3 = ( (df["ml_cape"]>0) & (df["dcape"]>50) & (df["dlm"]>15) & (df["dlm"]<26) & (df["s06"]>20 ))	#HF


	test  = True
	
	if test:
	    import itertools
	    ml_sf_list = list(np.array([-1,0]))
	    s06_sf_list = list(np.arange(25,40,5))
	    dcape_upper_sf_list = list(np.arange(250,800,50))
	    dlm_list = list(np.arange(20,32.5,2.5))
	    dcape_lower_sf_list = list(np.array([-1,0]))
	    ml_mf_list = list(np.arange(0,425,25))
	    dcape_mf_list = list(np.arange(0,525,25))
	    z = itertools.product(ml_sf_list,s06_sf_list,dcape_upper_sf_list,dlm_list,dcape_lower_sf_list,\
			ml_mf_list,dcape_mf_list)

	    ml_sf_out = []
	    s06_sf_out = []
	    dcape_upper_sf_out = []
	    dlm_out = []
	    dcape_lower_sf_out = []
	    ml_mf_out = []
	    dcape_mf_out = []
	    hr_out = []
	    far_out = []
	    csi_out = []
	    l_out = []
	    for l in z:
		ml_sf,s06_sf,dcape_upper_sf,dlm,dcape_lower_sf,ml_mf,dcape_mf = l
		#Generalised conditions:
		#SF
		cond1 = ( (df["ml_cape"]>ml_sf) & (df["s06"]>=s06_sf) & (df["dcape"]<dcape_upper_sf) & \
				(df["dlm"]>=dlm) & (df["dcape"]>dcape_lower_sf) )
		#MF
		cond2 = ( (df["ml_cape"]>ml_mf) & (df["dcape"]>dcape_mf) & (df["dlm"]<dlm) )

		hits = ((df.jdh==1) & (cond1 | cond2)).sum()
		misses = ((df.jdh==1) & ~(cond1 | cond2)).sum()
		fa = ((df.jdh==0) & (cond1 | cond2)).sum()
		cn = ((df.jdh==0) & ~(cond1 | cond2)).sum()
		hit_rate = hits / float(hits + misses)
		far = fa / float(cn + fa)
		csi = hits / float(hits+misses+fa)

		if (hit_rate > 0.6) & (far < 0.1):
			ml_sf_out.append(ml_sf)
			s06_sf_out.append(s06_sf)
			dcape_upper_sf_out.append(dcape_upper_sf)
			dlm_out.append(dlm)
			dcape_lower_sf_out.append(dcape_lower_sf)
			ml_mf_out.append(ml_mf)
			dcape_mf_out.append(dcape_mf)
			hr_out.append(hit_rate)
			far_out.append(far)
			csi_out.append(csi)
			l_out.append(l)
		
	    l_out = np.array(l_out)
	    far_out = np.array(far_out)
	    hr_out = np.array(hr_out)
	    csi_out = np.array(csi_out)

	print("LOWEST CSI: " + str(csi_out.max()))
	print("HR: " + str(hr_out[csi_out==csi_out.max()]))
	print("FAR: " + str(far_out[csi_out==csi_out.max()]))
	print("PARAMS: " + str(l_out[csi_out==csi_out.max()]))
	
	np.save("/short/eg3/ab4502/ExtremeWind/csi.npy",csi_out)
	np.save("/short/eg3/ab4502/ExtremeWind/hr.npy",hr_out)
	np.save("/short/eg3/ab4502/ExtremeWind/far.npy",far_out)
	np.save("/short/eg3/ab4502/ExtremeWind/params.npy",l_out)

	#Inspect station wind speed distributions for the two types of environmental conditions
	#plt.boxplot([df[(cond1)]["wind_gust"].dropna(),df[(cond2)]["wind_gust"].dropna()],\
	#	labels=["Synoptic forcing\n"+"("+str(cond1.sum())+")",\
	#	"Mesoscale forcing\n"+"("+str(cond2.sum())+")"])
	#plt.plot(np.ones(((cond1)&(df.jdh==1)).sum())+0.1,df[(cond1)&(df.jdh==1)]["wind_gust"].dropna(),\
	#	color="g",marker="^",linestyle="none")
	#plt.plot(2*np.ones(((cond2)&(df.jdh==1)).sum())+0.1,df[(cond2)&(df.jdh==1)]["wind_gust"].dropna(),\
	#	color="g",marker="^",linestyle="none")
	#plt.ylabel("Daily maximum station wind gust (m/s)")
	#plt.show()
