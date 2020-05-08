#All code related to the investigation of the COND parameter. I.e. Looking at the environments of extreme wind events
#in ERA-interim (environmental scatter) in terms of MLCAPE, S06, DLM/MLM and DCAPE, and testing a conditional
#parameter based on these elements

from event_analysis import *
import matplotlib.colors as colors

def environment_scatter(df,title,event,mlm=26,s06=30,dcape_mf=350,dcape_sf=500):
	#CAPE/S06/DCAPE/DLM

	for x in [s06]:
	   cnt=1
	   for c in [-1,0,200]:
	   	fig,ax = plt.subplots()
		#plt.subplot(3,2,cnt)
		h = plt.hist2d(df[(df[event]==0) & (df["ml_cape"]>c)]["mlm"],\
			df[(df[event]==0) & (df["ml_cape"]>c)]["dcape"],bins=20,\
			norm=colors.LogNorm(1,10000),cmap=plt.get_cmap("Greys",8))
		#High S06 high CAPE
		if c == -1:
			lab_mf1 = "S06 >= "+str(x)+" m.s$^{-1}$, MLCAPE >= 0" + " J.kg$^{-1}$"
			lab_mf2 = "S06 < "+str(x)+" m.s$^{-1}$, MLCAPE >= 0" + " J.kg$^{-1}$"
			lab_null = "MLCAPE >= 0" + " J.kg$^{-1}$"
		elif c == 0:
			lab_mf1 = "S06 >= "+str(x)+" m.s$^{-1}$, MLCAPE > 0" + " J.kg$^{-1}$"
			lab_mf2 = "S06 < "+str(x)+" m.s$^{-1}$, MLCAPE > 0" + " J.kg$^{-1}$"
			lab_sf1 = "S06 >= "+str(x)+" m.s$^{-1}$, MLCAPE = 0" + " J.kg$^{-1}$"
			lab_sf2 = "S06 < "+str(x)+" m.s$^{-1}$, MLCAPE = 0" + " J.kg$^{-1}$"
			lab_null = "MLCAPE > 0" + " J.kg$^{-1}$"
		else:
			lab_mf1 = "S06 >= "+str(x)+" m.s$^{-1}$, MLCAPE >= " +str(c)+ " J.kg$^{-1}$"
			lab_mf2 = "S06 < "+str(x)+" m.s$^{-1}$, MLCAPE >= " + str(c) + " J.kg$^{-1}$"
			lab_sf1 = "S06 >= "+str(x)+" m.s$^{-1}$, MLCAPE < " + str(c) + " J.kg$^{-1}$"
			lab_sf2 = "S06 < "+str(x)+" m.s$^{-1}$, MLCAPE < " + str(c) + " J.kg$^{-1}$"
			lab_null = "MLCAPE > "+ str(c) + " J.kg$^{-1}$"
		plt.scatter(df[(df[event]==1) & (df.s06>=x) & (df.ml_cape>c)]["mlm"],df[(df[event]==1) & \
			(df.s06>x) & (df.ml_cape>c)]["dcape"],color="darkblue",\
			label=lab_mf1)
		#High S06 low CAPE
		if c >= 0:
			plt.scatter(df[(df[event]==1) & (df.s06>=x) & (df.ml_cape<=c)]["mlm"],df[(df[event]==1) & \
			   (df.s06>x) & (df.ml_cape<=c)]["dcape"],color="lightblue",marker="o",\
			   label=lab_sf1) 
		#Low S06 high CAPE
		plt.scatter(df[(df[event]==1) & (df.s06<x) & (df.ml_cape>c)]["mlm"],df[(df[event]==1) & \
			(df.s06<x) & (df.ml_cape>c)]["dcape"],color="red",\
			label=lab_mf2)
		#Low S06 low CAPE
		if c >= 0:
			plt.scatter(df[(df[event]==1) & (df.s06<x) & (df.ml_cape<=c)]["mlm"],df[(df[event]==1) & \
			   (df.s06<x) & (df.ml_cape<=c)]["dcape"],color="orange",marker="o",\
			   label=lab_sf2)
		#plt.title("X = " + str(c) + " J/kg")
		plt.plot([-100,-100],[-100,-100],color="k",marker="s",linestyle="none",label=lab_null)
		plt.xlabel("MLM (m.s$^{-1}$)",fontsize="xx-large")
		plt.ylabel("DCAPE (J.kg$^{-1}$)",fontsize="xx-large")
		plt.ylim([-10,2000])
		plt.xlim([-1,40])
		#DRAW CONDS
		plt.plot([mlm,mlm],[0,dcape_sf],color="b",linestyle="--")
		plt.plot([mlm,mlm],[dcape_mf,2000],color="r",linestyle="--")
		plt.plot([mlm,40],[dcape_sf,dcape_sf],"b--")
		plt.plot([mlm,0],[dcape_mf,dcape_mf],"r--")
		#cnt=cnt+1
	   	#plt.subplot(3,2,cnt-1)
	   	plt.legend(bbox_to_anchor=(-0.5,0),fontsize="xx-large")
	   	plt.subplots_adjust(bottom=0.2)
	   	#plt.suptitle(title)
	   	cax = fig.add_axes([0.1,0.03,0.6,0.01])
	   	cb = plt.colorbar(h[-1],cax,orientation="horizontal",extend="max")
	   	cb.ax.tick_params(labelsize="xx-large")
	   	ax.tick_params(labelsize="xx-large")
	   	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/scatter/environment_scatter_sa_small_"+str(c)+"_"+str(x)+".png",bbox_inches="tight")
	   	plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/environment_scatter_sa_small_"+str(c)+"_"+str(x)+".tiff",bbox_inches="tight")

def environment_scatter2(domain,event,param1,param2,param3=False,fit_points=False):
	#CAPE/S06/DCAPE/DLM

	if event == "jdh":
		df = analyse_events(event,domain)
	elif (event == "extreme") | (event == "strong"):
		df = analyse_events("aws",domain)
	else:
		print("EVENT MUST BE jdh, strong OR extreme")

	fig,ax = plt.subplots()
	#plt.subplot(3,2,cnt)

	[cmap1,mean_levels1,extreme_levels1,cb_lab1,range1,log_plot1,threshold1] = contour_properties(param1)
	[cmap2,mean_levels2,extreme_levels2,cb_lab2,range2,log_plot2,threshold2] = contour_properties(param2)
	if (log_plot1) & (~log_plot2):
		ax.set_xscale("symlog")
		plt.xlim([0,range1[1]])
		base = len(str(int(((range1[1]/10)))))
		factor = range1[1] / np.power(10,base)
		xbin = factor*np.logspace(1,base,10)
		ybin = np.linspace(0,range2[1],20)
		h = ax.hist2d(df[(df[event]==0)][param1],\
			df[(df[event]==0)][param2],\
			bins = [np.concatenate(([0],xbin)),ybin],\
			norm=colors.LogNorm(.1,10000),cmap=plt.get_cmap("Greys",14))
		#try:
		x = np.linspace(plt.xlim()[0]+1,plt.xlim()[1]+1)
		c = fit_points[0][1]
		m = (fit_points[1][1]-float(fit_points[0][1])) / np.log(fit_points[1][0]/float(fit_points[0][0]))
		plt.plot(x, m * np.log(x) + c, color="k" ) 
		df["eqn"] = m * np.log(df[param1]) + c
		misses = ((df[param2] < df["eqn"]) & (df[event] == 1)).sum()
		hits = ((df[param2] > df["eqn"]) & (df[event] == 1)).sum()
		cn = ((df[param2] < df["eqn"]) & (df[event] == 0)).sum()
		fa = ((df[param2] > df["eqn"]) & (df[event] == 0)).sum()
		title1 = "m = "+str(round(float(m),3))+", c = "+str(round(c,3))
		title2 = "Below line: Events = "+str(misses)+", Non-events = "+str(cn)
		title3 = "Above line: Events = "+str(hits)+", Non-events = "+str(fa)
		plt.title(title1 + "\n" + title2+ "\n" + title3)
		#except:
		#	pass
	elif (~log_plot1) & (log_plot2):
		ax.set_yscale("symlog",linthreshy=0.1)
		plt.ylim([0,range2[1]])
		base = len(str(int(((range2[1]/10)))))
		factor = range2[1] / np.power(10,base)
		xbin = np.linspace(0,range1[1],20)
		ybin = factor*np.logspace(1,base,10)
		h = ax.hist2d(df[(df[event]==0)][param1],\
			df[(df[event]==0)][param2],\
			bins = [xbin,np.concatenate(([0],ybin))],\
			norm=colors.LogNorm(.1,10000),cmap=plt.get_cmap("Greys",14))

		try:
			x = np.linspace(plt.xlim()[0],plt.xlim()[1]+1)
			lam = fit_points[0][1]
			gam = np.log(fit_points[1][1]/float(fit_points[0][1])) / float(fit_points[1][0]-fit_points[0][0])
			plt.plot(x, lam * \
				np.exp( x * gam )  \
				, color="k" ) 
			df["eqn"] = lam * np.exp(df[param1] * gam)
			misses = ((df[param2] < df["eqn"]) & (df[event] == 1)).sum()
			hits = ((df[param2] > df["eqn"]) & (df[event] == 1)).sum()
			cn = ((df[param2] < df["eqn"]) & (df[event] == 0)).sum()
			fa = ((df[param2] > df["eqn"]) & (df[event] == 0)).sum()
			title1 = "Lambda = "+str(round(float(lam),3))+", Gamma = "+str(round(gam,3))
			title2 = "Below line: Events = "+str(misses)+", Non-events = "+str(cn)
			title3 = "Above line: Events = "+str(hits)+", Non-events = "+str(fa)
			plt.title(title1 + "\n" + title2+ "\n" + title3)
		except:
			pass
	elif (log_plot1) & (log_plot2):
		print("loglog")
		ax.set_yscale("symlog",linthreshy=0.1)
		ax.set_xscale("symlog",linthreshy=0.1)
		plt.xlim([0,range1[1]])
		plt.ylim([0,range2[1]])
		base1 = len(str(int(((range1[1]/10)))))
		base2 = len(str(int(((range2[1]/10)))))
		factor1 = range1[1] / float(np.power(10,base1))
		factor2 = range2[1] / float(np.power(10,base2))
		xbin = factor1*np.logspace(.1,base1,10)
		ybin = factor2*np.logspace(.1,base2,10)
		h = ax.hist2d(df[(df[event]==0)][param1],\
			df[(df[event]==0)][param2],\
			bins = [np.concatenate(([0],xbin)),np.concatenate(([0],ybin))],\
			norm=colors.LogNorm(.1,10000),cmap=plt.get_cmap("Greys",14))
		
	else:
		base = len(str(int(((range2[1]/10)))))
		h = ax.hist2d(df[(df[event]==0)][param1],\
			df[(df[event]==0)][param2],\
			bins = 20,\
			norm=colors.LogNorm(.1,10000),cmap=plt.get_cmap("Greys",14))
		x = np.linspace(plt.xlim()[0],plt.xlim()[1]+1)
		try:
			c = fit_points[0][1]
			m = (fit_points[1][1] - fit_points[0][1]) / float(fit_points[1][0] - fit_points[0][0])
			plt.plot(x, m * x + c, color="k")
			df["eqn"] = c + (df[param1] * m)
			misses = ((df[param2] < df["eqn"]) & (df[event] == 1)).sum()
			hits = ((df[param2] > df["eqn"]) & (df[event] == 1)).sum()
			cn = ((df[param2] < df["eqn"]) & (df[event] == 0)).sum()
			fa = ((df[param2] > df["eqn"]) & (df[event] == 0)).sum()
			title1 = "m = "+str(round(float(m),3))+", c = "+str(round(c,3))
			title2 = "Below line: Events = "+str(misses)+", Non-events = "+str(cn)
			title3 = "Above line: Events = "+str(hits)+", Non-events = "+str(fa)
			plt.title(title1 + "\n" + title2+ "\n" + title3)
		except:
			pass

	plt.colorbar(h[3])
	plt.ylabel(param2)
	plt.xlabel(param1)
		
	
	if param3 == False:
		ax.scatter(df[(df[event]==1)][param1],df[(df[event]==1)][param2])
	else:	
		plt.scatter(df[(df[event]==1)][param1],df[(df[event]==1)][param2],c=df[(df[event]==1)][param3])
		plt.colorbar()



	plt.show()


def test_daily_cond():

	#Take daily convective parameter data at station locations, and test the application of a conditional 
	# parameter on it (in terms of JDH events)

	#LOAD DAILY DATA
	df,jdh_df,non_jdh_df = analyse_jdh_events()
	df["strong_gust"] = 0;df["extreme_gust"] = 0
	df.loc[(df.wind_gust >= 25) & (df.wind_gust < 30),"strong_gust"] = 1
	df.loc[(df.wind_gust >= 30),"extreme_gust"] = 1

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
	dlm=26;s06=30;dcape_mf=350;dcape_sf=500
	cond1 = ( (df["s06"]>=s06) & (df["dcape"]<dcape_sf) & (df["dlm"]>=dlm) & (df["dcape"] > 0) )
	cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>dcape_mf) & (df["dlm"]<dlm) )

	#4)
	#As above but with a non-zero MUCAPE for the SF case
	#dlm=26;s06=30;dcape_mf=350;dcape_sf=500
	#cond1 = ( (df["mu_cape"]>0) & (df["s06"]>=30) & (df["dcape"]<500) & (df["dlm"]>=26) & (df["dcape"] > 0) )
	#cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>250) & (df["dlm"]<26) )

	#5)
	#As in 3) but with a third, hybrid condition
	#cond1 = ( (df["s06"]>=30) & (df["dcape"]<500) & (df["dlm"]>=26) & (df["dcape"] > 0) )	#SF
	#cond2 = ( (df["ml_cape"]>120) & (df["dcape"]>250) & (df["dlm"]<=15) )	#MF
	#cond3 = ( (df["ml_cape"]>0) & (df["dcape"]>50) & (df["dlm"]>15) & (df["dlm"]<26) & (df["s06"]>20 ))	#HF

	#6)
	#As in 3 but with optimising values for CSI and HR over 60%
	#dlm=30;s06=35;dcape_mf=50;dcape_sf=400
	#cond1 = ( (df["s06"]>=s06) & (df["dcape"]<dcape_sf) & (df["dlm"]>=dlm) & (df["dcape"] > 0) )
	#cond2 = ( (df["ml_cape"]>175) & (df["dcape"]>dcape_mf) & (df["dlm"]<dlm) )

	#TEST
	hits = ((df.jdh==1) & (cond1 | cond2)).sum()
	misses = ((df.jdh==1) & ~(cond1 | cond2)).sum()
	fa = ((df.jdh==0) & (cond1 | cond2)).sum()
	cn = ((df.jdh==0) & ~(cond1 | cond2)).sum()
	hit_rate = hits / float(hits + misses)
	far = fa / float(cn + fa)
	csi = hits / float(hits+misses+fa)

def test_cond_6hr(df6,wind_events,append=False):

	#Take 6-hourly convective parameter data at station locations, and test the application of a conditional 
	# parameter on it. Conditional parameter is then resampled to daily intervals (asking is there a condition
	# met on that day) for comparison with the wind event dataset

	#If append == True, then overwrite the COND parameter in the saved, daily dataframe for ERA-Interim with
	# the COND defined in this function

	#SET COND
	mlm=26;s06=30;dcape_mf=350;dcape_sf=500
	sf = ( (df6["s06"]>=s06) & (df6["dcape"]<dcape_sf) & (df6["mlm"]>=mlm) )
	mf = ( (df6["ml_cape"]>120) & (df6["dcape"]>dcape_mf) & (df6["mlm"]<mlm) )
	cond = (sf | mf) * 1
	df6["sf"] = mf
	df6["mf"] = sf
	df6["cond"] = cond

	#RESAMPLE COND TO DAILY FOR COMPARISON WITH WIND EVENTS
	df = pd.DataFrame()
	for loc in np.unique(df6.loc_id):
		print(loc)
		temp_df = pd.DataFrame(df6[df6.loc_id==loc][["cond","sf","mf"]].resample("1D").max())
		temp_df["loc_id"] = loc
		df = pd.concat([df,temp_df])
	df = df.set_index("loc_id",append=True)
	df["jdh"] = wind_events.jdh
	df["wind_gust"] = wind_events.wind_gust
	
	#test metrics. First have to drop where there is no AWS data, as this is not part of the JDH dataset
	df = df.dropna(axis=0,subset=["wind_gust"])
	hits = ((df.jdh==1) & (df.cond==1)).sum()
	misses = ((df.jdh==1) & (df.cond==0)).sum()
	fa = ((df.jdh==0) & (df.cond==1)).sum()
	cn = ((df.jdh==0) & (df.cond==0)).sum()
	hit_rate = hits / float(hits + misses)
	far = fa / float(cn + fa)
	csi = hits / float(hits+misses+fa)
	print(hit_rate,far,csi)

	if append:
		print("APPENDING TO DAILY DATAFRAME...")
		df_daily = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"erai_points_1979_2017_daily_max.pkl").set_index(["date","loc_id"])
		df_daily["cond"] = df.cond
		df_daily["mf"] = df.mf
		df_daily["sf"] = df.sf
		df_daily.reset_index().to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
			"erai_points_1979_2017_daily_max.pkl")


if __name__ == "__main__":

	#6_HOURLY COND TESTING....
	#LOAD 6-HOURLY ERA-INTERIM DATA
	#df6 = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/erai_points_sa_small_1979_2017.pkl")
	df = analyse_events("jdh","sa_small","barra")
	#test_cond_6hr(df6,pd.DataFrame(df[["wind_gust","jdh"]]),False)

	test  = False
	plot = False
	plot_boxplot = False

	#environment_scatter2("sa_small","jdh","mmp","dcp")
	environment_scatter(df,"BARRA","jdh")
	
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

	if plot_boxplot:

		#Inspect station wind speed distributions for the two types of environmental conditions
		plt.figure(figsize=[8,5])
		#SF
		#sf = ( (df["s06"]>=30) & (df["mlm"]>=26) & (df["dcape"]<500) )
		#MF
		#mf = ( (df["ml_cape"]>120) & (df["dcape"]>350) & (df["mlm"]<26) )
		#df["sf"]=sf;df["mf"]=mf
		sf = df["sf"];mf=df["mf"]
		df=df.dropna(subset=["wind_gust"])
		plt.boxplot([df[(df.sf==1)]["wind_gust"].dropna(),df[(df.mf==1)]["wind_gust"].dropna(),\
			df[~((df.sf==1) | (df.mf==1))]["wind_gust"].dropna()],\
			labels=["Synoptic\nforcing\n"+"("+str(int(df.mf.sum()))+")",\
			"Mesoscale\nforcing\n"+"("+str(int(df.sf.sum()))+")",\
			"Neither condition\nmet\n"+"("+str((~((df.sf==1) | (df.mf==1))).sum())+")"],sym="+")
		plt.plot(np.ones(((df.sf==1)&(df.jdh==1)).sum())+0.1,df[(df.sf==1)&(df.jdh==1)]["wind_gust"].dropna(),\
			color="g",marker="^",linestyle="none")
		plt.plot(2*np.ones(((df.mf==1)&(df.jdh==1)).sum())+0.1,df[(df.mf==1)&(df.jdh==1)]["wind_gust"].dropna(),\
			color="g",marker="^",linestyle="none")
		plt.plot(3*np.ones(((~((df.sf==1) | (df.mf==1)))&(df.jdh==1)).sum())+0.1,df[(~((df.sf==1) | (df.mf==1)))&(df.jdh==1)]["wind_gust"].dropna(),\
			color="g",marker="^",linestyle="none")
		plt.ylabel("Daily maximum station\n wind gust (m.s$^{-1}$)",fontsize="xx-large")
		ax = plt.gca()
		ax.tick_params(labelsize="xx-large")
		#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/distributions/environmental_wind_gust_boxplot.png",bbox_inches="tight")
		plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/environmental_wind_gust_boxplot.tiff",bbox_inches="tight")

	if plot:
		#MAKE PLOTS
		environment_scatter(df,"All JDH Events","jdh",26,30,350,500)
		#Seasonal
		#environment_scatter(df[np.in1d(df.month,np.array([11,12,1,2,3,4]))],"November - April","jdh")
		#environment_scatter(df[np.in1d(df.month,np.array([5,6,7,8,9,10]))],"May - October","jdh")
		#Locations
		#environment_scatter(df[np.in1d(df.index.get_level_values(1),\
		#	np.array(["Adelaide AP","Mount Gambier","Edinburgh"]))],"Adelaide, Mt. Gambier, Edinburgh","jdh")
		#environment_scatter(df[np.in1d(df.index.get_level_values(1),\
			#np.array(["Woomera"]))],"Woomera","jdh")

