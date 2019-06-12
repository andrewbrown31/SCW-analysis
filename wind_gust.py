#Load point-based daily max wind gust data (reanalyisis and aws) and compare

from scipy.stats import spearmanr as rho
from statsmodels.distributions.empirical_distribution import ECDF
from event_analysis import *
from plot_param import *

def load_wind_gusts(include_barra,remove_incomplete_years='True'):
	#Load dataframes
	aws = pd.read_pickle("/short/eg3/ab4502/ExtremeWind/aws/all_daily_max_wind_gusts_sa_1979_2017.pkl")
	if remove_incomplete_years:
		aws = remove_incomplete_aws_years(aws,"Port Augusta")
	erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
		"erai_fc_points_1979_2017_daily_max.pkl")

	#Combine ERA-Interim, AWS and BARRA-R daily max wind gusts 
	if include_barra:
	    barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
			"barra_r_fc_points_daily_2003_2016.pkl")
	    barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
			"barra_ad_points_daily_2006_2016.pkl")
	    combined = pd.concat([aws.set_index(["date","stn_name"]).\
		rename(columns={"wind_gust":"aws_gust"})\
		,erai.set_index(["date","loc_id"]).rename(columns={"wg10":"erai_gust"}).erai_gust,\
		barra_ad.reset_index().set_index(["date","loc_id"]).\
		rename(columns={"max_max_wg10":"barra_ad_max_gust",\
		"max_wg10":"barra_ad_gust"})[["barra_ad_max_gust","barra_ad_gust"]],
		barra_r.reset_index().set_index(["date","loc_id"]).\
		rename(columns={"max_wg10":"barra_r_gust"}).barra_r_gust,\
		],axis=1).dropna()
	else:
	    combined = pd.concat([aws.set_index(["date","stn_name"]).\
		rename(columns={"wind_gust":"aws_gust"})\
		,erai.set_index(["date","loc_id"]).rename(columns={"wg10":"erai_gust"}).erai_gust,]\
		,axis=1).dropna()

	return combined

def resample_barra():
	#Take BARRA-R and BARRA-AD data at hourly frequency. Resample to daily max and save 
	#(HAS BEEN SAVED, NOW JUST LOAD)
	
	barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"+\
		"barra_r_fc_points_2006_2016.pkl")
	barra_r = eliminate_wind_gust_spikes(barra_r.set_index(["date","loc_id"]),"max_wg10")\
			.reset_index()
	barra_r_1d = pd.DataFrame()
	stns = np.unique(barra_r.loc_id)
	for stn in stns:
		print(stn)
		barra_r_1d = barra_r_1d.append(barra_r[barra_r.loc_id==stn].\
			set_index("date").resample("1D").max())
	barra_r_1d.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
		"barra_r_fc_points_daily_2006_2016.pkl")

	#barra_r = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"+\
	#	"barra_ad_points_mean_2006_2016.pkl")
	#barra_r = eliminate_wind_gust_spikes(barra_r.set_index(["date","loc_id"]),"max_wg10")\
	#		.reset_index()
	#barra_r_1d = pd.DataFrame()
	#stns = np.unique(barra_r.loc_id)
	#for stn in stns:
	#	print(stn)
	#	barra_r_1d = barra_r_1d.append(barra_r[barra_r.loc_id==stn].\
	#		set_index("date").resample("1D").max())
	#barra_r_1d.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"\
	#	"barra_ad_points_mean_daily_2006_2016.pkl")

#	barra_ad = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/barra_ad_points_2006_2016.pkl")
#	barra_ad_1d = pd.DataFrame()
#	stns = np.unique(barra_ad.loc_id)
#	for stn in stns:
#		print(stn)
#		barra_ad_1d = barra_ad_1d.append(barra_ad[barra_ad.loc_id==stn].\
#			set_index("date").resample("1D").max())
#	barra_ad_1d.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_ad/"\
#		"barra_ad_points_daily_2006_2016.pkl")

def eliminate_wind_gust_spikes(df,mod_name):

	#Take an hourly gust dataframe. Consider all gusts above 40 m/s (likely to be a wind gust 
	# spike). If the previous and next hour are both below 20 m/s, disregard the gust. Replace
	# with the mean of the previous and next hour

	gust_df = df[df[mod_name]>=40]
	for i in np.arange(0,gust_df.shape[0]):
		prev_gust = df.loc[(gust_df.index[i][0]+dt.timedelta(hours=-1),gust_df.index[i][1])][mod_name]
		next_gust = df.loc[(gust_df.index[i][0]+dt.timedelta(hours=1),gust_df.index[i][1])][mod_name]
		if (prev_gust < 20) & (next_gust < 20):
			df.loc[((gust_df.index[i][0],gust_df.index[i][1]),mod_name)] = \
				(prev_gust + next_gust) / 2
	return df

def quantile_match(combined,obs_name,model_name):
	#Quantile matching
	#Note that currently, scenario is the same data as used to construct the observed 
	#	distribution (i.e. no cross-validation)
	#Note that model is matched to a combined distribution of all AWS stations

	#Create cumuliative distribution functions
	obs_cdf = ECDF(combined[obs_name])
	model_cdf = ECDF(combined[model_name])
	obs_invcdf = np.percentile(obs_cdf.x,obs_cdf.y*100)

	#Match model wind gust to the observed quantile distribution
	scenario_model = combined[model_name]
	scenario_obs = combined[obs_name]
	model_p = np.interp(scenario_model,model_cdf.x,model_cdf.y)
	model_xhat = np.interp(model_p,obs_cdf.y,obs_invcdf)

	#The 0th percentile wind gust is always matched to a NaN (due to interp function). Match
	# to the min observed instead
	model_xhat[np.isnan(model_xhat)] = obs_cdf.x[1]
	
	#Add quantile matched values to dataframe
	combined[(model_name + "_qm")] = model_xhat
	
	return combined

def plot_scatter(combined,model_list,name_list,location=False):
	import matplotlib.colors as colors
	fig = plt.figure(figsize=[15,7])
	if location != False:
		combined = combined[combined.index.get_level_values(1) == location]

	for n in np.arange(0,len(model_list)):
		plt.subplot(1,len(model_list),n+1)
		mod_rho = rho(combined["aws_gust"],combined[model_list[n]]).correlation
		bias = np.mean(combined[model_list[n]]) / np.mean(combined["aws_gust"])
		mae = np.mean(combined[model_list[n]] - combined["aws_gust"])
		rmse = np.sqrt(np.mean(np.power((combined[model_list[n]] - combined["aws_gust"]),2)))
		leg = "r = " + str(round(mod_rho,3)) + "\nMAE = " + str(round(mae,3)) + "\nRMSE = "+\
				str(round(rmse,3))
		#plt.scatter(combined["aws_gust"],combined[model_list[n]],label=leg)
		plt.scatter(combined[combined["aws_gust"]>=30]["aws_gust"],\
				combined[combined["aws_gust"]>=30][model_list[n]],color="k",marker="o")
		h = plt.hist2d(combined["aws_gust"],combined[model_list[n]],\
				bins=20,range=[[0,40],[0,40]],norm=colors.LogNorm(.1,1000),cmap=plt.get_cmap("Greys",8))
		#plt.legend(numpoints=0,fontsize="xx-large",loc=2)
		plt.plot([0,50],[0,50],"r")
		plt.xlim([0,40])
		plt.ylim([0,40])
		ax = plt.gca()
		ax.tick_params(labelsize="xx-large")
		ax.annotate(leg, xy=(2,39), xycoords='data',size="xx-large",ha="left",va="top",\
				bbox=dict(boxstyle='round', fc='w'))
		if n==1:
			ax.set_title(location,fontsize="xx-large")
		if location != "Mount Gambier":
			ax = plt.gca()
			ax.set_xticklabels("")			
		else:
			ax = plt.gca()
			ax.set_xticks([0,10,20,30,40])			
			ax.set_xticklabels(['0','10','20','30','40'])			
	   		cax = fig.add_axes([0.2,0.05,0.6,0.01])
	   		cb = plt.colorbar(h[-1],cax,orientation="horizontal",extend="max")
	   		cb.ax.tick_params(labelsize="xx-large")
		if n>0:
			ax = plt.gca()
			ax.set_yticklabels("")			
	#plt.savefig("/short/eg3/ab4502/figs/ExtremeWind/scatter_"+location+".tiff",\
	#	bbox_inches="tight")
	#plt.savefig("/home/548/ab4502/working/ExtremeWind/figs/scatter/aws/"+location+".png",\
	#	bbox_inches="tight")
	plt.show()
    plt.close()

def plot_monthly_dist(combined,gust_list,threshold):
	for n in np.arange(0,len(gust_list)):
		cnt, months = get_monthly_dist(combined.rename(columns={gust_list[n]:"wind_gust"})\
				,20)
		plt.plot(months,cnt,label = gust_list[n])
		plt.legend(loc=2)
	plt.show()

def plot_extreme_dist(combined,model_list,threshold,bins,log=False,normed=True):

	#For days with high AWS gusts (defined by threshold), what do re-analysis dists look like

	data = list()
	for n in np.arange(0,len(model_list)):
		df = combined[combined[model_list[n]]>=threshold]
		data.append(df[model_list[n]].values)

	plt.hist(data,histtype="bar",bins=bins,label=model_list,\
			normed=normed,range=[0,50],log=log)
	#plt.axvline(x=df.aws_gust.mean(),color="k",linestyle="--",label="Observed Mean")
	plt.legend()
	#plt.axvline(x=threshold,color="k")
	plt.xlim([threshold,50])
	plt.show()

def plot_extreme_dist_loc(combined,model_list,locations,threshold,bins,log=False):

	#For days with high AWS gusts (defined by threshold), what do re-analysis dists look like
	#Do for 4 locations
	
	combined = combined.reset_index().rename(columns={"level_0":"date","level_1":"loc_id"})
	for l in np.arange(0,len(locations)):
	    plt.subplot(2,2,l+1)
	    data = list()
	    for n in np.arange(0,len(model_list)):
		df = combined[(combined[model_list[n]]>=threshold)&(combined.loc_id==locations[l])]
		data.append(df[model_list[n]].values)
	    plt.hist(data,histtype="bar",bins=bins,label=model_list,\
			normed=False,range=[0,50],log=log)
	    plt.title(locations[l])
	    #plt.axvline(x=df.aws_gust.mean(),color="k",linestyle="--",label="Observed Mean")
	    #plt.axvline(x=threshold,color="k")
	    plt.xlim([threshold,45])
	plt.legend()
	plt.show()

def test_dist_assumption(combined,gust_type,stns=False,threshold=0):

	#If we are to use quantile matching between AWS data and each model, and apply to a gridded
	# dataset, then an underlying assumption is that the distribution of wind gust is the same
	# accross the entire state, and is the same as the distribution of the combined AWS data

	#We can test this by lookig at the mean/variance/max for each station
	#Producing a map of mean/variance for the sa_small domain in BARRA-R

	combined = combined.reset_index().rename(columns={"level_0":"date","level_1":"stn_name"})
	gust_type.append("aws_gust")
	cols = ["r","b"]
	if stns == False:
		stns = combined.stn_name.unique()
	plt.figure()
	aws = list()
	stn_lab = list()
	plot_cols = list()
	for s in np.arange(0,len(stns)):
	   for n in np.arange(0,len(gust_type)):
		aws.append(combined[(combined.stn_name == stns[s]) & \
			(combined[gust_type[n]]>=threshold)][gust_type[n]].values)
		if n == 0:
			stn_lab.append(stns[s])
		else:
			stn_lab.append("")
		plot_cols.append(cols[n])
	
	for n in np.arange(0,len(gust_type)):
		if n == 0:
			stn_lab.append("Total")
		else:
			stn_lab.append("")
		aws.append(combined[combined[gust_type[n]]>=threshold][gust_type[n]].values)
		plot_cols.append(cols[n])
		plt.plot([0,1],[0,1],color=cols[n],label=gust_type[n])
	plt.legend()
	bp = plt.boxplot(aws,whis=1.5,labels=stn_lab)
	cnt = 0
	for patch in bp["boxes"]:
		patch.set(color=plot_cols[cnt])
		cnt=cnt+1
	plt.xticks(rotation="vertical")
	plt.subplots_adjust(bottom=0.30)
	#plt.title()
	plt.ylim([threshold-1,50])
	plt.axhline(30,color="grey")

	plt.show()

if __name__ == "__main__":
	df = load_wind_gusts(True,remove_incomplete_years=False)
	#test_dist_assumption(df,["barra_r_gust"],stns=False,threshold=20)
	[plot_scatter(df,["erai_gust","barra_r_gust","barra_ad_gust"],["ERA-Interim","BARRA-R","BARRA-AD"],\
		location=l) for l in ["Adelaide AP","Woomera","Port Augusta","Mount Gambier"]]
