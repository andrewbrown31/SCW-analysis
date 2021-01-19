#Use code from working/ExtremeWind to get skill scores from ERA5 for SCW reports. Get
#   -> Optimal HSS and corresponding threshold, with a minimum POD of 2/3
#   -> POD and FAR using this threshold
#   -> AUC
#All scores should have a 95% confidence interval

import numpy as np
import pandas as pd
from event_analysis import optimise_pss
from plot_param import add_confidence_bounds, resample_events
from event_analysis import calc_auc, pss
import multiprocessing
import itertools

def auc_test(df, v_list, N):

	auc = pd.DataFrame(columns=["auc","auc_ci"], index=v_list)

	for v in v_list:

		a, c1, c2 = calc_auc((df), v, "is_conv_aws", N)
		auc.loc[v, "auc"] = a; auc.loc[v, "auc_ci"] = "["+str(round(c1,3)) + ", " + str(round(c2, 3))+"]"

	return auc

def calc_far(df, v_list, thresh_df, event):

	#False alarm ratio

	out = []
	for v in v_list:
		hits = ((df[v] >= thresh_df[v]) & (df[event]==1)).sum()
		fa = ((df[v] >= thresh_df[v]) & (df[event]==0)).sum()
		out.append(fa / (hits+fa))
	return np.array(out), pd.DataFrame({"far":out}, index=v_list)

def calc_pod(df, v_list, thresh_df, event):

	out = []
	hits_out = []
	misses_out = []
	fa_out = []
	cn_out = []
	for v in v_list:
		hits = ((df[v] >= thresh_df[v]) & (df[event]==1)).sum()
		fa = ((df[v] >= thresh_df[v]) & (df[event]==0)).sum()
		misses = ((df[v] < thresh_df[v]) & (df[event]==1)).sum()
		cn = ((df[v] < thresh_df[v]) & (df[event]==0)).sum()
		out.append(hits / (hits+misses))
		hits_out.append(hits)
		misses_out.append(misses)
		fa_out.append(fa)
		cn_out.append(cn)
	return np.array(out), pd.DataFrame({"pod":out}, index=v_list), pd.DataFrame({"hits":hits_out, "misses":misses_out, "false_alarms":fa_out,\
		    "correct_negatives":cn_out}, index=v_list)

if __name__ == "__main__":

    #Settings
    N=1000	    #Bootstrap

    #Load HSS and thresholds for ERA5
    hss, df_aws, df_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/era5_allvars_v3_2005_2018.pkl",T=1000,\
		compute=True, l_thresh=2, is_pss="hss", model_name="era5_v5",time="floor")
    df_aws["logit"] = 1 / ( 1 + np.exp( -(
            df_aws["ebwd"] * 6.1e-2
            + df_aws["Umean800_600"] * 1.5e-1
            + df_aws["lr13"] * 9.4e-1
            + df_aws["rhmin13"] * 3.9e-2
            + df_aws["srhe_left"] * 1.7e-2
            + df_aws["q_melting"] * 3.8e-1
            + df_aws["eff_lcl"] * 4.7e-4
            - 1.3e+1) ) )
    df_aws["logit_sta"] = 1 / ( 1 + np.exp( -(
            df_aws["ebwd"] * 1.3e-1
            + df_aws["Umean06"] * 1.7e-1
            + df_aws["ml_cape"] * 1.6e-3
            + df_aws["lr13"] * 4.1e-1
            - 5.6) ) )

    #Calculate optimal HSS for each logistic regression model, and add to the HSS dataframe for all other diagnostics
    #Measured model
    pool = multiprocessing.Pool()
    temp_df = df_aws.loc[:,["is_conv_aws","logit"]]
    iterable = itertools.product(np.linspace(0,1,100), [temp_df], ["logit"], ["is_conv_aws"], ["hss"], [0.67])
    res = pool.map(pss, iterable)
    hss.loc["logit","pss_conv_aws"] = np.max([res[i][0] for i in np.arange(len(res))])
    hss.loc["logit","threshold_conv_aws"] = [res[i][1] for i in np.arange(len(res))][np.argmax([res[i][0] for i in np.arange(len(res))])]
    #Reported (STA) model
    temp_df = df_aws.loc[:,["is_conv_aws","logit_sta"]]
    iterable = itertools.product(np.linspace(0,1,100), [temp_df], ["logit_sta"], ["is_conv_aws"], ["hss"], [0.67])
    res = pool.map(pss, iterable)
    hss.loc["logit_sta","pss_conv_aws"] = np.max([res[i][0] for i in np.arange(len(res))])
    hss.loc["logit_sta","threshold_conv_aws"] = [res[i][1] for i in np.arange(len(res))][np.argmax([res[i][0] for i in np.arange(len(res))])]

    #Drop unnecessary columns
    v_list = ["stn_name","hourly_floor_utc","lat","lon","dcp","mucape*s06","mlcape*s06","t_totals","eff_sherb","sbcape*s06","mmp","effcape*s06","scp_fixed",\
	"sweat","ship","ml_cape","k_index","gustex","sb_cape","sherb","eff_cape","mu_cape","mwpi_ml","logit"]
    df_aws = df_aws[v_list+["is_conv_aws"]].reset_index().drop(columns="index")
    hss = hss.loc[v_list]
    df_aws.rename(columns={"dcp":"DCP","mucape*s06":"MUCS6","mlcape*s06":"MLCS6","t_totals":"Total-Totals","eff_sherb":"SHERBE","sbcape*s06":"SBCS6","mmp":"MMP","effcape*s06":"Eff-CS6","scp_fixed":"SCP","sweat":"SWEAT","ship":"SHIP","ml_cape":"MLCAPE","k_index":"K-index","gustex":"GUSTEX","sb_cape":"SBCAPE","sherb":"SHERB","eff_cape":"Eff-CAPE","mu_cape":"MUCAPE","mwpi_ml":"MWPI","logit":"BDSD","is_conv_aws":"SCW","hourly_floor_utc":"time","stn_name":"location"}).set_index("time").to_csv("/g/data/eg3/ab4502/ExtremeWind/brown_dowdy_jgr-a_daily_data.csv")
    df_aws = df_aws.drop(columns=["stn_name","hourly_floor_utc","lat","lon"])

    #Add 95% confidence intervals on HSS
    hss = add_confidence_bounds(hss, df_aws, "is_conv_aws", "conv_aws", N, 100)
    
    #Get the AUC
    auc = auc_test(df_aws, v_list, N)

    #Merge HSS, AUC and POD
    _,pod,cont_tbl = calc_pod(df_aws, v_list, hss["threshold_conv_aws"], "is_conv_aws")
    skill = pd.concat([hss[["threshold_conv_aws","pss_conv_aws","conv_aws_upper","conv_aws_lower"]], auc, pod],axis=1)

    #Get confidence intervals on POD
    x,y = resample_events(df_aws, "is_conv_aws", N, 100) 
    resampled_pod = []
    for i in np.arange(len(x)):
	    resample_temp_df = df_aws.iloc[np.append(x[i], y[i])]	
	    out, _, _ = calc_pod(resample_temp_df, v_list, hss["threshold_conv_aws"], "is_conv_aws")
	    resampled_pod.append(out)
    pod_low = pd.DataFrame({"pod_low":np.percentile(np.stack(resampled_pod),2.5, axis=0)}, index=v_list)
    pod_up = pd.DataFrame({"pod_up":np.percentile(np.stack(resampled_pod),97.5, axis=0)}, index=v_list)
    skill = pd.concat([skill, pod_low, pod_up], axis=1)

    #Fix up dataframe and save
    skill["hss_ci"] = "[" + round(skill["conv_aws_lower"],3).astype(str) + ", " + round(skill["conv_aws_upper"],3).astype(str) + "]"
    skill["pod_ci"] = "[" + round(skill["pod_low"],3).astype(str) + ", " + round(skill["pod_up"],3).astype(str) + "]"
    skill = skill.rename(columns={"pss_conv_aws":"hss", "threshold_conv_aws":"threshold"}).sort_values("hss", ascending=False)[["threshold","hss","hss_ci","auc","auc_ci","pod","pod_ci"]]
    skill["threshold"] = skill["threshold"].astype(float)
    skill["hss"] = skill["hss"].astype(float)
    skill["auc"] = skill["auc"].astype(float)
    skill["pod"] = skill["pod"].astype(float)
    skill = skill.round(3)
    skill.to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/era5_cmip_aws.csv")
    cont_tbl.sort_values("false_alarms").to_csv("/g/data/eg3/ab4502/ExtremeWind/skill_scores/cont_tbl_aws.csv")
