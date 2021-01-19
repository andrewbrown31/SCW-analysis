#For a number of diagnostics, plot the monthly count of measured SCW events against the monthly sum of an diagnostic, at 35 locations.

from matplotlib.ticker import MaxNLocator
from logit import *
from scipy.stats import pearsonr as rho
import matplotlib
matplotlib.rcParams.update({'font.size': 10})

np.random.seed(10)

def get_corr(x1,x2,N=10000):
	r=(rho(x1,x2)[0])
	r_resample = []
	for i in np.arange(N):
		r_resample.append(rho(np.random.choice(x1, size=len(x1), replace=True), np.random.choice(x2, size=len(x2), replace=True))[0])
	return r, (r_resample >= r).sum() / N

c1="tab:blue"
c2="k"

#Load daily dataframe with model and obs data
_, era5_aws, era5_sta = optimise_pss("/g/data/eg3/ab4502/ExtremeWind/points/"+
            "era5_allvars_v3_2005_2018.pkl", T=1000, compute=False, l_thresh=2,
            is_pss="hss", model_name="era5_v5", time="floor") 
era5_aws["year"] = pd.DatetimeIndex(era5_aws["hourly_floor_utc"]).year
era5_aws["month"] = pd.DatetimeIndex(era5_aws["hourly_floor_utc"]).month

#Fit statistical regression models to measured and reported (STA events)
era5_aws_preds = ["ebwd","Umean800_600","lr13","rhmin13","srhe_left","q_melting","eff_lcl"] 
era5_sta_preds = ["ebwd","Umean06","ml_cape","lr13"]
logit = LogisticRegression(class_weight="balanced", solver="liblinear",
                 max_iter=1000) 
era5_aws.loc[:,"logit"] = logit.fit(era5_aws[era5_aws_preds], era5_aws["is_conv_aws"]).predict_proba(era5_aws[era5_aws_preds])[:,1] 
era5_aws.loc[:,"logit_sta"] = logit.fit(era5_sta[era5_sta_preds], era5_sta["is_sta"]).predict_proba(era5_aws[era5_sta_preds])[:,1] 

#Plot results for the DCP
fig=plt.figure(figsize=[10,12])
plt.subplot(4,2,1)
v="dcp"
thresh=0.15
y1=[]; y2=[] 
for y in np.arange(2005,2019): 
         y1.append((era5_aws[(era5_aws.year==y)][v] >= thresh).sum()) 
         y2.append(era5_aws[(era5_aws.year==y)]["is_conv_aws"].sum()) 
m1=[]; m2=[] 
for m in np.arange(1,13): 
         m1.append((era5_aws[(era5_aws.month==m)][v] >= thresh).sum()) 
         m2.append(era5_aws[(era5_aws.month==m)]["is_conv_aws"].sum()) 
plt.plot(np.arange(1,13), m1, color=c1, marker="o"); plt.gca().grid(axis="both");plt.gca().tick_params(axis="y",labelcolor=c1); ax2=plt.gca().twinx(); ax2.plot(np.arange(1,13), m2, color=c2,marker="o")
plt.title("DCP")
plt.gca().set_xticks(np.arange(1,13,1))
plt.gca().set_xticklabels("")
plt.text(0.025, 0.1, 'a)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)

plt.subplot(4,2,2)
r, p = get_corr(y1,y2)
plt.title(("r={:.3f} ({:.3f})").format(r,p))
plt.plot(np.arange(2005,2019), y1, label=v, color=c1,marker="o");plt.gca().grid(axis="both"); plt.gca().tick_params(axis="y",labelcolor=c1);ax2=plt.gca().twinx(); ax2.plot(np.arange(2005,2019), y2, color=c2, label="SCW events",marker="o")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.text(0.025, 0.1, 'b)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.gca().set_xticklabels("")

#Plot results for SHERBE
plt.subplot(4,2,3)
v="eff_sherb"
thresh=0.47
y1=[]; y2=[] 
for y in np.arange(2005,2019): 
         y1.append((era5_aws[(era5_aws.year==y)][v] >= thresh).sum()) 
         y2.append(era5_aws[(era5_aws.year==y)]["is_conv_aws"].sum()) 
m1=[]; m2=[] 
for m in np.arange(1,13): 
         m1.append((era5_aws[(era5_aws.month==m)][v] >= thresh).sum()) 
         m2.append(era5_aws[(era5_aws.month==m)]["is_conv_aws"].sum()) 
plt.plot(np.arange(1,13), m1, color=c1, marker="o"); plt.gca().grid(axis="both");plt.gca().tick_params(axis="y",labelcolor=c1); ax2=plt.gca().twinx(); ax2.plot(np.arange(1,13), m2, color=c2,marker="o")
plt.title("SHERBE")
plt.gca().set_xticklabels("")
plt.gca().set_xticks(np.arange(1,13,1))
plt.text(0.025, 0.1, 'c)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.subplot(4,2,4)
r, p = get_corr(y1,y2)
plt.title(("r={:.3f} ({:.3f})").format(r,p))
plt.plot(np.arange(2005,2019), y1, label=v, color=c1,marker="o");plt.gca().grid(axis="both"); plt.gca().tick_params(axis="y",labelcolor=c1);ax2=plt.gca().twinx(); ax2.plot(np.arange(2005,2019), y2, color=c2, label="SCW events",marker="o")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.text(0.025, 0.1, 'd)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.gca().set_xticklabels("")

#Plot results for T-Totals
plt.subplot(4,2,5)
v="t_totals"
thresh=48.1
y1=[]; y2=[] 
for y in np.arange(2005,2019): 
         y1.append((era5_aws[(era5_aws.year==y)][v]>=thresh).sum()) 
         y2.append(era5_aws[(era5_aws.year==y)]["is_conv_aws"].sum()) 
m1=[]; m2=[] 
for m in np.arange(1,13): 
         m1.append((era5_aws[(era5_aws.month==m)][v] >= thresh).sum()) 
         m2.append(era5_aws[(era5_aws.month==m)]["is_conv_aws"].sum()) 
plt.plot(np.arange(1,13), m1, color=c1, marker="o"); plt.gca().grid(axis="both");plt.gca().tick_params(axis="y",labelcolor=c1); ax2=plt.gca().twinx(); ax2.plot(np.arange(1,13), m2, color=c2,marker="o")
plt.title("T-Totals")
plt.gca().set_xticks(np.arange(1,13,1))
plt.gca().set_xticklabels("")
plt.text(0.025, 0.1, 'e)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.subplot(4,2,6)
r, p = get_corr(y1,y2)
plt.title(("r={:.3f} ({:.3f})").format(r,p))
plt.plot(np.arange(2005,2019), y1, label=v, color=c1,marker="o");plt.gca().grid(axis="both"); plt.gca().tick_params(axis="y",labelcolor=c1);ax2=plt.gca().twinx(); ax2.plot(np.arange(2005,2019), y2, color=c2, label="SCW events",marker="o")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.text(0.025, 0.1, 'f)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.gca().set_xticklabels("")

#Plot results for MSTAT
plt.subplot(4,2,7)
v="logit"
thresh=0.83
y1=[]; y2=[]
for y in np.arange(2005,2019): 
         y1.append((era5_aws[(era5_aws.year==y)][v]>=thresh).sum()) 
         y2.append(era5_aws[(era5_aws.year==y)]["is_conv_aws"].sum()) 
m1=[]; m2=[] 
for m in np.arange(1,13): 
         m1.append((era5_aws[(era5_aws.month==m)][v] >= thresh).sum()) 
         m2.append(era5_aws[(era5_aws.month==m)]["is_conv_aws"].sum()) 
plt.plot(np.arange(1,13), m1, color=c1, marker="o"); plt.gca().grid(axis="both");plt.gca().tick_params(axis="y",labelcolor=c1); ax2=plt.gca().twinx(); ax2.plot(np.arange(1,13), m2, color=c2,marker="o")
plt.title("BDSD")
plt.gca().set_xticks(np.arange(1,13,1))
plt.gca().set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
plt.text(0.025, 0.1, 'g)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
plt.subplot(4,2,8)
ax1=plt.gca()
r, p = get_corr(y1,y2)
plt.title(("r={:.3f} ({:.3f})").format(r,p))
plt.plot(np.arange(2005,2019), y1, label=v, color=c1,marker="o");plt.gca().grid(axis="both"); plt.gca().tick_params(axis="y",labelcolor=c1);ax2=plt.gca().twinx(); ax2.plot(np.arange(2005,2019), y2, color=c2, label="SCW events",marker="o")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.text(0.025, 0.1, 'h)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
#plt.gca().set_xticklabels("")

#Plot results for RSTAT
#plt.subplot(5,2,9)
#plt.xlabel("Month")
#v="logit_sta"
#thresh=0.77
#y1=[]; y2=[]
#for y in np.arange(2005,2019): 
 #        y1.append((era5_aws[(era5_aws.year==y)][v]>=thresh).sum()) 
  #       y2.append(era5_aws[(era5_aws.year==y)]["is_conv_aws"].sum()) 
#m1=[]; m2=[] 
#for m in np.arange(1,13): 
#         m1.append((era5_aws[(era5_aws.month==m)][v] >= thresh).sum()) 
#         m2.append(era5_aws[(era5_aws.month==m)]["is_conv_aws"].sum()) 
#plt.plot(np.arange(1,13), m1, color=c1, marker="o"); plt.gca().grid(axis="both");plt.gca().tick_params(axis="y",labelcolor=c1); ax2=plt.gca().twinx(); ax2.plot(np.arange(1,13), m2, color=c2,marker="o")
#plt.title("RSTAT")
#plt.gca().set_xticks(np.arange(2,14,2))
#plt.text(0.025, 0.1, 'i)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)
#plt.subplot(5,2,10)
#ax1=plt.gca()
#plt.xlabel("Year")
#r, p = get_corr(y1,y2)
#plt.title(("r={:.3f} ({:.3f})").format(r,p))
#plt.plot(np.arange(2005,2019), y1, label=v, color=c1,marker="o");plt.gca().grid(axis="both"); plt.gca().tick_params(axis="y",labelcolor=c1);ax2=plt.gca().twinx(); ax2.plot(np.arange(2005,2019), y2, color=c2, label="SCW events",marker="o")
#ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
#plt.text(0.025, 0.1, 'j)', horizontalalignment='left',verticalalignment='center',transform=plt.gca().transAxes, size=12)

ax2.plot([0,0],[1,0], color="tab:blue", label="Diagnostic");
ax2.set_xlim([2005,2018])
plt.legend(bbox_to_anchor=(0.3,-0.5),ncol=2)

plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.2)
fig.text(0.025, 0.5, "Number of environments", ha="center", va="center", rotation=90, color=c1)
fig.text(0.975, 0.5, "Number of events", ha="center", va="center", rotation=90, color=c2)

plt.savefig("/g/data/eg3/ab4502/figs/ExtremeWind/era5_variability.png", bbox_inches="tight")
