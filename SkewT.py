import matplotlib
from scipy.interpolate import interp1d
from numpy import (ma, array, linspace, logspace, log, cos, sin, pi, zeros,
                   exp, arange, trapz, where, concatenate, nan, isnan, argsort, flip,
                   log10, meshgrid, interp, empty, moveaxis, copy, tile, arange, isnan, stack, nan, argmax,
                   nanargmax, nanmin, nanmax)
try:
	import wrf
except:
	pass
import wrf

# Local thermodynamics stuff, see thermodynamics.py
try:
    from .thermodynamics import (
        VirtualTemp, Latentc, VaporPressure, MixRatio, GammaW,
        VirtualTempFromMixR, MixR2VaporPress, DewPoint, Theta, TempK,
        Density, DensHumid, ThetaE, ThetaV, barometric_equation_inv)
    from .thermodynamics import Rs_da, Cp_da, Epsilon, degCtoK
except:
    from thermodynamics import (
        VirtualTemp, Latentc, VaporPressure, MixRatio, GammaW,
        VirtualTempFromMixR, MixR2VaporPress, DewPoint, Theta, TempK,
        Density, DensHumid, ThetaE, ThetaV, barometric_equation_inv)
    from thermodynamics import Rs_da, Cp_da, Epsilon, degCtoK

# SkewT version
__version__ = "1.2.0"

def get_dcape(startp, startt, startq, z, p_sfc, p_lvl=True, p=None, sfc_included=False,\
	    minthetae_inds=None):
    """Wrapper for the numerics of calculating DCAPE.

    This function was adapted from get_cape.
    
    INPUTS:
    startp,startt,startdp: Definition of the parcel that we will base
		       the calculations on. This can be the output
		       of Sounding.get_parcel() or it can be a user-
		       defined parcel.
    
    OUTPUTS:
    DCAPE                 : CAPE calculated from temperature
    
    HINT:
    parcel=S.get_parcel('mu')
    lcl,lfc,el,cape,cin=get_cape(*parcel)
    """

    pres = copy(startp)
    temp = copy(startt)

    # Take a parcel from each start point down to well below the surface pressure (1100) using wet bulb temperature
    # (which assumes the parcel is saturated)
    if sfc_included:	#This is what is used by obs_read.py for UA data
       startt = (array(wrf.wetbulb(startp*100, startt+273.15, startq)) - 273.15).T
       preswet, tempwet = moist_ascent(startp, startt, ptop=p_sfc, nsteps=51,dcape=True)
    else:		#For 3d model data
       startt = (array(wrf.wetbulb(startp*100, startt+273.15, startq)) - 273.15)
       preswet, tempwet = moist_ascent(startp, startt, ptop=1100, nsteps=51,dcape=True)
    tparcel = tempwet
    pparcel = preswet

    # Interpolate the environmental profile onto the parcel coordinates
    if p_lvl:
    #If the 3d model data is on pressure levels...
	    tempenv = empty(pparcel.shape)
	    hgtenv = empty(pparcel.shape)
	    DCAPE = zeros((pparcel.shape[0],pparcel.shape[2],pparcel.shape[3]))
	    if sfc_included:
	       f_temp = interp1d(p, temp, axis=0)
	       f_z = interp1d(p, z, axis=0)
	    else:
	       f_temp = interp1d(concatenate([[1100],p]), temp, axis=0)
	       f_z = interp1d(concatenate([[1100],p]), z, axis=0)
	    for i in arange(0,pparcel.shape[0]):
	       if pparcel[i,0,0,0] < 300:
		       pparcel[i,0,0,0] = 300
	       if pparcel[i,-1,0,0] > 1100:
		       pparcel[i,-1,0,0] = 1000
	       tempenv[i] = f_temp(pparcel[i,:,0,0])
	       hgtenv[i] = f_z(pparcel[i,:,0,0])

	    #Mask where the parcel is below ground level and when the parcel is warmer than
	    # the environment
	    xcond = (pparcel < p_sfc)
	    cond = (tparcel < tempenv)

	    y = 9.81*(tparcel-tempenv)/(tempenv + 273.15)
	    x = hgtenv
	    y[~(cond)] = 0
	    x[~(xcond)] = 0
	    
	    DCAPE = trapz(y,x,axis=1)
	    DCAPE[isnan(DCAPE)] = 0

	    if sfc_included:
	       return DCAPE
	    else:
	       tparcel[pparcel > p_sfc] = nan
	       ddraft_temp = nanmax(tparcel,axis=1)
	       return [ DCAPE, ddraft_temp]
    else:
    #If the 3d model data is not on pressure levels...
    #Only do this for levels with minimum thetae
	    tempenv = empty(pparcel[0].shape)
	    hgtenv = empty(pparcel[0].shape)
	    pparcel2 = empty(pparcel[0].shape)
	    tparcel2 = empty(pparcel[0].shape)
	    DCAPE = zeros((pparcel.shape[2],pparcel.shape[3]))
	    for i in arange(pparcel.shape[2]):
		    for j in arange(pparcel.shape[3]):
			    tempenv[:,i,j] = interp(pparcel[minthetae_inds[i,j], :, i, j], \
				    flip(pres[:, i, j]) ,\
				    flip(temp[:, i, j]) )
			    hgtenv[:,i,j] = interp(pparcel[minthetae_inds[i,j], :, i, j], \
				    flip(pres[:, i, j]) ,\
				    flip(z[:, i, j]) )
			    pparcel2[:,i,j] = pparcel[minthetae_inds[i,j],:,i,j]
			    tparcel2[:,i,j] = tparcel[minthetae_inds[i,j],:,i,j]
			    
	    tparcel = tparcel2
	    pparcel = pparcel2

	    xcond = (pparcel < p_sfc)
	    cond = (tparcel < tempenv)
	    y = 9.81*(tparcel-tempenv)/(tempenv + 273.15)
	    x = hgtenv
	    y[~(cond)] = 0
	    x[~(xcond)] = 0
	    DCAPE = trapz(y,x,axis=0)
	    DCAPE[isnan(DCAPE)] = 0
	    tparcel[pparcel > p_sfc] = nan
	    ddraft_temp = nanmax(tparcel,axis=0)
	    return [ DCAPE, ddraft_temp]


def get_cape(startp, startt, startdp, z, p, totalcape=False):
    """Wrapper for the numerics of calculating CAPE.
    
    INPUTS:
    startp,startt,startdp: Definition of the parcel that we will base
		       the calculations on. This can be the output
		       of Sounding.get_parcel() or it can be a user-
		       defined parcel.
    totalcape [=False]   : Flag defining method of identifying the so-
		       called "Equilibrium Level" (Reference).
		       If False  (default), use the first stable
		       layer above the LFC, and ignore any CAPE in
		       unstable layers above this. If True, use all
		       CAPE up to the highest equilibrium level.
    
    OUTPUTS:
    P_lcl                : The lifted condensation level (LCL)
    P_lfc                : The level of free convection (LFC). Can be
		       the same as the LCL, or can be NaN if there
		       are no unstable layers.
    P_el                 : The Equilibrium Level, used to determine the
		       CAPE. If totalcape=True, use the highest
		       equilibrium level, otherwise use the first
		       stable equilibrium level above the LFC.
    CAPE                 : CAPE calculated from virtual temperature
    CIN                  : CIN calculated from virtual temperature
    
    HINT:
    parcel=S.get_parcel('mu')
    lcl,lfc,el,cape,cin=get_cape(*parcel)
    """

#Limit to parcels under 300 hPa
    parcel_pind = p >= 300
    pres = copy(startp)
    temp = copy(startt)
    startp = startp[parcel_pind]
    startt = startt[parcel_pind]
    startdp = startdp[parcel_pind]
    
	#startp = startp*100
	#pres = pres*100

	#assert (startt >= startdp).any(), "Not a valid parcel. Check Td<Tc"

	# fundamental environmental variables
#        pres = self.soundingdata['pres']
#        temp = self.soundingdata['temp']

	# Get Sub-LCL traces
    presdry, tempdry, tempiso = dry_ascent(
    startp, startt, startdp, nsteps=101)

	# make lcl variables explicit
    P_lcl = copy(presdry[:,-1,:,:])
    T_lcl = copy(tempdry[:,-1,:,:])

	# Now lift a wet parcel from the intersection point
	# preswet=linspace(P_lcl,100,101)
    preswet, tempwet = moist_ascent(P_lcl, T_lcl, nsteps=101)

	# tparcel is the concatenation of tempdry and
	# tempwet, and so on.
    tparcel = concatenate((tempdry, tempwet[:,1:,:,:]),axis=1)
    pparcel = concatenate((presdry, preswet[:,1:,:,:]),axis=1)
    dpparcel = concatenate((tempiso, tempwet[:,1::,:]),axis=1)

	# Interpolating the environmental profile onto the
	# parcel pressure coordinate
	# tempenv=interp(preswet,pres[::-1],temp[::-1])
	# NEW, for total column
    tempenv = empty(pparcel.shape)
    hgtenv = empty(pparcel.shape)
    CAPE = empty((P_lcl.shape))
    CIN = empty((P_lcl.shape))
    P_el = empty((P_lcl.shape))
    P_lfc = empty((P_lcl.shape))
    for i in arange(0,pparcel.shape[0]):
      for j in arange(0,pparcel.shape[2]):
         for k in arange(0,pparcel.shape[3]):
                if isnan(pparcel[i,::-1,j,k]).sum():
                   eqlev = array([])
                   stab = array([])
                else:
                   tempenv[i,:,j,k] = interp(pparcel[i,:,j,k], pres[::-1,j,k], temp[::-1,j,k])
                   hgtenv[i,:,j,k] = interp(pparcel[i,:,j,k], pres[::-1,j,k], z[::-1,j,k])
                   eqlev, stab = solve_eq(pparcel[i,:,j,k][pparcel[i,:,j,k] <= P_lcl[i,j,k]][::-1],
                       (tparcel[i,:,j,k]-tempenv[i,:,j,k])[pparcel[i,:,j,k] <= P_lcl[i,j,k]][::-1])

		# now solve for the equlibrium levels above LCL
		# (all of them, including unstable ones)
		# eqlev,stab=solve_eq(preswet[::-1],(tempwet-tempenv)[::-1])
		# NEW, for total column:
		# On second thought, we don't really want/need
		# any equilibrium levels below LCL
		# eqlev,stab=solve_eq(pparcel[::-1],(tparcel-tempenv)[::-1])
		# This is equivalent to the old statement :


		# Sorting index by decreasing pressure
                I = argsort(eqlev)[::-1]
                eqlev = eqlev[I]
                stab = stab[I]

		# temperatures at the equilibrium level
		# tempeq=interp(eqlev,preswet[::-1],tempenv[::-1])
		# NEW, for total column:
                tempeq = interp(eqlev, pparcel[i,::-1,j,k], tparcel[i,::-1,j,k])

		# This helps with debugging
		# for ii,eq in enumerate(eqlev):
		# print "%5.2f  %5.2f  %2d"%(eq,tempeq[ii],stab[ii])

		# need environmental temperature at LCL
                tenv_lcl = interp(P_lcl[i,j,k], pparcel[i,::-1,j,k], tempenv[i,::-1,j,k])

                isstab = where(stab == 1., True, False)
                unstab = where(stab == 1., False, True)

                if eqlev.shape[0] == 0:
		    # no unstable layers in entire profile
		    # because the parcel never crosses the tenv
                    P_lfc[i,j,k] = nan
                    P_el[i,j,k] = nan
                elif T_lcl[i,j,k] > tenv_lcl:
		    # check LCL to see if this is unstable
                    P_lfc[i,j,k] = P_lcl[i,j,k]
                    if totalcape is True:
                        P_el[i,j,k] = eqlev[isstab][-1]
                    else:
                        P_el[i,j,k] = eqlev[isstab][0]
                elif eqlev.shape[0] > 1:
		    # Parcel is stable at LCL so LFC is the
		    # first unstable equilibrium level and
		    # "EQ" level is the first stable equilibrium
		    # level
                    P_lfc[i,j,k] = eqlev[unstab][0]
                    if totalcape is True:
                        P_el[i,j,k] = eqlev[isstab][-1]
                    else:
                        P_el[i,j,k] = eqlev[isstab][0]
                else:
		    # catch a problem... if there is only
		    # one eqlev and it's unstable (this is
		    # unphysical), then it could be a vertical
		    # resolution thing. This is a kind of
		    # "null" option
                    if isstab[0]:
                        P_el[i,j,k] = eqlev[isstab][0]
                        P_lfc[i,j,k] = nan
                    else:
                        P_lfc[i,j,k] = nan
                        P_el[i,j,k] = nan

		#if isnan(P_lfc):
		#   return P_lcl, P_lfc, P_el, 0, 0

	# need to handle case where dwpt is not available
	# above a certain level for any reason. Most simplest
	# thing to do is set it to a reasonably low value;
	# this should be a conservative approach!
       # dwpt = self.soundingdata['dwpt'].copy().soften_mask()
       # # raise ValueError
       # if dwpt[(pres >= P_el).data*(pres < P_lfc).data].mask.any():
       #     print("WARNING: substituting -200C for masked values of",
       #           "DWPT in this sounding")
       # dwpt[dwpt.mask] = -200
       # # dwptenv=interp(preswet,pres[::-1],dwpt[::-1])
       # # NEW:
       # dwptenv = interp(pparcel, pres[::-1], dwpt[::-1])

       # hght = self.soundingdata['hght']
       # if hght[(pres >= P_el).data].mask.any():
       #     raise NotImplementedError(
       #         "TODO: Implement standard atmos. to sub. missing heights")
       # # hghtenv=interp(preswet,pres[::-1],self.soundingdata['hght'][::-1])
       # # NEW:
       # hghtenv = interp(pparcel, pres[::-1], self.soundingdata['hght'][::-1])

		# Areas of POSITIVE Bouyancy
                cond1 = (tparcel[i,:,j,k] >= tempenv[i,:,j,k]) * (pparcel[i,:,j,k] <= P_lfc[i,j,k]) * \
				(pparcel[i,:,j,k] > P_el[i,j,k])
		# Areas of NEGATIVE Bouyancy
                if totalcape is True:
                    cond2 = (tparcel[i,:,j,k] < tempenv[i,:,j,k]) * (pparcel[i,:,j,k] > P_el[i,j,k])
                else:
                    cond2 = (tparcel[i,:,j,k] < tempenv[i,:,j,k]) * (pparcel[i,:,j,k] > P_lfc[i,j,k])
		# Do CAPE calculation
		# 1. Virtual temperature of parcel...
		# remember it's saturated above LCL.
		# e_parcel=VaporPressure(tempwet)
		# Tv_parcel=VirtualTemp(tempwet+degCtoK,preswet*100.,e_parcel)
		# e_env=VaporPressure(dwptenv)
		# Tv_env=VirtualTemp(tempenv+degCtoK,preswet*100.,e_env)
		# TODO: Implement CAPE calculation with virtual temperature
		# (This will affect the significant level calculations as well!!)
		# e_parcel=VaporPressure(dpparcel)
		# Tv_parcel=VirtualTemp(tparcel+degCtoK,pparcel*100.,e_parcel)
		# e_env=VaporPressure(dwptenv)
		# Tv_env=VirtualTemp(tempenv+degCtoK,pparcel*100.,e_env)
		# CAPE=trapz(9.81*(Tv_parcel[cond1]-Tv_env[cond1])/\
		# Tv_env[cond1],hghtenv[cond1])
		# CIN=trapz(9.81*(Tv_parcel[cond2]-Tv_env[cond2])/\
		# Tv_env[cond2],hghtenv[cond2])

                CAPE[i,j,k] = trapz(
		    9.81*(tparcel[i,cond1,j,k]-tempenv[i,cond1,j,k])/(tempenv[i,cond1,j,k] + 273.15),
		    hgtenv[i,cond1,j,k])
                CIN[i,j,k] = abs(trapz(
		    9.81*(tparcel[i,cond2,j,k]-tempenv[i,cond2,j,k])/(tempenv[i,cond2,j,k] + 273.15),
		    hgtenv[i,cond2,j,k]))

       # if False:
       #     print("%3s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s" %
       #           ("IX", "PRES", "TPARCEL", "DPPARCE", "TENV",
       #            "DPENV", "TV PARC", "TV ENV", "HEIGHT"))
       #     for ix, c2 in enumerate(cond2):
       #         if c2:
       #             print(
       #                 "%3d  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f" +
       #                 "  %7.3f  %7.3f" % (ix, pparcel[ix], tparcel[ix],
       #                                     dpparcel[ix], tempenv[ix],
       #                                     dwptenv[ix], Tv_parcel[ix],
       #                                     Tv_env[ix], hghtenv[ix]))
    return P_lcl, P_lfc, P_el, CAPE, CIN

def dry_ascent(startp, startt, startdp, nsteps=101, ptop=100):
    # -------------------------------------------------------------------
    # Lift a parcel dry adiabatically from startp to LCL.
    # Init temp is startt in C, Init dew point is stwrtdp,
    # pressure levels are in hPa

    #If start dewpoint is greater than or equal to T, then result is a 
    # profile of nans
    # -------------------------------------------------------------------

    #Initialise dry parcel lift output
    #[27,nsteps,16,14]
    presdry = empty((startp.shape[0],nsteps,startp.shape[1],startp.shape[2]))
    tempdry = empty((startp.shape[0],nsteps,startp.shape[1],startp.shape[2]))
    tempiso = empty((startp.shape[0],nsteps,startp.shape[1],startp.shape[2]))

    #if startdp == startt:
    #    return array([startp]), array([startt]), array([startdp]),

    # Pres=linspace(startp,600,nsteps)
    Pres = moveaxis(tile(stack([logspace(log10(startp[level,0,0]), log10(ptop), nsteps) \
		for level in arange(startp.shape[0])]),[startp.shape[1],startp.shape[2],1,1]),\
		[0,1,2,3],[2,3,0,1])

    # Lift the dry parcel
    T_dry = stack([(startt[level]+degCtoK) * (Pres[level]/startp[level])**(Rs_da/Cp_da)-degCtoK for level in arange(startp.shape[0])])

    # Mixing ratio isopleth
    starte = VaporPressure(startdp)
    startw = MixRatio(starte, startp*100)
    e = stack([Pres[level] * startw[level] / (.622+startw[level]) for level in arange(startp.shape[0])])
    T_iso = 243.5 / (17.67/log(e/6.112)-1)

    # Solve for the intersection of these lines (LCL).
    # interp requires the x argument (argument 2)
    # to be ascending in order!
    #P_lcl = [interp(0, T_iso-T_dry[level], Pres) for level in np.arange(startp.shape[0])]
    #T_lcl = interp(P_lcl, Pres[::-1], T_dry[::-1])
    P_lcl = stack([interpz3d(Pres[level],T_iso[level]-T_dry[level],0,missing=nan) for level in\
		 arange(startp.shape[0])])

    #INEFFICIENT
    # presdry=linspace(startp,P_lcl)
    for i in arange(0,startp.shape[0]):
       for j in arange(0,startp.shape[1]):
          for k in arange(0,startp.shape[2]):
                presdry[i,:,j,k] = logspace(log10(startp[i,j,k]), log10(P_lcl[i,j,k]), nsteps)
                tempdry[i,:,j,k] = interp(presdry[i,:,j,k], Pres[i,::-1,j,k], T_dry[i,::-1,j,k])
                tempiso[i,:,j,k] = interp(presdry[i,:,j,k], Pres[i,::-1,j,k], T_iso[i,::-1,j,k])

    return presdry, tempdry, tempiso


def moist_ascent(startp, startt, ptop=100, nsteps=101, dcape=False):
    # -------------------------------------------------------------------
    # Lift a parcel moist adiabatically from startp to endp.
    # Init temp is startt in C, pressure levels are in hPa
    # -------------------------------------------------------------------

    #preswet = empty((startp.shape[0],nsteps,startp.shape[1],startp.shape[2]))

    if dcape:
	#If doing a dcape calculation (dropping parcels to the surface), use a different "ptop" for each 
	#gridpoint, corresponding to the surface pressure
    	#preswet = moveaxis(logspace(log10(startp), log10(ptop), nsteps),[0,1,2,3],[1,0,2,3])
    	preswet = moveaxis(linspace(startp, ptop, nsteps),[0,1,2,3],[1,0,2,3])
    else:
	#Or else, it is a cape calculation, where we lift each parcel to the same ptop

    	preswet = moveaxis(logspace(log10(startp), log10(ptop), nsteps),[0,1,2,3],[1,0,2,3])

    temp = startt
    tempwet = zeros(preswet.shape)
    tempwet[:,0,:,:] = startt

    delp = preswet[:,0:-1,:,:] - preswet[:,1:,:,:]
    for ii in range(preswet.shape[1]-1):
                temp = temp + 100 * delp[:,ii,:,:] * GammaW(
		    temp+degCtoK, (preswet[:,ii,:,:]-delp[:,ii,:,:]/2)*100)
                tempwet[:,ii+1,:,:] = temp

    return preswet, tempwet