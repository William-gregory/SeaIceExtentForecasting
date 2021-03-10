### Regional September Sea Ice Forecasting with Complex Networks and Gaussian Processes (DOI: 10.1175)
### Author: William Gregory
### Last updated: 30/03/2020

import numpy as np
import struct
import glob
import warnings
import datetime
import itertools
import os
import operator
from scipy import stats
import random
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import scipy
from scipy.interpolate import griddata
import minimize
import ComplexNetworks as CN

#G L O B A L   V A R I A B L E S

dimX = 448 #Polar stereo grid dimensions (25km)
dimY = 304

dimXR = 57 #downsampled Polar stereo (100km)
dimYR = 57

datapath="./correlations/"

lat = (np.fromfile("./Data/psn25lats_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
lon = (np.fromfile("./Data/psn25lons_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
psa = (np.fromfile("./Data/psn25area_v3.dat",dtype='<i4').reshape(dimX,dimY))/1000

SIC = {}
GP = {}
regions = ['total', 'Beaufort_Sea', 'Chukchi_Sea', 'East_Siberian_Sea', 'Laptev_Sea', 'Kara_Sea', 'Barents_Sea', 'Greenland_Sea', 'Baffin_Bay', 'Canadian_Arch']

def readSIE():
    SIEs = {}
    SIEs_dt = {}
    SIEs_trend = {}
    for k in range(len(regions)):
        tag = regions[k]+'_sep'
        if regions[k] == 'total':
            SIEs[tag] = np.genfromtxt('./Data/N_'+str('%02d'%index)+'_extent_v3.0.csv',delimiter=',').T[4][1:]
        else:
            SIEs[tag] = np.genfromtxt('./Data/N_'+str(regions[k])+'_extent.csv',delimiter='\t')[1:,8]/1e6  

    for k in range(len(regions)):
        tag = regions[k]+'_sep'
        if any(np.isnan(SIEs[tag])):
            a = np.where(np.isnan(SIEs[tag]))[0][0]
            SIEs[tag][a] = (SIEs[tag][a+1]+SIEs[tag][a-1])/2
        if min(SIEs[tag])<0:
            a = np.where(SIEs[tag]<0)[0][0]
            SIEs[tag][a] = (SIEs[tag][a+1]+SIEs[tag][a-1])/2
        trend = np.zeros((2019-1984+1,2))
        dt = np.zeros((2019-1984+1,2019-1979+1))
        for year in range(1984,2019+1):
            nmax = year - 1979
            trendT, interceptT, r_valsT, probT, stderrT = scipy.stats.linregress(np.arange(nmax+1),SIEs[tag][range(nmax+1)])
            lineT = (trendT*np.arange(nmax+1)) + interceptT
            trend[year-1984,0] = trendT
            trend[year-1984,1] = interceptT
            dt[year-1984,range(nmax+1)] = SIEs[tag][range(nmax+1)]-lineT
        SIEs_trend[tag] = trend
        SIEs_dt[tag] = dt
            
    return SIEs,SIEs_dt,SIEs_trend

def readSIC(monthID, month, days, ymax):
    daily = np.zeros((dimX,dimY,ymax-2019+1,days)) ; daily[daily==0] = np.nan
    monthly = np.zeros((dimX,dimY,2018-1979+1)) ; monthly[monthly==0] = np.nan
    j = -1
    for year in range(2019,ymax+1):
        j = j + 1
        k = -1
        for day in range(1,days+1):
            k = k + 1
            icefile = open(glob.glob("./Data/nt_"+str(year)+str(monthID)+str("%02d"%day)+"*.bin")[0], "rb")
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = 300)
            daily[:,:,j,k] = np.array(z).reshape((dimX,dimY))
    daily = daily/250
    daily[daily>1]=np.nan       
    month_fm_daily = np.nanmean(daily, axis=3)
    k = 0
    for year in range(1979,2018+1):
        icefile = open(glob.glob("./Data/nt_"+str(year)+str(monthID)+"*.bin")[0], "rb")
        contents = icefile.read()
        icefile.close()
        s="%dB" % (int(dimX*dimY),)
        z=struct.unpack_from(s, contents, offset = 300)
        monthly[:,:,k] = np.array(z).reshape((dimX,dimY))
        k = k + 1
    monthly = monthly/250
    monthly[monthly>1]=np.nan
    data = np.concatenate((monthly,month_fm_daily),2)
    SIC[str(month)+'_data'] = data
    
def regrid(monthID, month):
    m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
    x,y = m(lon,lat)
    lonr, latr = m.makegrid(int((m.xmax-m.xmin)/100000)+1, int((m.ymax-m.ymin)/100000)+1)
    xr,yr=m(lonr,latr)
    
    data = SIC[str(month)+'_data']
    regrid = np.zeros((dimXR,dimYR,data.shape[2])) ; regrid[regrid==0] = np.nan
    fill = np.zeros((dimX,dimY,data.shape[2])) ; fill[fill==0] = np.nan
    for t in range(data.shape[2]):
        ice_copy = np.copy(data[:,:,t])
        #SMMR Pole Hole Mask: 84.5 November 1978 - June 1987
        #SSM/I Pole Hole Mask: 87.2 July 1987 - December 2007
        #SSMIS Pole Hole Mask: 89.18 January 2008 - present
        if t < 1988-1979:
            pmask=84.5
        elif (t > 1987-1979) & (t < 2008-1979):
            pmask=87.2
        else:
            pmask=89.2
        hole = np.nanmean(ice_copy[(lat_ori > pmask-0.5) & (lat_ori < pmask)]) #calculate the mean 0.5 degrees around polar hole
        fill[:,:,t] = np.ma.where((lat_ori >= pmask-0.5), hole, data[:,:,t]) #Fill polar hole with mean
        regrid[:,:,t] = griddata((x.ravel(), y.ravel()),fill[:,:,t].ravel(), (xr, yr), method='nearest') #downsample to 100km
        
    psar = griddata((x.ravel(), y.ravel()),psa.ravel(), (xr, yr), method='nearest')*16 #downsample to 100km
    SIC[str(month)+'_fill'] = fill
    SIC[str(month)+'_regrid'] = regrid
    
    return lonr, latr, psar
        
def detrend(ymax, key):
    data = SIC[str(key)]
    X = data.shape[0] ; Y = data.shape[1]
    for year in range(1984,ymax+1):
        nmax = year - 1979
        detrended = np.zeros((X,Y,nmax+1)) ; detrended[detrended==0] = np.nan
        trend = np.zeros((X,Y,2)) ; trend[trend==0] = np.nan
        for i,j in itertools.product(range(X),range(Y)):
            if ~np.isnan(data[i,j,range(nmax+1)]).all():
                trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(np.arange(nmax+1),data[i,j,range(nmax+1)])
                lineT = (trendT*np.arange(nmax+1)) + interceptT
                trend[i,j,0] = trendT
                trend[i,j,1] = interceptT
                detrended[i,j,range(nmax+1)]=data[i,j,range(nmax+1)]-lineT
                
        SIC[str(key)+'_dt_'+str(year)] = detrended  
        SIC[str(key)+'_trend_'+str(year)] = trend

def networks(month, areas, ymax):
    for year in range(1985,ymax+1):
        data = SIC[str(month)+'_regrid_dt_'+str(year)]
        print('Creating network: 1979 - ',year)
        nmax = year - 1979
        network = CN.Network(data)
        CN.Network.get_threshold(network)
        CN.Network.get_nodes(network)
        CN.Network.get_links(network, area=areas)
        SIC[str(month)+'_nodes_'+str(year)] = network.V
        SIC[str(month)+'_anoms_'+str(year)] = network.anomaly
        
def GPR(month):
    for k in range(len(regions)):
        print('Region: ',regions[k])
        print('input month: ',month)
        fmean = np.zeros(2019-1985+1)
        fmean_rt = np.zeros(2019-1985+1)
        fvar = np.zeros(2019-1985+1)
        for year in range(1985,2019+1):
            y = np.asarray([SIEs_dt[regions[k]+'_sep'][year-1984-1,range(year-1979)]]).T #n x 1
            n = len(y)
            X = []
            for area in SIC[month+'_anoms_'+str(year)]:
                r,p = stats.pearsonr(y[:,0],SIC[month+'_anoms_'+str(year)][area][range(year-1979)])
                if month == 'jun':
                    l_init = [np.logspace(-7,2,15)[7],np.logspace(-7,2,15)[0],np.logspace(-7,2,15)[4],np.logspace(-7,2,15)[5],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[6],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[6]]
                    sigma_init = [np.logspace(-3,9,15)[0],np.logspace(-3,9,15)[11],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[8],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[4],np.logspace(-3,9,15)[11],np.logspace(-3,9,15)[11],np.logspace(-3,9,15)[11],np.logspace(-3,9,15)[6]]
                    if r>0:
                        X.append(SIC[month+'_anoms_'+str(year)][area][range(year-1979+1)])   
                elif month == 'jul':
                    l_init = [np.logspace(-7,2,15)[7],np.logspace(-7,2,15)[5],np.logspace(-7,2,15)[4],np.logspace(-7,2,15)[0],np.logspace(-7,2,15)[3],np.logspace(-7,2,15)[3],np.logspace(-7,2,15)[3],np.logspace(-7,2,15)[3],np.logspace(-7,2,15)[0],np.logspace(-7,2,15)[4]]
                    sigma_init = [np.logspace(-3,9,15)[4],np.logspace(-3,9,15)[4],np.logspace(-3,9,15)[8],np.logspace(-3,9,15)[10],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[11],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[4],np.logspace(-3,9,15)[8]]
                    if k == 0:
                        X.append(SIC[month+'_anoms_'+str(year)][area][range(year-1979+1)])
                    else:
                        if (r>0) & (p/2<0.08):
                            X.append(SIC[month+'_anoms_'+str(year)][area][range(year-1979+1)])
                elif month == 'aug':
                    l_init = [np.logspace(-7,2,15)[6],np.logspace(-7,2,15)[9],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[2],np.logspace(-7,2,15)[1],np.logspace(-7,2,15)[5],np.logspace(-7,2,15)[0],np.logspace(-7,2,15)[0],np.logspace(-7,2,15)[0]]
                    sigma_init = [np.logspace(-3,9,15)[8],np.logspace(-3,9,15)[1],np.logspace(-3,9,15)[10],np.logspace(-3,9,15)[10],np.logspace(-3,9,15)[10],np.logspace(-3,9,15)[10],np.logspace(-3,9,15)[7],np.logspace(-3,9,15)[9],np.logspace(-3,9,15)[6],np.logspace(-3,9,15)[9]]
                    if k == 0:
                        X.append(SIC[month+'_anoms_'+str(year)][area][range(year-1979+1)])
                    else:
                        if (r>0) & (p/2 < 0.05):
                            X.append(SIC[month+'_anoms_'+str(year)][area][range(year-1979+1)])

            X = np.asarray(X).T
            Xs = np.asarray([X[-1,:]])
            X = X[:-1,:]

            M = np.abs(np.cov(X, rowvar=False, bias=True))
            np.fill_diagonal(M,0)
            np.fill_diagonal(M,-np.sum(M,axis=0))

            def MLII(hyperparameters): #Empirical Bayesian technique for optimisation of hyperparameters
                ℓ = np.exp(hyperparameters[0]) ; σn_tilde = np.exp(2*hyperparameters[1])
                try:
                    Σ = scipy.linalg.expm(ℓ*M)
                    L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn_tilde)
                    α = np.linalg.solve(L.T,np.linalg.solve(L,y)).reshape(n,1)
                    σf = np.dot(y.T,α)/n
                    σn = σf*σn_tilde
                    Σ = σf * scipy.linalg.expm(ℓ*M)
                    L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn)
                    α = np.linalg.solve(L.T,np.linalg.solve(L,y)).reshape(n,1)
                    nlML = np.dot(y.T,α)/2 + np.log(L.diagonal()).sum() + n*np.log(2*np.pi)/2

                    Q = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n))) - np.dot(α,α.T)
                    dKdθ1 = (Q*np.linalg.multi_dot([X,np.dot(M,Σ),X.T])).sum()/2
                    dKdθ2 = σn_tilde*np.trace(Q)
                except (np.linalg.LinAlgError,ValueError,OverflowError) as e:
                    nlML = np.inf
                    dKdθ1 = np.inf ; dKdθ2 = np.inf
                return nlML, np.asarray([dKdθ1,dKdθ2])

            θ = minimize.run(MLII,X=[np.log(l_init[k]),np.log(sigma_init[k])],length=100)

            ℓ = np.exp(θ[0]) ; σn_tilde = np.exp(2*θ[1])
            Σ = scipy.linalg.expm(ℓ*M)
            L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn_tilde)
            α = np.linalg.solve(L.T,np.linalg.solve(L,y)).reshape(n,1)
            σf = np.dot(y.T,α)/n
            σn = σf*σn_tilde
            Σ = σf * scipy.linalg.expm(ℓ*M)
            L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn)
            α = np.linalg.solve(L.T,np.linalg.solve(L,y))
            KXXs = np.linalg.multi_dot([X,Σ,Xs.T])
            KXsXs = np.linalg.multi_dot([Xs,Σ,Xs.T]) + σn
            v = np.linalg.solve(L,KXXs)

            fmean[year-1985] = np.dot(KXXs.T,α)
            fvar[year-1985] = (KXsXs - np.dot(v.T,v))
            lineT = (np.arange(year-1979+1)*SIEs_trend[regions[k]+'_sep'][year-1984-1,0]) + SIEs_trend[regions[k]+'_sep'][year-1984-1,1]
            fmean_rt[year-1985] = fmean[year-1985] + lineT[-1]

        GP[month+'-sep_'+regions[k]+'_fmean'] = fmean
        GP[month+'-sep_'+regions[k]+'_fvar'] = fvar
        GP[month+'-sep_'+regions[k]+'_fmean_rt'] = fmean_rt


print('Reading...')
print(datetime.datetime.now())
readSIC("06", 'jun', 30, 2019)
readSIC("07", 'jul', 31, 2019)
readSIC("08", 'aug', 31, 2019)
SIEs,SIEs_dt,SIEs_trend = readSIE()

print('Re-gridding...')
print(datetime.datetime.now())
regrid(6, 'jun', lat, lon, psa)
regrid(7, 'jul', lat, lon, psa)
lonr, latr, psar = regrid(8, 'aug', lat, lon, psa)

print('De-trending...')
print(datetime.datetime.now())
detrend(2019, 'jun_regrid')
detrend(2019, 'jul_regrid')
detrend(2019, 'aug_regrid')

print('Reading Networks...')
print(datetime.datetime.now())
networks('jun', psar, ymax=2019)
networks('jul', psar, ymax=2019)
networks('aug', psar, ymax=2019)

print('Running Forecast...')
print(datetime.datetime.now())
GPR('jun')
GPR('jul')
GPR('aug')
