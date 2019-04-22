
# coding: utf-8

# In[ ]:


### Code to accompany paper: "Regional September Sea Ice Forecasting with Complex Networks"
### Author: William Gregory
### Code Last updated: 22/04/2019

import numpy as np
import struct
import glob
import warnings
import matplotlib.pyplot as plt
import datetime
import itertools
import os
import math
import operator
from scipy import stats
import random
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import scipy
from scipy.interpolate import griddata
from scipy.optimize import minimize
from Complex_Networks import CN

#G L O B A L   V A R I A B L E S

hdr = 300
dimX = 448 #Polar stereo grid dimensions (25km)
dimY = 304

dimXR = 57 #downsampled Polar stereo (100km)
dimYR = 57

datapath='./Data/'
corrpath='./Correlations/'

latf = datapath+'psn25lats_v3.dat'
lat = np.fromfile(latf,dtype='<i4')
lat = lat.reshape(448,304)
lat = lat/100000

lonf = datapath+'psn25lons_v3.dat'
lon = np.fromfile(lonf,dtype='<i4')
lon = lon.reshape(448,304)
lon = lon/100000

regions = ['total', 'Beaufort_Sea', 'Chukchi_Sea', 'East_Siberian_Sea', 'Laptev_Sea', 'Kara_Sea', 'Barents_Sea', 'Greenland_Sea', 'Baffin_Bay', 'Canadian_Arch']
SIEs = {}
SIEs_dt = {}
SIEs_trend = {}
for k in range(len(regions)):
    tag = regions[k]
    if tag == 'total':
        file = np.genfromtxt(datapath+'september_SIE.txt')
        SIEs.setdefault(tag, []).append(file)
    else:
        file = np.genfromtxt(datapath+'september_SIE_'+str(tag)+'.txt')
        SIEs.setdefault(tag, []).append(file)
for k in range(len(regions)):
    tag = regions[k]
    trend = np.zeros((2018-1984+1,2))
    dt = np.zeros((2018-1984+1,2018-1979+1))
    for yend in range(1984,2018+1):
        nmax = yend - 1979
        trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(np.arange(nmax+1),SIEs[tag][0][range(nmax+1)])
        lineT = (trendT*np.arange(nmax+1)) + interceptT
        trend[yend-1984,0] = trendT
        trend[yend-1984,1] = interceptT
        dt[yend-1984,range(nmax+1)] = SIEs[tag][0][range(nmax+1)]-lineT
    SIEs_trend.setdefault(tag, []).append(trend)
    SIEs_dt.setdefault(tag, []).append(dt)

#D E F I N E   C L A S S E S

class SIC_daily:
    def __init__(self, data, monthly):
        self.data = data
        self.monthly = monthly

    def read_daily(self, month, days, ymax):
            j = -1
            for y in range(2017,ymax+1):
                j = j + 1
                k = -1
                for d in range(1,days+1):
                    k = k + 1
                    icefile = open(glob.glob(datapath+'nt_'+str(y)+str(month)+str('%02d'%d)+'*.bin')[0], 'rb')
                    contents = icefile.read()
                    icefile.close()
                    s="%dB" % (int(dimX*dimY),)
                    z=struct.unpack_from(s, contents, offset = hdr)
                    self.data[:,:,j,k] = np.array(z).reshape((dimX,dimY))
            self.data = self.data/250
            self.data[self.data>1]=np.nan       

    def create_monthly(self, ymax):
        for i,j in itertools.product(range(dimX),range(dimY)):
            for y in range(ymax):
                self.monthly[i,j,y] = np.nanmean(self.data[i,j,y,:])

class SIC_monthly:
    def __init__(self, data, regrid, trend, trend_regrid, dt, dt_regrid, nodes, anomalies):
        self.data = data
        self.regrid = regrid
        self.trend = trend
        self.trend_regrid = trend_regrid
        self.dt = dt
        self.dt_regrid = dt_regrid
        self.nodes = nodes
        self.anomalies = anomalies
        
    def read_monthly(self, month):
        k = 0
        for y in range(1979,2016+1):
            icefile = open(glob.glob(datapath+'nt_'+str(y)+str(month)+'*.bin')[0], 'rb')
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = hdr)
            self.data[:,:,k] = np.array(z).reshape((dimX,dimY))
            k = k + 1
        self.data = self.data/250
        self.data[self.data>1]=np.nan
        
    def regrid(self, month, lat_ori, lon_ori):
        m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
        x,y = m(lon_ori,lat_ori)
        dx_res = 100000 #100 km square
        new_x = int((m.xmax-m.xmin)/dx_res)+1 ; new_y = int((m.ymax-m.ymin)/dx_res)+1
        lonsG, latsG = m.makegrid(new_x, new_y)
        xt,yt=m(lonsG,latsG)
        
        for t in range(np.shape(self.data)[2]):
            ice_copy = np.copy(self.data[:,:,t])
            #SMMR Pole Hole Mask: 84.5 November 1978 - June 1987
            #SSM/I Pole Hole Mask: 87.2 July 1987 - December 2007
            #SSMIS Pole Hole Mask: 89.18 January 2008 - present
            if t < 1987-1979:
                pmask=84.5
            elif (t == 1987-1979) & (month <= 6):
                pmask=84.5
            elif (t == 1987-1979) & (month > 6):
                pmask=84.5#87.2
            elif (t > 1987-1979) & (t < 2008-1979):
                pmask=87.2
            else:
                pmask=89.2
            hole = np.nanmean(ice_copy[(lat_ori > pmask-0.5) & (lat_ori < pmask)]) #calculate the mean 0.5 degrees around polar hole
            self.data[:,:,t] = np.ma.where((lat_ori >= pmask-0.5), hole, self.data[:,:,t]) #Fill polar hole with mean
            self.regrid[:,:,t] = griddata((x.ravel(), y.ravel()),self.data[:,:,t].ravel(), (xt, yt), method='linear') #downsample to 100km
        
        return lonsG, latsG
        
    def detrend(self, ymax, data):
        self.dt = {}
        self.dt_regrid = {}
        self.trend = {}
        self.trend_regrid = {}
        X = data.shape[0] ; Y = data.shape[1]
        for yend in range(1984,ymax+1):
            nmax = yend - 1979
            detrended = np.zeros((dimX,dimY,nmax+1)) ; detrended_regrid = np.zeros((dimXR,dimYR,nmax+1))
            detrended[detrended==0] = np.nan ; detrended_regrid[detrended_regrid==0] = np.nan
            trend = np.zeros((dimX,dimY,nmax+1,2)) ; trend_regrid = np.zeros((dimXR,dimYR,nmax+1,2))
            for i,j in itertools.product(range(X),range(Y)):
                if all(~np.isnan(data[i,j,range(nmax+1)])):
                    trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(np.arange(nmax+1),data[i,j,range(nmax+1)])
                    lineT = (trendT*np.arange(nmax+1)) + interceptT
                    if X == 57:
                        trend_regrid[i,j,yend-1984,0] = trendT
                        trend_regrid[i,j,yend-1984,1] = interceptT
                        detrended_regrid[i,j,range(nmax+1)]=data[i,j,range(nmax+1)]-lineT
                    else:
                        trend[i,j,yend-1984,0] = trendT
                        trend[i,j,yend-1984,1] = interceptT
                        detrended[i,j,range(nmax+1)]=data[i,j,range(nmax+1)]-lineT

            self.dt.setdefault(yend, []).append(detrended)
            self.dt_regrid.setdefault(yend, []).append(detrended_regrid)
            self.trend.setdefault(yend, []).append(trend)
            self.trend_regrid.setdefault(yend, []).append(trend_regrid)
                
    def gen_networks(self, month, lats):
        self.nodes = {}
        self.anomalies = {}
        for yend in range(1985,2018+1):
            print('Creating network: 1979 - ',yend)
            nmax = yend - 1979
            network = CN.Network(dimX=dimXR,dimY=dimYR)
            CN.Network.cell_level(network, self.dt_regrid[yend][0][:,:,range(nmax+1)], str(month), "_100sqkm_79-"+str(yend), corrpath)
            CN.Network.tau(network, self.dt_regrid[yend][0][:,:,range(nmax+1)], str(month), 0.01, "_100sqkm_79-"+str(yend), corrpath)
            CN.Network.area_level(network, str(month))
            CN.Network.intra_links(network, self.dt_regrid[yend][0][:,:,range(nmax+1)], str(month), lats)
            self.nodes.setdefault(yend, []).append(network.V)
            self.anomalies.setdefault(yend, []).append(network.anomaly)
        
class GPR:
    def __init__(self, forecast, retrend, error, error_retrend_minus, error_retrend_plus):
        self.fmean = forecast
        self.fmean_rt = retrend   
        self.fvar = error
        self.fvar_rt_minus = error_retrend_minus
        self.fvar_rt_plus = error_retrend_plus
            
    def forecast(self, target, trends, month, anomalies, iterations):
        print('Running Forecast')
        for m in range(len(regions)):
            print('Forecasting: ',str(regions[m]),' SIE')
            print(datetime.datetime.now())
            for yend in range(1985,2018+1):
                nmax = yend - 1979
                X = np.zeros((nmax,len(anomalies[yend][0]))) #Predictors (n x N)
                Z = np.zeros((len(anomalies[yend][0]),1)) #Test case for forecast (N x 1)
                M = np.zeros((len(anomalies[yend][0]),len(anomalies[yend][0]))) #Adj matrix for Prior Covariance (N x N)
                #print('Forecast year: ',yend)
                k = -1
                for area in anomalies[yend][0]: #For each network node
                    k = k + 1
                    X[:,k] = anomalies[yend][0][area][0][range(nmax)] 
                    Z[k,0] = anomalies[yend][0][area][0][nmax]
                    l = -1
                    for area2 in anomalies[yend][0]: #Repeat loop to generate network links
                        l = l + 1
                        if area != area2:
                            M[k,l] = np.cov(anomalies[yend][0][area][0][range(nmax)],anomalies[yend][0][area2][0][range(nmax)],bias=True)[0][1]
                Xt = X.T
                m_prior = np.zeros((X.shape[1],1)) #Zero mean prior (N x 1)
                M[M<0] = 0 #only take positive correlations between network nodes
                for i in range(len(anomalies[yend][0])):
                    ii = -1*(np.nansum(M[i,:]))
                    for j in range(len(anomalies[yend][0])):
                        if np.isnan(M[i,j]):
                            M[i,j] = 0
                        elif i == j:
                            M[i,j] = ii #set diagonal elements to be the -sum of all link weights

                def gen_matrices(C, V):
                    mat1 = np.matmul(X,C)
                    mat2 = np.matmul(mat1,Xt) + V
                    mat2i = np.linalg.inv(mat2)
                    mat3 = y - np.matmul(X,m_prior)
                    mat4 = np.matmul(mat2i,mat3)
                    yKy = np.matmul(mat3.T,mat4)
                    evidence = -1*(-0.5*(np.log(2*math.pi)) - yKy/(2*nmax) - np.log(np.linalg.det(mat2))/(2*nmax)) #eq 6 Sollich
                    return evidence, yKy, mat2i, mat4

                def marginal_likelihood(hyperparameters):
                    gradient = np.zeros(2)
                    Y = y - np.matmul(X,m_prior)
                    sigma = np.exp(hyperparameters[0]) ; l = np.exp(hyperparameters[1])
                    C = scipy.linalg.expm(M*l) ; V = np.eye(nmax) * sigma
                    evidence, yKy, M1, M2 = gen_matrices(C, V)
                    a = yKy/nmax
                    Cgrad_sig = a * scipy.linalg.expm(M*l) ; Vgrad_sig = np.eye(nmax) * a
                    #Cgrad_sig = a * scipy.linalg.expm(M*l) ; Vgrad_sig = np.eye(nmax) * (2*a*np.sqrt(sigma))
                    Cgrad_l = a * np.multiply(scipy.linalg.expm(M*l),M) ; Vgrad_l = np.eye(nmax) * (a*sigma)
                    dkdsig = np.matmul(np.matmul(X,Cgrad_sig),Xt) + Vgrad_sig
                    dkdl = np.matmul(np.matmul(X,Cgrad_l),Xt) + Vgrad_l
                    C = a * scipy.linalg.expm(M*l) ; V = np.eye(nmax) * (a*sigma)
                    evidence, yKy, M1, M2 = gen_matrices(C, V)
                    gradient[0] = -1*(0.5*np.matmul(np.matmul(np.matmul(Y.T,M1),dkdsig),M2) - 0.5*np.trace(np.matmul(M1,dkdsig)))
                    gradient[1] = -1*(0.5*np.matmul(np.matmul(np.matmul(Y.T,M1),dkdl),M2) - 0.5*np.trace(np.matmul(M1,dkdl)))
                    return evidence, gradient

                y = np.reshape(target[regions[m]][0][yend-1984-1,range(nmax)],(nmax,1)) 
                #optimise hyperparameters sigma and l
                theta_sig = np.zeros(iterations) ; theta_l = np.zeros(iterations) ; evidence = np.zeros(iterations) ; evidence[evidence==0] = np.nan
                passed = False
                while passed == False:
                    for it in range(iterations):
                        theta_sig[it] = np.random.uniform(0.001,10)
                        theta_l[it] = np.random.uniform(0.001,100)
                        try:
                            result = scipy.optimize.minimize(marginal_likelihood, [np.log(theta_sig[it]), np.log(theta_l[it])], method='TNC', jac=True, options={'disp':False})
                            if result.success == True: #did the result converge?
                                evidence[it] = result.fun
                            else:
                                evidence[it] = np.nan
                        except ValueError:
                            evidence[it] = np.nan
                        except OverflowError:
                            evidence[it] = np.nan
                        except np.linalg.LinAlgError:
                            evidence[it] = np.nan
                    if ~np.isnan(evidence).all():
                        id0 = np.where(evidence==np.nanmin(evidence))
                        id0 = id0[0][0]
                        try:
                            result = scipy.optimize.minimize(marginal_likelihood, [np.log(theta_sig[id0]), np.log(theta_l[id0])], method='TNC', jac=True, options={'disp':False})
                            passed = True
                        except ValueError:
                            pass
                        except OverflowError:
                            pass
                        except np.linalg.LinAlgError:
                            pass
                sigma = np.exp(result.x[0]) ; l = np.exp(result.x[1])
                C = scipy.linalg.expm(M*l) ; V = np.eye(nmax) * sigma
                evidence, yKy, M1, M2 = gen_matrices(C, V)
                a = yKy/nmax
                C = a * scipy.linalg.expm(M*l) ; V = np.eye(nmax) * (a*sigma)
                evidence, yKy, M1, M2 = gen_matrices(C, V)
                mat1 = np.matmul(C,Xt)
                mat2 = np.matmul(mat1,M2)
                mat3 = np.matmul(X,C)

                m_posterior = m_prior + mat2 #eq 3.37 tarantola
                C_posterior = C - np.matmul(np.matmul(mat1,M1),mat3) #eq 3.38 tarantola

                mat_a = np.matmul(Z.T,C)
                mat_b = np.matmul(mat_a,Z)
                mat_c = np.matmul(Z.T,mat1)
                mat_d = np.matmul(mat3,Z)
                mat_e = np.matmul(mat_c,M1)

                self.fmean[m,yend-1985] = np.matmul(Z.T,m_posterior)
                self.fvar[m,yend-1985] = mat_b - np.matmul(mat_e,mat_d)
                lineT = (np.arange(nmax+1)*trends[regions[m]][0][yend-1984-1,0]) + trends[regions[m]][0][yend-1984-1,1]
                self.fmean_rt[m,yend-1985] = self.fmean[m,yend-1985] + lineT[-1]
                self.fvar_rt_minus[m,yend-1985] = (self.fmean[m,yend-1985] - np.sqrt(self.fvar[m,yend-1985])) + lineT[-1]
                self.fvar_rt_plus[m,yend-1985] = (self.fmean[m,yend-1985] + np.sqrt(self.fvar[m,yend-1985])) + lineT[-1]

        print('Done')
        print(datetime.datetime.now())


print('Reading...')
daily_SIC_june = SIC_daily(np.zeros((dimX,dimY,2,30)),np.zeros((dimX,dimY,2)))
daily_SIC_july = SIC_daily(np.zeros((dimX,dimY,2,31)),np.zeros((dimX,dimY,2)))
daily_SIC_august = SIC_daily(np.zeros((dimX,dimY,2,31)),np.zeros((dimX,dimY,2)))
SIC_june = SIC_monthly(np.zeros((dimX,dimY,2016-1979+1)),np.zeros((dimXR,dimYR,2018-1979+1)),{},{},{},{},{},{})
SIC_july = SIC_monthly(np.zeros((dimX,dimY,2016-1979+1)),np.zeros((dimXR,dimYR,2018-1979+1)),{},{},{},{},{},{})
SIC_august = SIC_monthly(np.zeros((dimX,dimY,2016-1979+1)),np.zeros((dimXR,dimYR,2018-1979+1)),{},{},{},{},{},{})

SIC_daily.read_daily(daily_SIC_june,"06",30, 2018)
SIC_daily.read_daily(daily_SIC_july,"07",31, 2018)
SIC_daily.read_daily(daily_SIC_august,"08",31, 2018)
SIC_daily.create_monthly(daily_SIC_june, 2)
SIC_daily.create_monthly(daily_SIC_july, 2)
SIC_daily.create_monthly(daily_SIC_august, 2)
SIC_monthly.read_monthly(SIC_june,"06")
SIC_monthly.read_monthly(SIC_july,"07")
SIC_monthly.read_monthly(SIC_august,"08")
SIC_june.data = np.concatenate((SIC_june.data,daily_SIC_june.monthly),2)
SIC_july.data = np.concatenate((SIC_july.data,daily_SIC_july.monthly),2)
SIC_august.data = np.concatenate((SIC_august.data,daily_SIC_august.monthly),2)
print('Re-gridding...')
lon_regrid, lat_regrid = SIC_monthly.regrid(SIC_june, 6, lat, lon)
lon_regrid, lat_regrid = SIC_monthly.regrid(SIC_july, 7, lat, lon)
lon_regrid, lat_regrid = SIC_monthly.regrid(SIC_august, 8, lat, lon)
print('De-trending...')
SIC_monthly.detrend(SIC_june, 2018, SIC_june.regrid)
SIC_monthly.detrend(SIC_july, 2018, SIC_july.regrid)
SIC_monthly.detrend(SIC_august, 2018, SIC_august.regrid)
SIC_monthly.gen_networks(SIC_june, "june", lat_regrid)
SIC_monthly.gen_networks(SIC_july, "july", lat_regrid)
SIC_monthly.gen_networks(SIC_august, "august", lat_regrid)

june = GPR(np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)))
july = GPR(np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)))
august = GPR(np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)),np.zeros((len(regions),2018-1985+1)))

GPR.forecast(june, SIEs_dt, SIEs_trend, "june", SIC_june.anomalies, 10)
GPR.forecast(july, SIEs_dt, SIEs_trend, "july", SIC_july.anomalies, 10)
GPR.forecast(august, SIEs_dt, SIEs_trend, "august", SIC_august.anomalies, 10)


