## Using Networks of June SIC in order to forecast September SIC and probability of sea ice in September; through
## a Gaussian Process Regression. 
##
## Author: William Gregory
## Last updated: 19/03/2019

import numpy as np
import struct
import glob
import warnings
import datetime
import itertools
import os
import math
import operator
import random
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from scipy import stats
from scipy.interpolate import griddata
from scipy.optimize import minimize
import CN
    
hdr = 300

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
                    icefile = open(glob.glob(datapath+"/nt_"+str(y)+str(month)+str("%02d"%d)+"*.bin")[0], "rb")
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
    def __init__(self, data, regrid, trend, trend_regrid, dt, nodes, anomalies):
        self.data = data
        self.regrid = regrid
        self.trend = trend
        self.trend_regrid = trend_regrid
        self.dt = dt
        self.nodes = nodes
        self.anomalies = anomalies
        
    def read_monthly(self, month):
        k = 0
        for y in range(1979,2016+1):
            icefile = open(glob.glob(datapath+"/nt_"+str(y)+str(month)+"*.bin")[0], "rb")
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = hdr)
            self.data[:,:,k] = np.array(z).reshape((dimX,dimY))
            k = k + 1
        self.data = self.data/250
        self.data[self.data>1]=np.nan
        
    def regrid(self, month, lon_ori, lat_ori):
        m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')
        x,y=m(lon_ori,lat_ori)
        x=np.array(x) ; y=np.array(y)
        dx_res = 100000 #100 km square
        new_x = int((m.xmax-m.xmin)/dx_res)+1 ; new_y = int((m.ymax-m.ymin)/dx_res)+1
        lonsG, latsG = m.makegrid(new_x, new_y)
        xt,yt=m(lonsG,latsG)
        xt=np.array(xt) ; yt = np.array(yt)
        
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
                pmask=87.2
            elif (t > 1987-1979) & (t < 2008-1979):
                pmask=87.2
            else:
                pmask=89.2
            hole = np.nanmean(ice_copy[(lat_ori > pmask-0.5) & (lat_ori < pmask)]) #calculate the mean 0.5 degrees around polar hole
            self.data[:,:,t] = np.ma.where((lat_ori >= pmask-0.5), hole, self.data[:,:,t]) #Fill polar hole with mean
            self.regrid[:,:,t] = griddata((x.ravel(), y.ravel()),self.data[:,:,t].ravel(), (xt, yt), method='linear') #downsample to 100km
        
        return lonsG, latsG
        
    def detrend(self, ymax, data, X=dimXR,Y=dimYR):
        self.dt[self.dt==0] = np.nan
        for yend in range(1984,ymax+1):
            nmax = yend - 1979
            for i,j in itertools.product(range(X),range(Y)):
                if all(~np.isnan(data[i,j,range(nmax+1)])):
                    trendT, interceptT, r_valsT, probT, stderrT = stats.linregress(np.arange(nmax+1),data[i,j,range(nmax+1)])
                    lineT = (trendT*np.arange(nmax+1)) + interceptT
                    if X == dimXR:
                        self.trend_regrid[i,j,yend-1984,0] = trendT
                        self.trend_regrid[i,j,yend-1984,1] = interceptT
                        if yend == 1984:
                            self.dt[i,j,range(nmax+1)]=data[i,j,range(nmax+1)]-lineT
                        else:
                            self.dt[i,j,nmax]=data[i,j,nmax]-lineT[nmax]
                    else:
                        self.trend[i,j,yend-1984,0] = trendT
                        self.trend[i,j,yend-1984,1] = interceptT
                        if yend == 1984:
                            self.dt[i,j,range(nmax+1)]=data[i,j,range(nmax+1)]-lineT
                        else:
                            self.dt[i,j,nmax]=data[i,j,nmax]-lineT[nmax]
                
    def gen_networks(self, month, lats):
        self.nodes = {}
        self.anomalies = {}
        for yend in range(2018,2018+1): #these are the forecast years
            print('Creating network: 1979 - ',yend)
            nmax = yend - 1979
            network = CN.Network(dimX=dimXR,dimY=dimYR)
            CN.Network.cell_level(network, self.dt[:,:,range(nmax+1)], str(month), "_100sqkm_79-"+str(yend), corrpath)
            CN.Network.tau(network, self.dt[:,:,range(nmax+1)], str(month), 0.01, "_100sqkm_79-"+str(yend), corrpath)
            CN.Network.area_level(network, str(month))
            CN.Network.intra_links(network, self.dt[:,:,range(nmax+1)], str(month), lats)
            self.nodes.setdefault(yend, []).append(network.V)
            self.anomalies.setdefault(yend, []).append(network.anomaly)
            
class Forecast_gridlevel:
    def __init__(self, forecast, retrend, error, error_retrend, probs):
        self.fmean = forecast
        self.fmean_rt = retrend   
        self.fvar = error
        self.fvar_rt = error_retrend
        self.probability = probs
            
    def forecast(self, target, anomalies):
        print('Running Grid-Level Forecast')
        print(datetime.datetime.now())
        for yend in range(2018,2018+1): #just forecast 2018
            nmax = yend - 1979
            X = np.zeros((nmax,len(anomalies[yend][0]))) #Predictors (n x N)
            Z = np.zeros((len(anomalies[yend][0]),1)) #Test case for forecast (N x 1)
            M = np.zeros((len(anomalies[yend][0]),len(anomalies[yend][0]))) #Adj matrix for Prior Covariance (N x N)
            print('Forecast year: ',yend)
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

            M[M<0] = 0 #only take positive correlations between network nodes
            for i in range(len(anomalies[yend][0])):
                ii = -1*(np.nansum(M[i,:]))
                for j in range(len(anomalies[yend][0])):
                    if np.isnan(M[i,j]):
                        M[i,j] = 0
                    elif i == j:
                        M[i,j] = ii #set diagonal elements to be the -sum of all link weights
            for i,j in itertools.product(range(dimX),range(dimY)):
                if (all(~np.isnan(target.dt[i,j,range(nmax)]))) & (max(abs(target.dt[i,j,range(nmax)])) > 0):
                    y = np.reshape(target.dt[i,j,range(nmax)],(nmax,1))
                    Xt = X.T
                    m_prior = np.zeros((X.shape[1],1)) #Zero mean prior (N x 1)

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
                        sigma = np.exp(hyperparameters[0]) ; l = np.exp(hyperparameters[1])
                        C = scipy.linalg.expm(M*l) ; V = np.eye(nmax) * sigma
                        evidence, yKy, M1, M2 = gen_matrices(C, V)
                        a = yKy/nmax
                        C = a * scipy.linalg.expm(M*l) ; V = np.eye(nmax) * (a*sigma)
                        evidence, yKy, M1, M2 = gen_matrices(C, V)
                        return evidence

                    #optimise hyperparameters sigma and l
                    iterations = 10
                    theta_sig = np.zeros(iterations) ; theta_l = np.zeros(iterations) ; evidence = np.zeros(iterations) ; evidence[evidence==0] = np.nan
                    for it in range(iterations):
                        theta_sig[it] = np.random.uniform(0.0001,100)
                        theta_l[it] = np.random.uniform(0.0001,100)
                        try:
                            result = scipy.optimize.minimize(marginal_likelihood, [np.log(theta_sig[it]), np.log(theta_l[it])], method='TNC', options={'disp':False})
                            if result.success == True: #did the result converge?
                                evidence[it] = result.fun
                            else:
                                evidence[it] = np.nan
                        except ValueError: #NaNs in covariance matrix
                            evidence[it] = np.nan
                        except LinAlgError: #Singular matrix, hence cannot invert
                            evidence[it] = np.nan
                        except OverflowError: #infs in covariance matrix
                            evidence[it] = np.nan
                    id0 = np.where(evidence==np.nanmin(evidence)) #select smallest negative log marginal likelihood
                    id0 = id0[0][0]
                    result = scipy.optimize.minimize(marginal_likelihood, [np.log(theta_sig[id0]), np.log(theta_l[id0])], method='TNC', options={'disp':False})
                    sigma = np.exp(result.x[0]) ; l = np.exp(result.x[1])
                    C = scipy.linalg.expm(M*l) ; V = np.eye(nmax) * sigma
                    evidence, yKy, M1, M2 = gen_matrices(C, V)
                    a = yKy/nmax #Sollich, 2005
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

                    self.fmean[i,j,yend-1985] = np.matmul(Z.T,m_posterior)
                    self.fvar[i,j,yend-1985] = mat_b - np.matmul(mat_e,mat_d)
                    
                    self.fmean_rt[i,j,yend-1985] = self.fmean[i,j,yend-1985] + target.trend[i,j,yend-1984-1,1] + np.multiply(target.trend[i,j,yend-1984-1,0],yend-1984)
                    self.fvar_rt[i,j,yend-1985] = np.sqrt(self.fvar[i,j,yend-1985]) + target.trend[i,j,yend-1984-1,1] + np.multiply(target.trend[i,j,yend-1984-1,0],yend-1984)
                    
                    sic_vals = np.arange(0,1.01,0.01)
                    CDF = stats.norm.cdf(sic_vals,self.fmean_rt[i,j,yend-1985],self.fvar_rt[i,j,yend-1985])
                    id1 = np.where(sic_vals==0.15)
                    if ~np.isnan(CDF[id1]):
                        self.probability[i,j,yend-1985] = 1 - CDF[id1]
                    else:
                        self.probability[i,j,yend-1985] = np.nan
            
                else:
                    self.fmean[i,j,:] = np.nan
                    self.fmean_rt[i,j,:] = np.nan
                    self.fvar[i,j,:] = np.nan
                    self.fvar_rt[i,j,:] = np.nan
                    self.probability[i,j,:] = np.nan
                    
        print('Done')
        print(datetime.datetime.now())