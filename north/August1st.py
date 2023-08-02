import xarray as xr
import numpy as np
import datetime
import shutil
import urllib.request as request
from contextlib import closing
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.linalg import expm
import glob
import struct
import pandas as pd
from scipy.stats import linregress
import os
import warnings
warnings.filterwarnings('ignore')

def make_npstere_grid(boundinglat,lon_0,grid_res=25e3):
    import pyproj as proj
    p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0='+str(float(lon_0))+' +lat_ts=90.0 +lat_0=90.0',\
                  preserve_units=True)
    llcrnrlon = lon_0 - 45
    urcrnrlon = lon_0 + 135
    y_ = p(lon_0,boundinglat)[1]
    llcrnrlat = p(np.sqrt(2.)*y_,0.,inverse=True)[1]
    urcrnrlat = llcrnrlat
    llcrnrx,llcrnry = p(llcrnrlon,llcrnrlat)
    p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0='+str(float(lon_0))+' +lat_ts=90.0 +lat_0=90.0 +x_0='\
                  +str(-llcrnrx)+' +y_0='+str(-llcrnry), preserve_units=True)
    urcrnrx,urcrnry = p(urcrnrlon,urcrnrlat)

    nx = int(urcrnrx/grid_res)+1
    ny = int(urcrnry/grid_res)+1
    dx = urcrnrx/(nx-1)
    dy = urcrnry/(ny-1)

    x = dx*np.indices((ny,nx),np.float32)[1,:,:]
    y = dy*np.indices((ny,nx),np.float32)[0,:,:]
    lon,lat = p(x,y,inverse=True)
    return lon,lat,x,y,p

def read_SIE(): 
    SIEs = {}
    SIEs_dt = {}
    SIEs_trend = {}
    with closing(request.urlopen(sie_ftp+'/north/monthly/data/N_09_extent_v3.0.csv')) as r:
        with open(home+'/DATA/N_09_extent_v3.0.csv', 'wb') as f:
                shutil.copyfileobj(r, f)
    with closing(request.urlopen(sie_ftp+'/seaice_analysis/N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx')) as r:
        with open(home+'/DATA/N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx', 'wb') as f:
                shutil.copyfileobj(r, f)
    xls = pd.ExcelFile(home+'/DATA/N_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx')
    SIEs['Pan-Arctic'] = np.genfromtxt(home+'/DATA/N_09_extent_v3.0.csv',delimiter=',').T[4][1:]
    SIEs['Beaufort'] = np.array(np.array(pd.read_excel(xls, 'Beaufort-Extent-km^2')['September'])[3:-1]/1e6,dtype='float32')
    SIEs['Chukchi'] = np.array(np.array(pd.read_excel(xls, 'Chukchi-Extent-km^2')['September'])[3:-1]/1e6,dtype='float32')
    
    n = (fyear-1)-1979+1
    for tag in SIEs:
        trend = np.zeros(2)
        dt = np.zeros(n)
        reg = linregress(np.arange(n),SIEs[tag])
        lineT = (reg[0]*np.arange(n)) + reg[1]
        trend[0] = reg[0]
        trend[1] = reg[1]
        dt[:] = SIEs[tag][:]-lineT
        SIEs_trend[tag] = trend
        SIEs_dt[tag] = dt
            
    return SIEs,SIEs_dt,SIEs_trend

def readNSIDC(ymax):
    dimX = 448
    dimY = 304
    SIC = {}
    SIC['lat'] = (np.fromfile(home+"/misc/psn25lats_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['lon'] = (np.fromfile(home+"/misc/psn25lons_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['psa'] = (np.fromfile(home+"/misc/psn25area_v3.dat",dtype='<i4').reshape(dimX,dimY))/1000
    SIC['lonr'],SIC['latr'],SIC['xr'],SIC['yr'],p = make_npstere_grid(65,360,1e5)
    SIC['x'],SIC['y'] = p(SIC['lon'],SIC['lat'])
    dXR,dYR = SIC['xr'].shape
    SIC['psar'] = 16*griddata((SIC['x'].ravel(),SIC['y'].ravel()),SIC['psa'].ravel(),(SIC['xr'],SIC['yr']),'linear')
    data_regrid = np.zeros((dXR,dYR,ymax-1979+1))*np.nan

    #NRT data#
    if os.path.exists('DATA/NSIDC0081_SEAICE_PS_N25km_'+str(ymax)+'0731_v2.0.nc') == False:
        os.system("sed -i '' 's/DATEMIN/"+str(ymax)+"-07-01/g' nsidc_download_nrt.py")
        os.system("sed -i '' 's/DATEMAX/"+str(ymax)+"-07-31/g' nsidc_download_nrt.py")
        os.system("python nsidc_download_nrt.py")
        while os.path.exists('NSIDC0081_SEAICE_PS_N25km_'+str(ymax)+'0731_v2.0.nc')==False:
            pass
        os.system('rm NSIDC0081*.xml')
        os.system('rm NSIDC0081*.png')
        os.system('mv NSIDC0081*.nc DATA')
        os.system("sed -i '' 's/"+str(ymax)+"-07-01/DATEMIN/g' nsidc_download_nrt.py")
        os.system("sed -i '' 's/"+str(ymax)+"-07-31/DATEMAX/g' nsidc_download_nrt.py")
            
    NRT = xr.open_mfdataset('DATA/NSIDC0081*.nc',concat_dim='time',combine='nested').F18_ICECON
    NRT = NRT.where(NRT<=1).mean('time').to_numpy()
            
    #MONTHLY
    if os.path.exists('DATA/NSIDC0051_SEAICE_PS_N25km_'+str((ymax-1))+'07_v2.0.nc') == False:
        os.system("sed -i '' 's/YMAX/"+str((ymax-1))+"/g' nsidc_download_monthly.py")
        os.system("python nsidc_download_monthly.py")
        while os.path.exists('NSIDC0051_SEAICE_PS_N25km_'+str((ymax-1))+'07_v2.0.nc')==False:
            pass
        os.system('rm NSIDC0051*.xml')
        os.system('mv NSIDC0051*.nc DATA')
        os.system("sed -i '' 's/"+str((ymax-1))+"-07-31/YMAX-07-31/g' nsidc_download_monthly.py")

    for k,year in enumerate(range(1979,ymax+1)):
        if year <= 1987:
            key = 'N07_ICECON'
            hole = 84.5
        elif (year>1987) & (year<=1991):
            key = 'F08_ICECON'
            hole = 87.2
        elif (year>1991) & (year<=1995):
            key = 'F11_ICECON'
            hole = 87.2
        elif (year>1995) & (year<=2007):
            key = 'F13_ICECON'
            hole = 87.2
        elif year>2007:
            key = 'F17_ICECON'
            hole = 89.2
        if year == ymax:
            data = NRT
        else:
            data = xr.open_dataset('DATA/NSIDC0051_SEAICE_PS_N25km_'+str(year)+'07_v2.0.nc')[key].isel(time=0).to_numpy()
            data[data>1] = np.nan

        phole = np.nanmean(data[(SIC['lat'] > hole-0.5) & (SIC['lat'] < hole)]) #calculate the mean 0.5 degrees around polar hole
        filled = np.ma.where((SIC['lat'] >= hole-0.5), phole, data)
        data_regrid[:,:,k] = griddata((SIC['x'].ravel(),SIC['y'].ravel()),filled.ravel(),\
                                             (SIC['xr'],SIC['yr']),'linear')
    SIC['data'] = data_regrid
    return SIC

def detrend(dataset):
    import itertools
    data = dataset['data']
    X = data.shape[0] ; Y = data.shape[1] ; T = data.shape[2]
    detrended = np.zeros(data.shape)*np.nan
    trend = np.zeros((X,Y,2))*np.nan
    for i,j in itertools.product(range(X),range(Y)):
        if ~np.isnan(data[i,j,range(T)]).all():
            reg = linregress(np.arange(T),data[i,j,range(T)])
            lineT = (reg[0]*np.arange(T)) + reg[1]
            trend[i,j,0] = reg[0]
            trend[i,j,1] = reg[1]
            detrended[i,j,range(T)]=data[i,j,range(T)]-lineT

    dataset['dt'] = detrended
    dataset['trend'] = trend

def networks(dataset):
    from ComplexNetworks import CN
    network = CN.Network(data=dataset['dt'])
    CN.Network.tau(network, 0.01)
    CN.Network.area_level(network,latlon_grid=False)
    CN.Network.intra_links(network, area=dataset['psar'])
    dataset['nodes'] = network.V
    dataset['anoms'] = network.anomaly

def forecast(ymax):
    regions = ['Pan-Arctic','Beaufort','Chukchi']
    l_init = [np.logspace(-7,2,20)[9],np.logspace(-7,2,20)[7],np.logspace(-7,2,20)[3]]
    sigma_init = [np.logspace(-3,9,20)[4],np.logspace(-3,9,20)[13],np.logspace(-3,9,20)[13]]
    alaska = 0
    for k in range(3):
        y = np.asarray([SIEs_dt[regions[k]]]).T #n x 1
        n = len(y)
        X = []
        for area in SIC['anoms']:
            r,p = pearsonr(y[:,0],SIC['anoms'][area][:-1])
            if k == 0:
                X.append(SIC['anoms'][area])
            else:
                if (r>0) & (p/2<0.08):
                    X.append(SIC['anoms'][area])         

        X = np.asarray(X).T #n x N
        Xs = np.asarray([X[-1,:]])
        X = X[:-1,:]

        M = np.abs(np.cov(X, rowvar=False, bias=True))
        np.fill_diagonal(M,0)
        np.fill_diagonal(M,-np.sum(M,axis=0))

        def MLII(hyperparameters): #Empirical Bayesian technique for optimisation of hyperparameters
            ℓ = np.exp(hyperparameters[0]) ; σn_tilde = np.exp(hyperparameters[1])
            try:
                Σ_tilde = expm(ℓ*M)
                L_tilde = np.linalg.cholesky(np.linalg.multi_dot([X,Σ_tilde,X.T]) + np.eye(n)*σn_tilde)
                A_tilde = np.linalg.solve(L_tilde.T,np.linalg.solve(L_tilde,y))
                σf = (np.dot(y.T,A_tilde)/n)[0][0]
                σn = σf*σn_tilde
                Σ = σf * expm(ℓ*M)
                L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn)
                α = np.linalg.solve(L.T,np.linalg.solve(L,y))
                nlML = np.dot(y.T,α)/2 + np.log(L.diagonal()).sum() + n*np.log(2*np.pi)/2

                dKdℓ = np.linalg.multi_dot([X,np.dot(M,Σ),X.T]) + np.eye(n)*σn
                dKdσ_tilde = np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σf

                dKdθ1 = ((np.trace(np.linalg.solve(L.T,np.linalg.solve(L,dKdℓ)))/2 - np.linalg.multi_dot([α.T,dKdℓ,α])/2))[0][0]
                dKdθ2 = ((np.trace(np.linalg.solve(L.T,np.linalg.solve(L,dKdσ_tilde)))/2 - np.linalg.multi_dot([α.T,dKdσ_tilde,α])/2))[0][0]

            except (np.linalg.LinAlgError,ValueError,OverflowError) as e:
                nlML = np.inf
                dKdθ1 = np.inf ; dKdθ2 = np.inf
            return np.squeeze(nlML), np.asarray([dKdθ1,dKdθ2])

        #θ = minimize(MLII,x0=[np.log(l_init[k]),np.log(sigma_init[k])],\
        #                                     method='CG',jac=True,options={'disp':False}).x

        #ℓ = np.exp(θ[0]) ; σn_tilde = np.exp(θ[1])
        ℓ = l_init[k] ; σn_tilde = sigma_init[k]
        Σ_tilde = expm(ℓ*M)
        L_tilde = np.linalg.cholesky(np.linalg.multi_dot([X,Σ_tilde,X.T]) + np.eye(n)*σn_tilde)
        A_tilde = np.linalg.solve(L_tilde.T,np.linalg.solve(L_tilde,y))
        σf = (np.dot(y.T,A_tilde)/n)[0][0]
        σn = σf*σn_tilde
        Σ = σf * expm(ℓ*M)
        L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn)
        α = np.linalg.solve(L.T,np.linalg.solve(L,y))
        KXXs = np.linalg.multi_dot([X,Σ,Xs.T])
        KXsXs = np.linalg.multi_dot([Xs,Σ,Xs.T]) + σn
        v = np.linalg.solve(L,KXXs)

        fmean = np.dot(KXXs.T,α)[0][0]
        fvar = (KXsXs - np.dot(v.T,v))[0][0]
        lineT = (np.arange(ymax-1979+1)*SIEs_trend[regions[k]][0]) + SIEs_trend[regions[k]][1]
        fmean_rt = fmean + lineT[-1]
        
        if k == 0:
            print(regions[k]+' September '+str(ymax)+' forecast:')
            print('Extent: '+str(fmean_rt.round(2))+' +/- '+str(np.sqrt(fvar).round(2))+' million km squared')
            print('Extent anomaly: '+str(fmean.round(2))+' +/- '+str(np.sqrt(fvar).round(2))+' million km squared')
        else:
            alaska += fmean_rt
    print('Alaska region '+str(ymax)+' forecast:')
    print('Extent: '+str(alaska.round(2))+' million km squared (total Alaska area is 4 million km squared)')

home = os.getcwd()
if os.path.exists(home+'/DATA')==False:
    os.mkdir(home+'/DATA')
    os.chmod(home+'/DATA',0o0777)
sie_ftp = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135'
fyear = int(datetime.date.today().year)

print('Downloading and reading data...')
SIEs,SIEs_dt,SIEs_trend = read_SIE()
SIC = readNSIDC(ymax=fyear)
print('Processing data...')
detrend(SIC)
networks(SIC)
print('Running forecast...')
forecast(ymax=fyear)
cleanup = input('Would you like to remove all the downloaded data files to save disk space? y  n:\n')
if cleanup == 'y':
    shutil.rmtree(home+'/DATA',ignore_errors=True)
