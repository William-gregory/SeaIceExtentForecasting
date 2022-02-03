import numpy as np
import datetime
import shutil
from mpl_toolkits.basemap import Basemap
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
warnings.filterwarnings("ignore")

def read_SIE(): 
    SIEs = {}
    SIEs_dt = {}
    SIEs_trend = {}
    with closing(request.urlopen(sie_ftp+'/south/monthly/data/S_02_extent_v3.0.csv')) as r:
        with open(home+'/DATA/S_02_extent_v3.0.csv', 'wb') as f:
                shutil.copyfileobj(r, f)
    with closing(request.urlopen(sie_ftp+'/seaice_analysis/S_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx')) as r:
        with open(home+'/DATA/S_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx', 'wb') as f:
                shutil.copyfileobj(r, f)
    xls = pd.ExcelFile(home+'/DATA/S_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx')
    SIEs['Pan-Antarctic'] = np.genfromtxt(home+'/DATA/S_02_extent_v3.0.csv',delimiter=',').T[4][1:]
    SIEs['Ross'] = np.array(np.array(pd.read_excel(xls, 'Ross-Extent-km^2')['February'])[3:-1]/1e6,dtype='float32')
    SIEs['Weddell'] = np.array(np.array(pd.read_excel(xls, 'Weddell-Extent-km^2')['February'])[3:-1]/1e6,dtype='float32')
    
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
    dimX = 332
    dimY = 316
    SIC = {}
    SIC['lat'] = (np.fromfile(home+"/misc/pss25lats_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['lon'] = (np.fromfile(home+"/misc/pss25lons_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['psa'] = (np.fromfile(home+"/misc/pss25area_v3.dat",dtype='<i4').reshape(dimX,dimY))/1000
    SIC['x'],SIC['y'] = m(SIC['lon'],SIC['lat'])
    SIC['lonr'],SIC['latr'] = m.makegrid(int((m.xmax-m.xmin)/1e5)+1, int((m.ymax-m.ymin)/1e5)+1)
    SIC['xr'],SIC['yr']=m(SIC['lonr'],SIC['latr'])
    dXR,dYR = SIC['xr'].shape
    SIC['psar'] = 16*griddata((SIC['x'].ravel(),SIC['y'].ravel()),SIC['psa'].ravel(),(SIC['xr'],SIC['yr']),'nearest')
    data_regrid = np.zeros((dXR,dYR,ymax-1979+1))*np.nan
    month = 7
    k = 0
    for year in range(1979,ymax+1):
        if (year == ymax) or (year == ymax-1):
            if len(glob.glob(home+'/DATA/nt_'+str(year)+'10*nrt_s.bin'))==0:
                for day in range(1,31+1):
                    with closing(request.urlopen(sic_ftp1+'/nt_'+str(year)+'10'+str('%02d'%day)+'_f18_nrt_s.bin')) as r:
                        with open(home+'/DATA/nt_'+str(year)+'10'+str('%02d'%day)+'_f18_nrt_s.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
                    
            files = sorted(glob.glob(home+'/DATA/nt_'+str(year)+'10'+'*nrt_s.bin'))
            daily = np.zeros((dimX,dimY,len(files)))*np.nan
            f = 0
            for file in files:
                icefile = open(file,'rb')
                contents = icefile.read()
                icefile.close()
                s="%dB" % (int(dimX*dimY),)
                z=struct.unpack_from(s, contents, offset = 300)
                daily[:,:,f] = (np.array(z).reshape((dimX,dimY)))/250
                f += 1
            monthly = np.nanmean(daily,2)
        else:
            if year < 1988:
                sat = 'n07'
            elif (year > 1987) & (year < 1992):
                sat = 'f08'
            elif (year > 1991) & (year < 1996):
                sat = 'f11'
            elif (year > 1995) & (year < 2008):
                sat = 'f13'
            elif year > 2007:
                sat = 'f17'
            files = glob.glob(home+'/DATA/nt_'+str(year)+'10*.1_s.bin')
            if len(files) == 0:
                with closing(request.urlopen(sic_ftp2+'/nt_'+str(year)+'10_'+sat+'_v1.1_s.bin')) as r:
                    with open(home+'/DATA/nt_'+str(year)+'10_'+sat+'_v1.1_s.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
            icefile = open(glob.glob(home+'/DATA/nt_'+str(year)+'10*.1_s.bin')[0], 'rb')
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = 300)
            monthly = (np.array(z).reshape((dimX,dimY)))/250
        monthly[monthly>1] = np.nan
        data_regrid[:,:,k] = griddata((SIC['x'].ravel(),SIC['y'].ravel()),monthly.ravel(),\
                                             (SIC['xr'],SIC['yr']),'nearest')
        k += 1
    SIC['data'] = data_regrid
    return SIC

def readERA5(ymax,min_lat=40):
    if os.path.exists(home+'/DATA/ERA5_Oct_SST_'+str(ymax)+'.nc')==False:
        import cdsapi

        c = cdsapi.Client()
        dates = np.arange(1979,ymax+1)
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'sea_surface_temperature',
                'year': [str(date) for date in dates],
                'month': '10',
                'time': '00:00',
                'grid':'4/2',
                'area': [
                    -min_lat, -180, -90,
                    180,
                ],
            },
            home+'/DATA/ERA5_Oct_SST_'+str(ymax)+'.nc')

    SST = {}
    file = Dataset(home+'/DATA/ERA5_Oct_SST_'+str(ymax)+'.nc')
    SST['lon'],SST['lat'] = np.meshgrid(np.arange(-180,180,4),np.arange(90,38,-2))
    data1 = np.array(file['sst'])[:-1,0,:,:].transpose(1,2,0)
    data2 = np.array(file['sst'])[-1,1,:,:]
    data = np.dstack((data1,data2[:,:,np.newaxis]))
    #data = np.array(file['sst']).transpose(1,2,0)
    data[SST['lat']<min_lat] = np.nan
    data = data - 273.15
    data[data<0] = np.nan
    SST['data'] = data
    return SST

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

def networks(dataset,latlon=True):
    import ComplexNetworks as CN
    dimXR = dataset['dt'].shape[0] ; dimYR = dataset['dt'].shape[1]
    network = CN.Network(dimX=dimXR,dimY=dimYR)
    CN.Network.tau(network, dataset['dt'], 0.01)
    CN.Network.area_level(network, dataset['dt'],latlon_grid=latlon)
    if latlon:
        CN.Network.intra_links(network, dataset['dt'], lat=dataset['lat'])
    else:
        CN.Network.intra_links(network, dataset['dt'], area=dataset['psar'])
    dataset['nodes'] = network.V
    dataset['anoms'] = network.anomaly

def forecast(ymax):
    regions = ['Pan-Antarctic','Ross','Weddell']
    alaska = 0
    for k in range(3):
        y = np.asarray([SIEs_dt[regions[k]]]).T #n x 1
        n = len(y)
        X = []
        for area in SIC['anoms']:
            r,p = pearsonr(y[:,0],SIC['anoms'][area][:-1])
            if r>0:
                X.append(SIC['anoms'][area])            
        for area in SST['anoms']:
            r,p = pearsonr(y[:,0],SST['anoms'][area][:-1])
            if r<0:
                X.append(-SST['anoms'][area])  

        X = np.asarray(X).T #n x N
        X = (X - np.mean(X,0))/np.std(X,0)
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

        l_init = [np.logspace(-2,5,15)[5],np.logspace(-2,5,15)[4],np.logspace(-2,5,15)[2]]
        sigma_init = [np.logspace(-4,1,15)[1],np.logspace(-4,1,15)[1],np.logspace(-4,1,15)[11]]

        θ = minimize(MLII,x0=[np.log(l_init[k]),np.log(sigma_init[k])],\
                                             method='CG',jac=True,options={'disp':False}).x

        ℓ = np.exp(θ[0]) ; σn_tilde = np.exp(θ[1])
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
        
        print(regions[k]+' February '+str(ymax)+' forecast:')
        print('Extent: '+str(fmean_rt.round(2))+' +/- '+str(np.sqrt(fvar).round(2))+' million km squared')
        print('Extent anomaly: '+str(fmean.round(2))+' +/- '+str(np.sqrt(fvar).round(2))+' million km squared')

home = os.getcwd()
if os.path.exists(home+'/DATA')==False:
    os.mkdir(home+'/DATA')
    os.chmod(home+'/DATA',0o0777)
sie_ftp = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135'
sic_ftp1 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0081_nrt_nasateam_seaice/south'
sic_ftp2 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0051_gsfc_nasateam_seaice/final-gsfc/south/monthly'

m = Basemap(projection='spstere',boundinglat=-55,lon_0=180,resolution='l')
fyear = int(datetime.date.today().year)+1

print('Downloading and reading data...')
SIEs,SIEs_dt,SIEs_trend = read_SIE()
SIC = readNSIDC(ymax=fyear-1)
SST = readERA5(ymax=fyear-1)
print('Processing data...')
detrend(SIC)
detrend(SST)
networks(SIC,latlon=False)
networks(SST,key="_SST_40S_79-"+str((fyear-1)))
print('Running forecast...')
forecast(ymax=fyear)
cleanup = input('Would you like to remove all the downloaded data files to save disk space? y  n:\n')
if cleanup == 'y':
    shutil.rmtree(home+'/DATA',ignore_errors=True)








