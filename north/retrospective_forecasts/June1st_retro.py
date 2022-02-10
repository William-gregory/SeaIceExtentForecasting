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

def read_SIE(fmin,fmax): 
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
    SIEs['Pan-Arctic'] = (np.genfromtxt(home+'/DATA/N_09_extent_v3.0.csv',delimiter=',').T[4][1:])[:fmax-1979+1]
    SIEs['Beaufort'] = (np.array(np.array(pd.read_excel(xls, 'Beaufort-Extent-km^2')['September'])[3:-1]/1e6,dtype='float64')[:fmax-1979+1]).round(3)
    SIEs['Chukchi'] = (np.array(np.array(pd.read_excel(xls, 'Chukchi-Extent-km^2')['September'])[3:-1]/1e6,dtype='float64')[:fmax-1979+1]).round(3)
    
    for tag in SIEs:
        trend = np.zeros((fmax-(fmin-1)+1,2))
        dt = np.zeros((fmax-(fmin-1)+1,fmax-1979+1))
        for year in range(fmin-1,fmax+1):
            n = year-1979+1
            reg = linregress(np.arange(n),SIEs[tag][range(n)])
            lineT = (reg[0]*np.arange(n)) + reg[1]
            trend[year-(fmin-1),0] = reg[0]
            trend[year-(fmin-1),1] = reg[1]
            dt[year-(fmin-1),range(n)] = SIEs[tag][range(n)]-lineT
        SIEs_trend[tag] = trend
        SIEs_dt[tag] = dt.round(3)
    return SIEs,SIEs_dt,SIEs_trend

def readNSIDC(fmin,fmax):
    dimX = 448
    dimY = 304
    SIC = {}
    SIC['lat'] = (np.fromfile(home+"/misc/psn25lats_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['lon'] = (np.fromfile(home+"/misc/psn25lons_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['psa'] = (np.fromfile(home+"/misc/psn25area_v3.dat",dtype='<i4').reshape(dimX,dimY))/1000
    SIC['x'],SIC['y'] = m(SIC['lon'],SIC['lat'])
    SIC['lonr'],SIC['latr'] = m.makegrid(int((m.xmax-m.xmin)/1e5)+1, int((m.ymax-m.ymin)/1e5)+1)
    SIC['xr'],SIC['yr']=m(SIC['lonr'],SIC['latr'])
    dXR,dYR = SIC['xr'].shape
    SIC['psar'] = 16*griddata((SIC['x'].ravel(),SIC['y'].ravel()),SIC['psa'].ravel(),(SIC['xr'],SIC['yr']),'linear')
    data_regrid = np.zeros((dXR,dYR,fmax-1979+1))*np.nan
    k = 0
    for year in range(1979,fmax+1):
        if (year == ymax) or (year == ymax-1):
            if len(glob.glob(home+'/DATA/nt_'+str(year)+'05*nrt_n.bin'))==0:
                for day in range(1,31+1):
                    with closing(request.urlopen(sic_ftp1+'/nt_'+str(year)+'05'+str('%02d'%day)+'_f18_nrt_n.bin')) as r:
                        with open(home+'/DATA/nt_'+str(year)+'05'+str('%02d'%day)+'_f18_nrt_n.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
                    
            files = sorted(glob.glob(home+'/DATA/nt_'+str(year)+'05'+'*nrt_n.bin'))
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
            files = glob.glob(home+'/DATA/nt_'+str(year)+'05*.1_n.bin')
            if len(files) == 0:
                with closing(request.urlopen(sic_ftp2+'/nt_'+str(year)+'05_'+sat+'_v1.1_n.bin')) as r:
                    with open(home+'/DATA/nt_'+str(year)+'05_'+sat+'_v1.1_n.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
            icefile = open(glob.glob(home+'/DATA/nt_'+str(year)+'05*.1_n.bin')[0], 'rb')
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = 300)
            monthly = (np.array(z).reshape((dimX,dimY)))/250
        monthly[monthly>1] = np.nan
        if year <= 1987:
            hole=84.5
        elif (year > 1987) & (year < 2008):
            hole=87.2
        else:
            hole=89.2
        phole = np.nanmean(monthly[(SIC['lat'] > hole-0.5) & (SIC['lat'] < hole)]) #calculate the mean 0.5 degrees around polar hole
        filled = np.ma.where((SIC['lat'] >= hole-0.5), phole, monthly)
        data_regrid[:,:,k] = griddata((SIC['x'].ravel(),SIC['y'].ravel()),filled.ravel(),\
                                             (SIC['xr'],SIC['yr']),'linear')
        k += 1
    SIC['data'] = data_regrid
    return SIC

def readERA5(fmax,min_lat=40):
    if os.path.exists(home+'/DATA/ERA5_May_SST_'+str(fmax)+'.nc')==False:
        import cdsapi

        c = cdsapi.Client()
        dates = np.arange(1979,fmax+1)
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'sea_surface_temperature',
                'year': [str(date) for date in dates],
                'month': '05',
                'time': '00:00',
                'grid':'4/2',
                'area': [
                    90, -180, min_lat,
                    180,
                ],
            },
            home+'/DATA/ERA5_May_SST_'+str(fmax)+'.nc')

    SST = {}
    file = Dataset(home+'/DATA/ERA5_May_SST_'+str(fmax)+'.nc')
    SST['lon'],SST['lat'] = np.meshgrid(np.arange(-180,180,4),np.arange(90,38,-2))
    #data1 = np.array(file['sst'])[:-1,0,:,:].transpose(1,2,0)
    #data2 = np.array(file['sst'])[-1,1,:,:]
    #data = np.dstack((data1,data2[:,:,np.newaxis]))
    data = np.array(file['sst']).transpose(1,2,0)
    data[SST['lat']<min_lat] = np.nan
    data = data - 273.15
    data[data<0] = np.nan
    SST['data'] = data
    return SST

def detrend(dataset,fmin,fmax):
    import itertools
    for year in range(fmin,fmax+1):
        n = year-1979+1
        data = dataset['data'][:,:,range(n)]
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

        dataset['dt_'+str(year)] = detrended
        dataset['trend_'+str(year)] = trend

def networks(dataset,fmin,fmax,latlon=True):
    import ComplexNetworks as CN
    for year in range(fmin,fmax+1):
        network = CN.Network(data=dataset['dt_'+str(year)])
        CN.Network.tau(network, 0.01)
        CN.Network.area_level(network,latlon_grid=latlon)
        if latlon:
            CN.Network.intra_links(network, lat=dataset['lat'])
        else:
            CN.Network.intra_links(network, area=dataset['psar'])
        dataset['nodes_'+str(year)] = network.V
        dataset['anoms_'+str(year)] = network.anomaly

def forecast(fmin,fmax):
    regions = ['Pan-Arctic','Beaufort','Chukchi']
    GPR = {}
    l_init = [np.logspace(-7,2,20)[16],np.logspace(-7,2,20)[14],np.logspace(-7,2,20)[12]]
    sigma_init = [np.logspace(-3,9,20)[1],np.logspace(-3,9,20)[4],np.logspace(-3,9,20)[6]]
    for k in range(3):
        fmean = np.zeros(fmax-fmin+1)
        fvar = np.zeros(fmax-fmin+1)
        fmean_rt = np.zeros(fmax-fmin+1)
        for year in range(fmin,fmax+1):
            y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(year-1979)]]).T #n x 1
            n = len(y)
            X = []
            for area in SIC['anoms_'+str(year)]:
                r,p = pearsonr(y[:,0],SIC['anoms_'+str(year)][area][:-1])
                if r>0:
                    X.append(SIC['anoms_'+str(year)][area])            
            for area in SST['anoms_'+str(year)]:
                r,p = pearsonr(y[:,0],SST['anoms_'+str(year)][area][:-1])
                if r<0:
                    X.append(-SST['anoms_'+str(year)][area])  

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

            fmean[year-fmin] = (np.dot(KXXs.T,α)[0][0]).round(3)
            fvar[year-fmin] = ((KXsXs - np.dot(v.T,v))[0][0]).round(3)
            lineT = (np.arange(year-1979+1)*SIEs_trend[regions[k]][year-(fmin-1)-1,0]) + SIEs_trend[regions[k]][year-(fmin-1)-1,1]
            fmean_rt[year-fmin] = (fmean[year-fmin] + lineT[-1]).round(3)
                    
        GPR[regions[k]+'_fmean'] = fmean
        GPR[regions[k]+'_fvar'] = fvar
        GPR[regions[k]+'_fmean_rt'] = fmean_rt
    return GPR

def skill(fmin,fmax):
    regions = ['Pan-Arctic','Beaufort','Chukchi']
    skill_rt = []
    skill_dt = []
    dt_obs = []
    for k in range(3):
        dt = []
        for t in range(fmin,fmax+1):
            n = t - 1979
            dt.append(SIEs_dt[regions[k]][t-(fmin-1),n])
        dt_obs.append(dt)
        forecast_rt = GPR[regions[k]+'_fmean_rt']
        obs_rt = SIEs[regions[k]][fmin-1979:]
        a = np.mean((obs_rt-forecast_rt)**2)
        b = np.mean((obs_rt-np.nanmean(obs_rt))**2)
        skill_rt.append((1 - (a/b)).round(3))

        forecast_dt = GPR[regions[k]+'_fmean']
        c = np.mean((dt-forecast_dt)**2)
        d = np.mean((dt-np.nanmean(dt))**2)
        skill_dt.append((1 - (c/d)).round(3))
    return skill_rt,skill_dt,dt_obs
        

home = os.getcwd()
if os.path.exists(home+'/DATA')==False:
    os.mkdir(home+'/DATA')
    os.chmod(home+'/DATA',0o0777)
sie_ftp = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135'
sic_ftp1 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0081_nrt_nasateam_seaice/north'
sic_ftp2 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0051_gsfc_nasateam_seaice/final-gsfc/north/monthly'

ymax = int(datetime.date.today().year)
m = Basemap(projection='npstere',boundinglat=65,lon_0=360,resolution='l')
fmin = int(input('Please specify first year you would like to forecast (must be > 1980):\n'))
fmax = int(input('Please specify last year you would like to forecast (must be < '+str(ymax)+'):\n'))
if fmin < 1981:
    fmin = 1981
if fmax > ymax-1:
    fmax = ymax-1

print('Downloading and reading data...')
SIEs,SIEs_dt,SIEs_trend = read_SIE(fmin,fmax)
SIC = readNSIDC(fmin,fmax)
SST = readERA5(fmax)
print('Processing data...')
detrend(SIC,fmin,fmax)
detrend(SST,fmin,fmax)
networks(SIC,fmin,fmax,latlon=False)
networks(SST,fmin,fmax,latlon=True)
print('Running forecast...')
GPR = forecast(fmin,fmax)
skill_rt,skill_dt,dt_obs = skill(fmin,fmax)

years = np.arange(fmin,fmax+1).tolist()
years.append('Skill')

def prep(data,skill=None):
    if type(data)!=list:
        data = data.tolist()
    if skill is not None:
        data.append(skill)
    else:
        data.append('')
    return data

columns1 = ['Pan-Arctic$_o$','Pan-Arctic$_f$','Pan-Arctic$_f$ unc','Beaufort$_o$','Beaufort$_f$','Beaufort$_f$ unc','Chukchi$_o$','Chukchi$_f$','Chukchi$_f$ unc']
columns2 = ['Pan-Arctic$_o$','Pan-Arctic$_f$','Beaufort$_o$','Beaufort$_f$','Chukchi$_o$','Chukchi$_f$']
data_dt = list(zip(prep(dt_obs[0]),prep(GPR['Pan-Arctic_fmean'],skill_dt[0]),prep(np.sqrt(GPR['Pan-Arctic_fvar']).round(3)),prep(dt_obs[1]),prep(GPR['Beaufort_fmean'],\
                skill_dt[1]),prep(np.sqrt(GPR['Beaufort_fvar']).round(3)),prep(dt_obs[2]),prep(GPR['Chukchi_fmean'],skill_dt[2]),prep(np.sqrt(GPR['Chukchi_fvar']).round(3))))
df_dt = pd.DataFrame(data_dt, index=years, columns=columns1)

data_rt = list(zip(prep(SIEs['Pan-Arctic'][fmin-1979:]),prep(GPR['Pan-Arctic_fmean_rt'],skill_rt[0]),prep(SIEs['Beaufort'][fmin-1979:]),\
                   prep(GPR['Beaufort_fmean_rt'],skill_rt[1]),prep(SIEs['Chukchi'][fmin-1979:]),prep(GPR['Chukchi_fmean_rt'],skill_rt[2])))
df_rt = pd.DataFrame(data_rt, index=years, columns=columns2)

df_dt.to_csv('./June1st_detrended_forecasts_'+str(fmin)+'-'+str(fmax)+'.csv')
df_rt.to_csv('./June1st_forecasts_with_trend_'+str(fmin)+'-'+str(fmax)+'.csv')

cleanup = input('Would you like to remove all the downloaded data files to save disk space? y  n:\n')
if cleanup == 'y':
    shutil.rmtree(home+'/DATA',ignore_errors=True)
