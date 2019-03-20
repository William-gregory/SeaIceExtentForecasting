import GPR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

dimX = 448 #Polar stereo grid dimensions (25km)
dimY = 304

dimXR = 57 #downsampled Polar stereo (100km)
dimYR = 57

datapath = "./Data"
corrpath = "./Correlations"

latf = datapath+"/psn25lats_v3.dat"
lat = np.fromfile(latf,dtype='<i4')
lat = lat.reshape(448,304)
lat = lat/100000

lonf = datapath+"/psn25lons_v3.dat"
lon = np.fromfile(lonf,dtype='<i4')
lon = lon.reshape(448,304)
lon = lon/100000


daily_SIC_june = GPR.SIC_daily(np.zeros((dimX,dimY,2,30)),np.zeros((dimX,dimY,2)))
daily_SIC_sept = GPR.SIC_daily(np.zeros((dimX,dimY,2,30)),np.zeros((dimX,dimY,2)))
SIC_june = GPR.SIC_monthly(np.zeros((dimX,dimY,2016-1979+1)),np.zeros((dimXR,dimYR,2018-1979+1)),np.zeros((dimX,dimY,2018-1984+1,2)),np.zeros((dimXR,dimYR,2018-1984+1,2)),np.zeros((dimXR,dimYR,2018-1979+1)),{},{})
SIC_sept = GPR.SIC_monthly(np.zeros((dimX,dimY,2016-1979+1)),np.zeros((dimXR,dimYR,2018-1979+1)),np.zeros((dimX,dimY,2018-1984+1,2)),np.zeros((dimXR,dimYR,2018-1984+1,2)),np.zeros((dimX,dimY,2018-1979+1)),{},{})

GPR.SIC_daily.read_daily(daily_SIC_june,"06",30, 2018)
GPR.SIC_daily.read_daily(daily_SIC_sept,"09",30, 2018)
GPR.SIC_daily.create_monthly(daily_SIC_june, 2)
GPR.SIC_daily.create_monthly(daily_SIC_sept, 2)
GPR.SIC_monthly.read_monthly(SIC_june,"06")
GPR.SIC_monthly.read_monthly(SIC_sept,"09")
GPR.SIC_june.data = np.concatenate((SIC_june.data,daily_SIC_june.monthly),2)
GPR.SIC_sept.data = np.concatenate((SIC_sept.data,daily_SIC_sept.monthly),2)
GPR.lon_regrid, lat_regrid = SIC_monthly.regrid(SIC_june, 6, lon, lat)
GPR.lon_regrid, lat_regrid = SIC_monthly.regrid(SIC_sept, 9, lon, lat)
GPR.SIC_monthly.detrend(SIC_june, 2018, SIC_june.regrid, dimXR, dimYR)
GPR.SIC_monthly.detrend(SIC_sept, 2018, SIC_sept.data, dimX, dimY)

GPR.SIC_monthly.gen_networks(SIC_june, "june", lat_regrid)

june_GL = GPR.Forecast_gridlevel(np.zeros((dimX,dimY,2018-1985+1)),np.zeros((dimX,dimY,2018-1985+1)),np.zeros((dimX,dimY,2018-1985+1)),np.zeros((dimX,dimY,2018-1985+1)),np.zeros((dimX,dimY,2018-1985+1)))
GPR.Forecast_gridlevel.forecast(june_GL, SIC_sept, SIC_june.anomalies)


#Plot forecast results (obs, forecast (with trend) and probability)
m = Basemap(projection='npstere',boundinglat=65,lon_0=0, resolution='l')

%matplotlib inline
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 8

m.drawcoastlines(linewidth=0.25)
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))

xt,yt=m(lon,lat)
xt=np.array(xt)
yt=np.array(yt)
m.pcolormesh(xt, yt, SIC_sept.data[:,:,2018-1979])
plt.colorbar()                                      
plt.show()

m.drawcoastlines(linewidth=0.25)
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))

xt,yt=m(lon,lat)
xt=np.array(xt)
yt=np.array(yt)
m.pcolormesh(xt, yt, june_GL.fmean_rt[:,:,2018-1985])
plt.colorbar()                                      
plt.show()

m.drawcoastlines(linewidth=0.25)
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))

xt,yt=m(lon,lat)
xt=np.array(xt)
yt=np.array(yt)
m.pcolormesh(xt, yt, june_GL.probability[:,:,2018-1985])
plt.colorbar()                                      
plt.show()