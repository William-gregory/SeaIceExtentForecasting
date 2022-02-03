This directory contains a series of python scripts to run sea ice extent
forecasts in line with the Sea Ice Prediction Networks annual call for
submissions. The call for submissions begins each June (June 1st) where 
forecasts of the September sea ice extent are made based on data from
May -- the deadline to submit the forecast to SIPN is typically the 12th 
of each call month. This then proceeds for July 1st, August 1st, and 
September 1st.

The forecast method in this case is the statistical model outlined in 
Gregory et al., 2020. To run this code for the first time, please follow
the steps below:

1) cp -r /home/wjg/SIPN_forecasts ~/.
2) cp ~/SIPN_forecasts/north/misc/ComplexNetworks.py ~/.local/lib/python3.7/site-packages/.
3) conda activate base
4) pip install --user netCDF4
5) If you don't have a NASA EarthData account, get one here: https://urs.earthdata.nasa.gov/users/new
6) Once this is done, type command 7) below into the terminal, replacing uid and psswd with your details
7) echo 'machine urs.earthdata.nasa.gov login uid password psswd' >> ~/.netrc
8) chmod 0600 ~/.netrc
9) Follow the steps here https://cds.climate.copernicus.eu/api-how-to to get a Copernicus CDS account
10) Place the hidden file .cdsapirc containing the url information and your api key into ~/. on the CPOM server
11) You may need to run pip install --user cdsapi
12) python July1st.py



 