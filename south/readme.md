This directory contains a series of python scripts to run sea ice extent
forecasts in line with the Sea Ice Prediction Networks (south) annual call for
submissions. The call for submissions begins each November (November 1st) where
forecasts of the February sea ice extent are made based on data from
October -- the deadline to submit the forecast to SIPN is typically the 12th
of each call month. This then proceeds for December 1st, January 1st, and
February 1st. Note that these scripts are setup to run forecasts in real-time,
hence they should only be executed after the latest data become available. In
other words they should be run in the 12 day window after each call month.

The forecast method in this case is the statistical model outlined in
Gregory et al., 2020. To run this code for the first time, please follow
the steps below:

1) cp ../ComplexNetworks.py ~/.local/lib/python3.7/site-packages/.
2) pip install --user netCDF4
3) pip install --user openpyxl
4) If you don't have a NASA EarthData account, get one here: https://urs.earthdata.nasa.gov/users/new
5) Once this is done, type command 5) below into the terminal, replacing uid and psswd with your details
6) echo 'machine urs.earthdata.nasa.gov login uid password psswd' >> ~/.netrc
7) chmod 0600 ~/.netrc
8) Follow the steps here https://cds.climate.copernicus.eu/api-how-to to get a Copernicus CDS account
9) Place the hidden file .cdsapirc containing the url information and your api key into ~/.
10) You may need to run pip install --user cdsapi
11) python November1st.py (the forecasts will be printed to the terminal)



 
