# Sea Ice Extent Forecasting
This repository contains Python code to generate probabilistic seasonal forecasts of regional and pan-Arctic September sea ice extent, in line with the Sea Ice Prediction Network's annual calls for submissions (Pan-Arctic and Alaskan seas). Each python script can be executed on the first day of June, July, August, or September each year, where it will subsequently download the relevant data and generate pan-Arctic and Alaska predictions of the following September sea ice extents for the current year. Furthermore, the retrospective forecast scripts can also be executed to generate forecasts over a range of past years (the user will be prompted in the terminal which years are to be forecast).

The same scripts for forecasting Antarctic summer (February) sea ice (Pan-Antarctic, Ross and Weddell seas), are also available.

The method is based on Complex Networks and Gaussian Process Regression, as outlined in the article by [Gregory et al., 2020](https://discovery.ucl.ac.uk/id/eprint/10091542/1/Gregory_wafd190107.pdf)

