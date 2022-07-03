# Wasserstein Distributional Learning


# Data

## Abstract
There are two real-world applications in our paper.

The data sets for the climate modeling are the daily temperature data from Berkeley Earth and the physical driver data from Intergovernmental Panel on Climate Change (IPCC).

The data sets for the income modeling are the income distribution data from American Community Survey (ACS) and the county-level health indices of from the County Health Rankings & Roadmaps program
 
In our experiment, we first calculated quantile and density functions from the raw point data. We then performed analyses using these transformed distribution functions.

## Availability
The data sets used in this paper are publicly availalable and the access links are provided in the Description section.

## Description
The data sets used in this paper are publicly available and the links to download them are provided below.

The daily temperature data set can be downloaded at 
[http://berkeleyearth.org/data/](http://berkeleyearth.org/data/).

The physical driver data can be downloaded at
[http://www.climatechange2013.org/images/report/WG1AR5_AIISM_Datafiles.xlsx](http://www.climatechange2013.org/images/report/WG1AR5_AIISM_Datafiles.xlsx).

The income distribution data of Year 2014 from ACS can be downloaded at
[https://www2.census.gov/programs-surveys/acs/data/pums/](https://www2.census.gov/programs-surveys/acs/data/pums/).

The  county-level health indices of Year 2014 from the County Health Rankings & Roadmaps program can be downloaded at
[https://www.countyhealthrankings.org/](https://www.countyhealthrankings.org/).


# Code

## Description
The WDL algorithm is implemented in Python 3. All the function definitions can be found in folder **lib/WDL.py**.

The first comparison method Frechet regression is implemented using the R package `frechet`.

The second comparison method CLR regression is implemented by adapting the R codes from the reference paper **Talská, R., Menafoglio, A., Machalová, J., Hron, K., & Fišerová, E. (2018). Compositional regression with functional response. Computational Statistics & Data Analysis, 123, 66-85**.

The third comparison method MDN is implemented in Python 3 with Tensorflow. All the function definitions are contained in Python notebooks.

# Instructions for Use

## Reproducibility
1. The subfolder named **codes/simulation** contains the codes to reproduce all the results in simulations, including Table 1 and Figure 2&3.
2. The subfolder named **codes/climate** contains the codes to reproduce all the results in climate modeling, including Figure 1, Figure 4&5&6, and Figure D in the Appendix.
3. The subfolder named **codes/income** contains the codes to reproduce all the results in income modeling, including Table 2 and Figure 7&8.
