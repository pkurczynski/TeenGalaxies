#!/usr/bin/env python
"""
sfr_life_vs_m_line_clip_sim.py                       Python3 script

THIS VERSION FOR 2015-05-16-A/sample01/run03  lin-exp SFH FITS.

usage: sfr_life_vs_m_line_clip_sim.py [-h] [-v] [-d] [-i ITERATIONS]
                                 [-c CLIP_SIGMA] [-z ZSPEAGLE2014]
                                 input_file output_file

Analyzes SFR-M* relation of input data with various methods and compares
results to the literature (Speagle+2014). Performs clipping of outliers to a
specitied threshold (default 3 sigma), and re-fits the data. Estimates
intrinsic scatter of the data using various methods. Performs simulations to
estimate errors. Outputs pdf files showing data, best fit model(s) and
residuals, and optional text data to console.

positional arguments:
  input_file            The input .fits file. See program listing for columns
                        used.
  output_file           The output .pdf file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Print descriptive text, including results of analysis.
  -d, --display         Display the plot.
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations in simulation. Default = 1000
  -c CLIP_SIGMA, --clip_sigma CLIP_SIGMA
                        Outlier threshold (sigma). Default = 3.0
  -z ZSPEAGLE2014, --zSpeagle2014 ZSPEAGLE2014
                        Redshift for Speagle+2014 Main Sequence correlation


example:  bash$ 
./sfr_life_vs_m_line_clip_sim.py -v -i 1000 -c 3.0 -z 1.2 2014-11-13-A__sample01__run04__speedymc_results.fits sfr_life_vs_m_line_clip_sim_z12.pdf > sfr_life_vs_m_line_clip_sim_z12.log.txt


Execution time for 10000 realizations is about 30 seconds.

See Experiment 2014-11-26-A, 2014-11-13-A, 2014-09-05-A. 

v0 01/05/2015
v1 02/16/2015  include residual plots, improved output format, anova
v3 04/03/2015  updated with most recent versions of 
v4 04/21/2015  minor cosmetic changes
v5 05/30/2015  estimate parameters using covariances; output refit data
                with covariances, for upload to khuseti...run IDL program.
v6 06/30/2015	new clipping method.  residuals saved as txt file
"""

import argparse
import astropy
from astropy.cosmology import WMAP9 as cosmo
import astropy.io.fits as pyfits
from datetime import datetime
from linfit_multi_methods import linfit_f87, linfit_multi, linfit_multi_simerr, scatter_variance_adhoc
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import EllipseCollection
from matplotlib.ticker import NullFormatter, MultipleLocator
import numpy as np
import sys
import scipy.stats as s
 
if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description="Analyzes SFR-M* relation of input data with various methods and compares results to the literature (Speagle+2014).  Performs clipping of outliers to a specitied threshold (default 3 sigma), and re-fits the data.  Estimates intrinsic scatter of the data using various methods.  Performs simulations to estimate errors.  Outputs pdf files showing data, best fit model(s) and residuals, and optional text data to console.")
    parser.add_argument('input_file', help='The input .fits file.  See program listing for columns used.')
    parser.add_argument('output_file', help='The output .pdf file')
    parser.add_argument('-v','--verbose', action = 'store_true', help='Print descriptive text, including results of analysis.')
    parser.add_argument('-d','--display', action = 'store_true', help='Display the plot.')
    parser.add_argument('-i','--iterations', type = int, default = '1000', help='Number of iterations in simulation.  Default = 1000')
    parser.add_argument('-c','--clip_sigma', type = float, default = '3.0', help='Outlier threshold (sigma).  Default = 3.0')
    parser.add_argument('-z','--zSpeagle2014', type = float, default = '1.0', help='Redshift for Speagle+2014 Main Sequence correlation line.  Default = 1.1.')
    
    args = parser.parse_args()
    
    if (args.verbose):
        print("sfr_life_vs_m_line_clip_sim.py \n")
        
        print("Run date and time :  ", str(datetime.now()))
        print("Python Version    :  ", sys.version)
        print("Astropy version   :  ", astropy.__version__)
        
    """    
    Read the input data:  retrieve SFR, M* values
    """
    hduList = pyfits.open(args.input_file)
    tbData = hduList[1].data
        
    logStellarMass_Input_Value = tbData.field('LogStellarMass_Msun_Expected_Value')
    logStellarMass_Input_Err = tbData.field('LogStellarMass_Msun_Expected_Err68')
    
    logSFR_Input_Value = tbData.field('LogSFR_Life_Expected_Value')
    logSFR_Input_Err = tbData.field('LogSFR_Life_Expected_Err68')
    
    """
    Parameter indices for determining covariance matrix elements:
    
    Table 1.  Parameters in 2014-11-13-A/sample01/run03
    
    lab1 = LogAge_Yr
    lab2 = LogGalaxyMass_Msun
    lab3 = EBV
    lab4 = LogTau_Yr
    lab5 = LogStellarMass_Msun
    lab6 = LogSFR_Inst
    lab7 = LogSFR_Life
    lab8 = LogSFR_100
    lab9 = LogSFR_Max
    lab10 = LogSFR_MaxAvg
    lab11 = LogGalaxyMass_Inf_Msun
    
    Caption:  Parameters copied from file on khuseti:
    /Volumes/users/pkurczynski/analysis/candels/2014-11-13-A/sample01/run03/analysis_v2/
    sed__candels_2014a_sample01__id_fixed_distparams.ini.  
    Use parameter indices to identify the proper correlation matrix 
    element, e.g. Cov(5,7) is Cov(log StellarMass, logSFR_Life).
    """

    Covariance_LogStellarMass_LogSFR = tbData.field('Cov_5_7')
    Variance_LogStellarMass = tbData.field('Cov_5_5')
    Variance_LogSFR = tbData.field('Cov_7_7')

    Chisq_BestFit = tbData.field('Chisq_BestFit')
    
    Correlation_LogStellarMass_LogSFR = tbData.field('Corr_5_7')
    Source_ID = tbData.field('ID')

    """    
    Slice the data:  Select subset of input data for fitting
    
    Description  Selection criterion syntax
    (1)             (2)
    
    good chisq      goodData = (tbData.field('Chisq_BestFit') < 50.0)
    only
        
    good chisq
    good GR conv.   goodData = ((tbData.field('Chisq_BestFit') < 50.0) & (tbData.field('GR_StellarMass') < 0.2))

    """
    #goodData = (tbData.field('Chisq_BestFit') < 50.0)
    goodData = ((tbData.field('Chisq_BestFit') < 50.0) & (tbData.field('LogStellarMass_Msun_GR_Converge') < 0.2))
    
    """
    Compute Chisq statistics of SED fits
    
    Table 1.5.  Parameters in SED fits; chisq degrees of freedom
    
    	RunID	df		Free Parameters
    	(1)	(2)		(3)
    	
    	run02	14		Age, EBV, GalMass
    	run03	13		Age, EBV, GalMass, tau
    	run04	13		Age, EBV, GalMass, tau
    	
    Caption:  Free parameters used to determine degrees of freedom in 
    reduced chisq. (1) is the SpeedyMC Run ID.  (2) is the df for chisq 
    estimate:  17 - (Number of free parameters).  NB:  There are 17 
    photometry measurements in the SEDs.  (3) is the list of free 
    parameters, ie parameters in .ini file that are allowed to vary.
    """
    chisq_df = 13.0
    Chisqv_BestFit = Chisq_BestFit / chisq_df
    chisqv_goodData = Chisqv_BestFit[goodData]
    chisqv_goodData_median = np.median(chisqv_goodData)
    chisqv_goodData_min = np.min(chisqv_goodData)
    chisqv_goodData_max = np.max(chisqv_goodData)
    
    x_goodData_unscaled = logStellarMass_Input_Value[goodData]
    x_goodData_unscaled_err = logStellarMass_Input_Err[goodData]

    y_goodData_unscaled = logSFR_Input_Value[goodData]
    y_goodData_unscaled_err = logSFR_Input_Err[goodData]
    
    covxy_goodData = Covariance_LogStellarMass_LogSFR[goodData]
    varxx_goodData = Variance_LogStellarMass[goodData]
    varyy_goodData = Variance_LogSFR[goodData]
    
    corrxy_goodData = Correlation_LogStellarMass_LogSFR[goodData]
    
    source_id_goodData = Source_ID[goodData]

    num_input_data = tbData.size
    num_goodData = list(goodData).count(True)	
    x_goodData_unscaled_median = np.median(x_goodData_unscaled)
    y_goodData_unscaled_median = np.median(y_goodData_unscaled)

    x_goodData_unscaled_min = np.min(x_goodData_unscaled)
    y_goodData_unscaled_min = np.min(y_goodData_unscaled)

    x_goodData_unscaled_max = np.max(x_goodData_unscaled)
    y_goodData_unscaled_max = np.max(y_goodData_unscaled)

    if (args.verbose):  
        print("\n\nDescription of input data")    
        print("\tInput file                     : ", args.input_file)
        print("\tOutput file                    : ", args.output_file)
        print("\tNumber of input data           : ",num_input_data)
        print("\tNumber of good data            : ",num_goodData )
        print("\nGood data (used in subsequent analysis)")
        print("\tNumber of data                 : ", num_goodData)
        print("\tX data")
        print("\t   min                         : ",x_goodData_unscaled_min)
        print("\t   median                      : ",x_goodData_unscaled_median)
        print("\t   max                         : ",x_goodData_unscaled_max)
        print("\tY data")
        print("\t   min                         : ",y_goodData_unscaled_min)
        print("\t   median                      : ",y_goodData_unscaled_median)
        print("\t   max                         : ",y_goodData_unscaled_max)

    """ 
    Rescale the input data for fitting:  subtract median value in x,y
    """
    x_scale = x_goodData_unscaled_median
    y_scale = y_goodData_unscaled_median
    if (args.verbose):
        print("\nScaling (for decorrelating slope & intercept errors)")
        print("\tx_scale:  {:.3f}".format(x_scale) )
        print("\ty_scale:  {:.3f}".format(y_scale) )

        
    x_goodData_rescaled = x_goodData_unscaled - x_scale
    y_goodData_rescaled = y_goodData_unscaled - y_scale
    
    """ 
    fit the good input data only
    """
    x = x_goodData_rescaled
    y = y_goodData_rescaled
    sx = x_goodData_unscaled_err
    sy = y_goodData_unscaled_err
    
    x_unscaled = x_goodData_unscaled
    y_unscaled = y_goodData_unscaled    
    
    
    """ 
    INITIAL FITS
    """       
    
    (f87_rescaled_initialfit_slope,\
    f87_rescaled_initialfit_intercept,\
    f87_rescaled_initialfit_scatter_variance) = linfit_f87(x,y,sx,sy)

    """
    Convert rescaled fit quantities back to unscaled values 
    """
    f87_unscaled_initialfit_slope = f87_rescaled_initialfit_slope
    f87_unscaled_initialfit_intercept = f87_rescaled_initialfit_intercept - x_scale * f87_rescaled_initialfit_slope

    x_rescaled_initialfit_model = x
    y_rescaled_initialfit_model = f87_rescaled_initialfit_intercept + f87_rescaled_initialfit_slope * x_rescaled_initialfit_model
    y_rescaled_initialfit_residual = y - y_rescaled_initialfit_model
    y_rescaled_initialfit_residual_sigma = y_rescaled_initialfit_residual / sy

    if (args.verbose):  
        print("\nInitial fit to good data")    
        print("\tBest fit slope                 : {:.2f}".format(f87_unscaled_initialfit_slope))
        print("\tBest fit intercept             : {:.2f}".format(f87_unscaled_initialfit_intercept))
        print("\tIntrinsic scatter variance F87 : {:.3f}".format(f87_rescaled_initialfit_scatter_variance))
        print("\tResiduals (ie residual = (y-y_fit))")        
        print("\t     Mean                      : {:.2f}".format(y_rescaled_initialfit_residual.mean()))
        print("\t     Std deviation             : {:.2f}".format(y_rescaled_initialfit_residual.std()))
        print("\tResiduals (unitless; ie residual = (y-y_fit)/y_err)")        
        print("\t     Mean                      : {:.2f}".format(y_rescaled_initialfit_residual_sigma.mean()))
        print("\t     Std deviation             : {:.2f}".format(y_rescaled_initialfit_residual_sigma.std()))
        print("\nInitial fit to good data")
        scatter_variance_adhoc(x,y,sx,sy,f87_rescaled_initialfit_slope,f87_rescaled_initialfit_intercept, verbose = True)
   
    """    
    CLIPPING - Compute the residuals to the best fit model and identify outliers 

    new method of clipping
    pk 6/29/2015
    """
        
    outlierThreshold_sigma = float(args.clip_sigma) * y_rescaled_initialfit_residual.std()
    outliers = ((y_rescaled_initialfit_residual > outlierThreshold_sigma) | (y_rescaled_initialfit_residual < -1*outlierThreshold_sigma))
    nonoutliers = ((y_rescaled_initialfit_residual < outlierThreshold_sigma) & (y_rescaled_initialfit_residual > -1*outlierThreshold_sigma))

    outliers_positive = ((y_rescaled_initialfit_residual > outlierThreshold_sigma))
    outliers_negative = ((y_rescaled_initialfit_residual < -1*outlierThreshold_sigma))
    num_outliers_positive = list(outliers_positive).count(True)
    num_outliers_negative = list(outliers_negative).count(True)
    outlier_fraction_positive = num_outliers_positive / num_goodData
    outlier_fraction_negative = num_outliers_negative / num_goodData
    
    
    if (args.verbose):
        print("\nClipping of outliers")
        print("Data outside the clipping region are classified as outliers")        
        print("\tClipping threshold (sigma)     : ", outlierThreshold_sigma)
        print("\tNumber of non-outliers         : ", nonoutliers.sum())    
        print("\tTotal number of outliers       : ", outliers.sum())    
        print("\tNumber of positive outliers    : ", num_outliers_positive)    
        print("\tNumber of negative outliers    : ", num_outliers_negative)    
        print("\tOutlier fraction (positive)    : {:.2f}".format(outlier_fraction_positive))
        print("\tOutlier fraction (negative)    : {:.2f}".format(outlier_fraction_negative))

        initial_fit_residual_filename = 'sfr_life_vs_m_initial_fit_residuals.txt'
        np.savetxt(initial_fit_residual_filename, \
                    np.column_stack((x,sx,y,sy,y_rescaled_initialfit_model,y_rescaled_initialfit_residual)), \
                    fmt=('%5.6f','%5.6f','%5.6f','%5.6f','%5.6f','%5.6f'), \
                    header='x,xerr,y,yerr,ymodel,yresidual')
        print("Initial fit data, model, residuals saved to file: ",initial_fit_residual_filename)
    
    """    
    REFIT - refit the nonoutlier data to determine a more
    robust estimate of the best fit relationship.
    """
    
    """ 
    refit the nonoutlier (from initial fit) data only
    """
    x_refit = x[nonoutliers]
    y_refit = y[nonoutliers]
    sx_refit = sx[nonoutliers]
    sy_refit = sy[nonoutliers]
    covxy_refit = covxy_goodData[nonoutliers]
    varxx_refit = varxx_goodData[nonoutliers]
    varyy_refit = varyy_goodData[nonoutliers]
    
    corrxy_refit = corrxy_goodData[nonoutliers]
    source_id_refit = source_id_goodData[nonoutliers]
    
    f87_rescaled_refit_cov_slope,\
    f87_rescaled_refit_cov_intercept,\
    f87_rescaled_refit_cov_scatter_variance = linfit_f87(x_refit,\
                y_refit,\
                sx_refit,\
                sy_refit,\
                covxy = covxy_refit)
                
    """
    Convert rescaled refit quantities back to unscaled values 
    """
    f87_unscaled_refit_cov_slope = f87_rescaled_refit_cov_slope
    f87_unscaled_refit_cov_intercept = f87_rescaled_refit_cov_intercept - x_scale * f87_rescaled_refit_cov_slope
 
    """
    Compute model values and residuals
    """
    x_rescaled_refit_model = x_refit
    y_rescaled_refit_model = f87_rescaled_refit_cov_intercept + f87_rescaled_refit_cov_slope * x_rescaled_refit_model
    y_rescaled_refit_residual = y_refit - y_rescaled_refit_model
    y_rescaled_refit_residual_sigma = y_rescaled_refit_residual / sy_refit

    if (args.verbose):  
        print("\nRefit to good data")    
        print("\tBest fit slope                 : {:.2f}".format(f87_unscaled_refit_cov_slope))
        print("\tBest fit intercept             : {:.2f}".format(f87_unscaled_refit_cov_intercept))
        print("\tIntrinsic scatter variance F87 : {:.3f}".format(f87_rescaled_refit_cov_scatter_variance))
        print("\tResiduals (ie residual = (y-y_fit))")        
        print("\t     Mean                      : {:.2f}".format(y_rescaled_refit_residual.mean()))
        print("\t     Std deviation             : {:.2f}".format(y_rescaled_refit_residual.std()))
        print("\tResiduals (unitless; ie residual = (y-y_fit)/y_err)")        
        print("\t     Mean                      : {:.2f}".format(y_rescaled_refit_residual_sigma.mean()))
        print("\t     Std deviation             : {:.2f}".format(y_rescaled_refit_residual_sigma.std()))
        print("\nRefit to good data")
        scatter_variance_adhoc(x_refit,y_refit,sx_refit,sy_refit,f87_rescaled_refit_cov_slope,f87_rescaled_refit_cov_intercept, verbose = True)


    (bces_rescaled_refit_slope,\
    bces_rescaled_refit_intercept,\
    bces_rescaled_refit_scatter_variance,\
    f87_rescaled_refit_slope,\
    f87_rescaled_refit_intercept,\
    f87_rescaled_refit_scatter_variance,\
    k07_rescaled_refit_slope,\
    k07_rescaled_refit_intercept,\
    k07_rescaled_refit_scatter_variance,\
    mle_rescaled_refit_slope,\
    mle_rescaled_refit_intercept,\
    mle_rescaled_refit_scatter_variance, \
    odr_rescaled_refit_slope,\
    odr_rescaled_refit_intercept,\
    odr_rescaled_refit_scatter_variance,\
    ols_rescaled_refit_slope,\
    ols_rescaled_refit_intercept,\
    ols_rescaled_refit_scatter_variance,\
    t02_rescaled_refit_slope,\
    t02_rescaled_refit_intercept,\
    t02_rescaled_refit_scatter_variance, \
    wls_rescaled_refit_slope,\
    wls_rescaled_refit_intercept,\
    wls_rescaled_refit_scatter_variance) = linfit_multi(x_refit,\
                                                y_refit,\
                                                sx_refit,\
                                                sy_refit)

    """
    Compute errors to estimated quantities from simulation
    """
    (bces_rescaled_refit_slope_bias,
    bces_rescaled_refit_slope_error,
    f87_rescaled_refit_slope_bias,
    f87_rescaled_refit_slope_error,
    k07_rescaled_refit_slope_bias,
    k07_rescaled_refit_slope_error,
    mle_rescaled_refit_slope_bias,
    mle_rescaled_refit_slope_error,
    odr_rescaled_refit_slope_bias,
    odr_rescaled_refit_slope_error,
    ols_rescaled_refit_slope_bias,
    ols_rescaled_refit_slope_error,
    t02_rescaled_refit_slope_bias,
    t02_rescaled_refit_slope_error,
    wls_rescaled_refit_slope_bias,
    wls_rescaled_refit_slope_error,
    bces_rescaled_refit_intercept_bias,
    bces_rescaled_refit_intercept_error,
    f87_rescaled_refit_intercept_bias,
    f87_rescaled_refit_intercept_error,
    k07_rescaled_refit_intercept_bias,
    k07_rescaled_refit_intercept_error,
    mle_rescaled_refit_intercept_bias,
    mle_rescaled_refit_intercept_error,
    odr_rescaled_refit_intercept_bias,
    odr_rescaled_refit_intercept_error,
    ols_rescaled_refit_intercept_bias,
    ols_rescaled_refit_intercept_error,
    odr_rescaled_refit_intercept_bias,
    odr_rescaled_refit_intercept_error,
    t02_rescaled_refit_intercept_bias,
    t02_rescaled_refit_intercept_error,
    wls_rescaled_refit_intercept_bias,
    wls_rescaled_refit_intercept_error,
    bces_rescaled_refit_scatter_variance_bias,
    bces_rescaled_refit_scatter_variance_error,
    f87_rescaled_refit_scatter_variance_bias,
    f87_rescaled_refit_scatter_variance_error,
    k07_rescaled_refit_scatter_variance_bias,
    k07_rescaled_refit_scatter_variance_error,
    mle_rescaled_refit_scatter_variance_bias,
    mle_rescaled_refit_scatter_variance_error,
    odr_rescaled_refit_scatter_variance_bias,
    odr_rescaled_refit_scatter_variance_error,
    ols_rescaled_refit_scatter_variance_bias,
    ols_rescaled_refit_scatter_variance_error,
    odr_rescaled_refit_scatter_variance_bias,
    odr_rescaled_refit_scatter_variance_error,
    t02_rescaled_refit_scatter_variance_bias,
    t02_rescaled_refit_scatter_variance_error,
    wls_rescaled_refit_scatter_variance_bias,
    wls_rescaled_refit_scatter_variance_error)= linfit_multi_simerr(x_refit,\
                                    sx_refit,\
                                    sy_refit,\
                                    f87_rescaled_refit_slope,\
                                    f87_rescaled_refit_intercept,\
                                    f87_rescaled_refit_scatter_variance,\
                                    iterations=args.iterations,\
                                    plot_realization=None,\
                                    plot_results=None,\
                                    write_tables=None,\
                                    xerr_type = 'normal',\
                                    yerr_type = 'normal', \
                                    verbose=None )

    """
    Model parameter estimates from Kelly (2007)
    Run in IDL and hard-coded here.
    See khsueti:~/analysis/CANDELS/2015-05-16-A/sample01/analysis/run03

    Table 2.  Fit results for method of Kelly (2007) on 
    log SFR vs. log M* data for sample01 run03
    
    Version Scatter Var.    Slope           Intercept
    (1)     (2)             (3)             (4)    
    
                    --- SFR(100) ---
    v8      0.0570 0.0045   0.9739 0.0143   0.07738 0.0114 
    v9*     0.0586 0.0044   0.9723 0.0142   0.0761  0.0113
                    --- SFR(Inst) ---
    v10     0.0552 0.0052   0.8402 0.0150   0.0976  0.0116
    v11*    0.0573 0.0052   0.8375 0.0143   0.1000  0.0117
                    --- SFR(Life) ---
    v12     0.0024 0.00019  1.0103 0.00039  0.03318 0.00082
    v13*    0.0264 0.00259  1.0385 0.01240  0.08114 0.01000
    
    Caption:  (1) is the version of the fit run, see corresponding .sav file 
    for results.  *fits done without covariance. (2) is the best fit 
    intrinsic scatter variance and error, determined from mean and std 
    deviation of the posterior distribution.  (3) is the best fit slope 
    (beta, in usage of IDL program) and error.  (4) is Intercept and error 
    (alpha in usage of IDL program).
    """
    k07_rescaled_refit_slope = 1.0385
    k07_rescaled_refit_slope_error = 0.01240
    
    k07_rescaled_refit_cov_slope = 1.0103
    k07_rescaled_refit_cov_slope_error = 0.00039
    
    k07_rescaled_refit_intercept = 0.08114
    k07_rescaled_refit_intercept_error = 0.01000
    
    k07_rescaled_refit_cov_intercept = 0.03318
    k07_rescaled_refit_cov_intercept_error = 0.00082
    
    k07_rescaled_refit_scatter_variance = 0.0264
    k07_rescaled_refit_scatter_variance_error = 0.00259

    k07_rescaled_refit_cov_scatter_variance = 0.0024
    k07_rescaled_refit_cov_scatter_variance_error = 0.00019
    
    k07_rescaled_refit_scatter_variance_frac_error = k07_rescaled_refit_scatter_variance_error / k07_rescaled_refit_scatter_variance
    
    k07_rescaled_refit_scatter_sigma = np.sqrt(k07_rescaled_refit_cov_scatter_variance)
    k07_rescaled_refit_scatter_sigma_frac_error = 0.5 * k07_rescaled_refit_scatter_variance_frac_error
    k07_rescaled_refit_scatter_sigma_error = k07_rescaled_refit_scatter_sigma * k07_rescaled_refit_scatter_sigma_frac_error
    
    k07_refit_slope = k07_rescaled_refit_slope
    k07_refit_intercept = k07_rescaled_refit_intercept - x_scale * k07_rescaled_refit_slope

    """
    Output text data files with refit logM, logSFR data for use in external
    fitting routines(eg. Kelly (2007) method implemented in IDL)
    """
    if (args.verbose):
       refit_data_table_id_cov_corr_filename = 'sfr_life_vs_m_refit_id_cov_corr.txt'    
       refit_data_table_cov_filename = 'sfr_life_vs_m_refit_cov.txt'
       refit_data_table_filename = 'sfr_life_vs_m_refit.txt'
       np.savetxt(refit_data_table_filename, \
                   np.column_stack((x_refit,sx_refit,y_refit,sy_refit)), \
                   fmt=('%5.6f','%5.6f','%5.6f','%5.6f'), \
                   header='x_refit,sx_refit,y_refit,sy_refit')
       print("Refit data saved to file: ",refit_data_table_filename)

       np.savetxt(refit_data_table_cov_filename, \
                   np.column_stack((x_refit,sx_refit,y_refit,sy_refit, covxy_refit)), \
                   fmt=('%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f'), \
                   header='x_refit,sx_refit,y_refit,sy_refit,covxy_refit')

       print("Refit data and covariances saved to file: ",refit_data_table_cov_filename)
 
       np.savetxt(refit_data_table_id_cov_corr_filename,
                  np.column_stack((source_id_refit,x_refit,sx_refit,y_refit,sy_refit,covxy_refit,corrxy_refit)),
                  fmt=('%d', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f'), 
                  header = 'id,x_refit,sx_refit,y_refit,sy_refit,covxy_refit,corrxy_refit')

       print("Refit id, data, covariances, correlations saved to file: ",refit_data_table_id_cov_corr_filename)
   
    """
    Convert rescaled fit quantities back to unscaled values 
    """
    x_unscaled_refit = x_unscaled[nonoutliers]
    y_unscaled_refit = y_unscaled[nonoutliers]

    bces_unscaled_refit_slope = bces_rescaled_refit_slope
    f87_unscaled_refit_slope = f87_rescaled_refit_slope
    k07_unscaled_refit_slope = k07_rescaled_refit_slope
    mle_unscaled_refit_slope = mle_rescaled_refit_slope
    odr_unscaled_refit_slope = odr_rescaled_refit_slope
    ols_unscaled_refit_slope = ols_rescaled_refit_slope
    t02_unscaled_refit_slope = t02_rescaled_refit_slope
    wls_unscaled_refit_slope = wls_rescaled_refit_slope
    
    f87_unscaled_refit_cov_slope = f87_rescaled_refit_cov_slope
    k07_unscaled_refit_cov_slope = k07_rescaled_refit_cov_slope

    bces_unscaled_refit_intercept = bces_rescaled_refit_intercept - x_scale * bces_rescaled_refit_slope
    f87_unscaled_refit_intercept = f87_rescaled_refit_intercept - x_scale * f87_rescaled_refit_slope
    k07_unscaled_refit_intercept = k07_rescaled_refit_intercept - x_scale * k07_rescaled_refit_slope
    mle_unscaled_refit_intercept = mle_rescaled_refit_intercept - x_scale * mle_rescaled_refit_slope
    odr_unscaled_refit_intercept = odr_rescaled_refit_intercept - x_scale * odr_rescaled_refit_slope
    ols_unscaled_refit_intercept = ols_rescaled_refit_intercept - x_scale * ols_rescaled_refit_slope
    t02_unscaled_refit_intercept = t02_rescaled_refit_intercept - x_scale * t02_rescaled_refit_slope
    wls_unscaled_refit_intercept = wls_rescaled_refit_intercept - x_scale * wls_rescaled_refit_slope

    f87_unscaled_refit_cov_intercept = f87_rescaled_refit_cov_intercept - x_scale * f87_rescaled_refit_cov_slope
    k07_unscaled_refit_cov_intercept = k07_rescaled_refit_cov_intercept - x_scale * k07_rescaled_refit_cov_slope


    bces_unscaled_refit_scatter_variance = bces_rescaled_refit_scatter_variance
    f87_unscaled_refit_scatter_variance = f87_rescaled_refit_scatter_variance
    k07_unscaled_refit_scatter_variance = k07_rescaled_refit_scatter_variance
    mle_unscaled_refit_scatter_variance = mle_rescaled_refit_scatter_variance
    odr_unscaled_refit_scatter_variance = odr_rescaled_refit_scatter_variance
    ols_unscaled_refit_scatter_variance = ols_rescaled_refit_scatter_variance
    t02_unscaled_refit_scatter_variance = t02_rescaled_refit_scatter_variance
    wls_unscaled_refit_scatter_variance = wls_rescaled_refit_scatter_variance

    f87_unscaled_refit_cov_scatter_variance = f87_rescaled_refit_cov_scatter_variance
    k07_unscaled_refit_cov_scatter_variance = k07_rescaled_refit_cov_scatter_variance

    bces_unscaled_refit_scatter_variance_snr = bces_unscaled_refit_scatter_variance/bces_rescaled_refit_scatter_variance_error
    f87_rescaled_refit_scatter_variance_snr = f87_unscaled_refit_scatter_variance/f87_rescaled_refit_scatter_variance_error
    k07_rescaled_refit_scatter_variance_snr = k07_unscaled_refit_scatter_variance/k07_rescaled_refit_scatter_variance_error
    mle_rescaled_refit_scatter_variance_snr = mle_unscaled_refit_scatter_variance/mle_rescaled_refit_scatter_variance_error
    odr_rescaled_refit_scatter_variance_snr = odr_unscaled_refit_scatter_variance/odr_rescaled_refit_scatter_variance_error
    ols_rescaled_refit_scatter_variance_snr = ols_unscaled_refit_scatter_variance/ols_rescaled_refit_scatter_variance_error
    t02_rescaled_refit_scatter_variance_snr = t02_unscaled_refit_scatter_variance/t02_rescaled_refit_scatter_variance_error
    wls_rescaled_refit_scatter_variance_snr = wls_unscaled_refit_scatter_variance/wls_rescaled_refit_scatter_variance_error

    f87_rescaled_refit_cov_scatter_variance_snr = f87_unscaled_refit_cov_scatter_variance/f87_rescaled_refit_scatter_variance_error

    """
    correlation coefficients of data set
    """
    spearman_rho, spearman_pvalue = s.spearmanr(x_refit,y_refit)
    pearson_rho, pearson_pvalue = s.pearsonr(x_refit,y_refit)
    
    if (args.verbose):  
        print("\nResults of re-fit to model data")
        print("\tNumber of data in fit          : ", len(y_refit))
        print("\tx_scale                        : {:.3f}".format(x_scale))
        print("\ty_scale                        : {:.3f}".format(y_scale))
        print("\tSpearman correlation (rho,p)   : {:.2f}".format(spearman_rho), ", {:.6f}".format(spearman_pvalue))
        print("\tPearson correlation (r,p)      : {:.2f}".format(pearson_rho), ", {:.6f}".format(pearson_pvalue))
        print("\tQuality of fit")
        print("\tr^2 (OLS)                      : {:.3f}".format(pearson_rho**2))
        print("\nSimulation Inputs and Execution Time (Errors)")
        print("\tNumber of Realizations         : {:d}".format(args.iterations))
        print("\txerr_type                      :",'normal')
        print("\tyerr_type                      :",'normal')    
        #print("\tElapsed time (seconds)         : {:.3f}".format(elapsed))
        print("Parameter Estimates, Bias and Errors from Simulation")
        print("\tSlope Estimates")
        print("\t                         BCES  : {:.2f}".format(bces_unscaled_refit_slope))
        print("\t                          F87  : {:.2f}".format(f87_unscaled_refit_slope))
        print("\t                          k07  : {:.2f}".format(k07_unscaled_refit_slope))
        print("\t                          mle  : {:.2f}".format(mle_unscaled_refit_slope))
        print("\t                          odr  : {:.2f}".format(odr_unscaled_refit_slope))
        print("\t                          ols  : {:.2f}".format(ols_unscaled_refit_slope))
        print("\t                          t02  : {:.2f}".format(t02_unscaled_refit_slope))
        print("\t                          wls  : {:.2f}".format(wls_unscaled_refit_slope))
        print("\tSlope Bias From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_slope_bias))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_slope_bias))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_slope_bias))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_slope_bias))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_slope_bias))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_slope_bias))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_slope_bias))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_slope_bias))
        print("\tSlope Error From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_slope_error))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_slope_error))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_slope_error))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_slope_error))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_slope_error))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_slope_error))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_slope_error))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_slope_error))
        print("\tIntercept Estimates")
        print("\t                         BCES  : {:.2f}".format(bces_unscaled_refit_intercept))
        print("\t                          F87  : {:.2f}".format(f87_unscaled_refit_intercept))
        print("\t                          k07  : {:.2f}".format(k07_unscaled_refit_intercept))
        print("\t                          mle  : {:.2f}".format(mle_unscaled_refit_intercept))
        print("\t                          odr  : {:.2f}".format(odr_unscaled_refit_intercept))
        print("\t                          ols  : {:.2f}".format(ols_unscaled_refit_intercept))
        print("\t                          t02  : {:.2f}".format(t02_unscaled_refit_intercept))
        print("\t                          wls  : {:.2f}".format(wls_unscaled_refit_intercept))
        print("\tIntercept Bias From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_intercept_bias))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_intercept_bias))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_intercept_bias))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_intercept_bias))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_intercept_bias))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_intercept_bias))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_intercept_bias))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_intercept_bias))
        print("\tIntercept Error From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_intercept_error))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_intercept_error))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_intercept_error))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_intercept_error))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_intercept_error))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_intercept_error))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_intercept_error))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_intercept_error))
        print("\tScatter Variance Estimates")
        print("\t                         BCES  : {:.5f}".format(bces_unscaled_refit_scatter_variance))
        print("\t                          F87  : {:.5f}".format(f87_unscaled_refit_scatter_variance))
        print("\t                          k07  : {:.5f}".format(k07_unscaled_refit_scatter_variance))
        print("\t                          mle  : {:.5f}".format(mle_unscaled_refit_scatter_variance))
        print("\t                          odr  : {:.5f}".format(odr_unscaled_refit_scatter_variance))
        print("\t                          ols  : {:.5f}".format(ols_unscaled_refit_scatter_variance))
        print("\t                          t02  : {:.5f}".format(t02_unscaled_refit_scatter_variance))
        print("\t                          wls  : {:.5f}".format(wls_unscaled_refit_scatter_variance))
        print("\tScatter Variance Bias From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_scatter_variance_bias))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_scatter_variance_bias))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_scatter_variance_bias))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_scatter_variance_bias))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_scatter_variance_bias))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_scatter_variance_bias))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_scatter_variance_bias))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_scatter_variance_bias))
        print("\tScatter Variance Estimate (Bias Adjusted)")
        print("\t                         BCES  : {:.5f}".format(bces_unscaled_refit_scatter_variance - bces_rescaled_refit_scatter_variance_bias))
        print("\t                          F87  : {:.5f}".format(f87_unscaled_refit_scatter_variance - f87_rescaled_refit_scatter_variance_bias))
        print("\t                          k07  : {:.5f}".format(k07_unscaled_refit_scatter_variance - k07_rescaled_refit_scatter_variance_bias))
        print("\t                          mle  : {:.5f}".format(mle_unscaled_refit_scatter_variance - mle_rescaled_refit_scatter_variance_bias))
        print("\t                          odr  : {:.5f}".format(odr_unscaled_refit_scatter_variance - odr_rescaled_refit_scatter_variance_bias))
        print("\t                          ols  : {:.5f}".format(ols_unscaled_refit_scatter_variance - ols_rescaled_refit_scatter_variance_bias))
        print("\t                          t02  : {:.5f}".format(t02_unscaled_refit_scatter_variance - t02_rescaled_refit_scatter_variance_bias))
        print("\t                          wls  : {:.5f}".format(wls_unscaled_refit_scatter_variance - wls_rescaled_refit_scatter_variance_bias))
        print("\tScatter Variance Error From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_rescaled_refit_scatter_variance_error))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_scatter_variance_error))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_scatter_variance_error))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_scatter_variance_error))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_scatter_variance_error))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_scatter_variance_error))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_scatter_variance_error))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_scatter_variance_error))
        print("\tScatter Variance SNR From Simulations")
        print("\t                         BCES  : {:.5f}".format(bces_unscaled_refit_scatter_variance/bces_rescaled_refit_scatter_variance_error))
        print("\t                          F87  : {:.5f}".format(f87_rescaled_refit_scatter_variance/f87_rescaled_refit_scatter_variance_error))
        print("\t                          k07  : {:.5f}".format(k07_rescaled_refit_scatter_variance/k07_rescaled_refit_scatter_variance_error))
        print("\t                          mle  : {:.5f}".format(mle_rescaled_refit_scatter_variance/mle_rescaled_refit_scatter_variance_error))
        print("\t                          odr  : {:.5f}".format(odr_rescaled_refit_scatter_variance/odr_rescaled_refit_scatter_variance_error))
        print("\t                          ols  : {:.5f}".format(ols_rescaled_refit_scatter_variance/ols_rescaled_refit_scatter_variance_error))
        print("\t                          t02  : {:.5f}".format(t02_rescaled_refit_scatter_variance/t02_rescaled_refit_scatter_variance_error))
        print("\t                          wls  : {:.5f}".format(wls_rescaled_refit_scatter_variance/wls_rescaled_refit_scatter_variance_error))
        print("Parameter Estimates - Fits with covariances")
        print("\tSlope Estimates")
        print("\t                          F87  : {:.2f}".format(f87_rescaled_refit_cov_slope))
        print("\t                          K07  : {:.2f}".format(k07_rescaled_refit_cov_slope))
        print("\tIntercept Estimates")
        print("\t                          F87  : {:.2f}".format(f87_unscaled_refit_cov_intercept))
        print("\t                          k07  : {:.2f}".format(k07_unscaled_refit_intercept))
        print("\tScatter Variance Estimates")
        print("\t                          F87  : {:.5f}".format(f87_unscaled_refit_cov_scatter_variance))
        print("\t                          k07  : {:.5f}".format(k07_unscaled_refit_cov_scatter_variance))
    
        scatter_variance_adhoc(x_refit,y_refit,sx_refit,sy_refit,f87_rescaled_refit_slope,f87_rescaled_refit_intercept, verbose = True)


    spearman_rho, spearman_pvalue = s.spearmanr(x_refit,y_refit)
    pearson_rho, pearson_pvalue = s.pearsonr(x_refit,y_refit)
    

    """
    Best fit model values
    """
    y_refit_model_f87 = f87_rescaled_refit_intercept + f87_rescaled_refit_slope * x_refit 
    y_refit_model_odr = odr_rescaled_refit_intercept + odr_rescaled_refit_slope * x_refit
    y_refit_model_ols = ols_rescaled_refit_intercept + ols_rescaled_refit_slope * x_refit

    """
    Best fit models regularized to 100 data values
    """
    x_refit_model100 = np.linspace(x_refit.min(),x_refit.max(),100)
    y_refit_model100_f87 = f87_rescaled_refit_intercept + f87_rescaled_refit_slope * x_refit_model100 
    y_refit_model100_odr = odr_rescaled_refit_intercept + odr_rescaled_refit_slope * x_refit_model100
    y_refit_model100_ols = ols_rescaled_refit_intercept + ols_rescaled_refit_slope * x_refit_model100
    y_refit_model100_bces = bces_rescaled_refit_intercept + bces_rescaled_refit_slope * x_refit_model100
    y_refit_model100_k07 = k07_rescaled_refit_intercept + k07_rescaled_refit_slope * x_refit_model100

    """
    Best fit, regularized models converted back to unscaled data
    """
    x_unscaled_refit_model100 = x_refit_model100 + x_scale    
    y_unscaled_refit_model100_f87 = y_scale + f87_rescaled_refit_intercept - x_scale * f87_rescaled_refit_slope + f87_rescaled_refit_slope * x_unscaled_refit_model100 
    y_unscaled_refit_model100_odr = y_scale + odr_rescaled_refit_intercept - x_scale * odr_rescaled_refit_slope + odr_rescaled_refit_slope * x_unscaled_refit_model100
    y_unscaled_refit_model100_ols = y_scale + ols_rescaled_refit_intercept - x_scale * ols_rescaled_refit_slope + ols_rescaled_refit_slope * x_unscaled_refit_model100
    y_unscaled_refit_model100_bces = y_scale + bces_rescaled_refit_intercept - x_scale * bces_rescaled_refit_slope + bces_rescaled_refit_slope * x_unscaled_refit_model100
    y_unscaled_refit_model100_k07 = y_scale + k07_rescaled_refit_intercept - x_scale * k07_rescaled_refit_slope + k07_rescaled_refit_slope * x_unscaled_refit_model100
 
    intercept_high_err = f87_rescaled_refit_intercept + f87_rescaled_refit_intercept_error
    intercept_low_err = f87_rescaled_refit_intercept - f87_rescaled_refit_intercept_error    
    slope_high_err = f87_rescaled_refit_slope + f87_rescaled_refit_slope_error
    slope_low_err = f87_rescaled_refit_slope - f87_rescaled_refit_slope_error
    
    y_refit_model100_f87_higherr = intercept_high_err + slope_high_err * x_refit_model100
    y_refit_model100_f87_lowerr = intercept_low_err + slope_low_err * x_refit_model100
    
    y_unscaled_refit_model100_f87_higherr = y_scale + y_refit_model100_f87_higherr 
    y_unscaled_refit_model100_f87_lowerr = y_scale + y_refit_model100_f87_lowerr 
    

    """    
    Generate the model fit results figure (output file)
    """    
    # figsize in inches
    # use [4,4] by default
    # use [5,4] when including colorbar
    fig = plt.figure(dpi=300, figsize=[5,4])
    fig.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
        
    """
    Figure - subplot - Data +  Model
    """
    plot_xlim = [6.25,11]
    plot_ylim= [-3.0,3.0]

    axis5 = fig.add_subplot(111)
    axis5.set_xlabel(r"log Stellar Mass, $M_\odot$", fontsize = 12)
    axis5.set_ylabel(r"log SFR$_{Life}$, $M_\odot$ yr$^{-1}$", fontsize = 12)
    axis5.set_xlim(plot_xlim)
    axis5.set_ylim(plot_ylim)
    
    """
     Main Sequence (Speagle+2014)
     
      NB:  Fitted mass range is logM = [9.7,11.1]
      see Speagle+2014, pg. 21, Equation 26 and discussion
      
      
     log SFR = slope * logMass - intercept
     
         time = age of universe at redshift, Gyr
           NB:  time(z=1.0) = 5.75
                time(z=2.0) = 3.23
                
         slope = 0.83 pm 0.03 - (0.027 pm 0.004) x time
         intercept = 6.38 pm 0.27 - (0.12 pm 0.04) x time
         
     scaling to M/M^10.5 ...
     
     log SFR = xi * log (Mass/10^10.5) - eta
     
       xi = slope
       eta = intercept - 10.5 * slope
       
       errors (from e-mail exchange with J. Speagle): 
       sigma_xi = sigma_slope (no change)
       sigma_eta = 0.003 (time independent term); 0.01 (time dependent term)
    """
            
    logMass_Speagle2014 = np.linspace(x_unscaled_refit.min(),x_unscaled_refit.max(),100)
    time_Gyr = cosmo.age(args.zSpeagle2014).value             
    slope_Speagle2014 = 0.83 - 0.027 * time_Gyr
    intercept_Speagle2014 = 6.38 - 0.12 * time_Gyr   
    logSFR_Speagle2014 = slope_Speagle2014 * logMass_Speagle2014 - intercept_Speagle2014
    
    lowExtrapolation = ((logMass_Speagle2014 > x.min()) & (logMass_Speagle2014 <= 9.7))
    noExtrapolation = ((logMass_Speagle2014 > 9.7) & (logMass_Speagle2014 <= 11.1))
    highExtrapolation= (logMass_Speagle2014 > 11.1)
    
    
    """
     Compute errors in Speagle+2014 relationship:
     These are un-rescaled values (taken directly from
     Speagle+2014) they are replaced by mass rescaled
     analysis below.
     
     HighErr - maximize slope and intercept
     LowErr - minimize slope and intercept
    """
    slope_HighErr_Speagle2014 = 0.86 - 0.023 * time_Gyr
    intercept_HighErr_Speagle2014 = 6.65 - 0.08 * time_Gyr   
    slope_LowErr_Speagle2014 = 0.80 - 0.031 * time_Gyr
    intercept_LowErr_Speagle2014 = 6.11 - 0.16 * time_Gyr   
    
    """
     recompute Speagle+2014 relation rescaled to M/10^10.5 Msun.  
     Rescaling is better for the error analysis.  The MS relation
     then takes the form:  
     logSFR = xi * log( M/10^10.5 Msun) - eta
    """
    xi = slope_Speagle2014
    eta = intercept_Speagle2014 - 10.5 * slope_Speagle2014
    xi_HighErr = slope_HighErr_Speagle2014
    xi_LowErr = slope_LowErr_Speagle2014
    
    """
     empirical error terms for mass scaled MS relationship.  These
     numbers are from Josh Speagle, e-mail 10/7/2014
     
     eta0_err = time independent term error = 0.01
     eta1_err = time dependent term error = 0.003
    """
    eta0_err = 0.01
    eta1_err = 0.003
    eta_err = math.sqrt(eta0_err**2 + time_Gyr**2 * eta1_err**2)
    eta_LowErr = eta - eta_err
    eta_HighErr = eta + eta_err
    
    logMass_Rescaled = logMass_Speagle2014 - 10.5
    logSFR_Rescaled = xi * logMass_Rescaled - eta

    """
    error curves incorporate errors to both rescaled slope and intercept
    """
    logSFR_Speagle2014_HighErr_MassRescaled = xi_HighErr * logMass_Rescaled - eta_LowErr
    logSFR_Speagle2014_LowErr_MassRescaled = xi_LowErr * logMass_Rescaled - eta_HighErr

    """
    plot the MS line from Speagle+2014 and error region
    """
    axis5.plot(logMass_Speagle2014[lowExtrapolation],\
             logSFR_Speagle2014[lowExtrapolation],\
             color='Red',\
             linestyle='dashed')
    axis5.plot(logMass_Speagle2014[noExtrapolation],\
             logSFR_Speagle2014[noExtrapolation],\
             color='Red',\
             linestyle='solid',\
             label='Literature')
    axis5.plot(logMass_Speagle2014[highExtrapolation],\
             logSFR_Speagle2014[highExtrapolation],\
             color='Red',\
             linestyle='dashed')
    axis5.fill_between(logMass_Speagle2014, \
                logSFR_Speagle2014_LowErr_MassRescaled, \
                logSFR_Speagle2014_HighErr_MassRescaled, \
                facecolor = 'Pink',\
            #    hatch = "X",\
                edgecolor = 'Pink', \
                linewidth=0.0)       
     
     
    """
    Plot error region (bottom layer), data and outliers (middle layer), 
    best fit lines  and text label (top layer)
    """
    axis5.fill_between(x_unscaled_refit_model100, y_unscaled_refit_model100_f87_lowerr, y_unscaled_refit_model100_f87_higherr, \
                facecolor = 'Thistle',\
                edgecolor = 'Thistle', \
                linewidth=0.0)

    """
    plot points with error bars
    """
    """    
    axis5.errorbar(x_unscaled_refit, \
                 y_unscaled_refit, \
                 xerr=sx_refit, \
                 yerr=sy_refit, \
                 linestyle='None', \
                 color = 'gray', \
                 capsize = 0, \
                 marker='s', \
                 markerfacecolor='gray', \
                 markeredgecolor='gray', \
                 markersize=2, \
                 markeredgewidth=0.5, \
                 alpha = 0.3)
    """

    """
    plot points with no error bars
    """
    """    
    axis5.plot(x_unscaled_refit, \
                 y_unscaled_refit, \
                 linestyle='None', \
                 color = 'black', \
                 alpha = 0.25, \
                 marker='o', \
                 markerfacecolor='black', \
                 markeredgecolor='black', \
                 markersize=2, \
                 markeredgewidth=0.5)
    """
    """
    plot points as error ellipses:
    
    ellipses are color-coded according to the correlation coeff.
    and a color bar is added on the rhs of the plot.
    
    change figure dimensions above to [5,4] when using colorbar    
    
    https://m.youtube.com/watch?v=717fVhFKn8E
    http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib

    http://matplotlib.org/examples/pylab_examples/ellipse_collection.html
    http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    http://matplotlib.org/api/collections_api.html
    http://matplotlib.org/users/transforms_tutorial.html
    http://matplotlib.org/examples/api/colorbar_only.html
    """
    
    
    xydata = np.vsplit(np.transpose(np.vstack([x_unscaled_refit,y_unscaled_refit])),1)
    sigx = sx_refit
    sigy = sy_refit
    sigxy = covxy_refit   
    sigxyterm = np.sqrt(0.25 * (sigx**2 - sigy**2)**2 + sigxy**2)    
    sigxp2 = 0.5 * (sigx**2 + sigy**2) + sigxyterm
    sigyp2 = 0.5 * (sigx**2 + sigy**2) - sigxyterm
    semi_major_axis = np.sqrt(sigxp2)
    semi_minor_axis = np.sqrt(sigyp2)
    theta = np.degrees(0.5 * np.arctan2(2*sigxy,(sigx**2 - sigy**2))) 
    #norm = mpl.colors.Normalize(vmin = -1.0, vmax = 1.0)  
    norm = mpl.colors.Normalize(vmin = corrxy_refit.min(), vmax = corrxy_refit.max())  
    cmap = mpl.cm.jet
    ec = EllipseCollection(2*semi_major_axis,
                        2*semi_minor_axis,
                        theta,
                        units='x',
                        array = corrxy_refit, 
                        cmap = cmap,
                        norm = norm, 
                        alpha = 0.10,
                        offsets=xydata,
                        transOffset=axis5.transData)
    #cbar = plt.colorbar(ec, label = 'corr',ticks = [-1,-0.5,0,0.5,1])
    cbar = plt.colorbar(ec, label = 'Corr(SFR,M*)')
    axis5.add_collection(ec)
    

    """
    Draw representative error ellipse in the upper left
    
    https://m.youtube.com/watch?v=717fVhFKn8E
    http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    """
    """    
    typical_point_x_value = 7.0
    typical_point_y_value = 1.25
    typical_point_x_error = np.median(sx_refit)
    typical_point_y_error = np.median(sy_refit)
    typical_point_xy_cov = np.median(covxy_refit)
    
    sigx = typical_point_x_error
    sigy = typical_point_y_error
    sigxy = typical_point_xy_cov   
    sigxyterm = np.sqrt(0.25 * (sigx**2 - sigy**2)**2 + sigxy**2)    
    sigxp2 = 0.5 * (sigx**2 + sigy**2) + sigxyterm
    sigyp2 = 0.5 * (sigx**2 + sigy**2) - sigxyterm
    semiMajorAxis = np.sqrt(sigxp2)
    semiMinorAxis = np.sqrt(sigyp2)
    theta = np.degrees(0.5 * np.arctan2(2*sigxy,(sigx**2 - sigy**2))) 
    
    ell = Ellipse(xy = (typical_point_x_value,typical_point_y_value), \
                    width = 2*semiMajorAxis, \
                    height = 2*semiMinorAxis, \
                    angle = theta, \
                    alpha = 0.5, \
                    color = 'blue')
    ell.set_facecolor('blue')
    axis5.add_artist(ell)
    """
    """
    plot a representative point with error bars in the upper left
    """
    """    
    typical_point_x_value = 7.0
    typical_point_y_value = 1.25
    typical_point_x_error = np.median(sx_refit)
    typical_point_y_error = np.median(sy_refit)
    axis5.plot(typical_point_x_value, \
                 typical_point_y_value, \
                 linestyle='None', \
                 color = 'black', \
                 alpha = 0.6, \
                 marker='o', \
                 markerfacecolor='black', \
                 markeredgecolor='black', \
                 markersize=2, \
                 markeredgewidth=0.5)

    axis5.errorbar(typical_point_x_value, \
                 typical_point_y_value, \
                 xerr=typical_point_x_error, \
                 yerr=typical_point_y_error, \
                 linestyle='None', \
                 color = 'gray', \
                 capsize = 0, \
                 alpha = 0.6, \
                 marker='s', \
                 markerfacecolor='gray', \
                 markeredgecolor='gray', \
                 markersize=2, \
                 markeredgewidth=0.5)
    """
    """
    plot outliers from fit (clipped points)
    """
    """             
    axis5.errorbar(x_unscaled[outliers], \
                 y_unscaled[outliers], \
                 xerr=sx[outliers], \
                 yerr=sy[outliers], \
                 linestyle='None', \
                 color = 'red', \
                 capsize = 0, \
                 marker='s', \
                 markerfacecolor='lightgray', \
                 markeredgecolor='lightgray', \
                 markersize=2, \
                 markeredgewidth=0.5)
    """
    axis5.plot(x_unscaled[outliers], \
                 y_unscaled[outliers], \
                 linestyle='None', \
                 color = 'red', \
                 alpha = 0.25, \
                 marker='o', \
                 markerfacecolor='red', \
                 markeredgecolor='red', \
                 markersize=2, \
                 markeredgewidth=0.5)
    
    axis5.plot(x_unscaled_refit_model100,\
             y_unscaled_refit_model100_ols,\
             linewidth = 3, \
             color='Indigo',\
             label='Best Fit')      
    """
    axis5.plot(x_unscaled_refit_model100,\
             y_unscaled_refit_model100_k07,\
             color='blueviolet',\
             label='Kelly (2007)')  
    axis5.plot(x_unscaled_refit_model100,\
             y_unscaled_refit_model100_bces,\
             color='slateblue',\
             label='BCES')  
    """
    """
    Plot text label:  intrinsic scatter
    """    
    text_label_scatter =r"$\sigma_{is} = " + r"{:.3f}".format(k07_rescaled_refit_scatter_sigma) + r"\pm"+ r"{:.3f}".format(k07_rescaled_refit_scatter_sigma_error)+r"$"
    text_label_r2 = r"$r^2 = " + r"{:.2f}".format(pearson_rho**2) + r"$"
    
    axis5.text(8.75,-2.0,text_label_r2,fontsize='small', fontweight="bold")
    axis5.text(8.75,-2.3,text_label_scatter,fontsize='small', fontweight="bold")
    
    axis5.legend(loc='upper left', fontsize='small')

    fig.savefig(args.output_file, format='pdf')    
    if (args.display):
        plt.show()



    """
    ANALYZE FINAL MODEL FIT RESIDUALS  
    
    Compute the residuals to the best refit model and identify outliers    
    """ 
    y_refit_residual = y_refit - y_refit_model_f87
    y_refit_residual_squared = y_refit_residual**2
    y_refit_residual_sigma = y_refit_residual / sy_refit
    y_refit_residual_sigma_squared = y_refit_residual_sigma**2
    if (args.verbose):
        print("\nSummary of Refit Residuals")
        print("Units of data, ie residual = y-y_fit")      
        print("\tMin                            : {:.2f}".format(y_refit_residual.min()))
        print("\tMean                           : {:.2f}".format(y_refit_residual.mean()))
        print("\tMax                            : {:.2f}".format(y_refit_residual.max()))
        print("\tStd deviation                  : {:.2f}".format(y_refit_residual.std()))
        print("\tSum of squares (ESS)           : {:.2f}".format(y_refit_residual_squared.sum()))
        print("Unitless ('sigma'), ie residual = (y-y_fit)/y_err")
        print("\tMin                            : {:.2f}".format(y_refit_residual_sigma.min()))
        print("\tMean                           : {:.2f}".format(y_refit_residual_sigma.mean()))
        print("\tMax                            : {:.2f}".format(y_refit_residual_sigma.max()))
        print("\tStd deviation                  : {:.2f}".format(y_refit_residual_sigma.std()))
        print("\tSum of squares (chisq)         : {:.2f}".format(y_refit_residual_sigma_squared.sum()))

    """    
    RESIDUALS - BIN SCAT  
    
    Histogram analysis of refit residuals
    """
    
    """
    x_bins must be consistent with y_binscat_binX definitions below
    eg. x_bins = np.array([1.0, 8.0, 9.0, 20.0])
        where   bin1 = [1.0, 8.0]
                bin2 = [8.0, 9.0]
                bin3 = [9.0, 20.0]
    """
    x_bins = np.array([1.0, 8.0, 9.0, 20.0])
    
    
    x_binscat = x_unscaled_refit
    """ 
    Old method of plotting residuals
    """
    #y_binscat = y_refit_residual_sigma
    """
    New method of plotting residuals
    pk 7/1/2015
    """    
    y_binscat = y_refit_residual
    
    ndata_in_x_bin, _ = np.histogram(x_binscat, bins=x_bins)
    sum_y_binscat, _ = np.histogram(x_binscat, bins=x_bins, weights=y_binscat)
    sum_y2_binscat, _ = np.histogram(x_binscat, bins=x_bins, weights=y_binscat*y_binscat)
    
    """
    mean, stddev of residuals in each x_bin
    """
    mean_binscat = sum_y_binscat / ndata_in_x_bin
    std_binscat = np.sqrt(sum_y2_binscat/ndata_in_x_bin - mean_binscat*mean_binscat)
    
    """
    indices of galaxies in each x_bin.  Values are 1...Number of 
    x Bins, where the integer specifies which bin for each galaxy
    """
    x_indices = np.digitize(x_binscat,x_bins)
    
    """
    compute descriptive statistics of x for each x bin
    
    e.g. element i of mean_x_bin is the mean x
    of all elements in bin i of x_bins (ie. data with 
    x_bins[i-1] <= x < x_bins[i]) 
    """
    mean_x_bin = np.zeros(x_bins.size)
    median_x_bin = np.zeros(x_bins.size)
    std_x_bin = np.zeros(x_bins.size)
    for index, value in np.ndenumerate(x_bins):
        if index[0] > 0:    
            mean_x_bin[index] = np.nanmean(x_binscat[(x_indices == index)])
            median_x_bin[index] = np.median(x_binscat[(x_indices == index)])
            std_x_bin[index] = np.std(x_binscat[(x_indices == index)])
    
    """
    assign y values in each of 3 x bins (for boxplot).
    This binning scheme must match definition of x_bins above
    """
    y_binscat_bin1 = y_binscat[(x_indices == 1)]
    y_binscat_bin2 = y_binscat[(x_indices == 2)]
    y_binscat_bin3 = y_binscat[(x_indices == 3)]
    y_boxplot_data = [y_binscat_bin1, y_binscat_bin2, y_binscat_bin3]
    
    if (args.verbose):
        print("\nSummary of x binned residuals")
        print("Number of bins: ",np.size(x_bins)-1)
        print("\nBin 1")
        print("\tNumber of data in bin          : ", y_binscat_bin1.size)
        print("\tx Range                        : ", x_bins[0:2])
        print("\tx Bin     - Mean               : {:.2f}".format(mean_x_bin[1]))
        print("\tx Bin     - Median             : {:.2f}".format(median_x_bin[1]))
        print("\tx Bin     - Std Dev            : {:.2f}".format(std_x_bin[1]))   
        print("\tResidual - Mean           : {:.2f}".format(y_binscat_bin1.mean()))
        print("\tResidual - Median         : {:.2f}".format(np.median(y_binscat_bin1)))
        print("\tResidual - Std Dev        : {:.2f}".format(np.std(y_binscat_bin1)))
    
        print("\nBin 2")
        print("\tNumber of data in bin          : ", y_binscat_bin2.size)
        print("\tx Range                        : ", x_bins[1:3])
        print("\tx Bin     - Mean               : {:.2f}".format(mean_x_bin[2]))
        print("\tx Bin     - Median             : {:.2f}".format(median_x_bin[2]))
        print("\tx Bin     - Std Dev            : {:.2f}".format(std_x_bin[2]))   
        print("\tResidual - Mean           : {:.2f}".format(y_binscat_bin2.mean()))
        print("\tResidual - Median         : {:.2f}".format(np.median(y_binscat_bin2)))
        print("\tResidual - Std Dev        : {:.2f}".format(np.std(y_binscat_bin2)))
    
        print("\nBin 3")
        print("\tNumber of data in bin          : ", y_binscat_bin3.size)
        print("\tx Range                        : ", x_bins[2:4])
        print("\tx Bin     - Mean               : {:.2f}".format(mean_x_bin[3]))
        print("\tx Bin     - Median             : {:.2f}".format(median_x_bin[3]))
        print("\tx Bin     - Std Dev            : {:.2f}".format(std_x_bin[3]))   
        print("\tResidual - Mean           : {:.2f}".format(y_binscat_bin3.mean()))
        print("\tResidual - Median         : {:.2f}".format(np.median(y_binscat_bin3)))
        print("\tResidual - Std Dev        : {:.2f}".format(np.std(y_binscat_bin3)))
        
    
    """
    CREATE FIGURE - RESIDUAL PLOT
    
    Generate combined box plot and residual plot
    
    See David Wittman's example page:
    http://www.physics.ucdavis.edu/~dwittman/Matplotlib-examples/
    http://www.physics.ucdavis.edu/~dwittman/Matplotlib-examples/plotexpresids.txt
    
    """
    
    # figsize in inches
    fig2 = plt.figure(dpi=300, figsize=[4,4])
    #fig.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0.05,0.05,0.95,0.95)})
    #fig.suptitle('y vs x (sample01/run01; ODR Estimation)', fontsize=12)
    
    plot_xlim= [6.25,11]
    plot_ylim = [-6.0,2.0]

    """
    FIGURE - RESIDUAL PLOT
     
     Plot y residuals vs x.  Outliers that were discarded
     from the clipping/refit process are shown in red.
    """
    axis6 = fig2.add_subplot(211)
    #axis1.set_title("Data", fontsize = 12)
    #axis1.set_xlabel(r"log Stellar Mass, $M_\odot$", fontsize = 12)
    #axis1.set_ylabel(r"residual ($\sigma$) in log SFR", fontsize = 12)
    """
    Plot the data with error bars in x, y
        marker options:  
            's'  square
            'o'  filled circle
    """
    plt.scatter(x_unscaled_refit, \
                y_refit_residual, \
                 color = 'blue', \
                 marker='o', \
                 alpha = 0.5)
    
    plt.scatter(x_unscaled[outliers], \
                y_rescaled_initialfit_residual[outliers], \
                 color = 'red', \
                 marker='o', \
                 alpha = 0.5)
    axis6.set_xlim(plot_xlim)
    axis6.set_ylim(plot_ylim)
    axis6.xaxis.set_major_formatter( NullFormatter() )

    
    """
    FIGURE - BOX PLOT
     
     Draw the box plot, illustrating quartiles of the y residuals
     in bins of x.  Boxes are labeld 
    
     see:  http://en.wikipedia.org/wiki/Box_plot
     http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
     
     median value:  red line
     IQR (inter-quartile range):  box (upper Q3, lower Q1)
     Q1 - 1.5xIQR, Q3 + 1.5 x IQR: upper, lower whiskers
     
     NB:  Quartile
     http://en.wikipedia.org/wiki/Quartile
     
     quartiles of a ranked set of data values are the three points that divide 
     the data set into four equal groups, each group comprising a quarter of 
     the data. A quartile is a type of quantile. The first quartile (Q1) is 
     defined as the middle number between the smallest number and the median 
     of the data set. The second quartile (Q2) is the median of the data. 
     The third quartile (Q3) is the middle value between the median and 
     the highest value of the data set.
    """
        
    axis7 = fig2.add_subplot(212)
    plot_ylim = [-1.5,1.5]
    bp = plt.boxplot(y_boxplot_data, positions = median_x_bin[1:], notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='blue', marker='+')
    axis7.set_xlim(plot_xlim)
    axis7.set_ylim(plot_ylim)
    axis7.xaxis.set_major_locator(MultipleLocator(1.0))
        
    plt.figtext(0.3,0.02,"log Stellar Mass, $M_\odot$",fontdict={'fontsize':12})
    #plt.figtext(0.01,0.7,"residual ($\sigma$) in log SFR",fontdict={'fontsize':12},rotation=90)
    plt.figtext(0.01,0.7,"residual in log SFR",fontdict={'fontsize':12},rotation=90)
    plt.subplots_adjust(wspace=0,hspace=0)
    
    """
     Save the plot as a .pdf file
    """
    residual_output_file= args.output_file[:-4]+'_residual.pdf'
    fig2.savefig(residual_output_file, format='pdf')
    
    if (args.display):
        plt.show()

    if (args.verbose):  
        print("\nExecutive summary")        
        
        print("\tSlope                     K07  : {:.3f}".format(k07_unscaled_refit_slope) + "$\pm${:.3f}".format(k07_rescaled_refit_slope_error))
        print("\tNormalizaton              K07  : {:.3f}".format(k07_unscaled_refit_intercept) + "$\pm${:.3f}".format(k07_rescaled_refit_intercept_error))
        print("\tIntrinsic scatter (sigma) K07  : {:.3f}".format(k07_rescaled_refit_scatter_sigma) + "$\pm${:.3f}".format(k07_rescaled_refit_scatter_sigma_error))
        print("\tTotal scatter (dex)       F87  : {:.3f}".format(y_rescaled_refit_residual.std()))
        print("\tOutlier fraction (positive)    : {:.3f}".format(outlier_fraction_positive))
        print("\tOutlier fraction (negative)    : {:.3f}".format(outlier_fraction_negative))
        print("\tMedian (reduced) chisq         : {:.3f}".format(chisqv_goodData_median))

        print("\nsfr_life_vs_m_line_clip_sim.py Done!")