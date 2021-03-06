#!/usr/bin/env python
"""
sfr_100_vs_m_line_clip_sim_linexp.py                       Python3 script

THIS VERSION FOR 2015-05-16-A/sample01/run03  lin-exp SFH FITS.

usage: sfr_100_vs_m_line_clip_sim.py [-h] [-v] [-d] [-i ITERATIONS]
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


Example:

Execution time for 10000 realizations is about 30 seconds.

See Experiment 2014-11-26-A, 2014-11-13-A, 2014-09-05-A, 2015-05-16-A. 

v0 01/05/2015
v1 02/16/2015  include residual plots, improved output format, anova
v3 04/03/2015  updated with most recent versions of 
v4 04/21/2015  minor cosmetic changes
v5 05/30/2015  estimate parameters using covariances; output refit data
                with covariances, for upload to khuseti...run IDL program.
v6 06/30/2015	new clipping method.  residuals saved as txt file
v7 08/04/2015  error ellipse estimates, and corrected a mistake in input
                of covariance data, see Table 1.
v8 09/17/2015  corrected minor inconsistencies in the way k07 fit results
                 are computed and reported.
v9 10/02/2015  updated revisions of v8 and changed output to F87 estimates
                 in Executive Summary.  Compute Y(XRef) for F87 best fit line.
                 Include SFR-M* correlation from Whitaker+2014 in plot.
v10 10/17/2015  Add selection functions computed in ...Manuscripts/2015/
                Kurczynski - SFR-M*/Analysis/2015-10-14-A/
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

    parser = argparse.ArgumentParser(description="Analyzes SFR-M* relation of input data with various methods and compares results to the literature (Speagle+2014, Whitaker+2014).  Performs clipping of outliers to a specitied threshold (default 3 sigma), and re-fits the data.  Estimates intrinsic scatter of the data using various methods.  Performs simulations to estimate errors.  Outputs pdf files showing data, best fit model(s) and residuals, and optional text data to console.")
    parser.add_argument('input_file', help='The input .fits file.  See program listing for columns used.')
    parser.add_argument('output_file', help='The output .pdf file')
    parser.add_argument('-v','--verbose', action = 'store_true', help='Print descriptive text, including results of analysis.')
    parser.add_argument('-d','--display', action = 'store_true', help='Display the plot.')
    parser.add_argument('-i','--iterations', type = int, default = '1000', help='Number of iterations in simulation.  Default = 1000')
    parser.add_argument('-c','--clip_sigma', type = float, default = '3.0', help='Outlier threshold (sigma).  Default = 3.0')
    parser.add_argument('-z','--zSpeagle2014', type = float, default = '1.0', help='Redshift for Speagle+2014 Main Sequence correlation line.  Default = 1.1.')
    
    args = parser.parse_args()
    
    if (args.verbose):
        print("sfr_100_vs_m_line_clip_sim_linexp.py \n")
        
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
    
    logSFR_Input_Value = tbData.field('LogSFR_100_Expected_Value')
    logSFR_Input_Err = tbData.field('LogSFR_100_Expected_Err68')
    
    """
    Parameter indices for determining covariance matrix elements:
    
    Table 1.  Parameters in 2015-05-16-A/sample00/run03
    
    Number of parameters         :  20
    Parameter list (use for indexing Cov and Corr matrix elements):
	1 = LogAge_Yr
	2 = Age_Gyr
	3 = LogGalaxyMass_Msun
	4 = GalaxyMass_9
	5 = EBV
	6 = LogTau_Yr
	7 = Tau_Gyr
	8 = LogStellarMass_Msun
	9 = StellarMass_9
	10 = LogSFR_Inst
	11 = SFR_Inst
	12 = LogSFR_Life
	13 = SFR_Life
	14 = LogSFR_100
	15 = SFR_100
	16 = LogT50_Yr
	17 = T50_Gyr
	18 = LogSFR_Max
	19 = LogSFR_MaxAvg
	20 = LogGalaxyMass_Inf_Msun
 
    Caption:  parameter list used to refer to estimated parameters 
    from getdist analysis of speedymc output.  See *fixed*.ini files
    in khuseti:~/analysis/candels/2015-05-16-A/sample00/run03/analysis/
    See also caption for table in database ecdfs.candels_2015a.
    sample00_run03_speedymc_results.
    
    Use parameter indices to identify the proper correlation matrix 
    element, e.g. Cov(8,12) is Cov(log StellarMass, logSFR_Life).
    """

    Covariance_LogStellarMass_LogSFR = tbData.field('Cov_8_14')
    Variance_LogStellarMass = tbData.field('Cov_8_8')
    Variance_LogSFR = tbData.field('Cov_14_14')

    Chisq_BestFit = tbData.field('Chisq_BestFit')
    
    Correlation_LogStellarMass_LogSFR = tbData.field('Corr_8_14')
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

    """
    Report clipping results, outliers.
    Save initial fit data to file
    """
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

        initial_fit_residual_filename = 'sfr_100_vs_m_initial_fit_residuals.txt'
        np.savetxt(initial_fit_residual_filename, \
                    np.column_stack((x,sx,y,sy,y_rescaled_initialfit_model,y_rescaled_initialfit_residual)), \
                    fmt=('%5.6f','%5.6f','%5.6f','%5.6f','%5.6f','%5.6f'), \
                    header='x xerr y yerr ymodel yresidual')
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
    Compute scatter of refit data with covariance
    """
    (ols_rescaled_refit_cov_scatter_variance, total_scatter_variance, xerr_scatter_term, yerr_scatter_term) = scatter_variance_adhoc(x_refit,\
                y_refit,\
                sx_refit,\
                sy_refit,\
                f87_rescaled_refit_cov_slope,\
                f87_rescaled_refit_cov_intercept,\
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
    See khsueti:~/analysis/CANDELS/2015-05-16-A/sample01/analysis/run03/
    summary__run03.txt
    
    Table 2.  Fit results for method of Kelly (2007) on 
    log SFR vs. log M* data for sample01 run03
    
    Version Scatter Var.    Slope           Intercept
    (1)     (2)             (3)             (4)    
    
                    --- SFR(100) ---
    v8      0.0570 0.0045   0.9739 0.0143   0.07738 0.0114
    v9*     0.0586 0.0044   0.9723 0.0142   0.0761  0.0113
    v14     0.0516 0.0038   0.9767 0.0141   0.08932 0.0078
    
                    --- SFR(Inst) ---
    v10     0.0552 0.0052   0.8402 0.0150   0.0976  0.0116
    v11*    0.0573 0.0052   0.8375 0.0143   0.1000  0.0117
    v15     0.0507 0.0041   0.8601 0.0120   0.0861  0.0089
    
                    --- SFR(Life) ---
    v12     0.0024 0.00019  1.0103 0.00039  0.03318 0.00082
    v13*    0.0264 0.00259  1.0385 0.01240  0.08114 0.01000
    v16     0.0228 0.00245  1.0498 0.01172  0.09787 0.00922
    
    Caption:  (1) is the version of the fit run, see corresponding .sav file 
    for results. v8, v10, v12 were done with mistaken covariance matrix
    elements.  This was corrected in v14, v15, v16 (pk 8/10/2015).  
    *fits done without covariance. (2) is the best fit 
    intrinsic scatter variance and error, determined from mean and std 
    deviation of the posterior distribution.  (3) is the best fit slope 
    (beta, in usage of IDL program) and error.  (4) is Intercept and error 
    (alpha in usage of IDL program).
    """
    k07_rescaled_refit_slope = 0.9723
    k07_rescaled_refit_slope_error = 0.0142
    
    k07_rescaled_refit_cov_slope = 0.9767
    k07_rescaled_refit_cov_slope_error = 0.0141
    
    k07_rescaled_refit_intercept = 0.0761
    k07_rescaled_refit_intercept_error = 0.0113
    
    k07_rescaled_refit_cov_intercept = 0.08932
    k07_rescaled_refit_cov_intercept_error = 0.0078
    
    k07_rescaled_refit_scatter_variance = 0.0586
    k07_rescaled_refit_scatter_variance_error = 0.0044
    k07_rescaled_refit_scatter_variance_frac_error = k07_rescaled_refit_scatter_variance_error / k07_rescaled_refit_scatter_variance

    k07_rescaled_refit_cov_scatter_variance = 0.0516
    k07_rescaled_refit_cov_scatter_variance_error = 0.0038
    k07_rescaled_refit_cov_scatter_variance_frac_error = k07_rescaled_refit_cov_scatter_variance_error / k07_rescaled_refit_cov_scatter_variance
    
    k07_rescaled_refit_scatter_sigma = np.sqrt(k07_rescaled_refit_scatter_variance)
    k07_rescaled_refit_scatter_sigma_frac_error = 0.5 * k07_rescaled_refit_scatter_variance_frac_error
    k07_rescaled_refit_scatter_sigma_error = k07_rescaled_refit_scatter_sigma * k07_rescaled_refit_scatter_sigma_frac_error
    
    k07_rescaled_refit_cov_scatter_sigma = np.sqrt(k07_rescaled_refit_cov_scatter_variance)
    k07_rescaled_refit_cov_scatter_sigma_frac_error = 0.5 * k07_rescaled_refit_cov_scatter_variance_frac_error
    k07_rescaled_refit_cov_scatter_sigma_error = k07_rescaled_refit_cov_scatter_sigma * k07_rescaled_refit_cov_scatter_sigma_frac_error

    k07_refit_slope = k07_rescaled_refit_slope
    k07_refit_intercept = k07_rescaled_refit_intercept - x_scale * k07_rescaled_refit_slope

    """
    Output text data files with refit logM, logSFR data for use in external
    fitting routines(eg. Kelly (2007) method implemented in IDL)
    """
    if (args.verbose):
       refit_data_table_id_cov_corr_filename = 'sfr_100_vs_m_refit_id_cov_corr.txt'    
       refit_data_table_cov_filename = 'sfr_100_vs_m_refit_cov.txt'
       refit_data_table_filename = 'sfr_100_vs_m_refit.txt'
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
       Output data for import to database
       
       Save as structured array because np.column_stack chokes
       on mixed string/float data types.
       """
       db_import_filename = 'sfr_100_vs_m_refit_db_table.txt'
       np.savetxt(db_import_filename,
                  np.column_stack((source_id_goodData,
                                   outliers,
                                   x_goodData_unscaled,
                                   x_goodData_unscaled_err,
                                   y_goodData_unscaled,
                                   y_goodData_unscaled_err,
                                   covxy_goodData,
                                   corrxy_goodData)),
                  fmt=('%d', '%d', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f'), 
                  header = 'id outlier x sx y sy covxy corrxy')
       
       print("Good data saved to file: ",db_import_filename)
           
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
    ols_unscaled_refit_cov_scatter_variance = ols_rescaled_refit_cov_scatter_variance
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

    f87_rescaled_refit_scatter_variance_frac_error = f87_rescaled_refit_scatter_variance_error / f87_rescaled_refit_scatter_variance
    f87_rescaled_refit_cov_scatter_variance_frac_error = f87_rescaled_refit_scatter_variance_error / f87_rescaled_refit_cov_scatter_variance
    f87_rescaled_refit_scatter_sigma = np.sqrt(f87_rescaled_refit_scatter_variance)
    f87_rescaled_refit_scatter_sigma_frac_error = 0.5 * f87_rescaled_refit_scatter_variance_frac_error
    f87_rescaled_refit_scatter_sigma_error = f87_rescaled_refit_scatter_sigma * f87_rescaled_refit_scatter_sigma_frac_error
    
    f87_rescaled_refit_cov_scatter_sigma = np.sqrt(f87_rescaled_refit_cov_scatter_variance)
    f87_rescaled_refit_cov_scatter_sigma_frac_error = 0.5 * f87_rescaled_refit_cov_scatter_variance_frac_error
    f87_rescaled_refit_cov_scatter_sigma_error = f87_rescaled_refit_cov_scatter_sigma * f87_rescaled_refit_cov_scatter_sigma_frac_error

    """
    correlation coefficients of data set
    """
    spearman_rho, spearman_pvalue = s.spearmanr(x_refit,y_refit)
    pearson_rho, pearson_pvalue = s.pearsonr(x_refit,y_refit)

    """
    y value at a specified x reference value
    
    This determines the value b, in the model equation
    
    y = a (x - x_reference_value) + b

    these values are for use in comparing with literature
    log_mass_reference_value = 9.0 means compute log_sfr(log_mass = 9.0) 
    (as opposed to using the intercept, which is log sfr(log mass = 0)
    """
    x_reference_value = 9.0
    f87_y_at_x_reference_value = f87_unscaled_refit_cov_intercept + x_reference_value * f87_unscaled_refit_cov_slope
    f87_y_at_x_reference_value_error = np.sqrt(f87_rescaled_refit_slope_error**2 + f87_rescaled_refit_intercept_error**2)
    
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
        print("\t                          k07  : {:.2f}".format(k07_unscaled_refit_cov_intercept))
        print("\tScatter Variance Estimates")
        print("\t                          F87  : {:.5f}".format(f87_unscaled_refit_cov_scatter_variance))
        print("\t                          OLS  : {:.5f}".format(ols_unscaled_refit_cov_scatter_variance))
        print("\t                          k07  : {:.5f}".format(k07_unscaled_refit_cov_scatter_variance))
        print("Intercept Estimates at log mass reference value")
        print("\tX reference value              : {:.5f}".format(x_reference_value))
        print("\tY(ref value)              F87  : {:.5f}".format(f87_y_at_x_reference_value))
        print("\tY(ref value) error        F87  : {:.5f}".format(f87_y_at_x_reference_value_error))
    
        scatter_variance_adhoc(x_refit,y_refit,sx_refit,sy_refit,f87_rescaled_refit_slope,f87_rescaled_refit_intercept, covxy = covxy_refit, verbose = True)


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
    fig = plt.figure(dpi=300, figsize=[4,4])
    fig.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
        
    """
    Figure - subplot - Data +  Model
    """
    plot_xlim = [6.25,11]
    plot_ylim= [-3.0,3.0]

    axis5 = fig.add_subplot(111)
    axis5.set_xlabel(r"Log Stellar Mass, $M_\odot$", fontsize = 12)
    axis5.set_ylabel(r"Log SFR$_{100}$, $M_\odot$ yr$^{-1}$", fontsize = 12)
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
     
     To scale relative to log(10.5), set to this:
     x_scale_Speagle2014 = 10.5 
     
     Otherwise, rescale value will be set to the same as the
     input data set     
     pk 7/30/2015
    """
    x_scale_Speagle2014 = x_scale    
    xi = slope_Speagle2014
    eta = intercept_Speagle2014 - x_scale_Speagle2014 * slope_Speagle2014
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
    
    logMass_Rescaled = logMass_Speagle2014 - x_scale_Speagle2014
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
             linewidth = 2, \
             linestyle='dashed')
    axis5.plot(logMass_Speagle2014[noExtrapolation],\
             logSFR_Speagle2014[noExtrapolation],\
             color='Red',\
             linestyle='solid',\
             linewidth = 2, \
             label='Speagle+14')
    axis5.plot(logMass_Speagle2014[highExtrapolation],\
             logSFR_Speagle2014[highExtrapolation],\
             color='Red',\
             linewidth = 2, \
             linestyle='dashed')
    """
    axis5.fill_between(logMass_Speagle2014, \
                logSFR_Speagle2014_LowErr_MassRescaled, \
                logSFR_Speagle2014_HighErr_MassRescaled, \
                facecolor = 'Pink',\
            #    hatch = "X",\
                edgecolor = 'Pink', \
                linewidth=0.0)       
    """ 
     
    """ 
    Main Sequence (Whitaker+2014)

    Plot the MS best fit curve from Whitaker+2014, based on 
    values in Whitaker+2014 Table 1:
    
    redshift	       alow		  ahigh	         b
    0.5<z<1.0	0.94 pm 0.03	 0.14 pm 0.08	1.11 pm 0.03
    1.0<z<1.5	0.99 pm 0.04	 0.51 pm 0.07	1.31 pm 0.02
    1.5<z<2.0	1.04 pm 0.05	 0.62 pm 0.06	1.49 pm 0.02
    2.0<z<2.5	0.91 pm 0.06	 0.67 pm 0.06	1.62 pm 0.02
  
    Model Equation:  Y = a (log M - 10.2) + b
    
    Lower limits to data in Whitaker+2014 (Taken by inspection 
    from their Figure 3)  
    
    redshift    log_mass (msun)
    0.5<z<1.0	8.5
    1.0<z<1.5	9.0
    1.5<z<2.0	9.2
    2.0<z<2.5	9.3

    """
    a_whitaker2014 = 0.99
    b_whitaker2014 = 1.31
    log_mass_lower_limit_whitaker2014 = 9.0
    
    log_mass_whitaker2014 = np.linspace(x_unscaled_refit.min(),x_unscaled_refit.max(),100)
    # compute log SFR using model equation above; add log(1.8) to convert
    # from Chabrier IMF used in Whitaker+2014 to Salpeter IMF used here.
    # SFR(Salpeter) = SFR(Chabrier) * 1.8  (Erb+2006)
    log_sfr_whitaker2014 = a_whitaker2014  * (log_mass_whitaker2014 - 10.2) + b_whitaker2014 + 0.2553

    whitaker2014_low_extrapolation = ((log_mass_whitaker2014 > x.min()) & (log_mass_whitaker2014 <= log_mass_lower_limit_whitaker2014))
    whitaker2014_no_extrapolation = ((log_mass_whitaker2014 > log_mass_lower_limit_whitaker2014))

    """
    plot the MS line from Whitaker+2014
    """
    axis5.plot(log_mass_whitaker2014[whitaker2014_low_extrapolation],\
             log_sfr_whitaker2014[whitaker2014_low_extrapolation],\
             color='darkcyan',\
             linewidth = 2, \
             linestyle='dashed')
    axis5.plot(log_mass_whitaker2014[whitaker2014_no_extrapolation],\
             log_sfr_whitaker2014[whitaker2014_no_extrapolation],\
             color='darkcyan',\
             linestyle='solid', \
             linewidth = 2, \
             label = "Whitaker+14")
     
    """
    Plot My Model Fit:
    error region (bottom layer), data and outliers (middle layer), 
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
    
    """    
    xydata = np.vsplit(np.transpose(np.vstack([x_unscaled_refit,y_unscaled_refit])),1)
    sigx = sx_refit
    sigy = sy_refit
    sigxy = covxy_refit   
    sigxyterm = np.sqrt(0.25 * (sigx**2 - sigy**2)**2 + sigxy**2)    
    sigxp2 = 0.5 * (sigx**2 + sigy**2) + sigxyterm
    # this value can go negative, which corresponds to a degenerate ellipse
    # that is a straight line.  max(..,0) prevents numerical error in sqrt()
    sigyp2 = max(0.5 * (sigx**2 + sigy**2) - sigxyterm, 0.0)
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

    """
    Output text data files with refit x,y values and covariance, correlation
    and ellipse theta values for debugging ellipse computation.
    """
    """
    if (args.verbose):
       refit_data_table_id_cov_corr_theta_filename = 'sfr_100_vs_m_refit_id_cov_corr_ellipse.txt'    
       np.savetxt(refit_data_table_id_cov_corr_theta_filename,
                  np.column_stack((source_id_refit,x_refit,sx_refit,y_refit,sy_refit,sx_refit2, sy_refit2, covxy_refit,corrxy_refit, sigxp2, sigyp2, theta)),
                  fmt=('%d', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f', '%5.6f'), 
                  header = 'id,x_refit,sx_refit,y_refit,sy_refit,sx_refit**2,sy_refit**2,covxy_refit,corrxy_refit,sigxp2,sigyp2,theta')

       print("Refit id, data, covariances, correlations, ellipse theta saved to file: ",refit_data_table_id_cov_corr_theta_filename)
    """

    """
    Draw representative error ellipse in the upper left
    
    https://m.youtube.com/watch?v=717fVhFKn8E
    http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    
    See error ellipses in action
    https://www.youtube.com/watch?v=E7rnPrwbLmI
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
    # this value can go negative, which corresponds to a degenerate ellipse
    # that is a straight line.  max(..,0) prevents numerical error in sqrt()
    sigyp2 = max(0.5 * (sigx**2 + sigy**2) - sigxyterm, 0.0)
    semiMajorAxis = np.sqrt(sigxp2)
    semiMinorAxis = np.sqrt(sigyp2)
    # arctan2:  "y" is the first parameter, "x" is the second parameter    
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
    Plot empirical selection functions from Experiment 2015-10-14-A

    parameters a-d, f_thresh are defined in Experiment 2015-10-14-A.
    See Summary file and documents therein.
    
    Table 1.  log F16W vs logMass		
    		
    Description	Slope, c	Intercept, d
    All	      0.81136	-7.396
    Age < 0.1 Gyr		
    Age > 1.0 Gyr		
    EBV > 0.5		
    EBV < 0.1		
    		
    Table 2.  log F435W vs logSFR		
    		
    Description	       Slope, a	Intercept, b
    SFR-M* non-outliers	0.59397	-1.16422
    Age < 0.1 Gyr     	0.58949	-1.12391
    Age >  1.0 Gyr 	0.49613	-1.10587
    
    Table 3.  Flux threshold
    
    Description     Value
    Orig. value     0.012
    Uniform value   0.025

    Caption:  Flux threshold used below as f_thresh.  See Excel
    Spreadsheet in ...Manuscripts/2015/.../Analysis/2015-10-14-A/

    Below parameters correspond to age > 1.0 Gyr subsample of sample00,
    which leads to the most extreme selection function (ie that which
    impinges most upon the range sampled by our data)
    """
    a=0.59397
    b=-1.16422
    c=0.81136
    d=-7.396
    f_thresh = 0.025

    x_selection_function = np.linspace(5.0,10.0,1000)  
    y_selection_function = -1*b/a + (1/a)*np.log10(2*f_thresh - 10**(c*x_selection_function + d))
    axis5.plot(x_selection_function, 
               y_selection_function,
               linewidth = 3, 
               color = 'Black')

    """
    Plot text label:  intrinsic scatter
    """    
    text_label_scatter =r"$\sigma_{is} = " + r"{:.3f}".format(f87_rescaled_refit_cov_scatter_sigma) + r"\pm"+ r"{:.3f}".format(f87_rescaled_refit_cov_scatter_sigma_error)+r"$"
    text_label_r2 = r"$r^2 = " + r"{:.2f}".format(pearson_rho**2) + r"$"
    
    """
    Text labels in lower left
    """
    """
    axis5.text(8.75,-2.0,text_label_r2,fontsize='small', fontweight="bold")
    axis5.text(8.75,-2.3,text_label_scatter,fontsize='small', fontweight="bold")
    """

    """
    Text labels in upper right
    """    
    axis5.text(6.5, 2.4,text_label_r2,fontsize='small', fontweight="bold")
    axis5.text(6.5, 2.0,text_label_scatter,fontsize='small', fontweight="bold")
    
    #axis5.legend(loc='upper left', fontsize='small')

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
    plot_ylim = [-3.5,2.0]

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
    plot_ylim = [-1.9,1.9]
    bp = plt.boxplot(y_boxplot_data, positions = median_x_bin[1:], notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='blue', marker='+')
    axis7.set_xlim(plot_xlim)
    axis7.set_ylim(plot_ylim)
    axis7.xaxis.set_major_locator(MultipleLocator(1.0))
        
    plt.figtext(0.3,0.02,"Log Stellar Mass, $M_\odot$",fontdict={'fontsize':12})
    #plt.figtext(0.01,0.7,"residual ($\sigma$) in log SFR",fontdict={'fontsize':12},rotation=90)
    plt.figtext(0.01,0.7,"Residual in Log SFR",fontdict={'fontsize':12},rotation=90)
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
        print("\tNumber of data in fit          : ", len(y_refit))        
        print("\tSlope                     F87  : {:.3f}".format(f87_unscaled_refit_cov_slope) + "$\pm${:.3f}".format(f87_rescaled_refit_slope_error))
        print("\tIntercept                 F87  : {:.3f}".format(f87_unscaled_refit_cov_intercept) + "$\pm${:.3f}".format(f87_rescaled_refit_intercept_error))
        print("\tXRef                           : {:.3f}".format(x_reference_value))        
        print("\tY(XRef)                   F87  : {:.3f}".format(f87_y_at_x_reference_value) +"$\pm${:.3f}".format(f87_y_at_x_reference_value_error))
        print("\tIntrinsic scatter (sigma) F87  : {:.3f}".format(f87_rescaled_refit_cov_scatter_sigma) + "$\pm${:.3f}".format(f87_rescaled_refit_cov_scatter_sigma_error))
        print("\tTotal scatter (dex)       F87  : {:.3f}".format(y_rescaled_refit_residual.std()))
        print("\tOutlier fraction (positive)    : {:.3f}".format(outlier_fraction_positive))
        print("\tOutlier fraction (negative)    : {:.3f}".format(outlier_fraction_negative))
        print("\tMedian (reduced) chisq         : {:.3f}".format(chisqv_goodData_median))

        print("\nsfr_100_vs_m_line_clip_sim_linexp.py Done!")