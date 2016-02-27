"""
linfit_multi_methods.py                           Python3 functions

Implements multiple methods of fitting a linear model to x,y data
with errors, including intrinsic scatter.

v1 12/08/2014
v2 02/13/2015 implements additional fitting methods from astronomical
              literature
v3 02/17/2015 implement additional scatter estimation methods
v4 02/25/2015 yet more edits and additions to scatter estimation methods.
v5 03/13/2015 estimates using BCES, F87, K07 (not implmented), MLE, ODR, OLS, T02, WLS
pk
"""
import lnr3 as lnr
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pidly
import random
import scipy.odr.odrpack as odrpack
from scipy.optimize import minimize, newton
import timeit


def scatter_variance_adhoc(x,y,xerr,yerr,slope,intercept, verbose = False):
    """
    Computes intrinsic scatter from a linear model fit by subtracting other components to the variance of the fit residuals.
    
    Intrinsic scatter is obtained from the fit residuals and xerr,yerr 
    values according to:
    
    < (y-yfit)^2 > = sig_is^2 + < yerr^2 > + m^2 * < xerr^2 >

    where yfit is the model fit (whereby the LHS is the variance of the 
    residuals to the fit), sig_is is the intrinsic scatter sigma and
    m is the slope of the best fit line.  This estimate of intrinsic scatter 
    includes any contribution due to regression lack of fit.  

    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
        slope (float):  slope of the corresponding linear model
        intercept (float):  intercept of the corresponding linear model
        
    Returns:  
        intrinsic_scatter_variance_estimate (float):  estimated intrinsic scatter
        total_scatter_variance (float):  total scatter:  < (y-yfit)^2 >
        xerr_scatter_term (float):  < xerr^2 >
        yerr_scatter_term (float):  < yerr^2 >
    """    

    yfit = intercept + slope * x
    residual = np.empty_like(yfit,dtype=np.float64)    
    residual[...] = y - yfit
    
    xerr2 = np.empty_like(xerr,dtype=np.float64)
    yerr2 = np.empty_like(yerr,dtype=np.float64)
    xerr2 = xerr**2
    yerr2 = yerr**2
    xerr_scatter_term = xerr2.mean()
    yerr_scatter_term = yerr2.mean()
    
    total_scatter_variance = residual.var()
    intrinsic_scatter_variance_term = total_scatter_variance - yerr_scatter_term - slope**2 * xerr_scatter_term   
    
    if intrinsic_scatter_variance_term > 0:
        intrinsic_scatter_variance_estimate = intrinsic_scatter_variance_term
    else:
        intrinsic_scatter_variance_estimate = 0.0

    if verbose:
        print("Components of scatter")
        print("\tTotal scatter, < (y-yfit)^2 >          : {:.5f}".format(total_scatter_variance))
        print("\tScatter due to y-err, <y_err^2>        : {:.5f}".format(yerr_scatter_term))
        print("\tBest fit slope, m, in linear model     : {:.5f}".format(slope))
        print("\tScatter due to x-err, <x_err^2>        : {:.5f}".format(xerr_scatter_term))
        print("\tIntrinsic  scatter, sig_IS^2           : {:.5f}".format(intrinsic_scatter_variance_estimate ))
        print("\tIntrinsic scatter, % of total          : {:.2f}%".format((intrinsic_scatter_variance_estimate/total_scatter_variance) * 100))
        
    return intrinsic_scatter_variance_estimate, total_scatter_variance, xerr_scatter_term, yerr_scatter_term

def chisqexy(x, y, xerr, yerr, slope, intercept, scatter_variance):
    """
    Compute modified chisq function of Tremaine+2002
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
        slope (float):  slope of the corresponding linear model
        intercept (float):  intercept of the corresponding linear model
        scatter_variance (float): intrinsic scatter variance
    
    Returns:  
        chisqexy (float):  1/df * sum( (yi - a - b xi))^2/(s^2 + yerr^2 + b*xerr^2) )
        where a=intercept, b=slope
    """

    """
    chisq df = N - 2
    """    
    df = len(y) - 2
    numerator = (y - intercept - slope*x)**2
    denominator = scatter_variance + yerr**2 + slope**2 * xerr**2
    try:
        chisqexy = np.sum(numerator/denominator)
    except ZeroDivisionError:
        print("ERROR fitexy_chisq (1):  divide by zero")
        chisqexy = np.nan
    
    try:
        reduced_chisqexy = chisqexy/df
    except ZeroDivisionError:
        reduced_chisqexy = np.nan
        print("ERROR fitexy_chisq (2):  divide by zero")
    
    return reduced_chisqexy


def fitexy_scatter(x, y, xerr, yerr, slope, intercept):
    """
    Estimate intrinsic scatter using the method of Tremaine+2002
    
    scatter variance, s is obtained by finding a root of the equation 
    f(s) = 0 where f(s) = chisqexy(s) - 1.0 using Newton's method.
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
        slope (float):  slope of the corresponding linear model
        intercept (float):  intercept of the corresponding linear model
        
    Returns:  
        scatter_variance_estimate (float):  estimated intrinsic scatter
    """
    
    chisqexy_minus_one = lambda s2: chisqexy(x,y,xerr,yerr,slope,intercept,s2) - 1.0

    s2_initial_guess = 0.0
    scatter_variance_estimate = newton(chisqexy_minus_one,s2_initial_guess)    
       
    return scatter_variance_estimate
    

def linfit_bces(x, y, xerr, yerr):
    """
    Estimate a linear model using method of Akritas and Bershady (BCES).  Provide ad-hoc estimate of intrinsic scatter variance.

    See Akritas and Bershady 1996 The Astrophysical Jounal 470 706.  This 
    implementation uses the lnr package, which I adapted for use in Python 3.
    See http://home.strw.leidenuniv.nl/~sifon/pycorner/bces/
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data

    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter using
            above equation or 0 if a negative value is obtained.
    """
    
    (bces_intercept_tuple, \
    bces_slope_tuple, \
    bces_covar) = lnr.bces(x,y,x1err=xerr, x2err=yerr,logify=False,bootstrap=10, verbose='quiet')
    bces_slope_estimate = bces_slope_tuple[0]
    #bces_slope_err_estimate = bces_slope_tuple[1]
    bces_intercept_estimate = bces_intercept_tuple[0]
    #bces_intercept_err_estimate = bces_intercept_tuple[1]
    (bces_scatter_variance_estimate, \
    total_scatter_variance, \
    xerr_scatter_terrm,\
    yerr_scatter_term) = scatter_variance_adhoc(x,y,xerr,yerr,bces_slope_estimate,bces_intercept_estimate) 

    return (bces_slope_estimate,bces_intercept_estimate,bces_scatter_variance_estimate)

    
def linfit_f87(x, y, xerr, yerr, covxy = None):
    """
    Estimate linear model parameters and intrinsic scatter using the maxium likelihood method of Fuller 1987.
    
    Estimate intrinsic scatter from Fuller 1987 (F87)    
    This code implements Equations 3.1.7 and 3.1.13 p. 187 (heteroscedastic)
    and Equation 2.2.21 on p. 107 (homoscedastic model)
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
        covxy (float array):  covariance between x,y values (for each pair)
        
    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter
    """ 


    """
    Implementation of best fit linear model with intrinsic scatter 
    for homoscedastic and uncorrelated errors.
    """    
    """
    ndata = len(x)
    x_mean = np.average(x)
    y_mean = np.average(y)
    xerr_mean = np.average(xerr)
    yerr_mean = np.average(yerr)    
    sxy = np.sum(x*y)
    sxx = np.sum(x*x) 

    # temporary usage
    Suu = xerr_mean**2
    Sww = yerr_mean**2
    Suw = 0.0
    
    numerator = sxy - ndata * Suw - ndata * x_mean * y_mean
    denominator = sxx - ndata * Suu - ndata * x_mean * x_mean
    slope_estimate = numerator / denominator

    Svv = 1/(ndata - 2) * np.sum( (y - y_mean - slope_estimate * (x - x_mean))**2)

    # This equation is the implementation of Fuller 1987 Eq. 2.2.21
    Sqq = Svv - (Sww - 2*slope_estimate*Suw + Suu * slope_estimate**2)
    

     
    scatter_variance_estimate = Sqq    
    intercept_estimate = y_mean - slope_estimate * x_mean
    """
    
    """
    Implementation of best fit linear model with intrinsic scatter
    for heteroscedastic (and possibly correlated) errors
    """
    
    ndata = len(x)

    # elements of covariance matrix for each observed x,y pair
    if covxy is not None:    
        sigxy = covxy
    else:
        sigxy = np.zeros_like(x)        
    sigx2 = xerr**2
    sigy2 = yerr**2
    
    xy_avg = np.average(x*y)
    x_avg = np.average(x)
    y_avg = np.average(y)
    x2_avg = np.average(x**2)
    sigx2_avg = np.average(sigx2)
    sigy2_avg = np.average(sigy2)
    sigxy_avg = np.average(sigxy)

    numerator = xy_avg - x_avg*y_avg - sigxy_avg
    denominator = x2_avg - x_avg**2 - sigx2_avg
    try:
        slope_estimate = numerator/denominator
    except ZeroDivisionError:
        slope_estimate = float('Inf')
        print("ERROR - linfit_f87:  slope_estimate is infinite")
        
    intercept_estimate = y_avg - slope_estimate * x_avg
    
    yfit = slope_estimate * x + intercept_estimate
    residual= y - yfit
    try:
        Svv =  np.sum(residual**2) / (ndata - 2)
    except ZeroDivisionError:
        Svv = float('Inf')
        print("ERROR - linfit_f87:  insufficient data for scatter estimate")
    
    scatter_variance_estimate = Svv \
                                - sigy2_avg \
                                + 2 * slope_estimate * sigxy_avg \
                                - slope_estimate**2 * sigx2_avg
    
    return (slope_estimate,intercept_estimate,scatter_variance_estimate)
    
def linfit_k07_from_file(input_file_path, input_file_name):
    """
    fits a data set to a linear model with intrinsic scatter using method of Kelly (2007)
    
    This function uses an IDL interface to read a text file, do the fitting in IDL and return the result.
    This function calls the IDL procedure linmix_wrap.pro which must be present in the IDL PATH.
    
    parameters:
    file_path (string):  complete file path to the data file
    file_name (string):  name of the data file
    
    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter
    """
    
    idl = pidly.IDL()
    print("Invoking linmix_err.pro in IDL.  This may take a while...")
    cmd1 = "input_file_path = \'%s\'"%input_file_path
    idl(cmd1)
    cmd2 = "input_file_path = \'%s\'"%input_file_name
    idl(cmd2)
    cmd3 = 'linmix_wrap, input_file_path, input_file_name, fit'
    idl(cmd3)
    k07_slope_posterior = idl.ev('fit.beta')
    k07_intercept_posterior = idl.ev('fit.alpha')
    k07_scatter_variance_posterior = idl.ev('fit.sigsqr')
    k07_slope = k07_slope_posterior.mean()
    k07_intercept = k07_intercept_posterior.mean()
    k07_scatter_variance = k07_scatter_variance_posterior.mean()
    #k07_scatter_sigma = np.sqrt(k07_scatter_variance)

    return (k07_slope,k07_intercept,k07_scatter_variance)


def linfit_mle(x, y, xerr, yerr):
    """
    Estimate a linear model using method of maximum likelihood found in lnr.py
    
    No documentation is found for this function; however it includes intrinsic scatter
    estimation.  This method numerically optimizes the likelihood function:
    logLike = 2 sum w_i + Sum (y_i - yfit)/w_i)^2  + log N sqrt(2 pi)/2
    where w_i = sqrt(b^2x_err^2 + y_err^2 + s^2) and yfit = a + b*x
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data

    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter using
            above equation or 0 if a negative value is obtained.
    """

    (ols_slope_estimate,\
    ols_intercept_estimate,\
    ols_scatter_variance_estimate) = linfit_ols(x, y, xerr, yerr)
    ols_scatter_sigma_estimate = np.sqrt(ols_scatter_variance_estimate)
    
    mle_po = (ols_intercept_estimate, ols_slope_estimate, ols_scatter_sigma_estimate)
    (mle_intercept_estimate, mle_slope_estimate, mle_scatter_sigma_estimate) = lnr.mle(x,y,x1err=xerr,x2err=yerr, logify=False,s_int = True, po=mle_po)
    mle_scatter_variance_estimate = mle_scatter_sigma_estimate**2
    
    return (mle_slope_estimate,mle_intercept_estimate,mle_scatter_variance_estimate)


def linfit_ols(x, y, xerr, yerr):
    """
    Estimate a linear model using Ordinary Least Squares (OLS).  Provide ad-hoc estimate of intrinsic scatter variance.
    
    The standard method of OLS model estimation is done to estimate slope
    and intercept.  See Draper & Smith Applied Regression Analysis p.14-15.
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data

    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter using
            above equation or 0 if a negative value is obtained.
    """
    ndata = len(x) 
    x_mean = np.average(x)
    y_mean = np.average(y)

    Sww = 1/(ndata-1) * np.sum((x - x_mean)*(x - x_mean))   
    Syw = 1/(ndata-1) * np.sum((x - x_mean)*(y - y_mean))
    ols_slope_estimate = Syw/Sww
    ols_intercept_estimate = y_mean - ols_slope_estimate * x_mean       
    (ols_scatter_variance_estimate, \
    total_scatter_variance, \
    xerr_scatter_terrm,\
    yerr_scatter_term)  = scatter_variance_adhoc(x,y,xerr,yerr,ols_slope_estimate,ols_intercept_estimate, verbose = False)       

    return (ols_slope_estimate,ols_intercept_estimate,ols_scatter_variance_estimate)

def linfit_odr(x, y, xerr, yerr, verbose = False):
    """
    Estimate a linear model using Orthogonal Distance Regression (ODR).  Provide ad-hoc estimate of intrinsic scatter variance.
    
    This implementation of ODR using Python package ODRPack.  Inital guess
    to linear model parameters is obtained from Ordinary Least Squares.
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data

    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter variance
    """

    (ols_slope_estimate,\
    ols_intercept_estimate,\
    ols_scatter_variance_estimate) = linfit_ols(x, y, xerr, yerr)

    def f(B, x):
        return B[0]*x + B[1]

    odr_linear_model = odrpack.Model(f)
    slope_initial_guess =   ols_slope_estimate
    intercept_initial_guess = ols_intercept_estimate
    odr_data = odrpack.RealData(x, y, sx=xerr, sy=yerr)
    odr_fit = odrpack.ODR(odr_data, odr_linear_model, beta0=[slope_initial_guess, intercept_initial_guess])
    odr_fit_results = odr_fit.run()
    
    if (verbose):  
        print("\nODR Estimation (raw output):\n")
        odr_fit_results.pprint()
    
    odr_slope_estimate = odr_fit_results.beta[0]
#    odr_slope_err = odr_fit_results.sd_beta[0]
    odr_intercept_estimate = odr_fit_results.beta[1]
#    odr_intercept_err = odr_fit_results.sd_beta[1]

    (odr_scatter_variance_estimate, \
    total_scatter_variance, \
    xerr_scatter_terrm,\
    yerr_scatter_term)  = scatter_variance_adhoc(x,y,xerr,yerr,ols_slope_estimate,ols_intercept_estimate, verbose = False)       

    return (odr_slope_estimate,odr_intercept_estimate,odr_scatter_variance_estimate)

def linfit_t02(x, y, xerr, yerr, scatter_variance = None):
    """
    Fit a linear model with intrinsic scatter to x,y data with errors using method of Tremaine+2002
    
    The method estimates slope and intercept of a linear model with intrinsic
    scatter by optimizing an adjusted chisq statistic.  If no input value
    of inrinsic scatter is supplied, then the method estimates intrinsic 
    scatter with an iterative procedure that effectively adjusts the 
    modified chisq statistic until it is equal to 1.0.  See Tremaine, S. 
    et al. The Astrophysical Journal, 574:740–753, 2002 
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
        scatter_variance (float, optional):  default = None.  Variance of 
            the intrinsic scatter. If a value is supplied, it is a 
            constant that is incorporated into the fitting model.  If
            no value is supplied then intrinsic scatter will be 
            estimated, using the method of Tremaine+2002.
    
    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the scatter (if fit); other-
            wise the input value of the scatter variance will be returned.
    """
    if scatter_variance is None:
        scatter_variance_model_parameter = 0.0
    else:
        scatter_variance_model_parameter = scatter_variance
    
    f87_slope, f87_intercept, f87_scatter_variance = linfit_f87(x,y,xerr,yerr)
    linmodel_initial_guess = (f87_intercept,f87_slope)
    func = lambda linmodel: chisqexy(x,y,xerr,yerr,linmodel[1],linmodel[0],scatter_variance_model_parameter)    
    optimize_result = minimize(func,linmodel_initial_guess, method='Nelder-Mead')
    
    (fitexy_intercept,fitexy_slope) = optimize_result.x
    
    if scatter_variance is not None:
        fitexy_scatter_variance = scatter_variance_model_parameter
    else:
        fitexy_scatter_variance = fitexy_scatter(x,y,xerr,yerr,fitexy_slope, fitexy_intercept)
        
    return (fitexy_slope, fitexy_intercept, fitexy_scatter_variance)

def linfit_wls(x, y, xerr, yerr):
    """
    Estimate a linear model using Weighted Least Squares (WLS).  Provide ad-hoc estimate of intrinsic scatter variance.

    The standard method of WLS model estimation is done to estimate slope
    and intercept.  See Bevington, Dada Reduction and Error Analyis for the
    Physical Sciences, p.104.
    
    Intrinsic scatter is obtained from the fit residuals, y - yfit, and
    the xerr,yerr values according to:
    
    < (y-yfit)^2 > = sig_is^2 + < yerr^2 > + m^2 * < xerr^2 >

    where sig_is is the intrinsic scatter sigma and m is the slope of the 
    best fit line.  <.> is computed as unweighted mean.

    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data

    Returns:
        slope (float):  estimate of the slope
        intercept (float):  estimate of the intercept
        scatter_variance (float):  estimate of the intrinsic scatter using
            above equation or 0 if a negative value is obtained.

    """
    
    sxxe = np.sum(x**2/yerr**2)
    sxye = np.sum(x*y/yerr**2)

    sxe = np.sum(x/yerr**2)
    sye = np.sum(y/yerr**2)
    see = np.sum(1/yerr**2)
    
    determinant = see * sxxe - sxe**2
    wls_intercept_estimate = 1/determinant * (sxxe * sye - sxe*sxye)
    wls_slope_estimate = 1/determinant * (see * sxye - sxe*sye)
    (wls_scatter_variance_estimate, \
    total_scatter_variance, \
    xerr_scatter_terrm,\
    yerr_scatter_term)  = scatter_variance_adhoc(x,y,xerr,yerr,wls_slope_estimate,wls_intercept_estimate) 
    
    return (wls_slope_estimate,wls_intercept_estimate,wls_scatter_variance_estimate)


def linfit_multi(x, y, xerr, yerr):
    """Fits a data set to a linear model (with intrinsic scatter) with multiple methods.
    
    Fits the data to a linear model with the following methods: 
    BCES, F87, K07, MLE, ODR, OLS, T02, WLS.  F87, K07, MLE, T02 include 
    independent estimates of intrinsic scatter.  BCES, ODR, OLS, WLS methods 
    do not explicitly incorporate intrinsic scatter; so an ad hoc estimate
    of intrinsic scatter is computed and returned for these methods.
    
    Parameters:
        x (float array): x-values of data to be fit (independent variable)
        y (float array): y-values of data to be fit (dependenent variable)
        xerr (float array):  errors to x-value data
        yerr (float array):  errors to y-value data
    
    Returns:
        bces_slope_estimate (float):  slope from BCES method
        bces_intercept_estimate (float):  intercept from BCES method
        bces_scatter_variance_estimate (float):  scatter variance (ad hoc)
        f87_slope_estimate (float):  slope from f87 method
        f87_intercept_estimate (float):  intercept from f87 method
        f87_scatter_variance_estimate (float):  scatter variance f87 method
        k07_slope_estimate (float):  slope from k07 method
        k07_intercept_estimate (float):  intercept from k07 method
        k07_scatter_variance_estimate (float):  scatter variance k07 method
        mle_slope_estimate (float):  slope from mle method
        mle_intercept_estimate (float):  intercept from mle method
        mle_scatter_variance_estimate (float):  scatter variance mle method
        odr_slope_estimate (float):  slope from odr method
        odr_intercept_estimate (float):  intercept from odr method
        odr_scatter_variance_estimate (float):  scatter variance (ad hoc)
        ols_slope_estimate (float):  slope from ols method
        ols_intercept_estimate (float):  intercept from ols method
        ols_scatter_variance_estimate (float):  scatter variance (ad hoc)
        t02_slope_estimate (float):  slope from t02 method
        t02_intercept_estimate (float):  intercept from t02 method
        t02_scatter_variance_estimate (float):  scatter variance T02 method
        wls_slope_estimate (float):  slope from wls method
        wls_intercept_estimate (float):  intercept from wls method
        wls_scatter_variance_estimate (float):  scatter variance (ad hoc)
    """
    
    (bces_slope,\
    bces_intercept,\
    bces_scatter_variance) = linfit_bces(x,y,xerr,yerr)

    (f87_slope,\
    f87_intercept,\
    f87_scatter_variance) = linfit_f87(x,y,xerr,yerr)
    
    """
    K07 not implemented yet
    """
    """
    (k07_slope,\
    k07_intercept,\
    k07_scatter_variance) = linfit_k07(x,y,xerr,yerr)
    """
    
    k07_slope = np.NaN
    k07_intercept = np.NaN
    k07_scatter_variance = np.NaN
    
    (mle_slope,\
    mle_intercept,\
    mle_scatter_variance) = linfit_mle(x,y,xerr,yerr)

    (odr_slope,\
    odr_intercept,\
    odr_scatter_variance) = linfit_odr(x,y,xerr,yerr)

    (ols_slope,\
    ols_intercept,\
    ols_scatter_variance) = linfit_ols(x,y,xerr,yerr)

    (t02_slope,\
    t02_intercept,\
    t02_scatter_variance) = linfit_t02(x,y,xerr,yerr)

    (wls_slope,\
    wls_intercept,\
    wls_scatter_variance) = linfit_wls(x,y,xerr,yerr)

    
    return  (bces_slope,\
            bces_intercept,\
            bces_scatter_variance,\
            f87_slope,\
            f87_intercept,\
            f87_scatter_variance,\
            k07_slope,\
            k07_intercept,\
            k07_scatter_variance,\
            mle_slope,\
            mle_intercept,\
            mle_scatter_variance,\
            odr_slope,\
            odr_intercept,\
            odr_scatter_variance,\
            ols_slope,\
            ols_intercept,\
            ols_scatter_variance,\
            t02_slope,\
            t02_intercept,\
            t02_scatter_variance,\
            wls_slope,\
            wls_intercept,\
            wls_scatter_variance)


        
def linfit_multi_simerr(x,\
                        xerr,\
                        yerr,\
                        slope,\
                        intercept,\
                        scatter_variance,\
                        iterations=1000,\
                        plot_realization=None,\
                        plot_results=None,\
                        write_tables=None,\
                        xerr_type = None,\
                        yerr_type = None,\
                        verbose=None):
    """
    Compute errors to linear model + intrinsic scatter estimated quantities by simulation.
        
    Fitting methods include BCES, F87, K07, MLE, ODR, OLS, T02, WLS.  Some of
    these methods may not be implemented, and will return placeholder values.
    
    Parameters:
    x               array of x-values of data to be fit.
    xerr            arry of errors to x-values of data to be fit.
    yerr            array of errors to y-values of data to be fit.
    slope           slope of the linear model
    intercept       intercept of the linear model
    scatter         sigma value of the intrinsic scatter
    iterations      number of iterations to perform in simulation
    plot_realization    set to generate a plot of the data from one realization
    plot_results    set to generate a plot of results
    write_table     set to generate text file output of results
    xerr_type       describes how the errors will be treated in simulation
    yerr_type        
    
    Acceptable values of xerr_type, yerr_type:    
        'None'              errors are set to the input values with no randomization
        'median'            errors set to the median of the input array (homoscedastic)
        'infinitesimal'     errors are set to an infinitesimal value (homoscedastic)
        'replacement'       errors are drawn from input array using sample with replacement (heteroscedastic)
        'normal'            errors are drawn from normal distribution scaled by input array (heteroscedastic)
    
    Returns:
    bias and errors determined from simulation for each estimated quantity.
    """
    #args_input_file = "None"
    args_output_prefix = "linfit_multisimerr"
    args_iterations= iterations
    args_scatter_mean = 0.0
    args_scatter_variance = scatter_variance
    args_slope_true = slope
    args_intercept_true = intercept

    if xerr_type is None: args_xerr_type = 'normal'
    else: args_xerr_type = xerr_type
    
    if yerr_type is None: args_yerr_type = 'normal'
    else:  args_yerr_type= yerr_type

    if plot_realization is not None:
        realization(x,\
            xerr,\
            yerr,\
            args_slope_true,\
            args_intercept_true,\
            args_scatter_mean,\
            args_scatter_variance,\
            xerr_type = args_xerr_type,\
            yerr_type = args_yerr_type, \
            figure =plot_realization)
    

    """
    Do Simulation: Perform a large number of random realizations
    """
    start_time = timeit.default_timer()
    
    (error_bces_slope,
    error_bces_intercept,
    error_bces_scatter_variance,
    error_f87_slope,
    error_f87_intercept,
    error_f87_scatter_variance,
    error_k07_slope,
    error_k07_intercept,
    error_k07_scatter_variance,
    error_mle_slope,
    error_mle_intercept,
    error_mle_scatter_variance,
    error_odr_slope,
    error_odr_intercept,
    error_odr_scatter_variance,
    error_ols_slope,
    error_ols_intercept,
    error_ols_scatter_variance,
    error_t02_slope,
    error_t02_intercept,
    error_t02_scatter_variance,
    error_wls_slope,
    error_wls_intercept,
    error_wls_scatter_variance) = zip(*(realization(x,\
                            xerr,\
                            yerr,\
                            args_slope_true,\
                            args_intercept_true,\
                            args_scatter_mean,\
                            args_scatter_variance,\
                            xerr_type = args_xerr_type,\
                            yerr_type = args_yerr_type \
                            ) for _ in range(iterations)))
    
    elapsed = timeit.default_timer() - start_time

    error_bces_slope_distrib_mean = np.mean(np.asarray(error_bces_slope))
    error_bces_slope_distrib_stddev = np.std(np.asarray(error_bces_slope))
    error_f87_slope_distrib_mean = np.mean(np.asarray(error_f87_slope))
    error_f87_slope_distrib_stddev = np.std(np.asarray(error_f87_slope))
    error_k07_slope_distrib_mean = np.mean(np.asarray(error_k07_slope))
    error_k07_slope_distrib_stddev = np.std(np.asarray(error_k07_slope))
    error_mle_slope_distrib_mean = np.mean(np.asarray(error_mle_slope))
    error_mle_slope_distrib_stddev = np.std(np.asarray(error_mle_slope))
    error_ols_slope_distrib_mean = np.mean(np.asarray(error_ols_slope))
    error_ols_slope_distrib_stddev = np.std(np.asarray(error_ols_slope))
    error_odr_slope_distrib_mean = np.mean(np.asarray(error_odr_slope))
    error_odr_slope_distrib_stddev = np.std(np.asarray(error_odr_slope))
    error_t02_slope_distrib_mean = np.mean(np.asarray(error_t02_slope))
    error_t02_slope_distrib_stddev = np.std(np.asarray(error_t02_slope))
    error_wls_slope_distrib_mean = np.mean(np.asarray(error_wls_slope))
    error_wls_slope_distrib_stddev = np.std(np.asarray(error_wls_slope))
    
    error_bces_intercept_distrib_mean = np.mean(np.asarray(error_bces_intercept))
    error_bces_intercept_distrib_stddev = np.std(np.asarray(error_bces_intercept))
    error_f87_intercept_distrib_mean = np.mean(np.asarray(error_f87_intercept))
    error_f87_intercept_distrib_stddev = np.std(np.asarray(error_f87_intercept))
    error_k07_intercept_distrib_mean = np.mean(np.asarray(error_k07_intercept))
    error_k07_intercept_distrib_stddev = np.std(np.asarray(error_k07_intercept))
    error_mle_intercept_distrib_mean = np.mean(np.asarray(error_mle_intercept))
    error_mle_intercept_distrib_stddev = np.std(np.asarray(error_mle_intercept))
    error_ols_intercept_distrib_mean = np.mean(np.asarray(error_ols_intercept))
    error_ols_intercept_distrib_stddev = np.std(np.asarray(error_ols_intercept))
    error_odr_intercept_distrib_mean = np.mean(np.asarray(error_odr_intercept))
    error_odr_intercept_distrib_stddev = np.std(np.asarray(error_odr_intercept))
    error_t02_intercept_distrib_mean = np.mean(np.asarray(error_t02_intercept))
    error_t02_intercept_distrib_stddev = np.std(np.asarray(error_t02_intercept))
    error_wls_intercept_distrib_mean = np.mean(np.asarray(error_wls_intercept))
    error_wls_intercept_distrib_stddev = np.std(np.asarray(error_wls_intercept))
    
    error_bces_scatter_variance_distrib_mean = np.mean(np.asarray(error_bces_scatter_variance))
    error_bces_scatter_variance_distrib_stddev = np.std(np.asarray(error_bces_scatter_variance))
    error_f87_scatter_variance_distrib_mean = np.mean(np.asarray(error_f87_scatter_variance))
    error_f87_scatter_variance_distrib_stddev = np.std(np.asarray(error_f87_scatter_variance))
    error_k07_scatter_variance_distrib_mean = np.mean(np.asarray(error_k07_scatter_variance))
    error_k07_scatter_variance_distrib_stddev = np.std(np.asarray(error_k07_scatter_variance))
    error_mle_scatter_variance_distrib_mean = np.mean(np.asarray(error_mle_scatter_variance))
    error_mle_scatter_variance_distrib_stddev = np.std(np.asarray(error_mle_scatter_variance))
    error_ols_scatter_variance_distrib_mean = np.mean(np.asarray(error_ols_scatter_variance))
    error_ols_scatter_variance_distrib_stddev = np.std(np.asarray(error_ols_scatter_variance))
    error_odr_scatter_variance_distrib_mean = np.mean(np.asarray(error_odr_scatter_variance))
    error_odr_scatter_variance_distrib_stddev = np.std(np.asarray(error_odr_scatter_variance))
    error_t02_scatter_variance_distrib_mean = np.mean(np.asarray(error_t02_scatter_variance))
    error_t02_scatter_variance_distrib_stddev = np.std(np.asarray(error_t02_scatter_variance))
    error_wls_scatter_variance_distrib_mean = np.mean(np.asarray(error_wls_scatter_variance))
    error_wls_scatter_variance_distrib_stddev = np.std(np.asarray(error_wls_scatter_variance))
    
    slope_table_data = (error_bces_slope_distrib_mean, 	
                        error_bces_slope_distrib_stddev,
                        error_f87_slope_distrib_mean, 	
                        error_f87_slope_distrib_stddev, 
                        error_k07_slope_distrib_mean, 	
                        error_k07_slope_distrib_stddev, 
                        error_mle_slope_distrib_mean, 	
                        error_mle_slope_distrib_stddev, 
                        error_ols_slope_distrib_mean, 	
                        error_ols_slope_distrib_stddev, 
                        error_odr_slope_distrib_mean, 	
                        error_odr_slope_distrib_stddev, 
                        error_t02_slope_distrib_mean, 	
                        error_t02_slope_distrib_stddev, 
                        error_wls_slope_distrib_mean, 	
                        error_wls_slope_distrib_stddev)

    intercept_table_data = (error_bces_intercept_distrib_mean, 	
                            error_bces_intercept_distrib_stddev, 
                            error_f87_intercept_distrib_mean, 	
                            error_f87_intercept_distrib_stddev, 
                            error_k07_intercept_distrib_mean, 	
                            error_k07_intercept_distrib_stddev, 
                            error_mle_intercept_distrib_mean, 	
                            error_mle_intercept_distrib_stddev, 
                            error_ols_intercept_distrib_mean, 	
                            error_ols_intercept_distrib_stddev, 
                            error_odr_intercept_distrib_mean, 	
                            error_odr_intercept_distrib_stddev, 
                            error_t02_intercept_distrib_mean, 	
                            error_t02_intercept_distrib_stddev, 
                            error_wls_intercept_distrib_mean, 	
                            error_wls_intercept_distrib_stddev)

    scatter_variance_table_data = (error_bces_scatter_variance_distrib_mean, 	
                                error_bces_scatter_variance_distrib_stddev, 
                                error_f87_scatter_variance_distrib_mean, 	
                                error_f87_scatter_variance_distrib_stddev, 
                                error_k07_scatter_variance_distrib_mean, 	
                                error_k07_scatter_variance_distrib_stddev, 
                                error_mle_scatter_variance_distrib_mean, 	
                                error_mle_scatter_variance_distrib_stddev, 
                                error_ols_scatter_variance_distrib_mean, 	
                                error_ols_scatter_variance_distrib_stddev, 
                                error_odr_scatter_variance_distrib_mean, 	
                                error_odr_scatter_variance_distrib_stddev, 
                                error_t02_scatter_variance_distrib_mean, 	
                                error_t02_scatter_variance_distrib_stddev, 
                                error_wls_scatter_variance_distrib_mean, 	
                                error_wls_scatter_variance_distrib_stddev)

    def format3(in_float):
        print("{:.3f}\t".format(in_float),end="")

    if verbose is not None:  

        print("\nSimulation Results:")
        print("\tElapsed time (seconds)       : {:.3f}".format(elapsed))
    
        print("\tFractional Error in Intr. Scatter (sigma) Estimate")
        print("\t                        F87  : {:.3f}".format(error_f87_scatter_variance_distrib_stddev))
        print("\t                        MLE  : {:.3f}".format(error_mle_scatter_variance_distrib_stddev))
        print("\t                        OLS  : {:.3f}".format(error_ols_scatter_variance_distrib_stddev))
        print("\t                        T02  : {:.3f}".format(error_t02_scatter_variance_distrib_stddev))
    
        print("\tBias in Intr. Scatter (sigma) Estimate")
        print("\t                        F87  : {:.3f}".format(error_f87_scatter_variance_distrib_mean))
        print("\t                        MLE  : {:.3f}".format(error_mle_scatter_variance_distrib_mean))
        print("\t                        OLS  : {:.3f}".format(error_ols_scatter_variance_distrib_mean))
        print("\t                        T02  : {:.3f}".format(error_t02_scatter_variance_distrib_mean))
   
    """
    print("\n")
    print("\tFractional Error Distributions")

    print("Fractional Error Distribution of Intrinsic Scatter Variance Estimates")
    print("\t\t\tScatter")
    print("\t\tbias\tscat")
    print(args.output_prefix,end="\t")    
    [format3(ss) for ss in scatter_variance_table_data]
    print("\n")
    
    print("Fractional Error Distributions of Slope Estimates")
    print("\t\t\tF87\t\t\t\tODR\t\t\t\tOLS\t\t\t\tWLS")
    print("\t\tbias\tscat\tbias\tscat\tbias\tscat\tbias\tscat")
    print(args.output_prefix,end="\t")    
    [format3(ss) for ss in slope_table_data]
    print("\n")
 
    print("Fractional Error Distributions of Intercept Estimates")
    print("\t\t\tF87\t\t\t\tODR\t\t\t\tOLS\t\t\t\tWLS")
    print("\t\tbias\tscat\tbias\tscat\tbias\tscat\tbias\tscat")
    print(args.output_prefix,end="\t")    
    [format3(ss) for ss in intercept_table_data]
    print("\n")
    """
    """
    OUPUT Table files (for use when running a set of simulations)
    """
    if write_tables is not None:
        with open("table_input_values.txt",'a') as table_args_file, \
             open("table_scatter.txt",'a') as table_scatter_file, \
             open("table_slope.txt",'a') as table_slope_file, \
             open("table_intercept.txt",'a') as table_intercept_file:

            #table_args_file.write(str(vars(args)))            
            input_values_str = str(args_output_prefix) + '\t' + \
                               str(args_iterations) + '\t' + \
                               str(args_scatter_variance) + '\t' + \
                               str(args_xerr_type) + '\t' + \
                               str(args_yerr_type) + '\n'           
            table_args_file.write(input_values_str)
            table_scatter_file.write(args_output_prefix + '\t' + '\t'.join(format(f, '.3f') for f in scatter_variance_table_data)+'\n')
            table_slope_file.write(args_output_prefix + '\t' + '\t'.join(format(f, '.3f') for f in slope_table_data)+'\n')
            table_intercept_file.write(args_output_prefix + '\t' + '\t'.join(format(f, '.3f') for f in intercept_table_data)+'\n')


    if plot_results is not None:

        output_fig1 = "linfit_multisimerr_scatter.pdf"
        output_fig2 = "linfit_multisimerr_slopes.pdf"
        output_fig3 = "linfit_multisimerr_intercepts.pdf"

        """
        OUTPUT FIGURE 1
        
        2x2 panel of errors to scatter_variance estimates from various methods.
        """    
        # figsize in inches
        fig1 = plt.figure(dpi=300, figsize=[8,8])
        figure_title = 'Simulation with ' + "{:,}".format(args_iterations) + ' Realizations'
        fig1.suptitle(figure_title,fontweight="bold", fontsize=14)
        fig1.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
    
        """
        Simulation Results:  Estimated Scatter Sigma (F87)
        """ 
        axis1 = fig1.add_subplot(221)
        axis1.set_title(r"F87 Estimated Scatter Sigma", fontweight="bold")
        axis1.set_xlabel(r"Fractional Error", fontsize = 12)
        axis1.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis1.hist(error_f87_scatter_variance, 100, normed=1, facecolor='indigo', alpha=0.75)
        y = mlab.normpdf( bins, error_f87_scatter_variance_distrib_mean, error_f87_scatter_variance_distrib_stddev)
        axis1.plot(bins, y, 'r', linewidth=1)
        axis1.axis([error_f87_scatter_variance_distrib_mean-4*error_f87_scatter_variance_distrib_stddev, error_f87_scatter_variance_distrib_mean+4*error_f87_scatter_variance_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Scatter Sigma (MLE)
        """
        axis2 = fig1.add_subplot(222)
        axis2.set_title(r"MLE Estimated Scatter Sigma", fontweight="bold")
        axis2.set_xlabel(r"Fractional Error", fontsize = 12)
        axis2.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis2.hist(error_mle_scatter_variance, 100, normed=1, facecolor='slateblue', alpha=0.75)
        y = mlab.normpdf( bins, error_mle_scatter_variance_distrib_mean, error_mle_scatter_variance_distrib_stddev)
        axis2.plot(bins, y, 'r', linewidth=1)
        axis2.axis([error_mle_scatter_variance_distrib_mean-4*error_mle_scatter_variance_distrib_stddev, error_mle_scatter_variance_distrib_mean+4*error_mle_scatter_variance_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Scatter Sigma (OLS)
        """
        axis3 = fig1.add_subplot(223)
        axis3.set_title(r"OLS Estimated Scatter Sigma", fontweight="bold")
        axis3.set_xlabel(r"Fractional Error", fontsize = 12)
        axis3.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis3.hist(error_ols_scatter_variance, 50, normed=1, facecolor='deepskyblue', alpha=0.75)
        y = mlab.normpdf( bins, error_ols_scatter_variance_distrib_mean, error_ols_scatter_variance_distrib_stddev)
        axis3.plot(bins, y, 'r', linewidth=1)
        axis3.axis([error_ols_scatter_variance_distrib_mean-4*error_ols_scatter_variance_distrib_stddev, error_ols_scatter_variance_distrib_mean+4*error_ols_scatter_variance_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Scatter Sigma (T02)
        """
        axis4 = fig1.add_subplot(224)
        axis4.set_title(r"T02 Estimated Scatter Sigma", fontweight="bold")
        axis4.set_xlabel(r"Fractional Error", fontsize = 12)
        axis4.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis4.hist(error_t02_scatter_variance, 100, normed=1, facecolor='darkseagreen', alpha=0.75)
        y = mlab.normpdf( bins, error_t02_scatter_variance_distrib_mean, error_t02_scatter_variance_distrib_stddev)
        axis4.plot(bins, y, 'r', linewidth=1)
        axis4.axis([error_t02_scatter_variance_distrib_mean-4*error_t02_scatter_variance_distrib_stddev, error_t02_scatter_variance_distrib_mean+4*error_t02_scatter_variance_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        fig1.savefig(output_fig1, format='pdf')
    
    
        """
        OUTPUT FIGURE 2
        
        2x2 panel of errors to slope estimates from various methods.
        """    
        # figsize in inches
        fig2 = plt.figure(dpi=300, figsize=[8,8])
        figure_title = 'Simulation with ' + "{:,}".format(args_iterations) + ' Realizations'
        fig2.suptitle(figure_title,fontweight="bold", fontsize=14)
        fig2.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
    
        """
        Simulation Results:  Estimated Slope (F87)
        """ 
        axis1 = fig2.add_subplot(221)
        axis1.set_title(r"F87 Estimated Slope", fontweight="bold")
        axis1.set_xlabel(r"Fractional Error", fontsize = 12)
        axis1.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis1.hist(error_f87_slope, 100, normed=1, facecolor='indigo', alpha=0.75)
        y = mlab.normpdf( bins, error_f87_slope_distrib_mean, error_f87_slope_distrib_stddev)
        axis1.plot(bins, y, 'r', linewidth=1)
        axis1.axis([error_f87_slope_distrib_mean-4*error_f87_slope_distrib_stddev, error_f87_slope_distrib_mean+4*error_f87_slope_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Slope (bces)
        """
        axis2 = fig2.add_subplot(222)
        axis2.set_title(r"BCES Estimated Slope", fontweight="bold")
        axis2.set_xlabel(r"Fractional Error", fontsize = 12)
        axis2.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis2.hist(error_bces_slope, 100, normed=1, facecolor='slateblue', alpha=0.75)
        y = mlab.normpdf( bins, error_bces_slope_distrib_mean, error_bces_slope_distrib_stddev)
        axis2.plot(bins, y, 'r', linewidth=1)
        axis2.axis([error_bces_slope_distrib_mean-4*error_bces_slope_distrib_stddev, error_bces_slope_distrib_mean+4*error_bces_slope_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Slope (OLS)
        """
        axis3 = fig2.add_subplot(223)
        axis3.set_title(r"OLS Estimated Slope", fontweight="bold")
        axis3.set_xlabel(r"Fractional Error", fontsize = 12)
        axis3.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis3.hist(error_ols_slope, 50, normed=1, facecolor='deepskyblue', alpha=0.75)
        y = mlab.normpdf( bins, error_ols_slope_distrib_mean, error_ols_slope_distrib_stddev)
        axis3.plot(bins, y, 'r', linewidth=1)
        axis3.axis([error_ols_slope_distrib_mean-4*error_ols_slope_distrib_stddev, error_ols_slope_distrib_mean+4*error_ols_slope_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Slope (T02)
        """
        axis4 = fig2.add_subplot(224)
        axis4.set_title(r"T02 Estimated Slope", fontweight="bold")
        axis4.set_xlabel(r"Fractional Error", fontsize = 12)
        axis4.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis4.hist(error_t02_slope, 100, normed=1, facecolor='darkseagreen', alpha=0.75)
        y = mlab.normpdf( bins, error_t02_slope_distrib_mean, error_t02_slope_distrib_stddev)
        axis4.plot(bins, y, 'r', linewidth=1)
        axis4.axis([error_t02_slope_distrib_mean-4*error_t02_slope_distrib_stddev, error_t02_slope_distrib_mean+4*error_t02_slope_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        fig2.savefig(output_fig2, format='pdf')
    
    
        """
        OUTPUT FIGURE 3
        
        2x2 panel of errors to intercept estimates from various methods.
        """    
        # figsize in inches
        fig3 = plt.figure(dpi=300, figsize=[8,8])
        figure_title = 'Simulation with ' + "{:,}".format(args_iterations) + ' Realizations'
        fig3.suptitle(figure_title,fontweight="bold", fontsize=14)
        fig3.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
    
        """
        Simulation Results:  Estimated Intercept (F87)
        """ 
        axis1 = fig3.add_subplot(221)
        axis1.set_title(r"F87 Estimated Intercept", fontweight="bold")
        axis1.set_xlabel(r"Fractional Error", fontsize = 12)
        axis1.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis1.hist(error_f87_intercept, 100, normed=1, facecolor='indigo', alpha=0.75)
        y = mlab.normpdf( bins, error_f87_intercept_distrib_mean, error_f87_intercept_distrib_stddev)
        axis1.plot(bins, y, 'r', linewidth=1)
        axis1.axis([error_f87_intercept_distrib_mean-4*error_f87_intercept_distrib_stddev, error_f87_intercept_distrib_mean+4*error_f87_intercept_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Intercept (bces)
        """
        axis2 = fig3.add_subplot(222)
        axis2.set_title(r"BCES Estimated Intercept", fontweight="bold")
        axis2.set_xlabel(r"Fractional Error", fontsize = 12)
        axis2.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis2.hist(error_bces_intercept, 100, normed=1, facecolor='slateblue', alpha=0.75)
        y = mlab.normpdf( bins, error_bces_intercept_distrib_mean, error_bces_intercept_distrib_stddev)
        axis2.plot(bins, y, 'r', linewidth=1)
        axis2.axis([error_bces_intercept_distrib_mean-4*error_bces_intercept_distrib_stddev, error_bces_intercept_distrib_mean+4*error_bces_intercept_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Intercept (OLS)
        """
        axis3 = fig3.add_subplot(223)
        axis3.set_title(r"OLS Estimated Intercept", fontweight="bold")
        axis3.set_xlabel(r"Fractional Error", fontsize = 12)
        axis3.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis3.hist(error_ols_intercept, 50, normed=1, facecolor='deepskyblue', alpha=0.75)
        y = mlab.normpdf( bins, error_ols_intercept_distrib_mean, error_ols_intercept_distrib_stddev)
        axis3.plot(bins, y, 'r', linewidth=1)
        axis3.axis([error_ols_intercept_distrib_mean-4*error_ols_intercept_distrib_stddev, error_ols_intercept_distrib_mean+4*error_ols_intercept_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        """
        Simulation Results:  Estimated Intercept (T02)
        """
        axis4 = fig3.add_subplot(224)
        axis4.set_title(r"T02 Estimated Intercept", fontweight="bold")
        axis4.set_xlabel(r"Fractional Error", fontsize = 12)
        axis4.set_ylabel(r"Normalized Histogram", fontsize = 12)
        n, bins, patches = axis4.hist(error_t02_intercept, 100, normed=1, facecolor='darkseagreen', alpha=0.75)
        y = mlab.normpdf( bins, error_t02_intercept_distrib_mean, error_t02_intercept_distrib_stddev)
        axis4.plot(bins, y, 'r', linewidth=1)
        axis4.axis([error_t02_intercept_distrib_mean-4*error_t02_intercept_distrib_stddev, error_t02_intercept_distrib_mean+4*error_t02_intercept_distrib_stddev, 0, n.max() + 0.10* n.max()])
    
        fig3.savefig(output_fig3, format='pdf')


    return(error_bces_slope_distrib_mean,
            error_bces_slope_distrib_stddev,
            error_f87_slope_distrib_mean,
            error_f87_slope_distrib_stddev,
            error_k07_slope_distrib_mean,
            error_k07_slope_distrib_stddev,
            error_mle_slope_distrib_mean,
            error_mle_slope_distrib_stddev,
            error_odr_slope_distrib_mean,
            error_odr_slope_distrib_stddev,
            error_ols_slope_distrib_mean,
            error_ols_slope_distrib_stddev,
            error_t02_slope_distrib_mean,
            error_t02_slope_distrib_stddev,
            error_wls_slope_distrib_mean,
            error_wls_slope_distrib_stddev,
            error_bces_intercept_distrib_mean,
            error_bces_intercept_distrib_stddev,
            error_f87_intercept_distrib_mean,
            error_f87_intercept_distrib_stddev,
            error_k07_intercept_distrib_mean,
            error_k07_intercept_distrib_stddev,
            error_mle_intercept_distrib_mean,
            error_mle_intercept_distrib_stddev,
            error_odr_intercept_distrib_mean,
            error_odr_intercept_distrib_stddev,
            error_ols_intercept_distrib_mean,
            error_ols_intercept_distrib_stddev,
            error_odr_intercept_distrib_mean,
            error_odr_intercept_distrib_stddev,
            error_t02_intercept_distrib_mean,
            error_t02_intercept_distrib_stddev,
            error_wls_intercept_distrib_mean,
            error_wls_intercept_distrib_stddev,
            error_bces_scatter_variance_distrib_mean,
            error_bces_scatter_variance_distrib_stddev,
            error_f87_scatter_variance_distrib_mean,
            error_f87_scatter_variance_distrib_stddev,
            error_k07_scatter_variance_distrib_mean,
            error_k07_scatter_variance_distrib_stddev,
            error_mle_scatter_variance_distrib_mean,
            error_mle_scatter_variance_distrib_stddev,
            error_odr_scatter_variance_distrib_mean,
            error_odr_scatter_variance_distrib_stddev,
            error_ols_scatter_variance_distrib_mean,
            error_ols_scatter_variance_distrib_stddev,
            error_odr_scatter_variance_distrib_mean,
            error_odr_scatter_variance_distrib_stddev,
            error_t02_scatter_variance_distrib_mean,
            error_t02_scatter_variance_distrib_stddev,
            error_wls_scatter_variance_distrib_mean,
            error_wls_scatter_variance_distrib_stddev)
            
def realization(in_x,\
                in_x_err,\
                in_y_err,\
                slope_true=0.46,\
                intercept_true=-4.09,\
                scatter_mean_true = 0.0,\
                scatter_sigma_true = 1.0,\
                xerr_type= None,\
                yerr_type= None,\
                figure = None,\
                txt = None):
    """
    Perform one realization of a simulation of data with a linear relationship 
    and intrinic scatter that are fit with a variety of fitting methods.
        
    Parameters:
    in_x            array of x-values of data to be fit.
    in_x_err        arry of errors to x-values of data to be fit.
    in_y_err        array of errors to y-values of data to be fit.
    slope_true      true value of the slope of the linear model
    intercept_true  true value of the intercept of the linear model
    scatter_mean    true value of the mean of the intrinic scatter distribution
    scatter_sigma   true value of the sigma (width) of the intrinsic scatter
    xerr_type       describes how the errors will be treated in simulation:
    yerr_type           see Different error types below
    figure          Generate a 2x2 panel figure showing the simulated data and fits
    txt             Save the realized data set as a text file with this name
    
        Different error types:    
        'None'              errors are set to the input values with no randomization
        'median'            errors set to the median of the input array (homoscedastic)
        'infinitesimal'     errors are set to an infinitesimal value (homoscedastic)
        'replacement'       errors are drawn from input array using sample with replacement (heteroscedastic)
        'normal'            errors are drawn from normal distribution scaled by input array (heteroscedastic)

    Returns:
    Fractional errors for each estimated quantity.
    """

    ndata = len(in_x)
            
    x_true = in_x
    y_true = intercept_true + slope_true * x_true
        
    """
    observed measurement errors
    """
    x_obs_err = np.empty(ndata)
    y_obs_err = np.empty(ndata)
  
    """
    simulated errors that will be added to noisify the model data
    """
    x_sim_err = np.empty(ndata)
    y_sim_err = np.empty(ndata)
    
    """
    Select method of handling errors
    """
    def sample_wr(population, k):
        """
        Chooses k random elements (with replacement) from a population.        
        http://code.activestate.com/recipes/273085-sample-with-replacement/
        """
        n = len(population)
        _random, _int = random.random, int  # speed hack 
        result = [None] * k
        for i in range(k):
            j = _int(_random() * n)
            result[i] = population[j]
        return result  
            
    x_obs_err = in_x_err
    y_obs_err = in_y_err
    if xerr_type == None:
        x_obs_err= in_x_err
        x_sim_err = 0
    elif xerr_type == 'median':
        x_obs_err[...] = np.median(in_x_err)
        x_sim_err[...] = np.median(in_x_err) * np.random.normal(0,1,ndata)
    elif xerr_type == 'infinitesimal':
        x_obs_err[...] = 0.00001
        x_sim_err = x_obs_err
    elif xerr_type == 'replacement':
        rnd_sgn = np.random.random_integers(0,1,size=ndata)
        rnd_sgn[rnd_sgn == 0] = -1
        x_sim_err = rnd_sgn * np.asarray(sample_wr(in_x_err,ndata))
        x_obs_err[...] = np.fabs(x_sim_err)

    elif xerr_type == 'normal':
        x_obs_err= in_x_err
        x_sim_err = np.random.normal(0,1,ndata)*in_x_err
    else: 
        x_obs_err= in_x_err
        x_sim_err = np.random.normal(0,in_x_err,ndata)

    if yerr_type == None:
        y_obs_err= in_y_err
        y_sim_err[...] = 0
    elif yerr_type == 'median':
        y_obs_err[...] = np.median(in_y_err)
        y_sim_err[...] = np.median(in_y_err) * np.random.normal(0,1,ndata)
    elif yerr_type == 'infinitesimal':
        y_obs_err[...] = 0.00001
        y_sim_err = y_obs_err
    elif yerr_type == 'replacement':
        rnd_sgn = np.random.random_integers(0,1,size=ndata)
        rnd_sgn[rnd_sgn == 0] = -1
        y_sim_err = rnd_sgn * np.asarray(sample_wr(in_y_err,ndata))
        y_obs_err= np.fabs(y_sim_err)
    elif yerr_type == 'normal':
        y_obs_err= in_y_err
        y_sim_err = np.random.normal(0,1,ndata)*in_y_err
    else: 
        y_obs_err= in_y_err
        y_sim_err = np.random.normal(0,in_y_err,ndata)
    
    
    scatter = scatter_sigma_true * np.random.normal(scatter_mean_true, 1.0, ndata)
    
    
    x_obs_err_mean = np.average(x_obs_err)
    y_obs_err_mean = np.average(y_obs_err)
    x_obs_err_sigma = np.std(x_obs_err)
    y_obs_err_sigma = np.std(y_obs_err)
    
    
    """
    generate noisified, simulated observed data from 
    linear model, noisified error and intrinsic scatter
    """
    x_obs = x_true + x_sim_err
    y_obs = y_true + y_sim_err + scatter

    """
    fit noisified data to a linear model with intrinsic scatter 
    using a variety of methods
    """
    (bces_slope,\
    bces_intercept,\
    bces_scatter_variance,\
    f87_slope,\
    f87_intercept,\
    f87_scatter_variance,\
    k07_slope,\
    k07_intercept,\
    k07_scatter_variance,\
    mle_slope,\
    mle_intercept,\
    mle_scatter_variance, \
    odr_slope,\
    odr_intercept,\
    odr_scatter_variance,\
    ols_slope,\
    ols_intercept,\
    ols_scatter_variance,\
    t02_slope,\
    t02_intercept,\
    t02_scatter_variance, \
    wls_slope,\
    wls_intercept,\
    wls_scatter_variance) = linfit_multi(x_obs,\
                            y_obs,\
                            x_obs_err,\
                            y_obs_err)

    if txt is not None:
        """
        Save realization data as text file
        """
        #print("\nSaving realization data to file: ", txt)
        #print("\tColumns:  x, xerr, y, yerr")
        np.savetxt(txt, np.column_stack((x_obs,x_obs_err,y_obs,y_obs_err)),fmt=('%5.6f','%5.6f','%5.6f','%5.6f'))

        
    """
    OUTPUT OF SIMULATION
    
    Absolute errors of estimated quantities
    """
    scatter_variance_true = scatter_sigma_true**2
    
    error_bces_slope = (bces_slope - slope_true)
    error_bces_intercept = (bces_intercept - intercept_true)
    error_bces_scatter_variance = (bces_scatter_variance - scatter_variance_true)
    error_f87_slope = (f87_slope - slope_true)
    error_f87_intercept = (f87_intercept - intercept_true)
    error_f87_scatter_variance = (f87_scatter_variance - scatter_variance_true)
    error_k07_slope = (k07_slope - slope_true)
    error_k07_intercept = (k07_intercept - intercept_true)
    error_k07_scatter_variance = (k07_scatter_variance - scatter_variance_true)
    error_mle_slope = (mle_slope - slope_true)
    error_mle_intercept = (mle_intercept - intercept_true)
    error_mle_scatter_variance = (mle_scatter_variance - scatter_variance_true)
    error_odr_slope = (odr_slope - slope_true)
    error_odr_intercept = (odr_intercept - intercept_true)
    error_odr_scatter_variance = (odr_scatter_variance - scatter_variance_true)
    error_ols_slope = (ols_slope - slope_true)
    error_ols_intercept = (ols_intercept - intercept_true)
    error_ols_scatter_variance = (ols_scatter_variance - scatter_variance_true)
    error_t02_slope = (t02_slope - slope_true)
    error_t02_intercept = (t02_intercept - intercept_true)
    error_t02_scatter_variance = (t02_scatter_variance - scatter_variance_true)
    error_wls_slope = (wls_slope - slope_true)
    error_wls_intercept = (wls_intercept - intercept_true)
    error_wls_scatter_variance = (wls_scatter_variance - scatter_variance_true)

    """
    Fractional errors of estimated quantities, incl scatter_sigma
        
    bces_scatter_sigma = np.sqrt(bces_scatter_variance)
    f87_scatter_sigma = np.sqrt(f87_scatter_variance)
    k07_scatter_sigma = np.sqrt(k07_scatter_variance)
    mle_scatter_sigma = np.sqrt(mle_scatter_variance)
    odr_scatter_sigma = np.sqrt(odr_scatter_variance)    
    ols_scatter_sigma = np.sqrt(ols_scatter_variance)    
    t02_scatter_sigma = np.sqrt(t02_scatter_variance)    
    wls_scatter_sigma = np.sqrt(wls_scatter_variance)

    frac_error_bces_slope = (bces_slope - slope_true) / slope_true
    frac_error_bces_intercept = (bces_intercept - intercept_true) / intercept_true
    frac_error_bces_scatter_sigma = (bces_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_f87_slope = (f87_slope - slope_true) / slope_true
    frac_error_f87_intercept = (f87_intercept - intercept_true) / intercept_true
    frac_error_f87_scatter_sigma = (f87_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_k07_slope = (k07_slope - slope_true) / slope_true
    frac_error_k07_intercept = (k07_intercept - intercept_true) / intercept_true
    frac_error_k07_scatter_sigma = (k07_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_mle_slope = (mle_slope - slope_true) / slope_true
    frac_error_mle_intercept = (mle_intercept - intercept_true) / intercept_true
    frac_error_mle_scatter_sigma = (mle_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_odr_slope = (odr_slope - slope_true) / slope_true
    frac_error_odr_intercept = (odr_intercept - intercept_true) / intercept_true
    frac_error_odr_scatter_sigma = (odr_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_ols_slope = (ols_slope - slope_true) / slope_true
    frac_error_ols_intercept = (ols_intercept - intercept_true) / intercept_true
    frac_error_ols_scatter_sigma = (ols_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_t02_slope = (t02_slope - slope_true) / slope_true
    frac_error_t02_intercept = (t02_intercept - intercept_true) / intercept_true
    frac_error_t02_scatter_sigma = (t02_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    frac_error_wls_slope = (wls_slope - slope_true) / slope_true
    frac_error_wls_intercept = (wls_intercept - intercept_true) / intercept_true
    frac_error_wls_scatter_sigma = (wls_scatter_sigma - scatter_sigma_true) / scatter_sigma_true
    
    """


    if figure is None:   
        """ 
        No output figure generated
        """
    else:
        """
        Generate output figure:
        2x2 panel showing simulated data and fits, histograms of error values 
        that were added to x,y data and histogram of intrinsic scatter values
        """
        # figsize in inches
        fig = plt.figure(dpi=300, figsize=[8,8])
        fig.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
        
        """
        Simulated data, model and estimates of model parameters
        """
        axis1 = fig.add_subplot(221)
        axis1.set_title(r'One Realization of Simulated Data', fontweight="bold")
        axis1.set_xlabel(r"Simulated x-data", fontsize = 12)
        axis1.set_ylabel(r"Simulated y-data", fontsize = 12)    
        axis1.errorbar(x_obs, \
                     y_obs, \
                     xerr=x_obs_err, \
                     yerr=y_obs_err, \
                     linestyle='None', \
                     color = 'gray', \
                     capsize = 0, \
                     marker='s', \
                     markerfacecolor='gray', \
                     markeredgecolor='gray', \
                     markersize=2, \
                     markeredgewidth=0.5)

        x_model = x_obs
        y_true_model = intercept_true + slope_true * x_model
        y_bces_model = bces_intercept + bces_slope * x_model
        y_F87_model = f87_intercept + f87_slope * x_model
        y_OLS_model = ols_intercept + ols_slope * x_model
        y_mle_model = mle_intercept + mle_slope * x_model
        y_T02_model = t02_intercept + t02_slope * x_model
        y_WLS_model = wls_intercept + wls_slope * x_model
        
        axis1.plot(x_model,\
                 y_true_model,\
                 color='Red', \
                 label='true model')  
        
        axis1.plot(x_model, \
                 y_F87_model,\
                 color='purple',\
                 label='F87 estimate')  
        
        axis1.plot(x_model, \
                 y_T02_model,\
                 color='slateblue',\
                 label='T02 estimate')  
                           
        axis1.plot(x_model, \
                 y_OLS_model,\
                 color='deepskyblue',\
                 label='OLS estimate')  

        axis1.plot(x_model, \
                 y_WLS_model,\
                 color='darkseagreen',\
                 label='WLS estimate')  

        axis1.plot(x_model, \
                 y_mle_model,\
                 color='greenyellow',\
                 label='MLE estimate')  
        axis1.plot(x_model, \
                 y_bces_model,\
                 color='green',\
                 label='BCES estimate')  
        axis1.legend(loc='upper left', fontsize='small')

        axis1_xmin = x_model.min() - 0.10*x_model.min()
        axis1_xmax = x_model.max() + 0.10*x_model.max()
        axis1_ymin = y_true_model.min() - 1.0*(y_true_model.max() - y_true_model.min())
        axis1_ymax = y_true_model.max() + 2.25*(y_true_model.max() - y_true_model.min())

        text_label='true intr. scatter (sigma): ' + str(scatter_sigma_true)
        axis1.text(axis1_xmin,axis1_ymin + 0.10*(axis1_ymax - axis1_ymin),text_label,fontsize='small', fontweight="bold")
        axis1.axis([axis1_xmin, \
                    axis1_xmax, \
                    axis1_ymin, \
                    axis1_ymax])
                
        """
        Histogram of x_err
        """
        axis2 = fig.add_subplot(222)
        axis2.set_title(r'Realized X Error Distribution', fontweight="bold")
        axis2.set_xlabel(r"x error", fontsize = 12)
        axis2.set_ylabel(r"Normalized histogram", fontsize = 12)    
        n,bins,patches = axis2.hist(x_obs_err,50,normed = 1, facecolor='deepskyblue', alpha=0.75)

        if xerr_type not in ['homoscedastic', 'infinitesimal']:
            y = mlab.normpdf( bins, x_obs_err_mean, x_obs_err_sigma)
            axis2.plot(bins, y, 'r', linewidth=1, label = 'model')
            axis2.legend(loc='upper left', fontsize='small')
        
        """
        Histogram of y_err
        """
        axis3 = fig.add_subplot(223)
        axis3.set_title(r'Realized Y Error Distribution', fontweight="bold")
        axis3.set_xlabel(r"y error", fontsize = 12)
        axis3.set_ylabel(r"Normalized histogram", fontsize = 12)    
        n,bins,patches = axis3.hist(y_obs_err,50,normed = 1, facecolor='deepskyblue', alpha=0.75)
        if yerr_type not in ['homoscedastic', 'infinitesimal']:
            y = mlab.normpdf( bins, y_obs_err_mean, y_obs_err_sigma)
            axis3.plot(bins, y, 'r', linewidth=1, label = 'model')
            axis3.legend(loc='upper left', fontsize='small')

        """
        Histogram of intrinsic scatter
        """
        axis4 = fig.add_subplot(224)
        axis4.set_title(r'Realized Instrinsic Scatter', fontweight="bold")
        axis4.set_xlabel(r"intrinsic scatter", fontsize = 12)
        axis4.set_ylabel(r"Normalized histogram", fontsize = 12)    
        n,bins,patches = axis4.hist(scatter,50,normed = 1, facecolor='indigo', alpha=0.75)
        y = mlab.normpdf( bins, scatter_mean_true, scatter_sigma_true)
        axis4.plot(bins, y, 'r', linewidth=1, label = 'model')
        axis4.legend(loc='upper left', fontsize='small')

        fig.savefig(figure, format='pdf')
        
    return (error_bces_slope,		
            error_bces_intercept,	
            error_bces_scatter_variance,
            error_f87_slope,		
            error_f87_intercept,	
            error_f87_scatter_variance,
            error_k07_slope,		
            error_k07_intercept,	
            error_k07_scatter_variance,
            error_mle_slope,		
            error_mle_intercept,	
            error_mle_scatter_variance,
            error_odr_slope,		
            error_odr_intercept,	
            error_odr_scatter_variance,
            error_ols_slope,		
            error_ols_intercept,	
            error_ols_scatter_variance,
            error_t02_slope,		
            error_t02_intercept,	
            error_t02_scatter_variance,
            error_wls_slope,		
            error_wls_intercept,	
            error_wls_scatter_variance)
            



def linear_scatter_model(in_x,\
                in_x_err,\
                in_y_err,\
                slope_true=0.46,\
                intercept_true=-4.09,\
                scatter_mean_true = 0.0,\
                scatter_sigma_true = 1.0,\
                random_xvalues = None, \
                covxy = None, \
                xerr_type= None,\
                yerr_type= None,\
                verbose = None, \
                txt = None):
    """
    Generate a model a simulation of data with a linear relationship 
    and intrinic scatter.
        
    Parameters:
    in_x            array of x-values of data to be fit.
    in_x_err        arry of errors to x-values of data to be fit.
    in_y_err        array of errors to y-values of data to be fit.
    slope_true      true value of the slope of the linear model
    intercept_true  true value of the intercept of the linear model
    scatter_mean    true value of the mean of the intrinic scatter distribution
    scatter_sigma   true value of the sigma (width) of the intrinsic scatter
    random_xvalues  choose x values from random distribution (not input data)
    coxy            array of covariances between x, y values. 
    xerr_type       describes how the errors will be treated in simulation:
    yerr_type           see Different error types below
    figure          Generate a 2x2 panel figure showing the simulated data and fits
    txt             Save the realized data set as a text file with this name
    
        Different error types:    
        'None'              errors are set to the input values with no randomization
        'median'            errors set to the median of the input array (homoscedastic)
        'infinitesimal'     errors are set to an infinitesimal value (homoscedastic)
        'replacement'       errors are drawn from input array using sample with replacement (heteroscedastic)
        'normal'            errors are drawn from normal distribution scaled by input array (heteroscedastic)

    Returns:
    x,xerr,y,yerr
    """

    ndata = len(in_x)
            
    if random_xvalues is not None:
        """
        compute random x values from uniform distribution
        """
        x_true = np.random.uniform(in_x.min(),in_x.max(),ndata)
    else:
        x_true = in_x
    y_true = intercept_true + slope_true * x_true
        
    """
    observed measurement errors
    """
    x_obs_err = np.empty(ndata)
    y_obs_err = np.empty(ndata)
  
    """
    simulated errors that will be added to noisify the model data
    """
    x_sim_err = np.empty(ndata)
    y_sim_err = np.empty(ndata)
    
    """
    Select method of handling errors
    """
    def sample_wr(population, k):
        """
        Chooses k random elements (with replacement) from a population.        
        http://code.activestate.com/recipes/273085-sample-with-replacement/
        """
        n = len(population)
        _random, _int = random.random, int  # speed hack 
        result = [None] * k
        for i in range(k):
            j = _int(_random() * n)
            result[i] = population[j]
        return result  
            
    x_obs_err = in_x_err
    y_obs_err = in_y_err
    if xerr_type == None:
        x_obs_err= in_x_err
        x_sim_err = 0
    elif xerr_type == 'median':
        x_obs_err[...] = np.median(in_x_err)
        x_sim_err[...] = np.median(in_x_err) * np.random.normal(0,1,ndata)
    elif xerr_type == 'infinitesimal':
        x_obs_err[...] = 0.00001
        x_sim_err = x_obs_err
    elif xerr_type == 'replacement':
        rnd_sgn = np.random.random_integers(0,1,size=ndata)
        rnd_sgn[rnd_sgn == 0] = -1
        x_sim_err = rnd_sgn * np.asarray(sample_wr(in_x_err,ndata))
        x_obs_err[...] = np.fabs(x_sim_err)
    elif xerr_type == 'normal':
        x_obs_err= in_x_err
        x_sim_err = np.random.normal(0,1,ndata)*in_x_err
    elif xerr_type == 'test':
        x_obs_err[...]= 0.30
        x_sim_err = 0.30 * np.random.normal(0,1,ndata)
    else: 
        x_obs_err= in_x_err
        x_sim_err = np.random.normal(0,1,ndata)*in_x_err

    if yerr_type == None:
        y_obs_err= in_y_err
        y_sim_err[...] = 0
    elif yerr_type == 'median':
        y_obs_err[...] = np.median(in_y_err)
        y_sim_err[...] = np.median(in_y_err) * np.random.normal(0,1,ndata)
    elif yerr_type == 'infinitesimal':
        y_obs_err[...] = 0.00001
        y_sim_err = y_obs_err
    elif yerr_type == 'replacement':
        rnd_sgn = np.random.random_integers(0,1,size=ndata)
        rnd_sgn[rnd_sgn == 0] = -1
        y_sim_err = rnd_sgn * np.asarray(sample_wr(in_y_err,ndata))
        y_obs_err= np.fabs(y_sim_err)
    elif yerr_type == 'normal':
        y_obs_err= in_y_err
        y_sim_err = np.random.normal(0,1,ndata)*in_y_err
    elif yerr_type == 'test':
        y_obs_err[...]= 0.30
        y_sim_err = 0.30 * np.random.normal(0,1,ndata)
    else: 
        y_obs_err= in_y_err
        y_sim_err =  np.random.normal(0,1,ndata)*in_y_err
    
    
    scatter = scatter_sigma_true * np.random.normal(scatter_mean_true, 1.0, ndata)


    """
    DEBUG:
    """

    """
    Histogram of intrinsic scatter
    """
    
    # figsize in inches
    fig = plt.figure(dpi=300, figsize=[8,8])
    fig.set_tight_layout({'pad':1.08,'h_pad':0.25, 'w_pad':0.25, 'rect':(0,0,0.95,0.95)})
    
    axis4 = fig.add_subplot(111)
    axis4.set_title(r'Instrinsic Scatter Distribution', fontweight="bold")
    axis4.set_xlabel(r"intrinsic scatter", fontsize = 12)
    axis4.set_ylabel(r"Normalized histogram", fontsize = 12)    
    n,bins,patches = axis4.hist(scatter,50,normed = 1, facecolor='indigo', alpha=0.75)
    y = mlab.normpdf( bins, scatter_mean_true, scatter_sigma_true)
    axis4.plot(bins, y, 'r', linewidth=1, label = 'model')
    axis4.legend(loc='upper left', fontsize='small')
    fig.savefig('scatter_histogram.pdf', format='pdf')



    """
    generate noisified, simulated observed data from 
    linear model, noisified error and intrinsic scatter
    """
    if covxy is None:
        x_obs = x_true + x_sim_err
        y_obs = y_true + y_sim_err + scatter
    else:
        """
        generate data with specified covariance between x,y (errors).
        Use Cholesky decomposition to generate correlated random
        variables from a unit_normal ie N(0,1) distribution.
        """
        cov_matrix_array = np.empty(shape=(ndata,2,2))
        cov_matrix_array[:,0,0] = x_obs_err**2
        cov_matrix_array[:,0,1] = covxy
        cov_matrix_array[:,1,0] = covxy
        cov_matrix_array[:,1,1] = y_obs_err**2
        low_triangular_array = np.linalg.cholesky(cov_matrix_array)
        unit_normal_array = np.random.normal(size=(ndata,2,1))
        corr_random_array = np.empty_like(unit_normal_array)
        for i in range(ndata):
            corr_random_array[i,:,:] = np.dot(low_triangular_array[i,:,:],unit_normal_array[i,:,:])
            x_sim_err[i] = corr_random_array[i,0]
            y_sim_err[i] = corr_random_array[i,1]
        
        x_obs = x_true + x_sim_err
        y_obs = y_true + y_sim_err + scatter

        xy_obs = np.array([x_obs,y_obs])
        xy_true = np.array([x_true,y_true])
        xy_sim_err = np.array([x_sim_err,y_sim_err])
        cov_xy_obs = np.cov(xy_obs)
        cov_xy_true = np.cov(xy_true)
        cov_xy_sim_err = np.cov(xy_sim_err)
    
        """
        test values derived from the equations:
        See Kelly (2007) Equations 6,7.
        These quantities should be zero.
        
        Cov(x_obs,yobs) = Cov(x_true,y_true) + sig_xy
        Var(x_obs) = Var(x_true) + sig_x^2
        Var(y_obs) = Var(y_true) + sig_y^2 = sig_IS^2
        """
        test_covariance = cov_xy_obs[0,1] - (cov_xy_true[0,1] + np.mean(covxy)) 
        test_x_variance = cov_xy_obs[0,0] - (cov_xy_true[0,0] + np.mean(x_obs_err)**2)
        test_y_variance = cov_xy_obs[1,1] - (cov_xy_true[1,1] + np.mean(y_obs_err)**2 + scatter_sigma_true**2)
    
    if verbose is not None:
        print("\nSimulated data")
        print("Sample variances")
        print("x_obs         {:.5f}".format(np.var(x_obs)))
        print("y_obs         {:.5f}".format(np.var(y_obs)))
        print("x_true        {:.5f}".format(np.var(x_true)))
        print("y_true        {:.5f}".format(np.var(y_true)))

        if covxy is not None:
            print("Sample covariance matrices")
            print("Cov(x_obs,y_obs)")
            print(cov_xy_obs)
            print("Cov(x_true,y_true)")
            print(cov_xy_true)        
            print("Cov(x_sim_err,y_sim_err)")
            print(cov_xy_sim_err)        
        print("Measurement error covariances (homoscedastic)")
        print("sig_x^2       {:.5f}".format(np.mean(x_obs_err)**2))
        print("sig_y^2       {:.5f}".format(np.mean(y_obs_err)**2))  
        if covxy is not None:
            print("sig_xy        {:.5f}".format(np.mean(covxy))) 
            print("Variance/Covariance Tests (should be zero)")
            print("test Covariance  {:.5f}".format(test_covariance))
            print("test x Variance  {:.5f}".format(test_x_variance))
            print("test y Variance  {:.5f}".format(test_y_variance))
            print("\n")
    if txt is not None:
        """
        Save realization data as text file
        """
        print("\nSaving realization data to file: ", txt)
        print("\tColumns:  x, xerr, y, yerr")
        np.savetxt(txt, np.column_stack((x_obs,x_obs_err,y_obs,y_obs_err)),fmt=('%5.6f','%5.6f','%5.6f','%5.6f'))

    return (x_obs,x_obs_err,y_obs,y_obs_err)