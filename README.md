# TeenGalaxies
Do galaxies grow steadily or in spurts?  Data science can answer this question!

Do galaxies form their stars gradually over time or suddenly in fits and starts?  If galaxies form gradually, then their Star Formation Rates (SFR) should be highly correlated with their Stellar Masses (M*), with low scatter about this correlation.  However, if there is large scatter in the SFR-M* correlation, then it suggests that galaxies instead grow in fits and starts.  

So we need to measure the *scatter* in the correlation, not just the correlation itself. Our main innovation is to use a statistical methodology that separates the artificial scatter caused by the measurement errors in SFR and M*, as well as covariance between SFR and M* estimates, from the intrinsic scatter that relates to the galaxies themselves.  Our paper is available on arXiv [here](http://lanl.arxiv.org/abs/1602.03909) and is published in The Astrophysical Journal Letters (Kurczynski et al. 2016, ApJ 820, 1L)

Here are the data and code (in Python) to analyze intrinsic scatter in the Star Formation Rate (SFR) vs. Stellar Mass (M*) relationship of star-forming galaxies in the Hubble Ultradeep Field (HUDF) and the surrounding area (CANDELS/GOODS-S) in five redshift bins.
