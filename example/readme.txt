Experiment
2015-05-16-A

sample01/run03b/sfr_vs_m_linfit_py

example illustrating SFR vs M* analysis code.


OBJECTIVE
SFR vs M* analysis of galaxies in sample01, using data from run03b (linear-exponential, "delayed tau model" SFH).  Linear model parameters and intrinsic scatter.

METHOD
1.  Input Data

	see file: candels_2015a_sample01_run03b_speedymc_results_no_rejects_v1.fits
	
	
2.  Python scripts: 

	Execution command (from bash shell):
	
./sfr_inst_vs_m_line_clip_sim_linexp.py -v -i 1000 -c 2.0 -z 1.2 candels_2015a_sample01_run03b_speedymc_results_no_rejects_v1.fits candels_2015a_sample01_run03b_sfr_inst_vs_m_line_new_clip_sim_z12_median_ellipse_v2.pdf > candels_2015a_sample01_run03b_sfr_inst_vs_m_line_new_clip_sim_z12_median_ellipse_v2.log.txt


	Execution time:  about 2 minutes

RESULTS

FIGURE:  candels_2015a_sample01_run03b_sfr_inst_vs_m_line_new_clip_sim_z12_median_ellipse.pdf

CAPTION
Star Formation Rate (SFR$_{Inst}$; instantaneous) vs. Stellar Mass (M$_*$) in the redshift range $1.0 < z < 1.5$ for galaxies in combined CANDELS (spec-z) and UVUDF (photo-z) sample.  Outliers (red points) from an initial fit are clipped; Remaining galaxies (gray points) are used to determine the best fit (dark purple).  Results from Whitaker et al.~(2014; cyan) and the meta-analysis of Speagle et al.~(2014; red) are shown; dashed regions indicate extrapolations from the reported ranges in M$_*$.  Selection curves are shown in black; our data are insensitive to galaxies that would fall along the correlation, to the lower left of this curve.  The squared Pearson correlation coefficient and estimated intrinsic scatter (dex) are indicated by the text label.  A typical error ellipse is shown in the upper left, with half-width and half-height equal to the median error in log M$_*$ and log SFR respectively, and orientation determined by the median covariance.


CONCLUSION
Revised SFR-M* analysis for sample01 is done for SFR_Inst vs M*.  




