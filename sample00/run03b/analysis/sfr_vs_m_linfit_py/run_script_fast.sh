#!/usr/bin/env bash
./sfr_100_vs_m_line_clip_sim_linexp.py -v -i 100 -c 2.0 -z 1.2 candels_2015a_sample01_run03_speedymc_results_norejects_v1.fits candels_2015a_sample01_run03_sfr_100_vs_m_line_new_clip_sim_z12_median_ellipse.pdf 
./sfr_inst_vs_m_line_clip_sim_linexp.py -v -i 100 -c 2.0 -z 1.2 candels_2015a_sample01_run03_speedymc_results_norejects_v1.fits candels_2015a_sample01_run03_sfr_inst_vs_m_line_new_clip_sim_z12_median_ellipse.pdf
./sfr_life_vs_m_line_clip_sim_linexp.py -v -i 100 -c 2.0 -z 1.2 candels_2015a_sample01_run03_speedymc_results_norejects_v1.fits candels_2015a_sample01_run03_sfr_life_vs_m_line_new_clip_sim_z12_median_ellipse.pdf

