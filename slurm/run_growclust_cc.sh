#!/bin/bash
cd relocation/growclust
mkdir -p TT OUT

cat <<EOF > growclust.inp
****  Example GrowClust Control File  *****
********  Daniel Trugman, 2016   **********
*******************************************
*
*******************************************
*************  Event list  ****************
*******************************************
* evlist_fmt (0 = evlist, 1 = phase, 2 = GrowClust, 3 = HypoInverse)
1
* fin_evlist (event list file name)
evlist.txt
*
*******************************************
************   Station list   *************
*******************************************
* stlist_fmt (0 = SEED channel, 1 = station name)
1
* fin_stlist (station list file name)
stlist.txt
*
*******************************************
*************   XCOR data   ***************
*******************************************
* xcordat_fmt (0 = binary, 1 = text), tdif_fmt (21 = tt2-tt1, 12 = tt1-tt2)
1  12
* fin_xcordat
dt.cc
*
*******************************************
*** Velocity Model / Travel Time Tables ***
*******************************************
* fin_vzmdl (input vz model file)
vzmodel.txt
* fout_vzfine (output, interpolated vz model file)
TT/vzfine.txt
* fout_pTT (output travel time table, P phase)
TT/tt.pg
* fout_sTT (output travel time table, S phase)
TT/tt.sg
*
******************************************
***** Travel Time Table Parameters  ******
******************************************
* vpvs_factor  rayparam_min (-1 = default)
  1.732             0.0
* tt_dep0  tt_dep1  tt_ddep
   -2.        31.       1.
* tt_del0  tt_del1  tt_ddel
   0.        500.      2.
*
******************************************
***** GrowClust Algorithm Parameters *****
******************************************
* rmin  delmax rmsmax 
   0.6    120    1.0
* rpsavgmin, rmincut  ngoodmin   iponly 
    0          0.6         8        0
*
******************************************
************ Output files ****************
******************************************
* nboot  nbranch_min
   0         1
* fout_cat (relocated catalog)
OUT/out.growclust_cc_cat
* fout_clust (relocated cluster file)
OUT/out.growclust_cc_clust
* fout_log (program log)
OUT/out.growclust_cc_log
* fout_boot (bootstrap distribution)
OUT/out.growclust_cc_boot
******************************************
******************************************
EOF

cat <<EOF > vzmodel.txt
-2.0 5.30 0.00
0.0 5.30 0.00
1.0 5.65 0.00
3.0 5.93 0.00
5.0 6.20 0.00
7.0 6.20 0.00
9.0 6.20 0.00
11.0 6.20 0.00
13.0 6.20 0.00
17.0 6.20 0.00
21.0 6.20 0.00
31.00 7.50 0.00
31.10 8.11 0.00
100.0 8.11 0.00
EOF

../GrowClust/SRC/growclust  growclust.inp
cp OUT/out.growclust_cc_cat ../../results/growclust_cc_catalog.txt
cd ../../
