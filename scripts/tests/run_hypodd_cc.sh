#!/bin/bash
set -x
WORKING_DIR=$PWD
if [ $# -eq 2 ]; then
  root_path=$1
  region=$2
else
  root_path="."
  region="."
fi

if [ ! -d "$root_path/$region/hypodd" ]; then
  mkdir -p $root_path/$region/hypodd
fi

mv $root_path/$region/hypodd/dt.cc $root_path/$region/hypodd/dt_old.cc 
cp $root_path/$region/cctorch/dt.cc $root_path/$region/hypodd/dt.cc 
cd $root_path/$region/hypodd

if [ ! -d "HypoDD" ]; then
  git clone git@github.com:zhuwq0/HypoDD.git
  export PATH=$PATH:$PWD/HypoDD
  make -C HypoDD/src/
fi

cat <<EOF > cc.inp
* RELOC.INP:
*--- input file selection
* cross correlation diff times:
dt.cc
*
*catalog P diff times:

*
* event file:
events.dat
*
* station file:
stations.dat
*
*--- output file selection
* original locations:
hypodd_cc.loc
* relocations:
hypodd_cc.reloc
* station information:
hypodd.sta
* residual information:
hypodd.res
* source paramater information:
hypodd.src
*
*--- data type selection: 
* IDAT:  0 = synthetics; 1= cross corr; 2= catalog; 3= cross & cat 
* IPHA: 1= P; 2= S; 3= P&S
* DIST:max dist [km] between cluster centroid and station 
* IDAT   IPHA   DIST
    1     3     120
*
*--- event clustering:
* OBSCC:    min # of obs/pair for crosstime data (0= no clustering)
* OBSCT:    min # of obs/pair for network data (0= no clustering)
* OBSCC  OBSCT    
     0     0        
*
*--- solution control:
* ISTART:  	1 = from single source; 2 = from network sources
* ISOLV:	1 = SVD, 2=lsqr
* NSET:      	number of sets of iteration with specifications following
*  ISTART  ISOLV  NSET
    2        2      4
*
*--- data weighting and re-weighting: 
* NITER: 		last iteration to used the following weights
* WTCCP, WTCCS:		weight cross P, S 
* WTCTP, WTCTS:		weight catalog P, S 
* WRCC, WRCT:		residual threshold in sec for cross, catalog data 
* WDCC, WDCT:  		max dist [km] between cross, catalog linked pairs
* DAMP:    		damping (for lsqr only) 
*       ---  CROSS DATA ----- ----CATALOG DATA ----
* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP
   4      1    1    -9    -9    -9    -9     -9    -9  70
   4      1    1     6    -9    -9    -9     -9    -9  70
   4      1    0.8   3     4    -9    -9     -9    -9  70
   4      1    0.8   2     2    -9    -9     -9    -9  70  
*
*--- 1D model:
* NLAY:		number of model layers  
* RATIO:	vp/vs ratio 
* TOP:		depths of top of layer (km) 
* VEL: 		layer velocities (km/s)
* NLAY  RATIO 
   10     1.73
* TOP 
0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 30.0
* VEL
4.74 5.01 5.35 5.71 6.07 6.17 6.27 6.34 6.39 7.80
*
*--- event selection:
* CID: 	cluster to be relocated (0 = all)
* ID:	cuspids of event to be relocated (8 per line)
* CID    
    0      
* ID
EOF

./HypoDD/src/hypoDD/hypoDD cc.inp
cd $WORKING_DIR