#!/bin/bash

WORKING_DIR=$PWD
root_path="local"
region="Mendocino_8mon"

cd $root_path/$region/velest

if [ ! -d "Velest" ]; then
    echo $PATH
    git clone git@github.com:AI4EPS/Velest.git
#    git clone git@github.com:BORONG-SEIS/VELEST_ethz.git
fi
cd $PWD/Velest
gfortran velest.f -o velest --std legacy
export PATH=${PATH}:$PWD
cd ../
# fi

echo $PATH

cp ./Velest/regionskoord.dat ./
cp ./Velest/regionsnamen.dat ./

read olat olon neqs < center.txt

# single mode
cat <<EOF > velest.cmn
******* CONTROL-FILE FOR PROGRAM  V E L E S T  (28-SEPT1993) *******
***
*** ( all lines starting with  *  are ignored! )
*** ( where no filename is specified, 
***   leave the line BLANK. Do NOT delete!)
***
*** next line contains a title (printed on output):
CALAVERAS area7 1.10.93 EK startmodell vers. 1.1  
***      starting model 1.1 based on Castillo and Ellsworth 1993, JGR
***  olat       olon   icoordsystem      zshift   itrial ztrial    ised
  $olat     $olon      0            0.000      0     0.00       0
***
*** neqs   nshot   rotate
     $neqs      0      0.0
***
*** isingle   iresolcalc
       1          0
***
*** dmax    itopo    zmin     veladj    zadj   lowveloclay
   200.0     0     -0.20       0.20    5.00       0
***
*** nsp    swtfac   vpvs       nmod
     2      0.75    1.730        1
***
***   othet   xythet    zthet    vthet   stathet
      0.01    0.01      0.01     10.     1
***
*** nsinv   nshcor   nshfix     iuseelev    iusestacorr
       1       0       0           0            0
***
*** iturbo    icnvout   istaout   ismpout
       1         1         1         0
***
*** irayout   idrvout   ialeout   idspout   irflout   irfrout   iresout
       0         0         0         0         0         0         0
***
*** delmin   ittmax   invertratio
    0.010      99          0
***
*** Modelfile:
velocity.mod
***
*** Stationfile:
station.sta
***
*** Seismofile:
                                                                                
***
*** File with region names:
regionsnamen.dat
***
*** File with region coordinates:
regionskoord.dat
***
*** File #1 with topo data:
                                                                                
***
*** File #2 with topo data:
                                                                                
***
*** DATA INPUT files:
***
*** File with Earthquake data:
catalog.cnv
***
*** File with Shot data:
                                                                                
***
*** OUTPUT files:
***
*** Main print output file:
velest.OUT
***
*** File with single event locations:
single_out.CHECK
***
*** File with final hypocenters in *.cnv format:
velest.CNV
***
*** File with new station corrections:
stacrr.OUT
***
*** File with summary cards (e.g. for plotting):

***
*** File with raypoints:

***
*** File with derivatives:

***
*** File with ALEs:

***
*** File with Dirichlet spreads:

***
*** File with reflection points:

***
*** File with refraction points:

***
*** File with residuals:

***
******* END OF THE CONTROL-FILE FOR PROGRAM  V E L E S T  *******
EOF

# Gil7 model
cat <<EOF > velocity.mod
 CALAVERAS1D-modell (mod1.1 EK280993)   Ref. station HGSM      
 8        vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)
 3.20       -5.0    001.00            P-VELOCITY MODEL
 3.20        0.0    001.00
 4.50        1.0    001.00 
 4.80        3.0    001.00
 5.51        4.0    001.00
 6.21        5.0    001.00
 6.89       17.0    001.00
 7.83       25.0    001.00
 8                     
 1.50       -5.0    001.00            S-VELOCITY MODEL
 1.50        0.0    001.00
 2.40        1.0    001.00 
 2.78        3.0    001.00
 3.18        4.0    001.00
 3.40        5.0    001.00
 3.98       17.0    001.00
 4.52       25.0    001.00
EOF

velest