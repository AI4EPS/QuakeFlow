#!/bin/bash
hypo=../hyp1.40/source/hyp1.40
ph2dt=../HYPODD/src/ph2dt/ph2dt
hypoDD=../HYPODD/src/hypoDD/hypoDD

## gamma to hypoDD
python gamma2hypoDD.py
# $ph2dt ph2dt.inp
# $hypoDD hypoDD.inp

# ## gamma to hypoinverse
# python convert_stations.py
# python convert_picks.py
# $hypo < hyp.command

# ## hypoinvese to hypoDD
# python hypoinverse2hypoDD.py
# $ph2dt ph2dt.inp  
# $hypoDD hypoDD.inp
