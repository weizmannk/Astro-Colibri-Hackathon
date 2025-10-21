#!/bin/bash

source /home/weizmann.kiendrebeogo/anaconda3/bin/activate nmma

python Condor_nmma_fit.py --model nugent-hyper --svd-path ~/Dark-Matter/nmma/svdmodels --outdir ~/Dark-Matter/nmma_fitter/OUTDIR_Nugent --label nugent-hyper -- dataDir ~/Dark-Matter/ZTF-data/nmma_dot_dat_ZTF --trigger-time  --prior ~/Dark-Matter/nmma/priors/sncosmo-generic.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1.0 --nlive 1048 --Ebv-max 0.5724 --seed 42   $@
