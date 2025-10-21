# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@description    : This script is designed to perform detailed cosmological simulations and analyses
                  using a variety of astrophysical models. It includes functions to convert luminosity
                  distance to redshift and vice versa, manage and process astrophysical data, create
                  and manipulate priors for gravitational wave analyses, and submit batch jobs for
                  simulation tasks. The script utilizes data from the Zwicky Transient Facility (ZTF)
                  and integrates Bilby for Bayesian inference.

---------------------------------------------------------------------------------------------------
"""
        
    
import os
import sys
import subprocess
import logging
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time

def parse_csv(infile):
    """
    Reads photometric data from a CSV file and transforms it into a specific format.

    Parameters:
        infile (str): Path to the CSV file containing photometric data.

    Returns:
        list: A list of lists, where each sublist contains the transformed data for one observation.
    """

    # Read the CSV file using numpy.genfromtxt, skipping the first line (header)
    in_data = np.genfromtxt(
        infile, dtype=None, delimiter=",", skip_header=1, encoding=None
    )

    out_data = []
    for line in in_data:
        # Convert JD time to ISO format using astropy.time.Time
        time_iso = Time(line[1], format="jd").isot

        # Handle non-detections where mag is 99.0 by using limit_mag and setting error to infinity
        mag = line[5] if line[2] == 99.0 else line[2]
        error = np.inf if line[2] == 99.0 else line[3]

        # Append the transformed data to out_data
        out_data.append([time_iso, str(line[4]), str(mag), str(error)])

    return out_data

# Define the model
model = "salt3"   ## "nugent-hyper", "nugent-sn1a",   "Piro2021", "salt2", "salt3"
                   ## SNI models  : nugent-sn2p, nugent-sn2l, nugent-sn2n, nugent-sn1bc

# Supernova class 
SN_class =  "Ca_rich_SN"    ## "Ca_rich_SN" "IbIcSN" , "IaSN" ,  "IISN"

model_list = [m.strip() for m in model.split(",")]

output_directory = f"OUTDIR/{'_'.join(sorted(model_list)).upper()}/{SN_class}"


os.makedirs(output_directory, exist_ok=True)
log_dir = os.path.join(output_directory, "logs")
os.makedirs(log_dir, exist_ok=True)


# # SVD models directories for Bu2019nsbh, Bu2019lm ...  models 
# svdmodel =   os.path.join(os.path.dirname(__file__), "nmma-models/models")

# Model based configurations


if len(model_list) == 1 and model_list[-1] in ["nugent-sn1a", "nugent-hyper", "nugent-sn2p", "nugent-sn2l", "nugent-sn2n", "nugent-sn1bc"]:
        prior_name = "sncosmo-generic.prior"

elif "salt2" in model_list  or "salt3"  in model_list and len(model_list) == 1:
    prior_name = "salt2.prior"
    
elif "Sr2023" in model_list and "nugent-hyper" in model_list:
    prior_name = "Sr2023.prior"
    
elif "nugent-hyper" in model_list and "TrPi2018" in model_list:
    prior_name = "sncosmo-generic_TrPi2018.prior"
    
else:
    prior_name =  f"{'_'.join(sorted(model_list))}.prior"
                      
print("Model:", model)
print("Prior Name:", prior_name)



prior_file  = os.path.join(os.path.dirname(__file__), './priors', prior_name)
infile = "./lc_ZTF19abrdxbh.dat"

in_data = parse_csv(infile)


# Set the trigger time
trigger_time = np.inf
for line in np.atleast_1d(in_data):
    if np.isinf(float(str(line[0]).split()[3])):
        continue

    elif Time(str(line[0]).split()[0], format='isot').mjd -1 < trigger_time:
        trigger_time = Time(str(line[0]).split()[0], format='isot').mjd -1



#     label = f"{'_'.join(sorted(model_list))}"

#     # determine the --tmax,  "Days to stop analysing from the 
#     time_data = Table.read(data_file, format="ascii")
#     init_det=  Time(time_data[0][0], format='isot')
#     last_det = Time(time_data[-1][0], format='isot')
#     tmax = ((last_det - init_det).value - 0.1) +  30 # tmax  + thimeshit
#     tmax_axis = ((last_det - init_det).value) + 10



    # #--svd-path {svdmodel}

    #  # Setup job arguments for Condor submission #--trigger-time  {trigger_time}
    # command_string = f"light-curve-analysis --model {model}   --outdir {output_directory} --data {data_file} --prior {prior_file}  --label {label}  --tmin -10 --tmax {tmax} --dt 0.5 --error-budget 0.1 --nlive 2048 --Ebv-max --bestfit --generation-seed 42  --local-only"


    # command = subprocess.run(command_string, shell=True, capture_output=True)
    # sys.stdout.buffer.write(command.stdout)
    # sys.stderr.buffer.write(command.stderr)
