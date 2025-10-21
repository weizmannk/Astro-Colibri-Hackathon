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
import glob
import subprocess
import logging
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from astropy.cosmology import Planck15 as cosmo, z_at_value
import bilby

## function to determine the Ebv_max
from astropy.coordinates import SkyCoord
from dustmaps.planck import  PlanckGNILCQuery
from dustmaps.config import config



# Setup basic configuration for logging
logging.basicConfig(filename='nmma-simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for cosmological calculations
H0 = 67.9  # Hubble Constant in km/s/Mpc, from Planck 2015 results
Omega_m = 0.31  # Matter density parameter

# Fundamental physical constants
c = 299792458.0  # Speed of light in m/s
G = 6.6743e-11  # Newton's gravitational constant in N(m^2)/(kg^2)

def luminosity_distance_to_redshift(ld):
    """
    Convert luminosity distance to redshift using the Astropy cosmology module.
    Parameters:
        ld (float): Luminosity distance in Mpc.
    Returns:
        float: Computed redshift as a dimensionless quantity.
    """
    return z_at_value(cosmo.luminosity_distance, ld * u.Mpc)

def redshift_to_luminosity_distance(z):
    """
    Calculate luminosity distance for a given redshift based on a cosmological model.
    The formula used comes from http://arxiv.org/pdf/1111.6396v1.pdf

    Parameters:
    z (float): Redshift, dimensionless.

    Returns:
    float: Luminosity distance in Mpc.
    """
    x0 = (1. - Omega_m) / Omega_m
    xZ = x0 / (1. + z)**3

    Phi0 = (1. + 1.320 * x0 + 0.4415 * x0**2 + 0.02656 * x0**3)
    Phi0 /= (1. + 1.392 * x0 + 0.5121 * x0**2 + 0.03944 * x0**3)
    PhiZ = (1. + 1.320 * xZ + 0.4415 * xZ**2 + 0.02656 * xZ**3)
    PhiZ /= (1. + 1.392 * xZ + 0.5121 * xZ**2 + 0.03944 * xZ**3)

    d_luminosity = (2. * c / H0 * (1.0e-3) * (1. + z) / np.sqrt(Omega_m) * (Phi0 - PhiZ / np.sqrt(1. + z)))
    return d_luminosity

def data_filename(datapath):
    """
    Recursively find all .dat files in the specified directory.
    Parameters:
        datapath (str): Path to the directory where data files are stored.
    Returns:
        list: A list of paths to .dat files.
    """
    return glob.glob(os.path.join(datapath, '**', '*.dat'), recursive=True)

def create_prior(prior_file_path, outdir, label='prior', z=None, fix_z=True):
    """
    Create and save a prior file using Bilby's PriorDict.
    Parameters:
        prior_directory (str): Directory where prior files are located.
        outdir (str): Output directory to save the prior file.
        label (str): Base name for the prior file.
        z (float, optional): Redshift to set as a prior.
        fix_z (bool): Whether to fix the redshift at a given value.
    Returns:
        str: Path to the saved prior file.
    """
    
    logging.info(f"Attempting to create prior with path: {prior_file_path} and outdir: {outdir}")
    
    if not os.path.isfile(prior_file_path):
        logging.error(f"Prior file {prior_file_path} does not exist")
        return None
    
    priors = bilby.gw.prior.PriorDict(prior_file_path)
    
    if fix_z and z is not None:
        from astropy.coordinates import Distance
        distance = Distance(z=z, unit=u.Mpc)
        priors['luminosity_distance']  = bilby.core.prior.DeltaFunction(
             peak=distance.value, name='luminosity_distance', latex_label='$D_L$', unit='Mpc')
   
    prior_output_path = os.path.join(outdir, f"{label}.prior")
    priors.to_file(outdir=outdir, label=label)
    
    return prior_output_path


def get_planck_ebv(ra, dec):
    """
    Get E(B-V) value for given right ascension and declination using Planck GNILC map.

    Parameters:
    ra (float): Right Ascension in degrees
    dec (float): Declination in degrees

    Returns:
    float: E(B-V) value
    """
    # Create SkyCoord object for the given coordinates : ICRS (International Celestial Reference System)
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    # Load the Planck GNILC map
    planck = PlanckGNILCQuery()
    # Query the Planck GNILC map for the E(B-V) value
    ebv = planck(coords)

    return ebv



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

output_directory = f"OUTDIR/{'_'.join(sorted(model_list)).upper()}/{SN_class}_redshift"


os.makedirs(output_directory, exist_ok=True)
log_dir = os.path.join(output_directory, "logs")
os.makedirs(log_dir, exist_ok=True)

# Data path setup
datapath = os.path.join(os.path.dirname(__file__), 'ZTF-data')

# data_files = data_filename(os.path.join(datapath, "dat-files"))
data_files = data_filename(os.path.join(datapath, f"nmma-format/ZTF-SN-data-dat/{SN_class}"))

classified_dir = os.path.join(datapath, 'classified_unclassified_objet', 'ZTF_Gap_Trasient_classifications.dat')
classified_sources = Table.read(classified_dir, format='ascii.fast_tab')

# SVD models directories for Bu2019nsbh, Bu2019lm ...  models 
svdmodel =   os.path.join(os.path.dirname(__file__), "nmma-models/models")

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

prior_file_path  = os.path.join(os.path.dirname(__file__), 'nmma/priors', prior_name)

job =0

# Process data files
for data_file in data_files:
    candname = os.path.basename(data_file).split('.')[0].split('_')[-1]
    # if candname =="ZTF21aagteny":
        
    outdir = os.path.join(output_directory, candname)
    os.makedirs(outdir, exist_ok=True)

    # Fetch redshift and convert to luminosity distance
    indx = np.where(classified_sources['source_name'] == candname)[0][0]
    redshift = classified_sources['redshift'][indx]


    ## E(B-V) Calclation using dustmaps :https://github.com/gregreen/dustmaps/blob/master/README.md
    # Example usage
    config['data_dir'] = './dustmaps' # Set the path to your dustmaps data directory

    ra = classified_sources['ra'][indx]  # Right Ascension in degrees
    dec = classified_sources['dec'][indx]   # Declination in degrees
    ebv_value = get_planck_ebv(ra, dec)
    print(f'E(B-V) = {ebv_value}')


    if np.isnan(redshift):
        logging.info(f"Redshift is NaN for {candname}")
        continue

    # if need to fix the distance , put the fix_z at True
    prior_file = create_prior(prior_file_path, outdir, label=candname, z=redshift, fix_z=True)
    logging.info(f"Created prior file for model {model} at {prior_file}")

    # Set the trigger time
    in_data = pd.read_csv(data_file)
    trigger_time = np.inf
    for line in np.atleast_1d(in_data):
        if np.isinf(float(str(line[0]).split()[3])):
            continue

        elif Time(str(line[0]).split()[0], format='isot').mjd -1 < trigger_time:
            trigger_time = Time(str(line[0]).split()[0], format='isot').mjd -1



    label = f"{'_'.join(sorted(model_list))}_{candname}"

    # determine the --tmax,  "Days to stop analysing from the 
    time_data = Table.read(data_file, format="ascii")
    init_det=  Time(time_data[0][0], format='isot')
    last_det = Time(time_data[-1][0], format='isot')
    tmax = ((last_det - init_det).value - 0.1) +  30 # tmax  + thimeshit
    tmax_axis = ((last_det - init_det).value) + 10



    #--svd-path {svdmodel}

     # Setup job arguments for Condor submission #--trigger-time  {trigger_time}
    arguments = f"--model {model}   --outdir {outdir} --data {data_file} --prior {prior_file}  --label {label}  --tmin -10 --tmax {tmax} --dt 0.5 --error-budget 0.1 --nlive 2048 --Ebv-max {ebv_value} --bestfit --generation-seed 42 --plot --ylim 22,16 --xlim=-10,{tmax_axis}  --local-only"

    condor_submit_script = f'''
                            universe = vanilla
                            accounting_group = ligo.dev.o4.cbc.pe.bayestar
                            getenv = true
                            executable = {subprocess.check_output(["which", "lightcurve-analysis"]).decode().strip()}
                            arguments = {arguments}
                            output = {log_dir}/$(Cluster)_$(Process).out
                            error = {log_dir}/$(Cluster)_$(Process).err
                            log = {log_dir}/$(Cluster)_$(Process).log
                            request_memory = 8192 MB
                            request_disk = 2000 MB
                            JobBatchName = "Dark-Matter"
                            environment = "OMP_NUM_THREADS=1"
                            queue
                        '''
    with subprocess.Popen(['condor_submit'], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, stderr = proc.communicate(input=condor_submit_script)
        if stdout:
            print("Condor submit output:", stdout)
        if stderr or proc.returncode != 0:
            print("Condor submit error:", stderr)
    job+=1


print(f"\n\n{job} jobs have been submit.")

