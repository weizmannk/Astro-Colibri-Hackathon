import os
import glob
import pandas as pd
import numpy as np

from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value

from tqdm.auto import tqdm
from astropy.time import Time


# =============================================================================
# The cosmological values
# =============================================================================
H0      = 67.9         # the Hubble Cosmological constant, km/s/Mpc, Plank 2015 results
Omega_m = 0.31         # density parameter of matter

# =============================================================================
# The constants
# =============================================================================
c        = 299792458.0              # The speed of light in m/s
#Mpc      = 3.08568025e22            # Mega-Parsec in meters, m
#Sun_Mass = 1.988409870698051e+30    # Sun mass in kg (IS Units)
G        = 6.6743e-11               # Newton Gravitational constant N.m^2.kg^2



def luminosity_distance_to_readshift(LD):
    
    z = z_at_value(cosmo.luminosity_distance, LD * u.Mpc).to_value(u.dimensionless_unscaled)
    
    return z
    
def readshit_to_luminosity_distance(z):
    """calculate luminosity distance in geometrized units 
        see http://arxiv.org/pdf/1111.6396v1.pdf

    :return: the luminosity distance in Mpc
    :rtype: numpy.ndarray of the luminosity distance in (Mpc)
    """

    x0 = (1. - Omega_m)/Omega_m
    xZ = x0/(1. + z)**3

    Phi0 = (1. + 1.320*x0 + 0.4415*x0**2 + 0.02656*x0**3)
    Phi0 /= (1. + 1.392*x0 + 0.5121*x0**2 + 0.03944*x0**3)
    PhiZ = (1. + 1.320*xZ + 0.4415*xZ**2 + 0.02656*xZ**3)
    PhiZ /= (1. + 1.392*xZ + 0.5121*xZ**2 + 0.03944*xZ**3)

    d_luminosity = ( 2.*c/H0*(1.0e-3)*(1. + z)/np.sqrt(Omega_m)*(Phi0 - PhiZ/np.sqrt(1. + z)))
    return d_luminosity


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """

    return glob.glob(datapath+'/**/*.csv', recursive=True)



### Data dir
datapath = f"{os.path.dirname(os.path.realpath('__file__'))}/ZTF-data/nmma_ZTF"
classified_dir =  f"{os.path.dirname(os.path.realpath('__file__'))}/ZTF-data/classified_unclassified_objet/ZTF_Gap_Trasient_classifications.dat"

targetdir ='ZTF'
if not os.path.isdir(targetdir):
    os.makedirs(targetdir)


#### Read files 
ztf_files = data_filename(datapath)
classified_sources = Table.read(classified_dir, format='ascii.fast_tab') #['source_name']



# with tqdm(total=len(ztf_files)) as progress:
#     for forcedfile in ztf_files:

### Read the source names of the transients
#name = forcedfile.split('/')[-1].split('.')[0].split('_')[-1]
forcedfile = f'{datapath}/lc_ZTF22aboisvs.csv'
name = "ZTF22aboisvs"
in_data = pd.read_csv(forcedfile)
in_data.sort_values(by='jd', ascending=True)

# Redshift of the supernova
indx   = np.where(classified_sources['source_name'] == name)[0][0]
redshift = classified_sources[indx]['redshift']

#convertion to luminosity distance
LD = readshit_to_luminosity_distance(redshift)


