from subprocess import check_output
import sys
import os
import glob
import argparse

import numpy as np
import pandas as pd
from astropy.time import Time

#executable_file = "/home/weizmann.kiendrebeogo/Dark-Matter/nmma_fitter/condor/condor_run.sh" 


def data_filename(datapath):
    """[using recursively to find files]

    :param datapath: the filename directory
    :type datapath: [str]
    :return: all .csv files 
    :rtype: [list]
    """

    return glob.glob(datapath+'/**/*.dat', recursive=True)


def main():

    parser = argparse.ArgumentParser(
        description="Inference on on model parameters."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to be used"
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points (default: 2048)"
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "-d",
        "--dataDir", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        required=True, 
        help="Path to the output directory"
    )
    parser.add_argument(
        "--label", 
        type=str, 
        required=True, 
        help="Label for the run"
    )
    parser.add_argument(
        "--trigger-time",
        action="store_true", 
        default=False,
        help="Trigger time in modified julian day, not required if injection set is provided",
    )
    parser.add_argument(
        "--prior",
        type=str,
        help="The prior file from which to generate injections",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Days to start analysing from the trigger time (default: 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to stop analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt", 
        type=float, 
        default=0.1, 
        help="Time step in day (default: 0.1)"
    )
    parser.add_argument(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--Ebv-max",
        type=float,
        default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)",
    )
    parser.add_argument(
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--error-budget",
        type=str,
        default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="pymultinest",
        help="Sampler to be used (default: pymultinest)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)",
    )
    parser.add_argument(
        "--seed",
        metavar="seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default="0,14",
        help="Start and end time for light curve plot (default: 0-14)",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default="22,16",
        help="Upper and lower magnitude limit for light curve plot (default: 22-16)",
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        nargs='+', 
        default=42, 
        help="Injection generation seed (default: 42), this can take a list",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--bestfit", 
        help="Save the best fit parameters and magnitudes to JSON", 
        action="store_true", 
        default=False,
    )
    parser.add_argument(
        "--condor-dag-file",
        type=str,
        required=True,
        help="The condor dag file to be created"
    )
    parser.add_argument(
        "--condor-sub-file",
        type=str,
        required=True,
        help="The condor sub file to be created"
    )
    parser.add_argument(
        "--bash-file", type=str, required=True, help="The bash file to be created"
    )
    args = parser.parse_args()

    
    logdir = os.path.join(args.outdir, f"logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)  
    
    light_curve_analysis = (
        check_output(["which", "lightcurve-analysis"]).decode().replace("\n", "")
    )
    
    datapath = args.dataDir
    ztf_files = data_filename(datapath)

    number_jobs = len(ztf_files)

    job_number = 0
    fid =  open(args.condor_dag_file, "w")
    fid1 = open(args.bash_file, "w")

    for data_file in ztf_files:
        candname = data_file.split('/')[-1].split('.')[0]
        
        outdir = f'{args.outdir}/{candname}'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            
        # Set the trigger time
        in_data = pd.read_csv(f'{data_file}')
        if args.trigger_time:
            trigger_time = np.inf
            for line in np.atleast_1d(in_data):
                
                if np.isinf(float(str(line[0]).split()[3])):
                    continue
                elif Time(str(line[0]).split()[0], format='isot').mjd < trigger_time:
                    trigger_time = Time(str(line[0]).split()[0], format='isot').mjd
                    print(trigger_time)

        fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
        fid.write("RETRY %d 3\n" % (job_number))
        fid.write(
            'VARS %d jobNumber="%d" OUTDIR="%s" DATA="%s" TRIGGER="%f"\n'
            % (job_number, job_number, outdir, data_file, trigger_time)
        )
        fid.write("\n\n")
        job_number = job_number + 1

        if args.interpolation_type:
            fid1.write(
                f"{light_curve_analysis}  --model {args.model} --svd-path {args.svd_path} --interpolation-type {args.interpolation_type} --outdir {outdir} --label {args.label} --data {data_file} --prior {args.prior}  --trigger-time  {trigger_time}  --tmin {args.tmin} --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
            )

        else:
            fid1.write(
                f"{light_curve_analysis} --model {args.model} --svd-path {args.svd_path} --outdir {outdir} --label {args.label} --data {data_file} --prior {args.prior}  --trigger-time  {trigger_time}  --tmin {args.tmin} --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
            )
    fid.close()
    fid1.close()

    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n"%light_curve_analysis)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")

    if args.interpolation_type:
        fid.write(
            f"arguments =  --model {args.model} --svd-path {args.svd_path} --interpolation-type {args.interpolation_type} --outdir $(OUTDIR) --label {args.label} --data $(DATA) --prior {args.prior} --trigger-time  $(TRIGGER) --tmin {args.tmin}  --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
        ) 
    else:
        fid.write(
            f"arguments =  --model {args.model} --svd-path {args.svd_path}  --outdir $(OUTDIR) --label {args.label} --data $(DATA) --prior {args.prior} --trigger-time  $(TRIGGER) --tmin {args.tmin} --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
        )
    
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o3.burst.allsky.stamp\n") 
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/light_curve_analysis.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")

    
if __name__ == "__main__":
    main()  {trigger_time}  --tmin {args.tmin} --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
            )
    fid.close()
    fid1.close()

    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n"%light_curve_analysis)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")

    if args.interpolation_type:
        fid.write(
            f"arguments =  --model {args.model} --svd-path {args.svd_path} --interpolation-type {args.interpolation_type} --outdir $(OUTDIR) --label {args.label} --data $(DATA) --prior {args.prior} --trigger-time  $(TRIGGER) --tmin {args.tmin}  --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
        ) 
    else:
        fid.write(
            f"arguments =  --model {args.model} --svd-path {args.svd_path}  --outdir $(OUTDIR) --label {args.label} --data $(DATA) --prior {args.prior} --trigger-time  $(TRIGGER) --tmin {args.tmin} --tmax {args.tmax} --dt {args.dt} --error-budget {args.error_budget} --nlive {args.nlive} --Ebv-max {args.Ebv_max} --plot --seed {args.seed}\n".format()
        )
    
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o3.burst.allsky.stamp\n") 
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/light_curve_analysis.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")

    
if __name__ == "__main__":
    main()