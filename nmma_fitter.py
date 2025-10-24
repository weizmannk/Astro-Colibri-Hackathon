# python nmma_fitter.py  --model salt3  --outdir ./output  --datafile lc_ZTF19abrdxbh.dat    --priors-dir ./priors --candname ZTF19abrdxbh --tmin -10  --tmax 20  --dt 0.5  --error-budget 0.1 --nlive 1024 --Ebv-max 0.5724 --bestfit --generation-seed 42  --local-only  --cpus 4  --timeshift 30 --dry-run --fetch-Ebv-from-dustmap --ra 222.7275966 --dec 27.5825746


# python nmma_fitter.py \
#     --model salt3 \
#     --outdir ./OUTDIR_SALT3_radec \
#     --datafile lc_ZTF19abrdxbh.dat \
#     --priors-dir ./priors \
#     --candname ZTF19abrdxbh \
#     --tmin -10 \
#     --tmax 20 \
#     --dt 0.5 \
#     --error-budget 0.1 \
#     --nlive 1024 \
#     --Ebv-max 0.5724 \
#     --bestfit \
#     --generation-seed 42 \
#     --local-only \
#     --cpus 4 \
#     --timeshift 30 \
#     --dry-run
#     --fetch-Ebv-from-dustmap \
#     --ra 222.7275966\
#     --dec 27.5825746


from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import yaml
from astropy.table import Table
from astropy.time import Time

from plot_bestfit import lc_fit

# ----------------------------- Logging setup ----------------------------- #


def configure_logging(verbosity: int) -> None:
    """Configuration of the the logging level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ----------------------------- Data classes ----------------------------- #


@dataclass
class InferenceOptions:
    # --- Inference options ---
    model: str
    outdir: Path
    candname: str
    label: str
    datafile: Path
    prior_file: Path
    tmin: float
    tmax: float
    dt: float
    error_budget: float
    nlive: int
    generation_seed: int
    bestfit: bool
    local_only: bool

    # --- Optional / defaulted ---
    cpus: Optional[int] = None
    trigger_time: Optional[float] = None
    sampler: Optional[str] = None

    # --- Extinction / Ebv options ---
    use_ebv: bool = False
    ebv_max: float = 0.5724
    fetch_ebv_from_dustmap: bool = False
    ra: Optional[float] = None
    dec: Optional[float] = None

    # --- Plot & misc ---
    plot: bool = False
    ylim: Optional[str] = None
    xlim: Optional[str] = None
    timeshift: Optional[float] = None


# ----------------------------- Core helpers ----------------------------- #


def resolve_prior_file(model: str, priors_dir: Path) -> Path:
    """Return the path to the appropriate prior file based on the model name."""
    m = model.lower()

    # Explicit cases
    if m in {
        "nugent-sn1a",
        "nugent-hyper",
        "nugent-sn2p",
        "nugent-sn2l",
        "nugent-sn2n",
        "nugent-sn1bc",
    }:
        return priors_dir / "sncosmo-generic.prior"

    if ("salt2" in m) or ("salt3" in m):
        return priors_dir / "salt3.prior"

    if ("sr2023" in m) and ("nugent-hyper" in m):
        return priors_dir / "Sr2023.prior"

    if ("nugent-hyper" in m) and ("trpi2018" in m):
        return priors_dir / "sncosmo-generic_TrPi2018.prior"

    # Fallback: use model name directly
    return priors_dir / f"{model}.prior"


def build_lightcurve_cmd(opts: InferenceOptions) -> List[str]:
    """Construct the `lightcurve-analysis` command as a list of arguments."""
    cmd: List[str] = [
        "lightcurve-analysis",
        "--model",
        opts.model,
        "--outdir",
        str(opts.outdir / opts.candname),
        "--data",
        str(opts.datafile),
        "--prior",
        str(opts.prior_file),
        "--label",
        opts.label,
        "--tmin",
        str(opts.tmin),
        "--tmax",
        str(opts.tmax),
        "--dt",
        str(opts.dt),
        "--error-budget",
        str(opts.error_budget),
        "--nlive",
        str(opts.nlive),
        "--generation-seed",
        str(opts.generation_seed),
    ]

    # --- Extinction logic ---
    if opts.fetch_ebv_from_dustmap:
        assert opts.ra is not None and opts.dec is not None, (
            "--fetch-Ebv-from-dustmap requires --ra and --dec"
        )
        cmd.append("--fetch-Ebv-from-dustmap")
        cmd += ["--ra", str(opts.ra), "--dec", str(opts.dec)]

    # When fetching from dustmap: do NOT add --use-Ebv/--Ebv-max
    elif opts.use_ebv:
        cmd.append("--use-Ebv")

    if opts.ebv_max is not None:
        cmd += ["--Ebv-max", str(opts.ebv_max)]
    if opts.bestfit:
        cmd.append("--bestfit")
    if opts.local_only:
        cmd.append("--local-only")
    if opts.cpus and opts.cpus > 0:
        cmd += ["--cpus", str(opts.cpus)]
    if opts.sampler:
        cmd += ["--sampler", opts.sampler]
    if opts.trigger_time is not None:
        cmd += ["--trigger-time", str(opts.trigger_time)]
    if opts.plot:
        cmd.append("--plot")
    if opts.ylim:
        cmd += ["--ylim", opts.ylim]
    if opts.xlim:
        cmd += [f"--xlim={opts.xlim}"]

    return cmd


def run_command(cmd: Iterable[str], *, dry_run: bool = False) -> int:
    """Execute the command. If `dry_run` is True, print it instead of running."""
    readable = " ".join(map(str, cmd))
    logging.info("Command: %s", readable)

    if dry_run:
        print(readable)
        return 0

    try:
        completed = subprocess.run(
            list(cmd), check=False, capture_output=True, text=True
        )
    except Exception as exc:
        logging.error("Execution failed: %s", exc)
        return 1

    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    return int(completed.returncode)


# ----------------------------- CLI parsing ----------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run light-curve parameter inference with a clean, extensible CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration file containing parameter values.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the command without executing it"
    )

    # Required core arguments
    req = parser.add_argument_group("Required")
    req.add_argument(
        "--datafile",
        type=Path,
        required=True,
        help="Path to the data file (CSV or photometry)",
    )
    req.add_argument(
        "--candname",
        type=str,
        required=True,
        help="Name or ID of the transient candidate",
    )

    # Model & Priors
    model_g = parser.add_argument_group("Model & Priors")
    model_g.add_argument(
        "--model",
        type=str,
        default="Bu2019lm",
        help="Model name (e.g., Bu2019lm, salt2)",
    )
    model_g.add_argument(
        "--priors-dir",
        type=Path,
        default=Path("./priors"),
        help="Directory of prior files",
    )

    model_g.add_argument(
        "--label", type=str, help="Label for the run", default="injection"
    )
    model_g.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory with {model}.joblib",
        default="svdmodels",
    )

    # Output
    out_g = parser.add_argument_group("Output")
    out_g.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("./outputs"),
        help="Base output directory",
    )

    # Inference settings
    inf_g = parser.add_argument_group("Inference Settings")
    inf_g.add_argument(
        "--tmin",
        type=float,
        default=0.1,
        help="Start time relative to reference (days)",
    )
    inf_g.add_argument(
        "--tmax", type=float, default=14.0, help="End time relative to reference (days)"
    )
    inf_g.add_argument("--dt", type=float, default=0.5, help="Time step (days)")
    inf_g.add_argument(
        "--error-budget", type=float, default=1, help="Photometric error budget in mag"
    )
    inf_g.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points for the sampler"
    )
    inf_g.add_argument("--generation-seed", type=int, default=42, help="Random seed")
    inf_g.add_argument(
        "--sampler",
        type=str,
        choices=["pymultinest", "dynesty", "ultranest"],
        default=None,
        help="Override default sampler",
    )
    inf_g.add_argument(
        "--timeshift",
        type=float,
        default=None,
        help="Days to extend tmax beyond data span (time shift)",
    )

    inf_g.add_argument(
        "--sample-over-Hubble",
        action="store_true",
        default=False,
        help="To sample over Hubble constant and redshift",
    )

    # --- Extinction / Ebv with dustmap ---
    # Fetch Ebv from the SFD dust map at (--ra, --dec) and use it as a fixed prior.
    # Overrides --use-Ebv if both are provided.
    ebv_g = parser.add_argument_group("Extinction (Ebv)")
    ebv_g.add_argument(
        "--fetch-Ebv-from-dustmap",
        action="store_true",
        default=False,
        help="Fetch Ebv at (--ra,--dec) from SFD dust map and use as fixed-value prior.",
    )
    ebv_g.add_argument("--ra", type=float, help="Right ascension for Ebv (degrees).")
    ebv_g.add_argument("--dec", type=float, help="Declination for Ebv (degrees).")

    # If --use-Ebv is enabled, define a prior on the host extinction parameter E(B–V).
    # Otherwise, fix Ebv = 0.0 (no dust extinction). The prior shape is linear between 0 and Ebv_max.
    ebv_g.add_argument(
        "--use-Ebv",
        action="store_true",
        default=False,
        help="Enable the use of host-galaxy extinction (Ebv) during inference.",
    )
    ebv_g.add_argument(
        "--Ebv-max",
        type=float,
        default=0.5724,
        help="Maximum allowed value for Ebv when --use-Ebv is active.",
    )

    # Flags
    flag_g = parser.add_argument_group("Flags")
    flag_g.add_argument(
        "--bestfit", action="store_true", help="Save/print best-fit results"
    )
    flag_g.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local computational resources",
    )

    # Performance & Misc
    perf_g = parser.add_argument_group("Performance & Misc")
    perf_g.add_argument("--cpus", type=int, default=None, help="Number of CPUs to use")
    perf_g.add_argument(
        "--trigger-time", type=float, default=None, help="Trigger time (MJD or seconds)"
    )
    perf_g.add_argument(
        "--svdmodels",
        type=Path,
        default=Path("/home/cough052/shared/NMMA/svdmodels"),
        help="Path to SVD models (useful for kilonova models)",
    )

    # Plot options
    plot_g = parser.add_argument_group("Plot Options")
    plot_g.add_argument(
        "--plot", action="store_true", help="Generate plots after analysis"
    )
    plot_g.add_argument(
        "--ylim",
        type=str,
        default="22,16",
        help="Upper and lower magnitude limit for light curve plot, e.g. '22,16'",
    )
    plot_g.add_argument(
        "--xlim",
        type=str,
        default="0,14",
        help="Start and end time for light curve plot, eg. '-10,30'",
    )

    # Interpolation type (for SVD models)
    interp_g = interp_g = parser.add_argument_group("Interpolation")
    interp_g.add_argument(
        "--interpolation-type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    return parser


def read_config(args):
    """Read YAML config and update args."""
    yaml_dict = yaml.safe_load(Path(args.config).read_text())
    for analysis_set in yaml_dict.keys():
        params = yaml_dict[analysis_set]
        for key, value in params.items():
            key = key.replace("-", "_")
            if key not in args:
                print(f"{key} not a known argument... please remove")
                exit()
            setattr(args, key, value)
    return args


def validate_environment(datafile: Path, priors_dir: Path, outdir: Path) -> None:
    """Validate file paths and ensure required dependencies are available."""
    if not datafile.exists():
        raise FileNotFoundError(f"--datafile not found: {datafile}")

    if not priors_dir.exists():
        raise FileNotFoundError(f"--priors-dir not found: {priors_dir}")

    if shutil.which("lightcurve-analysis") is None:
        raise RuntimeError("Executable 'lightcurve-analysis' not found in PATH.")

    outdir.mkdir(parents=True, exist_ok=True)


def get_tmax(args):
    """Compute tmax and tmax_axis from data span (+ timeshift)."""
    time_data = Table.read(args.datafile, format="ascii")
    t = Time(time_data[time_data.colnames[0]], format="isot")
    span = (t[-1] - t[0]).to_value("day")
    tmax = span - 0.1 + args.timeshift
    tmax_axis = span + 10
    logging.info(
        f"tmax = {tmax:.2f} d | tmax_axis = {tmax_axis:.2f} d (timeshift = {args.timeshift} d)"
    )
    return tmax, tmax_axis


# ----------------------------- Main entry ----------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        args = read_config(args)

    configure_logging(args.verbose)

    # Validate environment early
    try:
        validate_environment(args.datafile, args.priors_dir, args.outdir)
    except Exception as exc:
        logging.error(str(exc))
        return 2

    # Resolve priors
    prior_file = resolve_prior_file(args.model, args.priors_dir)
    if not prior_file.exists():
        logging.error("Prior file not found: %s", prior_file)
        return 2

    # Optional auto-extend tmax/xlim from data span
    if args.timeshift is not None:
        tmax, tmax_axis = get_tmax(args)
        args.tmax = tmax
        args.xlim = f"-10,{tmax_axis:.2f}"

    # Enforce E(B–V) flag coherence
    if args.fetch_Ebv_from_dustmap and args.use_Ebv:
        logging.warning(
            "Both --fetch-Ebv-from-dustmap and --use-Ebv given; using dustmap and ignoring --use-Ebv."
        )
        args.use_Ebv = False

    if args.fetch_Ebv_from_dustmap and (args.ra is None or args.dec is None):
        logging.error("--fetch-Ebv-from-dustmap requires both --ra and --dec.")
        return 2

    opts = InferenceOptions(
        model=args.model,
        outdir=args.outdir,
        candname=args.candname,
        label=f"{args.model}_{args.candname}",
        datafile=args.datafile,
        prior_file=prior_file,
        tmin=args.tmin,
        tmax=args.tmax,
        dt=args.dt,
        timeshift=args.timeshift,
        error_budget=args.error_budget,
        nlive=args.nlive,
        generation_seed=args.generation_seed,
        bestfit=args.bestfit,
        local_only=args.local_only,
        cpus=args.cpus,
        trigger_time=args.trigger_time,
        sampler=args.sampler,
        # Ebv options
        use_ebv=args.use_Ebv,
        ebv_max=args.Ebv_max,
        fetch_ebv_from_dustmap=args.fetch_Ebv_from_dustmap,
        ra=args.ra,
        dec=args.dec,
        # Plot
        plot=args.plot,
        ylim=args.ylim,
        xlim=args.xlim,
    )

    cmd = build_lightcurve_cmd(opts)
    rc = run_command(cmd, dry_run=args.dry_run)
    return rc

    if rc == 0 and not args.dry_run and not opts.plot:
        lc_fit(opts)

    if not opts.plot:
        lc_fit(opts)


if __name__ == "__main__":
    rc = main()
    raise SystemExit(rc)
