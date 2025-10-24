import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nmma.em.io import loadEvent
from nmma.em.model import create_light_curve_model_from_args
from nmma.em.utils import getFilteredMag


def _as_array_dict(d):
    return {k: np.asarray(v) for k, v in d.items()}


# DYI astro colibri color palette
purple = (164 / 255, 75 / 255, 213 / 255)
light_purple = (194 / 255, 141 / 255, 245 / 255)
dark_blue = (101 / 255, 113 / 255, 253 / 255)
blue = (0 / 255, 141 / 255, 232 / 255)
light_blue = (0 / 255, 203 / 255, 224 / 255)
green = (45 / 255, 205 / 255, 80 / 255)

CMAP = {
    "atlas o": "orange",
    "atlas c": "floralwhite",
    "asas-sn g": purple,
    "asas-sn V": light_purple,
    "ztf g": blue,
    "ztf G": blue,
    "ztf r": light_blue,
    "ztf R": light_blue,
    "ztf i": dark_blue,
    "ztf I": dark_blue,
}


# # DYI astro colibri color palette
# light_blue = (0 / 255, 203 / 255, 224 / 255)


#     # Second axis with date at lower x-axis
#     secaxx = ax.secondary_xaxis("bottom", functions=(mjd_to_date, mjd_to_mjd))
#     secaxx.xaxis.set_major_formatter(
#         DateFormatter("%Y-%m-%d")
#     )  # Set date format
#     plt.setp(secaxx.get_xticklabels(), rotation=20, horizontalalignment="right")

#     # Set MJD to upper x-axis
#     ax.xaxis.tick_top()
#     ax.set_xlabel("MJD")
#     ax.xaxis.set_label_position("top")
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
#     ax.xaxis.set_minor_locator(AutoMinorLocator())
#     ax.yaxis.set_minor_locator(AutoMinorLocator())

#     # Avoid scientific notation
#     ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

#     # Explicitly set tick parameters
#     ax.tick_params(
#         axis="x",
#         which="both",
#         bottom=True,
#         top=True,
#         direction="in",
#         length=6,
#         width=1,
#         colors="white",
#     )
#     secaxx.tick_params(
#         axis="x",
#         which="both",
#         bottom=True,
#         top=True,
#         direction="in",
#         length=6,
#         width=1,
#         colors="white",
#     )

#     # Reduce the font size of the top x-axis label
#     ax.xaxis.label.set_size(10)

#     # Plot style
#     query_color = light_blue
#     ax.set_title(title, color=query_color)
#     ax.set_facecolor((64 / 255, 75 / 255, 96 / 255))
#     fig.patch.set_facecolor((64 / 255, 75 / 255, 96 / 255))
#     ax.spines["bottom"].set_color("white")
#     ax.spines["top"].set_color("white")
#     ax.spines["right"].set_color("white")
#     ax.spines["left"].set_color("white")
#     ax.tick_params(which="both", axis="x", colors="white")
#     ax.tick_params(which="both", axis="y", colors="white")
#     ax.yaxis.label.set_color("white")
#     ax.xaxis.label.set_color("white")

#     ax.set_ylabel("magnitude")
#     ax.yaxis.label.set_size(12)
#     ax.invert_yaxis()
#     ax.grid(False)

#     # Insert Astro-Colibri logo
#     def open_image(path):
#         image = Image.open(path)
#         img_array = np.array(image)
#         return img_array

#     res = files(logo_pkg) / "colibri.png"
#     with as_file(res) as path:
#         logo = open_image(path)

#     # open_image(os.path.abspath(os.getcwd()) + "/logo/colibri.png")

#     image_xaxis = 0.88
#     image_yaxis = 0.837
#     image_width = 0.12
#     image_height = 0.12

#     ax_image = fig.add_axes(
#         [image_xaxis, image_yaxis, image_width, image_height]
#     )
#     ax_image.imshow(logo)
#     ax_image.axis("off")
#     ax.text(
#         0.96,
#         0.94,
#         "https://astro-colibri.com",
#         horizontalalignment="right",
#         fontsize=8,
#         color="white",
#         transform=plt.gcf().transFigure,
#     )

#     current_time = datetime.now(timezone.utc).strftime(
#         "created on %Y-%m-%d %H:%M:%S"
#     )
#     plt.text(
#         1.03,
#         0.5,
#         current_time,
#         transform=ax.transAxes,
#         fontsize=8,
#         color="white",
#         rotation=90,
#         verticalalignment="center",
#     )


def lc_fit(args):
    """
    Reproduce the light-curve figure by reloading saved results.
    Priority:
      1) {outdir}/{label}_bestfit_params.json (already has magnitudes + times)
      2) {outdir}/{label}_posterior_samples.dat (recompute magnitudes)
    Output: {outdir}/{label}_lightcurves.png
    """

    bestfit_json = os.path.join(
        args.outdir, args.candname, f"{args.label}_bestfit_params.json"
    )
    posterior_dat = os.path.join(
        args.outdir, args.candname, f"{args.label}_posterior_samples.dat"
    )
    plotName = os.path.join(args.outdir, args.candname, f"{args.label}_lightcurves.png")

    # --- load observed data ---
    try:
        data = loadEvent(args.datafile)
    except ValueError:
        with open(args.datafile) as f:
            data = json.load(f)
        for k in data.keys():
            data[k] = np.array(data[k])

    # trigger time
    if getattr(args, "trigger_time", None) is None:
        trigger_time = min(np.min(arr[:, 0]) for arr in data.values())
        print(f"trigger_time not provided -> using {trigger_time}")
    else:
        trigger_time = args.trigger_time

    # optionally remove non-detections
    if getattr(args, "remove_nondetections", False):
        for filt in list(data.keys()):
            idx = np.where(np.isfinite(data[filt][:, 2]))[0]
            data[filt] = data[filt][idx, :]
            if len(idx) == 0:
                del data[filt]

    # need at least one detection
    detection = False
    notallnan = False

    for filt in data.keys():
        if np.any(np.isfinite(data[filt][:, 2])):
            detection = True
        if np.any(np.isfinite(data[filt][:, 1])):
            notallnan = True
        if detection and notallnan:
            break

    if (not detection) or (not notallnan):
        raise ValueError("Need at least one detection to do fitting.")

    # error budget per filter
    if isinstance(args.error_budget, (float, int)):
        eb_list = [float(args.error_budget)]
    else:
        eb_list = [float(x) for x in str(args.error_budget).split(",")]
    filters_to_analyze = list(data.keys())
    error_budget = dict(zip(filters_to_analyze, eb_list * len(filters_to_analyze)))

    # --- recover best-fit and magnitudes ---
    have_json = os.path.isfile(bestfit_json)
    have_dat = os.path.isfile(posterior_dat)
    if not (have_json or have_dat):
        raise FileNotFoundError(
            f"Missing both posteriors files:\n - {bestfit_json}\n - {posterior_dat}"
        )

    bestfit_params = {}
    timeshift = 0.0
    mag = {}
    mag_all = None
    sample_times = None

    # Read and extract the bestfit_params
    if have_json:
        with open(bestfit_json, "r") as f:
            best = json.load(f)
        bestfit_params = {
            k: v
            for k, v in best.items()
            if k
            not in [
                "Magnitudes",
                "chi2_per_dof",
                "chi2_per_dof_per_filt",
                "Best fit index",
            ]
        }
        timeshift = float(bestfit_params.get("timeshift", 0.0))
        mag = _as_array_dict(best["Magnitudes"])
        sample_times = np.asarray(mag["bestfit_sample_times"])

        model_names, models, light_curve_model = create_light_curve_model_from_args(
            args.model,
            args,
            sample_times,
            filters=filters_to_analyze,
            sample_over_Hubble=False,
        )

        if len(models) > 1:
            _, mag_all = light_curve_model.generate_lightcurve(
                sample_times, bestfit_params, return_all=True
            )
            # Apply distance modulus
            if bestfit_params["luminosity_distance"] > 0:
                dm = 5.0 * np.log10(bestfit_params["luminosity_distance"] * 1e6 / 10.0)
                for ii in range(len(mag_all)):
                    for filt in mag_all[ii].keys():
                        mag_all[ii][filt] = np.asarray(mag_all[ii][filt]) + dm

                model_colors = cm.Spectral(np.linspace(0, 1, len(models)))[::-1]
        # else:
        #     # regenerate from posterior .dat
        #     sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)
        #     model_names, models, light_curve_model = create_light_curve_model_from_args(
        #         args.model, args, sample_times, filters=filters_to_analyze, sample_over_Hubble=False
        #     )
        #     posterior_samples = pd.read_csv(posterior_dat, header=0, delimiter=" ")
        #     best_idx = int(np.argmax(posterior_samples["log_likelihood"].to_numpy()))
        #     bestfit_params = {k: v[best_idx] for k, v in posterior_samples.to_dict(orient="list").items()}
        #     timeshift = float(bestfit_params.get("timeshift", 0.0))

        #     _, mag = light_curve_model.generate_lightcurve(sample_times, bestfit_params)
        #     if bestfit_params["luminosity_distance"] > 0:
        #         dm = 5.0 * np.log10(bestfit_params["luminosity_distance"] * 1e6 / 10.0)
        #         for filt in mag.keys():
        #             mag[filt] = np.asarray(mag[filt]) + dm
        #     mag["bestfit_sample_times"] = sample_times

        #     if len(models) > 1:
        #         _, mag_all = light_curve_model.generate_lightcurve(
        #             sample_times, bestfit_params, return_all=True
        #         )
        #         if bestfit_params.get("luminosity_distance", 0) > 0:
        #             for ii in range(len(mag_all)):
        #                 for filt in mag_all[ii].keys():
        #                     mag_all[ii][filt] = np.asarray(mag_all[ii][filt]) + dm

        # Generate the lightcurve

        # --- plotting (identical geometry/style/limits) ---)
        filters_plot = []
        for filt in filters_to_analyze:
            if filt not in data:
                continue
            samples = data[filt]
            t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            if len(t) == 0:
                continue
            filters_plot.append(filt)

    colors = cm.Spectral(np.linspace(0, 1, len(filters_plot)))[::-1]
    if mag_all is not None:
        model_colors = cm.Spectral(np.linspace(0, 1, len(mag_all)))[::-1]

    # figure layout
    wspace = 0.6
    hspace = 0.3
    lspace = 1.0
    bspace = 0.7
    trspace = 0.2
    hpanel = 2.25
    wpanel = 3.0
    ncol = 2
    nrow = int(np.ceil(len(filters_plot) / ncol))
    figsize = (
        1.5 * (lspace + wpanel * ncol + wspace * (ncol - 1) + trspace),
        1.5 * (bspace + hpanel * nrow + hspace * (nrow - 1) + trspace),
    )
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    fig.subplots_adjust(
        left=lspace / figsize[0],
        bottom=bspace / figsize[1],
        right=1.0 - trspace / figsize[0],
        top=1.0 - trspace / figsize[1],
        wspace=wspace / wpanel,
        hspace=hspace / hpanel,
    )

    if len(filters_plot) % 2:
        axes[-1, -1].axis("off")
    best_times = np.asarray(mag["bestfit_sample_times"])

    if len(filters_plot) % 2:
        axes[-1, -1].axis("off")

    cnt = 0
    for filt, color in zip(filters_plot, colors):
        cnt = cnt + 1

        # summary plot
        row = (cnt - 1) // ncol
        col = (cnt - 1) % ncol
        ax_sum = axes[row, col]
        # adding the ax for the Delta
        divider = make_axes_locatable(ax_sum)
        ax_delta = divider.append_axes("bottom", size="30%", sharex=ax_sum)

        # configuring ax_sum
        ax_sum.set_ylabel("AB magnitude", rotation=90)
        ax_delta.set_ylabel(r"$\Delta (\sigma)$")
        if cnt == len(filters_plot) or cnt == len(filters_plot) - 1:
            ax_delta.set_xlabel("Time [days]")
        else:
            ax_delta.set_xticklabels([])

        # plotting the best-fit lc and the data in ax1
        samples = data[filt]
        t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
        t -= trigger_time + timeshift
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

        idx = np.where(np.isfinite(sigma_y))[0]
        det_idx = idx
        ax_sum.errorbar(
            t[idx],
            y[idx],
            sigma_y[idx],
            fmt="o",
            color=color,
        )

        idx = np.where(~np.isfinite(sigma_y))[0]
        ax_sum.scatter(
            t[idx],
            y[idx],
            marker="v",
            color=color,
        )

        mag_plot = getFilteredMag(mag, filt)

        # calculating the chi2
        mag_per_data = np.interp(t[det_idx], best_times, mag_plot)
        diff_per_data = mag_per_data - y[det_idx]
        sigma_per_data = np.sqrt((sigma_y[det_idx] ** 2 + error_budget[filt] ** 2))
        chi2_per_data = diff_per_data**2
        chi2_per_data /= sigma_per_data**2
        chi2_total = np.sum(chi2_per_data)
        N_data = len(det_idx)

        # plot the mismatch between the model and the data
        ax_delta.scatter(t[det_idx], diff_per_data / sigma_per_data, color=color)
        ax_delta.axhline(0, linestyle="--", color="k")

        ax_sum.plot(
            best_times,
            mag_plot,
            color="coral",
            linewidth=3,
            linestyle="--",
        )

        if len(models) > 1:
            ax_sum.fill_between(
                best_times,
                mag_plot + error_budget[filt],
                mag_plot - error_budget[filt],
                facecolor="coral",
                alpha=0.2,
                label="combined",
            )
        else:
            ax_sum.fill_between(
                best_times,
                mag_plot + error_budget[filt],
                mag_plot - error_budget[filt],
                facecolor="coral",
                alpha=0.2,
            )

        if len(models) > 1:
            for ii in range(len(mag_all)):
                mag_plot = getFilteredMag(mag_all[ii], filt)
                ax_sum.plot(
                    best_times,
                    mag_plot,
                    color="coral",
                    linewidth=3,
                    linestyle="--",
                )
                ax_sum.fill_between(
                    best_times,
                    mag_plot + error_budget[filt],
                    mag_plot - error_budget[filt],
                    facecolor=model_colors[ii],
                    alpha=0.2,
                    label=models[ii].model,
                )

        ax_sum.set_title(
            f"{filt}: " + rf"$\chi^2 / d.o.f. = {round(chi2_total / N_data, 2)}$"
        )

        ax_sum.set_xlim([float(x) for x in args.xlim.split(",")])
        ax_sum.set_ylim([float(x) for x in args.ylim.split(",")])
        ax_delta.set_xlim([float(x) for x in args.xlim.split(",")])

    plt.savefig(plotName, bbox_inches="tight")
    plt.close()
