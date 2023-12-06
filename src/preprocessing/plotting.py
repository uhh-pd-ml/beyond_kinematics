"""Plotting functions for preprocessing."""

import logging
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.plotting import get_good_linestyles

logger = logging.getLogger("plotting")
logging.basicConfig(level=logging.INFO)  # active logger

HIST_KWARGS = dict(bins=100, density=False, histtype="step", linewidth=2)

# specify binning for some features which are not well represented by the
# default binning
BINNINGS = {
    "part_d0val": np.linspace(-2.5, 2.5, 101),
    "part_dzval": np.linspace(-2.5, 2.5, 101),
    "part_d0err": np.linspace(-0.3, 1.5, 101),
    "part_dzerr": np.linspace(-0.3, 1.5, 101),
    "jet_nparticles": np.linspace(-0.5, 130.5, 132),
    "jet_nparticles_after_etarel_cut": np.linspace(-0.5, 130.5, 132),
    "jet_sdmass": np.linspace(0, 300, 101),
    "jet_pt": np.linspace(400, 1100, 103),
}
JET_TYPE_LABELS = {
    "QCD": "QCD",
    "Zqq": "$Z(\\rightarrow q{q})$",
    "Wqq": "$W(\\rightarrow q\\bar{q})$",
    "Tbqq": "$t(\\rightarrow bqq')$",
    "Tbl": "$t(\\rightarrow bl\\nu)$",
    "Hbb": "$H(\\rightarrow b\\bar{b})$",
    "Hcc": "$H(\\rightarrow c\\bar{c})$",
    "Hgg": "$H(\\rightarrow gg)$",
    "H4q": "$H(\\rightarrow 4q)$",
    "Hqql": "$H(\\rightarrow l\\nu qq')$",
}
LABELS = {
    "part_ptrel": "Particle $p_\\mathrm{T}^\\mathrm{rel}$",
    "jet_pt": "Jet $p_\\mathrm{T}$",
    "part_d0val": "Particle $d_0$ [mm]",
    "part_dzval": "Particle $d_z$ [mm]",
    "part_etarel": "Particle $\\eta^\\mathrm{rel}$",
    "part_phirel": "Particle $\\phi^\\mathrm{rel}$",
    "jet_nparticles": "Particle multiplicity",
    "jet_nparticles_after_etarel_cut": "Particle multiplicity",
    "jet_sdmass": "Jet softdrop mass [GeV]",
}

good_linestyles = get_good_linestyles()


def plot_h5file(h5files_dict: dict, output_dir, n_plot=70_000, also_png=False):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Plotting jet features")

    for i, (file_label, h5file) in enumerate(h5files_dict.items()):
        # np.random.seed(789)
        # idx_to_plot = np.sort(
        #     np.random.choice(
        #         np.arange(h5file["part_features"].shape[0]), n_plot, replace=False
        #     )
        # )
        n_jets_in_file = h5file["jet_features"].shape[0]
        if n_plot > n_jets_in_file or n_plot < 0:
            n_plot = n_jets_in_file
        logger.info(f"Number of jets in {file_label}: {n_jets_in_file:_}")
        logger.info(f"Using {n_plot:_} jets for plotting")

        logger.debug("Getting jet features")
        jet_features = h5file["jet_features"][:n_plot]
        logger.debug("Getting part features")
        part_features = h5file["part_features"][:n_plot]
        logger.debug("Getting part mask")
        part_mask = h5file["part_mask"][:n_plot]
        jet_features_names = h5file["jet_features"].attrs["names_jet_features"][:]

        if i == 0:
            n_jet_features = len(jet_features_names)
            logger.debug(h5file["labels"].attrs["names_labels"])
            plot_ncols = 3
            plot_nrows = n_jet_features // plot_ncols + 1
            fig, ax = plt.subplots(
                plot_nrows,
                plot_ncols,
                figsize=(15, plot_nrows * 4),
                layout="constrained",
            )
            ax = ax.flatten()
            jet_feat_plots_dict = {}
            for var in jet_features_names:
                figvar, axvar = plt.subplots(figsize=(5, 4))
                jet_feat_plots_dict[var] = {"fig": figvar, "ax": axvar}

        jet_type_masks = {
            label.split("_")[-1]: jet_features[:, 0]
            == list(h5file["labels"].attrs["names_labels"]).index(label)
            for label in h5file["labels"].attrs["names_labels"]
        }

        for j in tqdm(range(n_jet_features)):
            feature_name = h5file["jet_features"].attrs["names_jet_features"][j]
            for label, mask in jet_type_masks.items():
                hist_kwargs_tmp = {
                    **HIST_KWARGS,
                    "label": f"{JET_TYPE_LABELS.get(label, label)} {file_label}"
                    if j == 0
                    else None,
                    "linestyle": good_linestyles[i],
                }
                if feature_name in BINNINGS:
                    hist_kwargs_tmp["bins"] = BINNINGS[feature_name]
                ax[j].hist(jet_features[mask, j], **hist_kwargs_tmp)
                hist_kwargs_tmp["label"] = f"{JET_TYPE_LABELS.get(label, label)} {file_label}"
                jet_feat_plots_dict[feature_name]["ax"].hist(
                    jet_features[mask, j], **hist_kwargs_tmp
                )

            ax[j].set_xlabel(feature_name)
            ax[j].set_ylabel("Entries")

    for jet_feature in jet_feat_plots_dict:
        jet_feat_plots_dict[jet_feature]["ax"].set_xlabel(LABELS.get(jet_feature, jet_feature))
        jet_feat_plots_dict[jet_feature]["ax"].set_ylabel("Entries")

        jet_feat_plots_dict[jet_feature]["ax"].set_yscale("log")
        ymin, ymax = jet_feat_plots_dict[jet_feature]["ax"].get_ylim()
        yscale = (ymax / ymin) ** (1.6 - 0.99)
        jet_feat_plots_dict[jet_feature]["ax"].set_ylim(top=(ymax - ymin) * yscale + ymin)

        jet_feat_plots_dict[jet_feature]["ax"].legend(frameon=False, ncol=2, loc="upper right")
        jet_feat_plots_dict[jet_feature]["fig"].savefig(
            f"{output_dir}/jet_features_{jet_feature}.pdf", bbox_inches="tight"
        )
        if also_png:
            jet_feat_plots_dict[jet_feature]["fig"].savefig(
                f"{output_dir}/jet_features_{jet_feature}.png", bbox_inches="tight"
            )

    fig.legend(loc="outside upper right", ncol=len(h5files_dict) * 2, frameon=False)
    # fig.tight_layout()
    fig.savefig(f"{output_dir}/jet_features.pdf", bbox_inches="tight")
    if also_png:
        fig.savefig(f"{output_dir}/jet_features.png", bbox_inches="tight")

    # ---------------------------------------------
    logger.info("Plotting particle features")

    for i, (file_label, h5file) in enumerate(h5files_dict.items()):
        n_jets_in_file = h5file["jet_features"].shape[0]
        if n_plot > n_jets_in_file or n_plot < 0:
            n_plot = n_jets_in_file
        logger.info(f"Number of jets in {file_label}: {n_jets_in_file:_}")
        logger.info(f"Using {n_plot:_} jets for plotting")

        logger.debug("Getting jet features")
        jet_features = h5file["jet_features"][:n_plot]
        logger.debug("Getting part features")
        part_features = h5file["part_features"][:n_plot]
        logger.debug("Getting part mask")
        part_mask = h5file["part_mask"][:n_plot]
        jet_features_names = h5file["jet_features"].attrs["names_jet_features"][:]

        # setup figure
        if i == 0:
            part_features_names = h5file["part_features"].attrs["names_part_features"][:]
            n_part_features = len(part_features_names)
            logger.info(h5file["labels"].attrs["names_labels"])
            plot_ncols = 3
            plot_nrows = n_part_features // plot_ncols + 1
            fig_part, ax_part = plt.subplots(
                plot_nrows,
                plot_ncols,
                figsize=(15, plot_nrows * 4),
                layout="constrained",
            )
            ax_part = ax_part.flatten()
            part_feat_plots_dict = {}
            for var in part_features_names:
                figvar, axvar = plt.subplots(figsize=(5, 4))
                part_feat_plots_dict[var] = {"fig": figvar, "ax": axvar}

        masks_valid_and_jet_type = {
            label: np.logical_and(
                part_mask == 1,
                np.repeat(mask[:, None], h5file["part_mask"].shape[1], axis=1),
            )
            for label, mask in jet_type_masks.items()
        }

        logger.info("Looping over particle features")

        for j in tqdm(range(n_part_features)):
            feature_name = h5file["part_features"].attrs["names_part_features"][j]
            if i == 0:
                values = part_features[:, :, j][part_mask == 1]
                mean = np.mean(values)
                std = np.std(values)
                ax_part[j].set_title(f"mean: {mean:.3f}, std: {std:.3f}")

            for label, mask in masks_valid_and_jet_type.items():
                hist_kwargs_tmp = {
                    **HIST_KWARGS,
                    "label": f"{JET_TYPE_LABELS.get(label, label)} {file_label}"
                    if j == 0
                    else None,
                    "linestyle": good_linestyles[i],
                }
                if feature_name in BINNINGS:
                    hist_kwargs_tmp["bins"] = BINNINGS[feature_name]
                ax_part[j].hist(part_features[mask, j], **hist_kwargs_tmp)
                hist_kwargs_tmp["label"] = f"{JET_TYPE_LABELS.get(label, label)} {file_label}"
                part_feat_plots_dict[feature_name]["ax"].hist(
                    part_features[mask, j], **hist_kwargs_tmp
                )
            ax_part[j].set_yscale("log")
            ax_part[j].set_xlabel(feature_name)
            ax_part[j].set_ylabel("Entries")

    for part_feature in part_feat_plots_dict:
        part_feat_plots_dict[part_feature]["ax"].set_xlabel(LABELS.get(part_feature, part_feature))
        part_feat_plots_dict[part_feature]["ax"].set_ylabel("Entries")

        part_feat_plots_dict[part_feature]["ax"].set_yscale("log")
        ymin, ymax = part_feat_plots_dict[part_feature]["ax"].get_ylim()
        yscale = (ymax / ymin) ** (1.6 - 0.99)
        part_feat_plots_dict[part_feature]["ax"].set_ylim(top=(ymax - ymin) * yscale + ymin)

        part_feat_plots_dict[part_feature]["ax"].legend(frameon=False, ncol=2, loc="upper right")
        part_feat_plots_dict[part_feature]["fig"].savefig(
            f"{output_dir}/part_features_{part_feature}.pdf", bbox_inches="tight"
        )
        if also_png:
            part_feat_plots_dict[part_feature]["fig"].savefig(
                f"{output_dir}/part_features_{part_feature}.png", bbox_inches="tight"
            )

    fig_part.legend(loc="outside upper right", ncol=len(h5files_dict) * 2, frameon=False)
    # fig_part.tight_layout()
    fig_part.savefig(f"{output_dir}/part_features.pdf", bbox_inches="tight")
    if also_png:
        fig_part.savefig(f"{output_dir}/part_features.png", bbox_inches="tight")
