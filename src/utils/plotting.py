"""Plots for analysing generated data."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from jetnet.utils import efps
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from tqdm import tqdm

from src.data.components import (
    calculate_jet_features,
    get_pt_of_selected_multiplicities,
    get_pt_of_selected_particles,
)


def apply_mpl_styles() -> None:
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#B6BFC3",
            "#3B515B",
            "#0271BB",
            "#E2001A",
        ]
    )
    mpl.rcParams["font.size"] = 15
    mpl.rcParams["patch.linewidth"] = 1.25


JETCLASS_FEATURE_LABELS = {
    "part_pt": "Particle $p_\\mathrm{T}$",
    "part_eta": "Particle $\\eta$",
    "part_phi": "Particle $\\phi$",
    "part_mass": "Particle $m$",
    "part_etarel": "Particle $\\eta^\\mathrm{rel}$",
    "part_dphi": "Particle $\\phi^\\mathrm{rel}$",
    "part_ptrel": "Particle $p_\\mathrm{T}^\\mathrm{rel}$",
    "part_d0val": "Particle $d_0$",
    "part_dzval": "Particle $d_z$",
    "part_d0err": "Particle $\\sigma_{d_0}$",
    "part_dzerr": "Particle $\\sigma_{d_z}$",
}

JET_FEATURE_LABELS = {
    "jet_pt": "Jet $p_\\mathrm{T}$",
    "jet_y": "Jet $y$",
    "jet_eta": "Jet $\\eta$",
    "jet_eta": "Jet $\\eta$",
    "jet_mrel": "Jet $m_\\mathrm{rel}$",
    "jet_m": "Jet $m$",
    "jet_phi": "Jet $\\phi$",
}

BINNINGS = {
    "part_d0val": np.linspace(-5, 5, 100),
    "part_dzval": np.linspace(-5, 5, 100),
    "part_charge": np.linspace(-3.5, 3.5, 8),
}


def plot_single_jets(
    data: np.ndarray,
    color: str = "#E2001A",
    save_folder: str = "logs/",
    save_name: str = "sim_jets",
) -> plt.figure:
    """Create a plot with 16 randomly selected jets from the data.

    Args:
        data (_type_): Data to plot.
        color (str, optional): Color of plotted point cloud. Defaults to "#E2001A".
        save_folder (str, optional): Path to folder where the plot is saved. Defaults to "logs/".
        save_name (str, optional): File_name for saving the plot. Defaults to "sim_jets".
    """
    mask_data = np.ma.masked_where(
        data[:, :, 0] == 0,
        data[:, :, 0],
    )
    mask = np.expand_dims(mask_data, axis=-1)
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 4)

    for i in tqdm(range(16)):
        ax = fig.add_subplot(gs[i])

        idx = np.random.randint(len(data))
        x_plot = data[idx, :, :2]  # .cpu()
        s_plot = np.abs(data[idx, :, 2])  # .cpu())
        s_plot[mask[idx, :, 0] < 0.0] = 0.0

        ax.scatter(*x_plot.T, s=5000 * s_plot, color=color, alpha=0.5)

        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

    plt.tight_layout()

    plt.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
    return fig


def prepare_data_for_plotting(
    data: list[np.ndarray],
    calculate_efps: bool = False,
    selected_particles: list[int] = [1, 3, 10],
    selected_multiplicities: list[int] = [20, 30, 40],
):
    """Calculate the features for plotting, i.e. the jet features, the efps, the pt of selected
    particles and the pt of selected multiplicities.

    Args:
        data (list of np.ndarray): list of data where data is in the shape
            (n_jets, n_particles, n_features) with features (pt, eta, phi)
            --> this allows to process data in batches. Will be concatenated
            in the output
        calculate_efps (bool, optional): If efps should be calculated. Defaults to False.
        selected_particles (list[int], optional): Selected particles. Defaults to [1,3,10].
        selected_multiplicities (list[int], optional): Selected multiplicities.
            Defaults to [20, 30, 40].

    Returns:
        np.ndarray : jet_data, shape (len(data), n_jets, n_features)
        np.ndarray : efps, shape (len(data), n_jets, n_efps)
        np.ndarray : pt_selected_particles, shape (len(data), n_selected_particles, n_jets)
        dict : pt_selected_multiplicities
    """

    jet_data = []
    efps_values = []
    pt_selected_particles = []
    pt_selected_multiplicities = []
    for count, data_temp in enumerate(data):
        jet_data_temp = calculate_jet_features(data_temp)
        efps_temp = []
        if calculate_efps:
            efps_temp = efps(data_temp)
        pt_selected_particles_temp = get_pt_of_selected_particles(data_temp, selected_particles)
        # TODO: should probably set the number of jets in the function call below?
        pt_selected_multiplicities_temp = get_pt_of_selected_multiplicities(
            data_temp, selected_multiplicities
        )

        jet_data.append(jet_data_temp)
        efps_values.append(efps_temp)
        pt_selected_particles.append(pt_selected_particles_temp)
        pt_selected_multiplicities.append(pt_selected_multiplicities_temp)

    new_dict = {}
    for count, i in enumerate(selected_multiplicities):
        new_dict[f"{count}"] = []

    for dicts in pt_selected_multiplicities:
        for count, dict_items_array in enumerate(dicts):
            new_dict[f"{count}"].append(np.array(dicts[dict_items_array]))

    for count, i in enumerate(new_dict):
        new_dict[i] = np.array(new_dict[i])

    return np.array(jet_data), np.array(efps_values), np.array(pt_selected_particles), new_dict


def plot_substructure(
    tau21_gen: np.array,
    tau32_gen: np.array,
    d2_gen: np.array,
    tau21_sim: np.array,
    tau32_sim: np.array,
    d2_sim: np.array,
    bins: int = 100,
    save_fig: bool = True,
    close_fig: bool = True,
    save_folder: str = None,
    save_name: str = None,
    model_name: str = "Model",
    simulation_name: str = "JetClass",
) -> None:
    """Plot the tau21, tau32 and d2 distributions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    hist_kwargs_sim = {
        "label": simulation_name,
        "histtype": "stepfilled",
        "alpha": 0.5,
        "density": True,
        "bins": bins,
    }
    hist_kwargs_model = {"label": model_name, "histtype": "step", "density": True, "bins": bins}
    ax1.hist(tau21_sim, **hist_kwargs_sim)
    ax1.hist(tau21_gen, **hist_kwargs_model)
    ax1.set_xlabel("$\\tau_{21}$")
    ax1.legend(loc="best", frameon=False)

    ax2.hist(tau32_sim, **hist_kwargs_sim)
    ax2.hist(tau32_gen, **hist_kwargs_model)
    ax2.set_xlabel("$\\tau_{32}$")
    ax2.legend(loc="best", frameon=False)

    ax3.hist(d2_sim, **hist_kwargs_sim)
    ax3.hist(d2_gen, **hist_kwargs_model)
    ax3.set_xlabel("$d_2$")
    ax3.legend(loc="best", frameon=False)

    fig.tight_layout()
    if save_fig:
        fig.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
        fig.savefig(f"{save_folder}{save_name}.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def plot_full_substructure(
    data_substructure_gen: np.array,
    data_substructure_sim: np.array,
    keys: np.array,
    bins: int = 100,
    model_name: str = "Model",
    simulation_name: str = "JetClass",
    save_fig: bool = True,
    close_fig: bool = True,
    save_folder: str = None,
    save_name: str = None,
) -> None:
    """Plot all substructure distributions."""
    fig, axs = plt.subplots(4, 4, figsize=(15, 20))

    for i, ax in enumerate(range(len(data_substructure_sim))):
        ax = axs.flatten()[i]
        ax.hist(
            data_substructure_sim[i],
            bins=bins,
            label=simulation_name,
            histtype="stepfilled",
            alpha=0.5,
            density=True,
        )
        ax.hist(
            data_substructure_gen[i],
            bins=bins,
            label=f"{model_name}",
            histtype="step",
            density=True,
        )
        ax.set_title(keys[i])
        if i == 0:
            ax.legend(loc="best", frameon=False)

    # plt.legend(loc="best", frameon=False)
    fig.tight_layout()
    if save_fig:
        fig.savefig(f"{save_folder}{save_name}.png", bbox_inches="tight")
        fig.savefig(f"{save_folder}{save_name}.pdf", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def plot_particle_features(
    data_sim: np.array,
    data_gen: np.array,
    mask_sim: np.array,
    mask_gen: np.array,
    feature_names: list,
    legend_label_sim: str = "Sim. data",
    legend_label_gen: str = "Gen. data",
    plot_path: str = None,
    also_png: bool = False,
):
    """Plot the particle features.

    Args:
        data_sim (np.array): Simulated particle data of shape (n_jets, n_particles, n_features)
        data_gen (np.array): Generated particle data of shape (n_jets, n_particles, n_features)
        mask_sim (np.array): Mask for simulated particle data of shape (n_jets, n_particles, 1)
        mask_gen (np.array): Mask for generated particle data of shape (n_jets, n_particles, 1)
        feature_names (list): List of feature names (as in the file, e.g. `part_etarel`)
        legend_label_sim (str, optional): Label for the simulated data. Defaults to "Sim. data".
        legend_label_gen (str, optional): Label for the generated data. Defaults to "Gen. data".
        plot_path (str, optional): Path to save the plot. Defaults to None. Which means
            the plot is not saved.
        also_png (bool, optional): If True, also save the plot as png. Defaults to False.
    """
    # plot the generated features and compare sim. data to gen. data
    nvars = data_sim.shape[-1]
    plot_cols = 3
    plot_rows = nvars // 3 + 1 * int(bool(nvars % 3))
    fig, ax = plt.subplots(plot_rows, plot_cols, figsize=(11, 2.8 * plot_rows))
    ax = ax.flatten()
    hist_kwargs = {"density": True}
    for i in range(data_sim.shape[-1]):
        feature_name = feature_names[i]
        values_sim = data_sim[:, :, i][mask_sim[:, :, 0] != 0].flatten()
        values_gen = data_gen[:, :, i][mask_gen[:, :, 0] != 0].flatten()
        # use same binning for both histograms
        _, bin_edges = np.histogram(np.concatenate([values_sim, values_gen]), bins=100)
        # use explicitly specified binning if exists, otherwise use the one from above
        hist_kwargs["bins"] = BINNINGS.get(feature_name, bin_edges)

        ax[i].hist(values_sim, label=legend_label_sim, alpha=0.5, **hist_kwargs)
        ax[i].hist(
            values_gen,
            label=legend_label_gen,
            histtype="step",
            **hist_kwargs,
        )
        ax[i].set_yscale("log")
        ax[i].set_xlabel(JETCLASS_FEATURE_LABELS.get(feature_name, feature_name))
    ax[2].legend(frameon=False)
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path)
        if also_png and plot_path.endswith(".pdf"):
            fig.savefig(plot_path.replace(".pdf", ".png"))
    plt.close(fig)


def plot_jet_features(
    jet_data_sim: np.array,
    jet_data_gen: np.array,
    jet_feature_names: list,
    legend_label_sim: str = "Sim. data",
    legend_label_gen: str = "Gen. data",
    plot_path: str = None,
    also_png: bool = False,
):
    """Plot the particle features.

    Args:
        jet_data_sim (np.array): Simulated jet data of shape (n_jets, n_features)
        jet_data_gen (np.array): Generated jet data of shape (n_jets, n_features)
        jet_feature_names (list): List of feature names (as in the file, e.g. `jet_pt`)
        legend_label_sim (str, optional): Label for the simulated data. Defaults to "Sim. data".
        legend_label_gen (str, optional): Label for the generated data. Defaults to "Gen. data".
        plot_path (str, optional): Path to save the plot. Defaults to None. Which means
            the plot is not saved.
        also_png (bool, optional): If True, also save the plot as png. Defaults to False.
    """
    # plot the generated features and compare sim. data to gen. data
    # nvars = data_sim.shape[-1]
    # plot_cols = 3
    # plot_rows = nvars // 3 + 1 * int(bool(nvars % 3))
    plot_rows = 3
    fig, ax = plt.subplots(plot_rows, 3, figsize=(11, 2.8 * plot_rows))
    ax = ax.flatten()
    hist_kwargs = {}
    for i in range(jet_data_sim.shape[-1]):
        values_sim = jet_data_sim[:, i]
        values_gen = jet_data_gen[:, i]
        _, bin_edges = np.histogram(np.concatenate([values_sim, values_gen]), bins=100)
        hist_kwargs["bins"] = bin_edges
        ax[i].hist(values_sim, label=legend_label_sim, alpha=0.5, **hist_kwargs)
        ax[i].hist(
            values_gen,
            label=legend_label_gen,
            histtype="step",
            **hist_kwargs,
        )
        ax[i].set_yscale("log")
        feature_name = jet_feature_names[i]
        ax[i].set_xlabel(JET_FEATURE_LABELS.get(feature_name, feature_name))
    ax[2].legend(frameon=False)
    fig.tight_layout()
    if plot_path is not None:
        fig.savefig(plot_path)
        if also_png and plot_path.endswith(".pdf"):
            fig.savefig(plot_path.replace(".pdf", ".png"))
    plt.close(fig)


def get_good_colours():
    """List of good colours.

    Returns
    -------
    list
        list with colours
    """

    color_list = [
        "#4477AA",  # blue
        "#228833",  # green
        "#993299",  # purple
        "#CD4909",  # red
        "#DF9D0A",  # yellow
        "#88c0d0",  # light blue
        "#946317",  # brown
        "#D4429A",  # pink
    ]
    return color_list * 10


def set_mpl_colours():
    """Overwrite the default matplotlib colour cycle.

    Afterwards, the colour cycle will be the one from `get_good_colours()`
    """
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=get_good_colours())
    # print("Overwriting matplotlib colour cycle")


def reset_mpl_colours():
    """Reset the colour cycler to the matplotlib defaults."""
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )


def get_good_linestyles(names=None):
    """Returns a list of good linestyles.

    Parameters
    ----------
    names : list or str, optional
        List or string of the name(s) of the linestyle(s) you want to retrieve, e.g.
        "densely dotted" or ["solid", "dashdot", "densely dashed"], by default None

    Returns
    -------
    list
        List of good linestyles. Either the specified selection or the whole list in
        the predefined order.

    Raises
    ------
    ValueError
        If `names` is not a str or list.
    """
    linestyle_tuples = {
        "solid": "solid",
        "densely dashed": (0, (5, 1)),
        "densely dotted": (0, (1, 1)),
        "densely dashdotted": (0, (3, 1, 1, 1)),
        "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
        "dotted": (0, (1, 1)),
        "dashed": (0, (5, 5)),
        "dashdot": "dashdot",
        "loosely dashed": (0, (5, 10)),
        "loosely dotted": (0, (1, 10)),
        "loosely dashdotted": (0, (3, 10, 1, 10)),
        "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
        "dashdotted": (0, (3, 5, 1, 5)),
        "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    }

    default_order = [
        "solid",
        "densely dotted",
        "densely dashed",
        "densely dashdotted",
        "densely dashdotdotted",
        "dotted",
        "dashed",
        "dashdot",
        # "loosely dotted",
        # "loosely dashed",
        # "loosely dashdotted",
        # "loosely dashdotdotted",
        "dashdotted",
        "dashdotdotted",
    ]
    if names is None:
        names = default_order * 3
    elif isinstance(names, str):
        return linestyle_tuples[names]
    elif not isinstance(names, list):
        raise ValueError("Invalid type of `names`, has to be a list of strings or a string.")
    return [linestyle_tuples[name] for name in names]


def unified_binning(*args, bins=100, bins_range=None):
    """Get unified binning of several arrays.

    *args : arrays
        Several arrays of which you want the unified binning
    bins : int , optional
        Number of bins, by default 100
    bins_range : tuple , optional
        Range (min, max) in which you want to have the bins.
    """
    _, bin_edges = np.histogram(
        np.hstack([*args]),
        bins=bins,
        range=bins_range,
    )
    return bin_edges


def decorate_ax(
    ax,
    yscale=1.3,
    text=None,
    text_line_spacing=1.2,
    text_font_size=12,
    draw_legend=False,
    indent=0.7,
    top_distance=1.2,
):
    """Helper function to decorate the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to decorate
    yscale : float, optional
        Factor by which the y-axis is scaled, by default 1.3
    text : str, optional
        Text to add to the plot, by default None
    text_line_spacing : float, optional
        Spacing between lines of text, by default 1.2
    text_font_size : int, optional
        Font size of the text, by default 12
    draw_legend : bool, optional
        Draw the legend with `frameon=False`, by default False
    indent : float, optional
        Horizontal indent, by default 0.7
    top_distance : float, optional
        Vertical indent, by default 1.2
    """
    PT = 1 / 72  # 1 point in inches

    # reset the y-axis limits (if they were changed before, it can happen
    # that the y-axis is not scaled correctly. especially it happens that ymin
    # becomes 0 even after setting logscale, which raises an error below as we
    # divide by ymin for logscale)
    if yscale != 1:
        ax.relim()
        ax.autoscale()

        # This weird order is necessary to allow for later
        # saving in logscaled y-axis
        if ax.get_yscale() == "log":
            ymin, _ = ax.get_ylim()
            ax.set_yscale("linear")
            _, ymax = ax.get_ylim()
            ax.set_yscale("log")
            yscale = (ymax / ymin) ** (yscale - 0.99)
        else:
            ymin, ymax = ax.get_ylim()

        # scale the y-axis to avoid overlap with text
        ax.set_ylim(top=yscale * (ymax - ymin) + ymin)

    if text is None:
        pass
    elif isinstance(text, str):
        # translation from the left side of the axes (aka indent)
        trans_indent = ScaledTranslation(
            indent * text_line_spacing * PT * text_font_size,
            0,
            ax.figure.dpi_scale_trans,
        )
        # translation from the top of the axes
        trans_top = ScaledTranslation(
            0,
            -top_distance * text_line_spacing * PT * text_font_size,
            ax.figure.dpi_scale_trans,
        )

        # add each line of the tag text to the plot
        for line in text.split("\n"):
            # fmt: off
            ax.text(0, 1, line, transform=ax.transAxes + trans_top + trans_indent, fontsize=text_font_size)  # noqa: E501
            trans_top += ScaledTranslation(0, -text_line_spacing * text_font_size * PT, ax.figure.dpi_scale_trans)  # noqa: E501
            # fmt: on
    else:
        raise TypeError("`text` attribute of the plot has to be of type `str`.")

    if draw_legend:
        ax.legend(frameon=False)
