"""Module with functions related to calculating jet substructure."""

import logging
import os

import awkward as ak
import fastjet
import h5py
import numpy as np
import vector

vector.register_awkward()

pylogger = logging.getLogger("jet_substructure")
logging.basicConfig(level=logging.INFO)


def calc_deltaR(particles, jet):
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)


class JetSubstructure:
    """Class to calculate and store the jet substructure variables.

    Definitions as in slide 7 here:
    https://indico.cern.ch/event/760557/contributions/3262382/attachments/1796645/2929179/lltalk.pdf
    """

    def __init__(self, particles, R=0.8, beta=1.0):
        """Run the jet clustering and calculate the substructure variables. The clustering is
        performed with the kt algorithm and the WTA pt scheme.

        Parameters
        ----------
        particles : awkward array
            The particles that are clustered into jets.
        R : float, optional
            The jet radius, by default 0.8
        beta : float, optional
            The beta parameter for N-subjettiness, by default 1.0
        """
        self.R = R
        self.beta = beta
        self.particles = particles
        self.particles_sum = ak.sum(particles, axis=1)
        self.jet_mass = self.particles_sum.mass
        self.jet_pt = self.particles_sum.pt
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme)
        print("Clustering jets with fastjet")
        print("Jet definition:", jetdef)
        self.cluster = fastjet.ClusterSequence(particles, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)

        print("Calculating N-subjettiness")
        self._calc_d0()
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        self.tau21 = self.tau2 / self.tau1
        self.tau32 = self.tau3 / self.tau2
        print("Calculating D2")
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = self.cluster.exclusive_jets_energy_correlator(njets=1, func="d2")

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.particles.pt * self.R**self.beta, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_1[:, :1])
        self.pt_i = self.particles.pt
        # calculate the tau1 values
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i**self.beta, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_2[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_2[:, 1:2])
        self.pt_i = self.particles.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_3[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 1:2])
        delta_r_3i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 2:3])
        self.pt_i = self.particles.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                    delta_r_3i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0


def calc_substructure(
    particles_sim,
    particles_gen,
    R=0.8,
    filename=None,
):
    """Calculate the substructure variables for the given particles and save them to a file.

    Parameters
    ----------
    particles_sim : awkward array
        The particles of the simulated jets.
    particles_gen : awkward array
        The particles of the generated jets.
    R : float, optional
        The jet radius, by default 0.8
    filename : str, optional
        The filename to save the results to, by default None (don't save)
    """
    if filename is None:
        print("No filename given, won't save the results.")
    else:
        if os.path.exists(filename):
            print(f"File {filename} already exists, won't overwrite.")
            return
        print(f"Saving results to {filename}")

    substructure_sim = JetSubstructure(particles_sim, R=R)
    substructure_gen = JetSubstructure(particles_gen, R=R)
    names = [
        "tau1",
        "tau2",
        "tau3",
        "tau21",
        "tau32",
        "d2",
        "jet_mass",
        "jet_pt",
    ]
    with h5py.File(filename, "w") as f:
        for name in names:
            f[f"{name}_sim"] = substructure_sim.__dict__[name]
            f[f"{name}_gen"] = substructure_gen.__dict__[name]


def load_substructure_data(
    h5_file_path,
    keys=["tau1", "tau2", "tau3", "tau21", "tau32", "d2", "jet_mass", "jet_pt"],
):
    """Load the substructure data from the h5 file.

    Args:
        h5_file_path (str): Path to the h5 file
        keys (list, optional): List of keys to load from the h5 file. Defaults to ["tau1", "tau2", "tau3", "tau21", "tau32", "d2", "jet_mass", "jet_pt"].

    Returns:
        data_gen: Array of shape (n_features, n_jets) with the substructure data for the generated jets
        data_jetclass: Array of shape (n_features, n_jets) with the substructure data for the JetClass jets
    """

    # load substructure for model generated data
    data_substructure = []
    data_substructure_jetclass = []
    with h5py.File(h5_file_path) as f:
        tau21 = np.array(f["tau21_gen"])
        tau32 = np.array(f["tau32_gen"])
        d2 = np.array(f["d2_gen"])
        jet_mass = np.array(f["jet_mass_gen"])
        jet_pt = np.array(f["jet_pt_gen"])
        tau21_isnan = np.isnan(tau21)
        tau32_isnan = np.isnan(tau32)
        d2_isnan = np.isnan(d2)
        if np.sum(tau21_isnan) > 0 or np.sum(tau32_isnan) > 0 or np.sum(d2_isnan) > 0:
            pylogger.warning(f"Found {np.sum(tau21_isnan)} nan values in tau21")
            pylogger.warning(f"Found {np.sum(tau32_isnan)} nan values in tau32")
            pylogger.warning(f"Found {np.sum(d2_isnan)} nan values in d2")
            pylogger.warning("Setting nan values to zero.")
        tau21[tau21_isnan] = 0
        tau32[tau32_isnan] = 0
        d2[d2_isnan] = 0
        for key in keys:
            data_substructure.append(np.array(f[key + "_gen"]))
        # set nan values in tau21, tau32 and d2 to zero
        data_substructure[keys.index("tau21")][tau21_isnan] = 0
        data_substructure[keys.index("tau32")][tau32_isnan] = 0
        data_substructure[keys.index("d2")][d2_isnan] = 0

        # load substructure for JetClass data
        tau21_jetclass = np.array(f["tau21_sim"])
        tau32_jetclass = np.array(f["tau32_sim"])
        d2_jetclass = np.array(f["d2_sim"])
        jet_mass_jetclass = np.array(f["jet_mass_sim"])
        jet_pt_jetclass = np.array(f["jet_pt_sim"])
        tau21_jetclass_isnan = np.isnan(tau21_jetclass)
        tau32_jetclass_isnan = np.isnan(tau32_jetclass)
        d2_jetclass_isnan = np.isnan(d2_jetclass)
        if (
            np.sum(tau21_jetclass_isnan) > 0
            or np.sum(tau32_jetclass_isnan) > 0
            or np.sum(d2_jetclass_isnan) > 0
        ):
            pylogger.warning(f"Found {np.sum(tau21_jetclass_isnan)} nan values in tau21")
            pylogger.warning(f"Found {np.sum(tau32_jetclass_isnan)} nan values in tau32")
            pylogger.warning(f"Found {np.sum(d2_jetclass_isnan)} nan values in d2")
            pylogger.warning("Setting nan values to zero.")
        tau21_jetclass[tau21_jetclass_isnan] = 0
        tau32_jetclass[tau32_jetclass_isnan] = 0
        d2_jetclass[d2_jetclass_isnan] = 0
        for key in keys:
            data_substructure_jetclass.append(np.array(f[key + "_sim"]))
        # set nan values in tau21, tau32 and d2 to zero
        data_substructure_jetclass[keys.index("tau21")][tau21_jetclass_isnan] = 0
        data_substructure_jetclass[keys.index("tau32")][tau32_jetclass_isnan] = 0
        data_substructure_jetclass[keys.index("d2")][d2_jetclass_isnan] = 0

    data_substructure = np.array(data_substructure)
    data_substructure_jetclass = np.array(data_substructure_jetclass)
    return (
        data_substructure,
        data_substructure_jetclass,
    )
