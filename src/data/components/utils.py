import energyflow as ef
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(
    x: np.ndarray, categories: list = [[0, 1, 2, 3, 4]], num_other_features: int = 4
) -> np.array:
    """One hot encode the jet type and leave the rest of the features as is
        Note: The one_hot encoded value is based on the position in the categories list not the value itself,
        e.g. categories: [0,3] results in the two one_hot encoded values [1,0] and [0,1]

    Args:
        x (np.ndarray): jet data with shape (num_jets, num_features) that contains the jet type in the first column
        categories (list, optional): List with values in x that should be one hot encoded. Defaults to [[0, 1, 2, 3, 4]].
        num_other_features (int, optional): Number of features in x that are not one hot encoded. Defaults to 4.

    Returns:
        np.array: one_hot_encoded jet data (num_jets, num_features) with feature length len(categories) + 3 (pt, eta, mass)
    """
    enc = OneHotEncoder(categories=categories)
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, num_other_features)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)


def mask_data(particle_data, jet_data, num_particles, variable_jet_sizes=True):
    """Splits particle data in data and mask If variable_jet_sizes is True, the returned data only
    contains jets with num_particles constituents.

    Args:
        particle_data (_type_): Particle Data (batch, particles, features)
        jet_data (_type_): _description_
        num_particles (_type_): _description_
        variable_jet_sizes (bool, optional): _description_. Defaults to True.

    Returns:
        x (): masked particle data
        mask (): mask
        particle_data (): modified particle data with 4 features (3 + mask)
        jet_data (): masked jet data
    """
    x = None
    mask = None
    if not variable_jet_sizes:
        # take only jets with num_particles constituents
        jet_mask = np.ma.masked_where(
            np.sum(particle_data[:, :, 3], axis=1) == num_particles,
            np.sum(particle_data[:, :, 3], axis=1),
        )
        masked_particle_data = particle_data[jet_mask.mask]

        jet_data = jet_data[jet_mask.mask]

        x = torch.Tensor(masked_particle_data[:, :, :3])
        mask = torch.Tensor(masked_particle_data[:, :, 3:])
        particle_data = masked_particle_data
        # print(
        #    f"Jets with {num_particles} constituents:",
        #    np.ma.count_masked(jet_mask),
        #    f"({np.round(np.ma.count_masked(jet_mask)/(np.ma.count_masked(jet_mask)+np.ma.count(jet_mask))*100,2)}%)",
        # )
        # print(
        #    f"Jets with less than {num_particles} constituents:",
        #    np.ma.count(jet_mask),
        #    f"({np.round(np.ma.count(jet_mask)/(np.ma.count_masked(jet_mask)+np.ma.count(jet_mask))*100,2)}%)",
        # )

    elif variable_jet_sizes:
        particle_data = particle_data[:, :num_particles, :]
        x = torch.Tensor(particle_data[:, :, :3])
        mask = torch.Tensor(particle_data[:, :, 3:])

    mask[mask > 0] = 1
    mask[mask < 0] = 0

    return x, mask, particle_data, jet_data


# normalize


def normalize_tensor(tensor, mean, std, sigma=5):
    """Normalisation of every tensor feature.

        tensor[..., i] = (tensor[..., i] - mean[i]) / (std[i] / sigma)
    Args:
        tensor (_type_): (batch, particles, features)
        mean (_type_): _description_
        std (_type_): _description_
        sigma (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """

    for i in range(len(mean)):
        tensor[..., i] = (tensor[..., i] - mean[i]) / (std[i] / sigma)
    return tensor


def inverse_normalize_tensor(tensor, mean, std, sigma=5):
    """Inverse normalisation of each feature of a tensor.

        tensor[..., i] = (tensor[..., i] * (std[i] / sigma)) + mean[i]

    Args:
        tensor (_type_): _description_
        mean (_type_): _description_
        std (_type_): _description_
        sigma (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    for i in range(len(mean)):
        tensor[..., i] = (tensor[..., i] * (std[i] / sigma)) + mean[i]
    return tensor


def calculate_jet_features(particle_data):
    """Calculate the jet_features by transforming jet constituents to p4s, summing up and
    transforming back to hadrodic coordinates. Phi_ref is 0. Mask in input particle_data is
    allowed.

    Args:
        particle_data (_type_): particle data, shape: [events, particles, features], features: [eta,phi,pt,(mask)]

    Returns:
        jet_data _type_: jet data, shape: [events, features], features: [pt,y,phi,m]
    """
    particle_data = particle_data[..., [2, 0, 1]]
    p4s = ef.p4s_from_ptyphims(particle_data)
    sum_p4 = np.sum(p4s, axis=-2)
    jet_data = ef.ptyphims_from_p4s(sum_p4, phi_ref=0)
    return jet_data


def get_pt_of_selected_particles(particle_data, selected_particles=[1, 3, 10]):
    """Return pt of selected particles.

    Args:
        particle_data (np.array): Particle data of shape (n_jets, n_particles, n_features)
            The particle features are assumed to be in the order (eta_rel, phi_rel, pt_rel)
        selected_particles (list, optional): _description_. Defaults to [1, 3, 10].

    Returns:
        np.array: Array of shape (n_selected_indices, n_jets) where array[i, :] represents
            the pT values of the selected_particles[i]'th particle (after sorting by
            pT)
    """
    # sort along pt_rel (third feature) and invert the ordering (largest to smallest)
    particle_data_sorted = np.sort(particle_data[:, :, 2])[:, ::-1]
    pt_selected_particles = []
    for selected_particle in selected_particles:
        pt_selected_particle = particle_data_sorted[:, selected_particle - 1]
        pt_selected_particles.append(pt_selected_particle)
    return np.array(pt_selected_particles)


def get_pt_of_selected_multiplicities(
    particle_data, selected_multiplicities=[10, 20, 30], num_jets=150
) -> dict:
    """Return pt of jets with selected particle multiplicities.

    Args:
        particle_data (np.ndarray): Particle data of shape (num_jets, num_particles, num_features)
        selected_multiplicities (list, optional): List of selected particle
            multiplicities. Defaults to [20, 30, 40].
        num_jets (int, optional): Number of jets to consider. Defaults to 150.

    Returns:
        dict: Dict containing {selected_multiplicity: pt_selected_multiplicity} pairs
            where pt_selected_multiplicity is a masked array of shape (num_jets, num_particles).
    """
    data = {}
    for count, selected_multiplicity in enumerate(selected_multiplicities):
        # with that we select particles that have the selected multiplicity or more
        # --> is this what we want?
        particle_data_temp = particle_data[:, :selected_multiplicity, :]
        # we have to test for pt_rel non-zero to check if a particle is masked
        # particles with eta_rel = 0 can actually have pt_rel != 0, so those would
        # be masked even though they are valid particles
        mask = np.ma.masked_where(
            np.count_nonzero(particle_data_temp[:, :, 0], axis=1) == selected_multiplicity,
            np.count_nonzero(particle_data_temp[:, :, 0], axis=1),
        )
        masked_particle_data = particle_data_temp[mask.mask]
        masked_pt = masked_particle_data[:num_jets, :, 2]
        data[f"{count}"] = masked_pt
    return data
