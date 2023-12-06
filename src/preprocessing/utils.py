"""Utils for loading the JetClass dataset."""
import logging

import awkward as ak
import h5py
import numpy as np
import uproot
import vector
from tqdm import tqdm

logger = logging.getLogger("utils")

vector.register_awkward()

# helper function from https://github.com/jet-universe/particle_transformer/blob/main/dataloader.py  # noqa: E501


def read_file(
    filepath,
    max_num_particles=128,
    particle_features=["part_pt", "part_eta", "part_phi", "part_energy"],
    jet_features=["jet_pt", "jet_eta", "jet_phi", "jet_energy"],
    labels=[
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ],
):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet.
        Jets with fewer particles will be zero-padded,
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded.
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded.
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded.
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_),
        y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features
            in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
            in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth labels
            in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype="float32"):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, : len(trunc)] = trunc
            return x

    table = uproot.open(filepath)["tree"].arrays()

    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
        }
    )
    table["part_pt"] = p4.pt
    table["part_eta"] = p4.eta
    table["part_phi"] = p4.phi

    x_particles = np.stack(
        [ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features],
        axis=1,
    )
    x_jets = np.stack(
        [ak.to_numpy(table[feature_name]).astype("float32") for feature_name in jet_features],
        axis=1,
    )
    y = np.stack([ak.to_numpy(table[label_name]).astype("int") for label_name in labels], axis=1)

    return x_particles, x_jets, y


def merge_files(files_to_merge: list, output_file_name: str) -> None:
    """Merge numpy files into a single numpy file.

    Args:
        files_to_merge (list): List of numpy files to merge.
        output_file_name (str): Name of the output file.
    """
    array_dict = {}
    array_metadata_mapping = {
        "part_features": "names_part_features",
        "part_mask": None,
        "jet_features": "names_jet_features",
        "labels": "names_labels",
    }

    for i, filename in enumerate(files_to_merge):
        logger.info(f"Reading file {i+1}/{len(files_to_merge)}: {filename}")
        npfile = np.load(filename)
        if i == 0:
            for key in array_metadata_mapping:
                array_dict[key] = [npfile[key]]
            for key in array_metadata_mapping.values():
                if key is not None:
                    array_dict[key] = npfile[key]
        else:
            for key in array_metadata_mapping:
                array_dict[key].append(npfile[key])

    # concatenate all arrays
    for key in array_metadata_mapping:
        array_dict[key] = np.concatenate(array_dict[key])
    # shuffle
    np.random.seed(553)
    permutation = np.random.permutation(len(array_dict["part_features"]))
    for key in array_metadata_mapping:
        logger.info(f"Shuffling array: {key}")
        array_dict[key] = array_dict[key][permutation]

    # save to h5 file
    logger.info(f"Saving to file: {output_file_name}")

    with h5py.File(output_file_name, "w") as f:
        for array_key, metadata_key in array_metadata_mapping.items():
            f.create_dataset(array_key, data=array_dict[array_key], dtype=np.float32)
            # add the metadata
            if metadata_key is not None:
                f[array_key].attrs.create(
                    metadata_key,
                    data=array_dict[metadata_key],
                    dtype=h5py.special_dtype(vlen=str),
                )


def calc_means_and_stds(filename: str):
    """Calculate means and standard deviations of the features in a h5 file.

    Args:
        filename (str): Name of the h5 file.

    Returns:
        part_features_means (1-d numpy.ndarray): Means of the particle features.
        part_features_stds (1-d numpy.ndarray): Standard deviations of the particle
            features.
        jet_features_means (1-d numpy.ndarray): Means of the jet features.
        jet_features_stds (1-d numpy.ndarray): Standard deviations of the jet features.
    """

    with h5py.File(filename, "r") as f:
        part_features = f["part_features"][:]
        part_features_names = f["part_features"].attrs["names_part_features"]
        mask_valid = f["part_mask"][:] != 0

        logger.info(100 * "-")
        logger.info("Calculating means and standard deviations of particle features")
        part_features_means = [
            np.mean(part_features[:, :, i][mask_valid]) for i in range(len(part_features_names))
        ]
        part_features_stds = [
            np.std(part_features[:, :, i][mask_valid]) for i in range(len(part_features_names))
        ]
        logger.info(100 * "-")
        for i, part_feature_name in enumerate(part_features_names):
            logger.info(
                f"{part_feature_name}: mean={part_features_means[i]}, std={part_features_stds[i]}"
            )
        logger.info(100 * "-")

        logger.info("Calculating means and standard deviations of jet features")
        jet_features = f["jet_features"][:]
        jet_features_names = f["jet_features"].attrs["names_jet_features"]
        jet_features_means = np.mean(jet_features, axis=0)
        jet_features_stds = np.std(jet_features, axis=0)

        logger.info(100 * "-")
        for i, jet_feature_name in enumerate(jet_features_names):
            logger.info(
                f"{jet_feature_name}: mean={jet_features_means[i]}, std={jet_features_stds[i]}"
            )
        logger.info(100 * "-")

    return (
        part_features_means,
        part_features_stds,
        jet_features_means,
        jet_features_stds,
    )


def standardize_data(
    filename_dict: dict,
    standardize_particle_features: bool = True,
    standardize_jet_features: bool = False,
) -> None:
    """Standardize the features in the h5 files.

    Args:
        filename_dict (dict): Dictionary with the filenames of the h5 files.
            Has to contain the keys "train", "val", and "test".
        standardize_particle_features (bool, optional): Whether to standardize the
            particle features. Defaults to True.
        standardize_jet_features (bool, optional): Whether to standardize the jet
            features. Defaults to False.
    """
    # calculate the means and stds of the training data
    part_features_means, part_features_stds, _, _ = calc_means_and_stds(filename_dict["train"])

    # standardize the data
    for split, filename in filename_dict.items():
        logger.info(f"Standardizing {split} data")
        filename_standardized = filename.replace(".h5", "_standardized.h5")
        with h5py.File(filename) as h5file:
            part_features = np.array(h5file["part_features"][:])
            part_mask = h5file["part_mask"][:] == 1
            if standardize_particle_features:
                for i in tqdm(range(len(part_features_means))):
                    part_features[:, :, i][part_mask] = (
                        part_features[:, :, i][part_mask] - part_features_means[i]
                    ) / part_features_stds[i]
            if standardize_jet_features:
                raise NotImplementedError(
                    """Standardization of jet features is not implemented yet."""
                )
            # save standardized data to h5 file
            logger.info(f"Saving to file: {filename_standardized}")
            with h5py.File(filename_standardized, "w") as h5file_std:
                dataset_names = list(h5file.keys())

                h5file_std.create_dataset("part_means", data=part_features_means)
                h5file_std.create_dataset("part_stds", data=part_features_stds)
                h5file_std["part_means"].attrs.create(
                    "names_part_means",
                    data=h5file["part_features"].attrs["names_part_features"],
                    dtype=h5py.special_dtype(vlen=str),
                )
                h5file_std["part_stds"].attrs.create(
                    "names_part_stds",
                    data=h5file["part_features"].attrs["names_part_features"],
                    dtype=h5py.special_dtype(vlen=str),
                )

                if standardize_particle_features:
                    h5file_std.create_dataset(
                        "part_features",
                        data=part_features,
                    )
                    h5file_std["part_features"].attrs.create(
                        "names_part_features",
                        data=h5file["part_features"].attrs["names_part_features"],
                        dtype=h5py.special_dtype(vlen=str),
                    )
                    dataset_names.remove("part_features")

                if standardize_jet_features:
                    # TODO: implement standardization of jet features
                    pass

                # copy remaining datasets and metadata from the original file
                for dataset_name in dataset_names:
                    h5file_std.create_dataset(dataset_name, data=h5file[dataset_name][:])

                    if dataset_name == "part_mask":
                        continue

                    h5file_std[dataset_name].attrs.create(
                        f"names_{dataset_name}",
                        data=h5file[dataset_name].attrs[f"names_{dataset_name}"],
                        dtype=h5py.special_dtype(vlen=str),
                    )
