"""Callback for evaluating the model on the JetClass dataset."""

import os
import time
import warnings
from typing import Callable, Mapping, Optional

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
import vector
import wandb

from src.data.components import calculate_all_wasserstein_metrics
from src.data.components.metrics import wasserstein_distance_batched
from src.schedulers.logging_scheduler import (
    custom1,
    custom5000epochs,
    custom10000epochs,
    epochs10000,
    nolog10000,
)
from src.utils.data_generation import generate_data
from src.utils.jet_substructure import calc_substructure, load_substructure_data
from src.utils.plotting import (
    apply_mpl_styles,
    plot_full_substructure,
    plot_particle_features,
    plot_substructure,
)
from src.utils.pylogger import get_pylogger

from .ema import EMA

pylogger = get_pylogger("JetClassEvaluationCallback")

vector.register_awkward()


class JetClassEvaluationCallback(pl.Callback):
    """Create a callback to evaluate the model on the test dataset of the JetClass dataset and log
    the results to loggers. Currently supported are CometLogger and WandbLogger.

    Args:
        every_n_epochs (int, optional): Log every n epochs. Defaults to 10.
        additional_eval_epochs (list, optional): Log additional epochs. Defaults to [].
        num_jet_samples (int, optional): How many jet samples to generate.
            Negative values define the amount of times the whole dataset is taken,
            e.g. -2 would use 2*len(dataset) samples. Defaults to -1.
        image_path (str, optional): Folder where the images are saved. Defaults
            to "./logs/callback_images/".
        model_name (str, optional): Name for saving the model. Defaults to "model-test".
        log_times (bool, optional): Log generation times of data. Defaults to True.
        log_epoch_zero (bool, optional): Log in first epoch. Default to False.
        data_type (str, optional): Type of data to plot. Options are 'test' and 'val'.
            Defaults to "test".
        use_ema (bool, optional): Use exponential moving average weights for logging.
            Defaults to False.
        fix_seed (bool, optional): Fix seed for data generation to have better
            reproducibility and comparability between epochs. Defaults to True.
        w_dist_config (Mapping, optional): Configuration for Wasserstein distance
            calculation. Defaults to {'num_jet_samples': 10_000, 'num_batches': 40}.
        generation_config (Mapping, optional): Configuration for data generation.
            Defaults to {"batch_size": 256, "ode_solver": "midpoint", "ode_steps": 100}.
        plot_config (Mapping, optional): Configuration for plotting. Defaults to {}.
    """

    def __init__(
        self,
        every_n_epochs: int | Callable = 10,
        additional_eval_epochs: list[int] = None,
        num_jet_samples: int = -1,
        image_path: str = None,
        model_name: str = "model",
        log_times: bool = True,
        log_epoch_zero: bool = False,
        data_type: str = "val",
        use_ema: bool = False,
        fix_seed: bool = True,
        w_dist_config: Mapping = {
            "num_jet_samples": 10_000,
            "num_batches": 40,
        },
        generation_config: Mapping = {
            "batch_size": 256,
            "ode_solver": "midpoint",
            "ode_steps": 100,
        },
        plot_config: Mapping = {"plot_efps": False},
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.additional_eval_epochs = additional_eval_epochs
        self.num_jet_samples = num_jet_samples
        self.log_times = log_times
        self.log_epoch_zero = log_epoch_zero
        self.use_ema = use_ema
        self.fix_seed = fix_seed

        self.model_name = model_name
        self.data_type = data_type

        self.image_path = image_path
        apply_mpl_styles()

        self.w_dist_config = w_dist_config
        self.generation_config = generation_config
        self.plot_config = plot_config

        # loggers
        self.comet_logger = None
        self.wandb_logger = None

        # available custom logging schedulers
        self.available_custom_logging_scheduler = {
            "custom1": custom1,
            "custom5000epochs": custom5000epochs,
            "custom10000epochs": custom10000epochs,
            "nolog10000": nolog10000,
            "epochs10000": epochs10000,
        }

    def on_train_start(self, trainer, pl_module) -> None:
        # log something, so that metrics exists and the checkpoint callback doesn't crash
        self.log("w1m_mean", 0.005, sync_dist=True)
        self.log("w1p_mean", 0.005, sync_dist=True)

        if self.image_path is None:
            self.image_path = f"{trainer.default_root_dir}/plots/"
            os.makedirs(self.image_path, exist_ok=True)

        pylogger.info("Logging plots during training to %s", self.image_path)

        # set number of jet samples if negative
        if self.num_jet_samples < 0:
            self.datasets_multiplier = abs(self.num_jet_samples)
            if self.data_type == "test":
                self.num_jet_samples = len(trainer.datamodule.tensor_test) * abs(
                    self.num_jet_samples
                )
            if self.data_type == "val":
                self.num_jet_samples = len(trainer.datamodule.tensor_val) * abs(
                    self.num_jet_samples
                )
        else:
            self.datasets_multiplier = -1

        hparams_to_log = {
            "training_dataset_size": float(len(trainer.datamodule.tensor_train)),
            "validation_dataset_size": float(len(trainer.datamodule.tensor_val)),
            "test_dataset_size": float(len(trainer.datamodule.tensor_test)),
            "number_of_generated_val_jets": float(self.num_jet_samples),
        }
        # get loggers
        for logger in trainer.loggers:
            logger.log_hyperparams(hparams_to_log)
            if isinstance(logger, pl.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, pl.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        # get ema callback
        self.ema_callback = self._get_ema_callback(trainer)
        if self.ema_callback is None and self.use_ema:
            warnings.warn(
                "JetClass Evaluation Callbacks was told to use EMA weights, but EMA callback was"
                " not found. Using normal weights."
            )
        elif self.ema_callback is not None and self.use_ema:
            pylogger.info("Using EMA weights for evaluation.")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.fix_seed:
            # fix seed for better reproducibility and comparable results
            torch.manual_seed(9999)

        # Skip for all other epochs
        log_epoch = True
        if not self.log_epoch_zero and trainer.current_epoch == 0:
            log_epoch = False

        # determine if logging should happen
        log = False
        if type(self.every_n_epochs) is int:
            if trainer.current_epoch % self.every_n_epochs == 0 and log_epoch:
                log = True
        else:
            try:
                custom_logging_schedule = self.available_custom_logging_scheduler[
                    self.every_n_epochs
                ]
                log = custom_logging_schedule(trainer.current_epoch)
            except KeyError:
                raise KeyError("Custom logging schedule not available.")
        # log at additional epochs
        if self.additional_eval_epochs is not None:
            if trainer.current_epoch in self.additional_eval_epochs and log_epoch:
                log = True

        if log:
            time_eval_start = time.time()
            pylogger.info(f"Evaluating model after epoch {trainer.current_epoch}.")
            # Get background data for plotting and calculating Wasserstein distances
            # fmt: off
            if self.data_type == "test":
                pylogger.info("Using test data for evaluation.")
                background_data = np.array(trainer.datamodule.tensor_test)[: self.num_jet_samples]  # noqa: E501
                background_mask = np.array(trainer.datamodule.mask_test)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_test)[
                    : self.num_jet_samples
                ]
            elif self.data_type == "val":
                pylogger.info("Using validation data for evaluation.")
                background_data = np.array(trainer.datamodule.tensor_val)[: self.num_jet_samples]  # noqa: E501
                background_mask = np.array(trainer.datamodule.mask_val)[: self.num_jet_samples]
                background_cond = np.array(trainer.datamodule.tensor_conditioning_val)[
                    : self.num_jet_samples
                ]
            # fmt: on

            if trainer.datamodule.mask_gen is None:
                pylogger.info(
                    "No mask for generated data found. Using the same mask as for simulated data."
                )
                mask = background_mask
                cond = background_cond
            else:
                mask = trainer.datamodule.mask_gen[: self.num_jet_samples]
                cond = np.array(trainer.datamodule.tensor_conditioning_gen[: self.num_jet_samples])

            # maximum number of samples to plot is the number of samples in the dataset
            num_plot_samples = len(background_data)

            # Get EMA weights if available
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.replace_model_weights(pl_module)
            elif self.ema_callback and self.use_ema:
                warnings.warn("EMA Callback is not initialized. Using normal weights.")

            # Generate data
            data, generation_time = generate_data(
                model=pl_module,
                num_jet_samples=len(mask),
                cond=torch.tensor(cond),
                variable_set_sizes=trainer.datamodule.hparams.variable_jet_sizes,
                mask=torch.tensor(mask),
                normalized_data=trainer.datamodule.hparams.normalize,
                normalize_sigma=trainer.datamodule.hparams.normalize_sigma,
                means=trainer.datamodule.means,
                stds=trainer.datamodule.stds,
                **self.generation_config,
            )
            pylogger.info(f"Generated {len(data)} samples in {generation_time:.0f} seconds.")

            # If there are multiple jet types, plot them separately
            if trainer.datamodule.names_conditioning is not None:
                jet_types_dict = {
                    var_name.split("_")[-1]: i
                    for i, var_name in enumerate(trainer.datamodule.names_conditioning)
                    if "jet_type" in var_name and np.sum(background_cond[:, i] == 1) > 0
                }
            else:
                jet_types_dict = {}
            pylogger.info(f"Used jet types: {jet_types_dict.keys()}")

            plot_path_part_features = (
                f"{self.image_path}/particle_features_epoch_{trainer.current_epoch}.pdf"
            )
            plot_path_part_features_png = plot_path_part_features.replace(".pdf", ".png")
            plot_particle_features(
                data_sim=background_data,
                data_gen=data,
                mask_sim=background_mask,
                mask_gen=mask,
                feature_names=trainer.datamodule.names_particle_features,
                legend_label_sim="JetClass",
                legend_label_gen="Generated",
                plot_path=plot_path_part_features,
                also_png=True,
            )

            # only keep jets with at least three particles
            at_least_three_particles_gen = np.sum(np.array(mask[:, :, 0]) == 1, axis=1) >= 3
            data = data[at_least_three_particles_gen]
            mask = mask[at_least_three_particles_gen]
            cond = cond[at_least_three_particles_gen]
            at_least_three_particles_sim = (
                np.sum(np.array(background_mask[:, :, 0]) == 1, axis=1) >= 3
            )
            background_data = background_data[at_least_three_particles_sim]
            background_mask = background_mask[at_least_three_particles_sim]
            background_cond = background_cond[at_least_three_particles_sim]

            # print minimum of particles after filtering
            pylogger.info(
                "Minimum number of particles after filtering (sim):"
                f" {np.min(np.sum(np.array(background_mask[:, :, 0]) == 1, axis=1))}"
            )
            pylogger.info(
                "Minimum number of particles after filtering (gen):"
                f" {np.min(np.sum(np.array(mask[:, :, 0]) == 1, axis=1))}"
            )

            # fmt: off
            if self.comet_logger is not None:
                self.comet_logger.log_image(plot_path_part_features_png, name=f"epoch{trainer.current_epoch}_particle_features")  # noqa: E501
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"epoch{trainer.current_epoch}_particle_features": wandb.Image(plot_path_part_features_png)})  # noqa: E501

            # calculate substructure
            substructure_full_path = f"{self.image_path}/substructure_epoch_{trainer.current_epoch}_gen"  # noqa: E501
            substructure_full_path_jetclass = substructure_full_path.replace("_gen", "_sim")  # noqa: E501

            # first, create awkward arrays
            names_cond_features = list(trainer.datamodule.names_conditioning)
            names_part_features = list(trainer.datamodule.names_particle_features)
            idx_jet_pt = names_cond_features.index("jet_pt")
            idx_part_ptrel = names_part_features.index("part_ptrel")

            pt_sim = background_data[:, :, idx_part_ptrel] * background_cond[:, idx_jet_pt][:, None]
            # pt_gen = data[:, :, idx_part_ptrel] * cond[:, idx_jet_pt][:, None]
            # clipping the ptrel to be positive (with certain minimum value).
            # In the actual evaluation this is clipped to the min/max seen in the training data,
            # but since this is just for monitoring purposes, this should be fine
            ptrel_min = 1e-5
            pt_gen = np.clip(data[:, :, idx_part_ptrel], ptrel_min, None) * cond[:, idx_jet_pt][:, None] * np.array(mask[:, :, 0])

            idx_jet_eta = names_cond_features.index("jet_eta")
            idx_part_etarel = names_part_features.index("part_etarel")
            eta_gen = data[:, :, idx_part_etarel] + cond[:, idx_jet_eta][:, None]
            eta_sim = background_data[:, :, idx_part_etarel] + background_cond[:, idx_jet_eta][:, None]

            idx_part_dphi = names_part_features.index("part_dphi")
            dphi_gen = data[:, :, idx_part_dphi]
            dphi_sim = background_data[:, :, idx_part_dphi]
            # create awkward arrays
            particles_gen = ak.zip(
                {
                    "pt": pt_gen,
                    "eta": eta_gen,
                    "phi": dphi_gen,
                    "mass": np.zeros_like(pt_gen),
                },
                with_name="Momentum4D",
            )
            particles_sim = ak.zip(
                {
                    "pt": pt_sim,
                    "eta": eta_sim,
                    "phi": dphi_sim,
                    "mass": np.zeros_like(pt_sim),
                },
                with_name="Momentum4D",
            )
            # remove zero-padded entries
            particles_gen_mask = ak.mask(particles_gen, particles_gen.pt > 0)
            particles_sim_mask = ak.mask(particles_sim, particles_sim.pt > 0)
            particles_gen = ak.drop_none(particles_gen_mask)
            particles_sim = ak.drop_none(particles_sim_mask)

            # drop jets with less than 3 particles
            # particles_gen = particles_gen[ak.num(particles_gen) >= 3]
            # particles_sim = particles_sim[ak.num(particles_sim) >= 3]

            pylogger.info("Calculating substructure for generated data.")
            calc_substructure(
                particles_sim=particles_sim,
                particles_gen=particles_gen,
                R=0.8,
                filename=substructure_full_path + ".h5",
            )
            # fmt: on

            # load substructure data
            pylogger.info("Loading substructure data.")
            keys = ["tau1", "tau2", "tau3", "tau21", "tau32", "d2", "jet_mass", "jet_pt"]
            # load the substructure data
            data_substructure, data_substructure_jetclass = load_substructure_data(
                h5_file_path=substructure_full_path + ".h5",
                keys=keys,
            )

            tau21, tau32, d2, jet_mass, jet_pt = data_substructure[[3, 4, 5, 6, 7], :]
            (
                tau21_jetclass,
                tau32_jetclass,
                d2_jetclass,
                jet_mass_jetclass,
                jet_pt_jetclass,
            ) = data_substructure_jetclass[[3, 4, 5, 6, 7], :]

            # ---------------------------------------------------------------
            # using W1 distances for training monitoring
            # since at the beginning of training the two distributions sim/gen
            # won't have much overlap, the W1 distance is a better metric than
            # the KLD during training
            # ---------------------------------------------------------------
            pylogger.info("Calculating Wasserstein distances for substructure.")

            # Wasserstein distances
            # mass and particle features averaged
            w_dists = calculate_all_wasserstein_metrics(
                background_data, data, **self.w_dist_config
            )
            # substructure
            w_dist_config = {
                "num_eval_samples": self.w_dist_config["num_eval_samples"],
                "num_batches": self.w_dist_config["num_batches"],
            }
            w_dist_tau21_mean, w_dist_tau21_std = wasserstein_distance_batched(
                tau21_jetclass, tau21, **w_dist_config
            )
            w_dist_tau32_mean, w_dist_tau32_std = wasserstein_distance_batched(
                tau32_jetclass, tau32, **w_dist_config
            )
            w_dist_d2_mean, w_dist_d2_std = wasserstein_distance_batched(
                d2_jetclass, d2, **w_dist_config
            )
            self.log("w_dist_tau21_mean", w_dist_tau21_mean, sync_dist=True)
            self.log("w_dist_tau21_std", w_dist_tau21_std, sync_dist=True)
            self.log("w_dist_tau32_mean", w_dist_tau32_mean, sync_dist=True)
            self.log("w_dist_tau32_std", w_dist_tau32_std, sync_dist=True)
            self.log("w1m_mean", w_dists["w1m_mean"], sync_dist=True)
            self.log("w1p_mean", w_dists["w1p_mean"], sync_dist=True)
            self.log("w1m_std", w_dists["w1m_std"], sync_dist=True)
            self.log("w1p_std", w_dists["w1p_std"], sync_dist=True)

            if self.comet_logger is not None:
                text = (
                    f"W-Dist epoch:{trainer.current_epoch} "
                    f"W1m: {w_dists['w1m_mean']}+-{w_dists['w1m_std']}, "
                    f"W1p: {w_dists['w1p_mean']}+-{w_dists['w1p_std']}, "
                    f"W1efp: {w_dists['w1efp_mean']}+-{w_dists['w1efp_std']}"
                )
                self.comet_logger.log_text(text)

            for jet_type, jet_type_idx in jet_types_dict.items():
                jet_type_mask_sim = background_cond[:, jet_type_idx] == 1
                jet_type_mask_gen = cond[:, jet_type_idx] == 1
                path_part_feats_this_type = plot_path_part_features.replace(
                    ".pdf", f"_{jet_type}.pdf"
                )
                path_part_feats_this_type_png = path_part_feats_this_type.replace(".pdf", ".png")
                plot_particle_features(
                    data_sim=background_data[jet_type_mask_sim],
                    data_gen=data[jet_type_mask_gen],
                    mask_sim=background_mask[jet_type_mask_sim],
                    mask_gen=mask[jet_type_mask_gen],
                    feature_names=trainer.datamodule.names_particle_features,
                    legend_label_sim="JetClass",
                    legend_label_gen="Generated",
                    plot_path=path_part_feats_this_type,
                    also_png=True,
                )
                if self.comet_logger is not None:
                    self.comet_logger.log_image(
                        path_part_feats_this_type_png,
                        name=f"epoch{trainer.current_epoch}_particle_features_{jet_type}",
                    )  # noqa: E501
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            f"epoch{trainer.current_epoch}_particle_features_{jet_type}": wandb.Image(
                                path_part_feats_this_type_png
                            )
                        }
                    )
                # calculate the wasserstein distances for this jet type
                pylogger.info(f"Calculating Wasserstein distances for {jet_type} jets.")
                w_dists_tt = calculate_all_wasserstein_metrics(
                    background_data[jet_type_mask_sim],
                    data[jet_type_mask_gen],
                    **self.w_dist_config,
                )
                w_dist_tau21_mean_tt, w_dist_tau21_std_tt = wasserstein_distance_batched(
                    tau21_jetclass[jet_type_mask_sim],
                    tau21[jet_type_mask_gen],
                    **w_dist_config,
                )
                w_dist_tau32_mean_tt, w_dist_tau32_std_tt = wasserstein_distance_batched(
                    tau32_jetclass[jet_type_mask_sim],
                    tau32[jet_type_mask_gen],
                    **w_dist_config,
                )
                w_dist_d2_mean_tt, w_dist_d2_std_tt = wasserstein_distance_batched(
                    d2_jetclass[jet_type_mask_sim],
                    d2[jet_type_mask_gen],
                    **w_dist_config,
                )
                self.log(f"w_dist_tau21_mean_{jet_type}", w_dist_tau21_mean_tt, sync_dist=True)
                self.log(f"w_dist_tau21_std_{jet_type}", w_dist_tau21_std_tt, sync_dist=True)
                self.log(f"w_dist_tau32_mean_{jet_type}", w_dist_tau32_mean_tt, sync_dist=True)
                self.log(f"w_dist_tau32_std_{jet_type}", w_dist_tau32_std_tt, sync_dist=True)
                self.log(f"w1m_mean_{jet_type}", w_dists_tt["w1m_mean"], sync_dist=True)
                self.log(f"w1p_mean_{jet_type}", w_dists_tt["w1p_mean"], sync_dist=True)
                self.log(f"w1m_std_{jet_type}", w_dists_tt["w1m_std"], sync_dist=True)
                self.log(f"w1p_std_{jet_type}", w_dists_tt["w1p_std"], sync_dist=True)

                # plot substructure
                file_name_substructure = f"epoch{trainer.current_epoch}_subs_3plots_{jet_type}"
                file_name_full_substructure = f"epoch{trainer.current_epoch}_subs_full_{jet_type}"
                plot_substructure(
                    tau21_gen=tau21[jet_type_mask_gen],
                    tau32_gen=tau32[jet_type_mask_gen],
                    d2_gen=d2[jet_type_mask_gen],
                    tau21_sim=tau21_jetclass[jet_type_mask_sim],
                    tau32_sim=tau32_jetclass[jet_type_mask_sim],
                    d2_sim=d2_jetclass[jet_type_mask_sim],
                    save_fig=True,
                    save_folder=self.image_path,
                    save_name=file_name_substructure,
                    close_fig=True,
                    simulation_name="JetClass",
                    model_name="Generated",
                )
                plot_full_substructure(
                    data_substructure_gen=[
                        data_substructure[i][jet_type_mask_gen]
                        for i in range(len(data_substructure))
                    ],
                    data_substructure_sim=[
                        data_substructure_jetclass[i][jet_type_mask_sim]
                        for i in range(len(data_substructure_jetclass))
                    ],
                    keys=keys,
                    save_fig=True,
                    save_folder=self.image_path,
                    save_name=file_name_full_substructure,
                    close_fig=True,
                    simulation_name="JetClass",
                    model_name="Generated",
                )
                # upload image to comet
                img_path_3plots = f"{self.image_path}/{file_name_substructure}.png"
                img_path_full = f"{self.image_path}/{file_name_full_substructure}.png"
                if self.comet_logger is not None:
                    self.comet_logger.log_image(
                        img_path_3plots,
                        name=f"epoch{trainer.current_epoch}_substructure_3plots_{jet_type}",
                    )
                    self.comet_logger.log_image(
                        img_path_full,
                        name=f"epoch{trainer.current_epoch}_substructure_full_{jet_type}",
                    )

            # remove the additional particle features for compatibility with the rest of the code

            # Get normal weights back after sampling
            if (
                self.ema_callback is not None
                and self.ema_callback.ema_initialized
                and self.use_ema
            ):
                self.ema_callback.restore_original_weights(pl_module)

            # Prepare Data for Plotting

            time_eval_end = time.time()
            eval_time = time_eval_end - time_eval_start
            # Log jet generation time
            if self.log_times:
                if self.comet_logger is not None:
                    self.comet_logger.log_metrics({"Jet generation time": generation_time})
                    self.comet_logger.log_metrics({"Evaluation time": eval_time})
                if self.wandb_logger is not None:
                    self.wandb_logger.log({"Jet generation time": generation_time})
                    self.wandb_logger.log({"Evaluation time": eval_time})

        if self.fix_seed:
            torch.manual_seed(torch.seed())

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback
