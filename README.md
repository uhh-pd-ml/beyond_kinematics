<div align="center">

# Flow Matching Beyond Kinematics: Generating Jets with Particle-ID and Trajectory Displacement Information

Joschka Birk, Erik Buhmann, Cedric Ewen, Gregor Kasieczka, David Shih

[![arXiv](https://img.shields.io/badge/arXiv-2312.00123-b31b1b.svg)](https://arxiv.org/abs/2312.00123)
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

## Description

This repository contains the code for the results presented in the paper
'[Flow Matching Beyond Kinematics: Generating Jets with Particle-ID and Trajectory Displacement Information](https://arxiv.org/abs/2312.00123)'.

<img src=assets/beyond_kinematics_jet.png width=600 style="border-radius:10px">

**Abstract**:

> We introduce the first generative model trained on the JetClass dataset. Our model generates jets at the constituent level, and it is a permutation-equivariant continuous normalizing flow (CNF) trained with the flow matching technique. It is conditioned on the jet type, so that a single model can be used to generate the ten different jet types of JetClass. For the first time, we also introduce a generative model that goes beyond the kinematic features of jet constituents. The JetClass dataset includes more features, such as particle-ID and track impact parameter, and we demonstrate that our CNF can accurately model all of these additional features as well. Our generative model for JetClass expands on the versatility of existing jet generation techniques, enhancing their potential utility in high-energy physics research, and offering a more comprehensive understanding of the generated jets.

## How to run the code

If you just want to generate new jets with our model (i.e. for comparisons with
future work), the required setup is minimal and doesn't need an installation
if you use the provided Docker image.

### Clone repository and setup environment

```shell
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name
```

Create a `.env` file in the root directory to set paths and API keys

```bash
LOG_DIR="<your-log-dir>"
COMET_API_TOKEN="<your-comet-api-token>"
HYDRA_FULL_ERROR=1
```

### Install dependencies

**Docker image**: You can use the Docker image `jobirk/pytorch-image:v0.2.2`, which contains all dependencies.

On a machine with singularity installed, you can run the following command to
convert the Docker image to a Singularity image and run it:

```shell
singularity shell --nv -B <your_directory_with_this_repo> docker://jobirk/pytorch-image:v0.2.2
```

To activate the conda environment, run the following inside the singularity container:

```shell
source /opt/conda/bin/activate
```

**Manual installation**: Alternatively, you can install the dependencies manually:

```shell
# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Generating new jets with our model

To generate new jets with the model used in our paper, you can simply
run the script `scripts/generate_jets.py`:

```shell
python scripts/generate_jets.py \
    --output_dir <path_to_output_dir> \
    --n_jets_per_type <number_of_jets_to_generate_per_type> \
    --types <list_of_jet_types>
```

### Dataset preparation

First, you have to download the JetClass dataset.
For that, please follow the instructions found in the repository
[`jet-universe/particle_transformer`](https://github.com/jet-universe/particle_transformer).

After downloading the dataset, you can preprocess it.
Adjust the corresponding input and output paths in `configs/preprocessing/data.yaml`.
Then, run the following:

```shell
python scripts/prepare_dataset.py && python scripts/preprocessing.py
```

### Model training

Once you have the dataset prepared, you can train the model.

Set the path to your preprocessed dataset directory in
`configs/experiment/jetclass_cond.yaml` and run the following:

```bash
python src/train.py experiment=jetclass_cond
```

The results should be logged to [comet](https://www.comet.com/site/) and stored
locally in the `LOG_DIR` directory specified in your `.env` file.

### Model evaluation

To evaluate the model, you can run the following:

```bash
python scripts/eval_ckpt.py \
    --ckpt=<model_checkpoint_path> \
    --n_samples=<number_of_jets_to_generate> \
    --cond_gen_file=<file_with_the_conditioning_features>
```

This will store the generated jets in a subdirectory `evaluated_checkpoints` of the
checkpoint directory.

### Classifier test

After evaluating the model, you can run the classifier test.
For that, you have to set the path to the directory pointing to the generated jets
in `configs/experiment/jetclass_classifier.yaml` and run the following.
However, if you want to load the pre-trained version of ParT, you will
have to adjust the corresponding paths to the checkpoints as well (which
you'll find in the
[`jet-universe/particle_transformer`](https://github.com/jet-universe/particle_transformer)
repository).

After you've done all that, you can run the classifier training with

```bash
python src/train.py experiment=jetclass_cond
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{birk2023flow,
      title={Flow Matching Beyond Kinematics: Generating Jets with Particle-ID and Trajectory Displacement Information},
      author={Joschka Birk and Erik Buhmann and Cedric Ewen and Gregor Kasieczka and David Shih},
      year={2023},
      eprint={2312.00123},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
}
```
