# Diffusion

This repository contains code related to Diffusion models and its extensions.
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

## Setup

First, download the codebase and the submodules:

```bash
git clone https://github.com/DuskNgai/diffusion.git -o diffusion && cd diffusion
git submodule update --init --recursive
```

Second, install the dependencies by manually installing them:

```bash
conda create -n diffusion python=3.11
conda activate diffusion
pip install torch torchvision
pip install accelerate deepspeed diffusers fvcore h5py ipykernel jupyterlab lightning matplotlib numpy omegaconf pandas rich scikit-learn scipy seaborn tensorboard timm
```

## Usage

We have a unified training command for all the experiments:

```bash
python train.py \
--config-file <PATH_TO_CONFIG_FILE> \
--num-gpus <NUM_GPUS> \
--num-nodes <NUM_NODES> \
<KEY_TO_MODIFY> <VALUE_TO_MODIFY>
```

It takes approximately 14 hours to train UNet on the CIFAR-10 dataset for 200 epochs with 64 batch size using 1 RTX 3090 GPUs.

For example, you can train a unet model on cifar dataset using rectified flow with following command:
```bash
python train.py \
--config-file diffusion/configuration/rf_cifar.yaml \
--num-gpus 1 \
--num-nodes 1 \
OUTPUT_DIR output/rf_velocity_unet_cifar \
DATASET.ROOT $PATH_TO_DATASET_ROOT
```

We recommend naming the configuration file and output directory with the following format:
```txt
Configuration file: <MODEL_NAME>_<PREDICTION_TYPE>_<DATASET_NAME>.yaml
Output directory: output/<MODEL_NAME>_<PREDICTION_TYPE>_<DATASET_NAME>
```
