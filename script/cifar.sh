# # Debugging flags
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5
# export PYTHONFAULTHANDLER=1

export TIMM_FUSED_ATTN=1

PATH_TO_DATASET_ROOT=

python train.py \
--config-file diffusion/configuration/mean_flow_cifar.yaml \
OUTPUT_DIR output/mean_flow/velocity_dit_cifar_unconditional \
DATAMODULE.DATASET.ROOT $PATH_TO_DATASET_ROOT \
MODEL.NUM_CLASSES 0

python train.py \
--config-file diffusion/configuration/edm_cifar.yaml \
OUTPUT_DIR output/edm/sample_dit_cifar_unconditional \
DATAMODULE.DATASET.ROOT $PATH_TO_DATASET_ROOT \
MODEL.NUM_CLASSES 0

python train.py \
--config-file diffusion/configuration/edm_cifar.yaml \
OUTPUT_DIR output/edm/epsilon_dit_cifar_unconditional \
DATAMODULE.DATASET.ROOT $PATH_TO_DATASET_ROOT \
MODULE.NOISE_SCHEDULER.PREDICTION_TYPE epsilon \
MODEL.NUM_CLASSES 0

python train.py \
--config-file diffusion/configuration/rectified_flow_cifar.yaml \
OUTPUT_DIR output/rectified_flow/velocity_dit_cifar_unconditional \
DATAMODULE.DATASET.ROOT $PATH_TO_DATASET_ROOT \
MODEL.NUM_CLASSES 0
