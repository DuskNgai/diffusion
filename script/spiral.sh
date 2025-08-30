# # Debugging flags
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5
# export PYTHONFAULTHANDLER=1

python train.py \
--config-file diffusion/configuration/edm_spiral.yaml \
MODULE.NOISE_SCHEDULER.PREDICTION_TYPE sample \
OUTPUT_DIR output/edm/sample_mlp_spiral \

python train.py \
--config-file diffusion/configuration/edm_spiral.yaml \
MODULE.NOISE_SCHEDULER.PREDICTION_TYPE epsilon \
OUTPUT_DIR output/edm/epsilon_mlp_spiral \

python train.py \
--config-file diffusion/configuration/rectified_flow_spiral.yaml \
OUTPUT_DIR output/rectified_flow/velocity_mlp_spiral \

python train.py \
--config-file diffusion/configuration/mean_flow_spiral.yaml \
OUTPUT_DIR output/mean_flow/velocity_mlp_spiral \
