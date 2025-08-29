#!/bin/bash
set -e

read -p "Enter conda environment name, or just press enter for default name 'diffusion': " env_input
env_name="${env_input:-diffusion}"

echo "[1/4] Creating conda env '$env_name'..."
conda create -y -n "$env_name" python=3.11

echo "[2/4] Activating environment..."
conda activate "$env_name"

echo "[3/4] Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

echo "[4/4] Installing project requirements..."
pip install -r requirements.txt

echo "[✓] Setup complete. Activate environment with: conda activate $env_name"
