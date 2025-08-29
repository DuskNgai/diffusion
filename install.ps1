$ErrorActionPreference = "Stop"

$env_input = Read-Host "Enter conda environment name, or just press enter for default name 'diffusion': "
$env_name = if ($env_input -eq '') { "diffusion" } else { $env_input }

Write-Host "[1/4] Creating conda env '$env_name'..."
conda create -y -n "$env_name" python=3.11

Write-Host "[2/4] Activating environment..."
conda activate "$env_name"

Write-Host "[3/4] Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

Write-Host "[4/4] Installing project requirements..."
pip install -r requirements.txt

Write-Host "`n[✓] Setup complete. Activate environment with: conda activate $env_name"
