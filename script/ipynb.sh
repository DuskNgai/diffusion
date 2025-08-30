#! /bin/bash

#SBATCH --job-name=diffusion
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

jupyter notebook --no-browser --port=12345 --ip "*" --notebook-dir `pwd`

# ssh -N -f -L localhost:<LOCAL_PORT>:<NODE>:<REMOTE_PORT> root@id
# http://127.0.0.1:<LOCAL_PORT>
