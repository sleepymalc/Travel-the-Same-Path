#!/bin/bash

#SBATCH --job-name=data_generation

#SBATCH --mail-user=<username>@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb

#SBATCH --gres=gpu:1

#SBATCH --time=10-00:00:00

#SBATCH --account=<account>

#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate ecole

cd /home/pbb/Project/learn2branch/

python 02_generate_dataset.py -n 25 -j 24


/bin/hostname
sleep 60