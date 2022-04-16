#!/bin/bash

#SBATCH --job-name=imitation

#SBATCH --mail-user=<uniqname>@umich.edu

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

source /home/<uniqname>/anaconda3/etc/profile.d/conda.sh
conda activate ecole

cd /home/<uniqname>/Project/learn2branch/

python 03_train_gnn_imitation.py -n 15 -m False -g 0

/bin/hostname
sleep 60