#!/bin/bash

#SBATCH --job-name=imitation15

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

module load gcc/9.2.0
module load cuda/11.5.1

cd /home/<uniqname>/Project/learn2branch/

python 03_train_gnn_imitation.py -n 15 -g 0

/bin/hostname
sleep 60