#!/bin/bash

#SBATCH --job-name=imit20-0.5

#SBATCH --mail-user=pbb@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb

#SBATCH --gres=gpu:1

#SBATCH --time=10-00:00:00

#SBATCH --account=qmei3

#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate gnn

module load gcc/9.2.0
module load cuda/11.5.1

cd /home/pbb/Project/Test/

python 03_train_gnn_imitation.py -n 20 -g 0 

/bin/hostname
sleep 60