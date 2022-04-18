#!/bin/bash

#SBATCH --job-name=instances20

#SBATCH --mail-user=pbb@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=standard

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb

#SBATCH --time=3-00:00:00

#SBATCH --account=qmei3

#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate gnn

cd /home/pbb/Project/Test/

python 01_generate_instances.py

/bin/hostname
sleep 60