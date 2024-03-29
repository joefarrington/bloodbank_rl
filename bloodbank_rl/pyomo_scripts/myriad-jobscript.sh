#! /bin/bash -l

#$ -N pyomo-stochastic-sS
#$ -l h_rt=01:00:00
#$ -l mem=8G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py

