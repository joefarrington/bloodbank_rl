#! /bin/bash/ -l

#$ -N pyomo-stochastic-sS
#$ -l h_rt=00:10:00
#$ -l mem=1G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py

