#! /bin/bash -l

#$ -N pyomo_stochastic_sQ
#$ -l h_rt=38:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py +experiment=simulated_demand_data/sSbQ_weekly