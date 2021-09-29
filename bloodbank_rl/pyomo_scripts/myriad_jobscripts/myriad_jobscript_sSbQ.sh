#! /bin/bash -l

#$ -N pyomo_stochastic_sSbQ
#$ -l h_rt=24:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py model_constructor=sSbQ_PyomoModelConstructor n_scenarios=10 +solver_options.mipgap=0.05