#! /bin/bash -l

#$ -N pyomo_stochastic_sSaQ
#$ -l h_rt=38:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py model_constructor=sSaQ_PyomoModelConstructor n_scenarios=20 +solver_options.TimeLimit=129600 model_constructor_params.weekly_policy=False