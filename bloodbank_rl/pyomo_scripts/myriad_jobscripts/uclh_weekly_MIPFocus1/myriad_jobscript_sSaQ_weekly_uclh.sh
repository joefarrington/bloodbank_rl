#! /bin/bash -l

#$ -N pyomo_stochastic_sSaQ
#$ -l h_rt=38:00:00
#$ -l mem=4G
#$ -pe smp 8
#$ -cwd

conda activate optim && echo 'Conda environment activated' || echo 'Activating conda environment failed'
module load gurobi/8.1.1 && echo 'Gurobi loaded' || echo 'Loading Gurobi failed'
python3 run_pyomo_smilp.py model_constructor=sSaQ_PyomoModelConstructor n_scenarios=20 +solver_options.TimeLimit=129600 +solver_options.MIPFocus=1 model_constructor_params.weekly_policy=True model_constructor_params.initial_inventory.2=25 demand_provider=DFPyomoDemandProvider +demand_provider_kwargs=2015_2016_real_demand scenario_name_start=0 log_folder_name=sSaQ_uclh_weeklyMIPFocus1