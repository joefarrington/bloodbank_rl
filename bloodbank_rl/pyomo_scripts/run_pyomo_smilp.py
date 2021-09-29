from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import subprocess

from pathlib import Path
import os
import sys

path_root = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(path_root))

from bloodbank_rl.environments.platelet_bankSR import PoissonDemandProviderSR
from bloodbank_rl.pyomo_models.model_constructors import (
    sS_PyomoModelConstructor,
    sQ_PyomoModelConstructor,
    sSaQ_PyomoModelConstructor,
    sSbQ_PyomoModelConstructor,
)
from bloodbank_rl.pyomo_models.stochastic_model_runner import PyomoModelRunner

log = logging.getLogger(__name__)

# Function to get git hash from https://stackoverflow.com/a/21901260
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    n_scenarios = cfg.n_scenarios
    model_constructor = globals()[cfg.model_constructor]
    model_constructor_params = OmegaConf.to_container(cfg.model_constructor_params)
    demand_provider = globals()[cfg.demand_provider]
    solver_string = cfg.solver_string
    solver_options = OmegaConf.to_container(cfg.solver_options)

    log.info(f"Git revision hash: {get_git_revision_hash()}")

    model_runner = PyomoModelRunner(
        model_constructor=model_constructor,
        model_constructor_params=model_constructor_params,
        n_scenarios=n_scenarios,
        demand_provider=demand_provider,
        solver_string=solver_string,
        solver_options=solver_options,
        log=log,
    )
    model_runner.solve_program()
    model_runner.construct_results_dfs()
    model_runner.check_outputs(".")
    model_runner.save_results(".")


if __name__ == "__main__":
    main()
