import numpy as np
import pandas as pd

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm

from omegaconf import DictConfig, OmegaConf
import hydra

from pathlib import Path
import os
import sys

path_root = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(path_root))

from bloodbank_rl.environments.platelet_bankSR import PoissonDemandProviderSR
from bloodbank_rl.pyomo_models.model_constructors_nonweekly import (
    sS_PyomoModelConstructor,
)
from bloodbank_rl.pyomo_models.stochastic_model_runner import PyomoModelRunner


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    t_max = cfg.t_max
    a_max = cfg.a_max
    n_scenarios = cfg.n_scenarios
    results_filepath_string = cfg.hydra_logdir

    model_runner = PyomoModelRunner(
        model_constructor=sS_PyomoModelConstructor,
        n_scenarios=n_scenarios,
        t_max=t_max,
        a_max=a_max,
        demand_provider=PoissonDemandProviderSR,
    )
    model_runner.solve_program()
    model_runner.construct_results_dfs()
    model_runner.save_results(results_filepath_string)


if __name__ == "__main__":
    main()
