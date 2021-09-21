import numpy as np
import pandas as pd

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm

from pathlib import Path
import os
import sys

path_root = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(path_root))

from bloodbank_rl.environments.platelet_bankSR import PoissonDemandProviderSR
import bloodbank_rl.pyomo_models.model_constructors_nonweekly as pyomo_mc


class PyomoModelRunner:
    def __init__(self, model_constructor, n_scenarios, t_max, a_max, demand_provider):
        self.model_constructor = model_constructor
        self.n_scenarios = n_scenarios
        self.t_max = t_max
        self.a_max = a_max
        self.demand_provider = demand_provider

    def scenario_creator(self, scenario_name):
        prov = self.demand_provider(seed=int(scenario_name))
        prov.reset()
        demand = {t: prov.generate_demand() for t in range(1, self.t_max + 1)}

        model = self.model_constructor(
            demand=demand, t_max=self.t_max, a_max=self.a_max
        ).build_model()

        # Telling it which decisions belong to first stage - for us this could be all our policy parameters
        # because we can't change them during a trajectory

        first_stage_params = self._get_first_stage_decision_params(model)

        sputils.attach_root_node(model, 0, first_stage_params)
        # If we don't specify, assume that all equally likely
        model._mpisppy_probability = 1.0 / self.n_scenarios
        return model

    def _get_first_stage_decision_params(self, model):

        if self.model_constructor.policy_parameters() == ["s", "S"]:
            return [model.s, model.S]
        elif self.model_constructor.policy_parameters() == ["s", "Q"]:
            return [model.s, model.Q]
        elif self.model_constructor.policy_parameters() == ["s", "S", "a", "Q"]:
            return [model.s, model.S, model.a, model.Q]
        elif self.model_constructor.policy_parameters() == ["s", "S", "b", "Q"]:
            return [model.s, model.S, model.b, model.Q]
        else:
            raise ValueError("Policy parameters not recognised")

    def solve_program(self):
        options = {"solver": "gurobi"}
        all_scenario_names = [
            f"{i+310}" for i in range(1, self.n_scenarios + 1)
        ]  # seed 5 used for example run in Excel, so add const

        self.ef = ExtensiveForm(
            options=options,
            all_scenario_names=all_scenario_names,
            scenario_creator=self.scenario_creator,
        )

        self.results = self.ef.solve_extensive_form()

        objval = self.ef.get_objective_value()

        return objval

    def construct_results_dfs(self):
        self.results_list = []
        for tup in self.ef.scenarios():
            scen = tup[0]
            print(f"Scenario {scen}")
            prov = self.demand_provider(seed=int(scen))
            prov.reset()
            demand = {t: prov.generate_demand() for t in range(1, self.t_max + 1)}
            model = tup[1]

            # Add common variables to output
            res_dicts = [
                {
                    "opening_inventory": [model.IssB[t, a]() for a in model.A],
                    "received": [model.X[t, a]() for a in model.A],
                    "demand": demand[t],
                    "DSSR": [model.DssR[t, a]() for a in model.A],
                    "wastage": model.W[t](),
                    "shortage": model.E[t](),
                    "closing inventory": [model.IssE[t, a]() for a in model.A],
                    "inventory position": model.IP[t](),
                    "order quantity": model.OQ[t](),
                }
                for t in model.T
            ]

            # Add policy paramters to results
            for res_dict, t in zip(res_dicts, model.T):
                for param in self.model_constructor.policy_parameters():
                    res_dict[f"{param}"] = eval(f"model.{param}[t]()")

            self.results_list.append(pd.DataFrame(res_dicts))
            print(f"Variable cost: {model.variable_cost()}")
            print(f"Holding cost: {model.holding_cost()}")
            print(f"Fixed cost: {model.fixed_cost()}")
            print(f"Wastage cost: {model.wastage_cost()}")
            print(f"Shortage cost: {model.shortage_cost()}")
            print("")

    def save_results(self):
        for i, df in enumerate(self.results_list):
            df.to_csv(f"scenario_{i}_output.csv")
