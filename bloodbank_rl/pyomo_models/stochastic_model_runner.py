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

        self.all_scenario_names = [f"{i+310}" for i in range(1, self.n_scenarios + 1)]

        self.checks_to_perform = self._determine_checks_to_perform()

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
        # seed 5 used for example run in Excel, so add const

        self.ef = ExtensiveForm(
            options=options,
            all_scenario_names=self.all_scenario_names,
            scenario_creator=self.scenario_creator,
        )

        self.results = self.ef.solve_extensive_form()

        objval = self.ef.get_objective_value()

        return objval

    def construct_results_dfs(self):
        self.results_list = []
        self.costs_df = pd.DataFrame(
            columns=[
                "Seed",
                "Variable cost",
                "Holding cost",
                "Fixed cost",
                "Wastage cost",
                "Shortage cost",
            ]
        )
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
                    "opening_inventory": [
                        round(model.IssB[t, a](), 0) for a in model.A
                    ],
                    "received": [round(model.X[t, a](), 0) for a in model.A],
                    "demand": round(demand[t], 0),
                    "DSSR": [round(model.DssR[t, a](), 0) for a in model.A],
                    "wastage": round(model.W[t](), 0),
                    "shortage": round(model.E[t](), 0),
                    "closing inventory": [
                        round(model.IssE[t, a](), 0) for a in model.A
                    ],
                    "inventory position": round(model.IP[t](), 0),
                    "order quantity": round(model.OQ[t](), 0),
                }
                for t in model.T
            ]

            # Add policy paramters to results
            for res_dict, t in zip(res_dicts, model.T):
                for param in self.model_constructor.policy_parameters():
                    res_dict[f"{param}"] = round(eval(f"model.{param}[t]()"), 0)

            self.results_list.append(pd.DataFrame(res_dicts))

            # Record the costs for each scenario and store in a single Pandas DataFrame
            scen_costs_dict = {
                "Seed": scen,
                "Variable cost": round(model.variable_cost(), 0),
                "Holding cost": round(model.holding_cost(), 0),
                "Fixed cost": round(model.fixed_cost(), 0),
                "Wastage cost": round(model.wastage_cost(), 0),
                "Shortage cost": round(model.shortage_cost(), 0),
            }

            self.costs_df = self.costs_df.append(scen_costs_dict, ignore_index=True)

            # For now, also print the costs as useful for debugging
            print(f"Variable cost: {round(model.variable_cost(),0)}")
            print(f"Holding cost: {round(model.holding_cost(),0)}")
            print(f"Fixed cost: {round(model.fixed_cost(),0)}")
            print(f"Wastage cost: {round(model.wastage_cost(),0)}")
            print(f"Shortage cost: {round(model.shortage_cost(),0)}")
            print("")

    def save_results(self, directory_path_string):
        for scen, df in zip(self.all_scenario_names, self.results_list):
            filename = Path(directory_path_string) / f"scenario_{scen}_output.csv"
            df.to_csv(filename)

        filename = Path(directory_path_string) / f"all_costs.csv"
        self.costs_df.to_csv(filename)

    def check_outputs(self, directory_path_string):
        self.results_of_checks_list = []
        for scen, scenario_df in zip(self.all_scenario_names, self.results_list):

            # Ensure that entries in columns with array values are numpy arrays
            array_cols = ["opening_inventory", "received", "DSSR", "closing inventory"]
            for col in array_cols:
                scenario_df[f"{col}"] = scenario_df[f"{col}"].apply(
                    lambda x: np.array(x)
                )

            # Do a merge to easily run checks where we look at consecutive rows
            merged_results = pd.concat(
                [
                    scenario_df,
                    scenario_df.loc[:, ["opening_inventory", "received"]]
                    .shift(-1)
                    .add_prefix("next_"),
                ],
                axis=1,
            )

            # Run the necessary checks
            out_df = pd.DataFrame()
            for f in self.checks_to_perform:
                res = merged_results.apply(f, axis=1)
                out_df = pd.concat([out_df, res], axis=1)

            # Print the number of rows with failure and store
            # the results if any failures for a scenario
            fail_check_rows = out_df[~out_df.all(axis=1)]
            n_rows_with_fail = fail_check_rows.shape[0]
            print(f"Scenario {scen}: {n_rows_with_fail} rows with a failed check")
            if n_rows_with_fail > 0:
                filename = Path(directory_path_string) / f"scenario_{scen}_checks.csv"
                out_df.to_csv(filename)

            self.results_of_checks_list.append(out_df)

    ### Functions for checking the output is consistent with constraints ###
    # TODO: Could run a check that policy params same in each scenario

    def _determine_checks_to_perform(self):
        checks_to_run = [
            self._check_wastage,
            self._check_shortage,
            self._check_inventory_during_day,
            self._check_no_max_age_opening_inventory,
            self._check_close_to_next_open_inventory,
            self._check_order_to_next_received,
        ]
        if self.model_constructor.policy_parameters() == ["s", "S"]:
            return checks_to_run + [self._check_sS]
        elif self.model_constructor.policy_parameters() == ["s", "Q"]:
            return checks_to_run + [self._check_sQ]
        elif self.model_constructor.policy_parameters() == ["s", "S", "a", "Q"]:
            return checks_to_run + [self._check_sSaQ]
        elif self.model_constructor.policy_parameters() == ["s", "S", "b", "Q"]:
            return checks_to_run + [self._check_sSbQ]
        else:
            raise ValueError("Policy parameters not recognised")

    # High level wastage check
    def _check_wastage(self, row):
        return pd.Series(
            {
                "check_wastage": row["wastage"]
                == max(
                    0, row["opening_inventory"][0] + row["received"][0] - row["demand"]
                )
            }
        )

    # High level shortage check
    def _check_shortage(self, row):
        return pd.Series(
            {
                "check_shortage": row["shortage"]
                == max(
                    0,
                    row["demand"]
                    - row["opening_inventory"].sum()
                    - row["received"].sum(),
                )
            }
        )

    # Check closing inventory
    def _calculate_remaining_stock_and_demand(self, row):
        total_remaining_demand = row["demand"]
        inventory = row["opening_inventory"] + row["received"]
        remaining_demand = np.zeros_like(inventory)
        for idx, stock in enumerate(inventory):
            demand_filled = min(total_remaining_demand, stock)
            remaining_stock = stock - demand_filled
            total_remaining_demand = total_remaining_demand - demand_filled
            inventory[idx] = remaining_stock
            remaining_demand[idx] = total_remaining_demand

        return inventory, remaining_demand

    def _check_inventory_during_day(self, row):

        (
            calc_closing_inventory,
            calc_remaining_demand,
        ) = self._calculate_remaining_stock_and_demand(row)

        return pd.Series(
            {
                "check_closing_inventory": (
                    row["closing inventory"] == calc_closing_inventory
                ).all(),
                "check_DSSR": (row["DSSR"] == calc_remaining_demand).all(),
                "check_inventory_position": row["inventory position"]
                == row["closing inventory"][1:].sum(),
            }
        )

    def _check_no_max_age_opening_inventory(self, row):
        return pd.Series(
            {"check_no_max_age_opening_inventory": row["opening_inventory"][-1] == 0}
        )

    def _check_close_to_next_open_inventory(self, row):
        if row["next_opening_inventory"] is np.nan:
            return pd.Series({"check_close_to_next_open_inventory": None})
        else:
            return pd.Series(
                {
                    "check_close_to_next_open_inventory": (
                        row["closing inventory"][1:]
                        == row["next_opening_inventory"][:-1]
                    ).all()
                }
            )

    def _check_order_to_next_received(self, row):
        if row["next_received"] is np.nan:
            return pd.Series({"check_order_to_next_received": None})
        else:
            return pd.Series(
                {
                    "check_order_to_next_received": row["order quantity"]
                    == row["next_received"].sum()
                }
            )

    def _check_sS(self, row):

        S_gt_s = row["S"] >= row["s"] + 1

        if row["inventory position"] < row["s"]:
            order_quantity_to_params = (
                row["order quantity"] == row["S"] - row["inventory position"]
            )
        else:
            order_quantity_to_params = row["order quantity"] == 0

        return pd.Series(
            {
                "check_sS_S_gt_s": S_gt_s,
                "check_sS_order_quantity_to_params": order_quantity_to_params,
            }
        )

    def _check_sQ(self, row):
        if row["inventory position"] < row["s"]:
            order_quantity_to_params = row["order quantity"] == row["Q"]
        else:
            order_quantity_to_params = row["order quantity"] == 0

        return pd.Series(
            {"check_sQ_order_quantity_to_params": order_quantity_to_params}
        )

    def _check_sSaQ(self, row):

        S_gt_s = row["S"] >= row["s"] + 1

        s_gt_a = row["s"] >= row["a"] + 1

        if row["inventory position"] < row["a"]:
            order_quantity_to_params = (
                row["order quantity"] == row["S"] - row["inventory position"]
            )
        elif row["inventory position"] < row["s"]:
            order_quantity_to_params = row["order quantity"] == row["Q"]
        else:
            order_quantity_to_params = row["order quantity"] == 0

        return pd.Series(
            {
                "check_sSaQ_S_gt_s": S_gt_s,
                "check_sSaQ_s_gt_a": s_gt_a,
                "check_sSaQ_order_quantity_to_params": order_quantity_to_params,
            }
        )

    def _check_sSbQ(self, row):

        S_gt_s = row["S"] >= row["s"] + 1

        s_gt_b = row["s"] >= row["b"] + 1

        if row["inventory position"] < row["b"]:
            order_quantity_to_params = row["order quantity"] == row["Q"]
        elif row["inventory position"] < row["s"]:
            order_quantity_to_params = (
                row["order quantity"] == row["S"] - row["inventory position"]
            )
        else:
            order_quantity_to_params = row["order quantity"] == 0

        return pd.Series(
            {
                "check_sSbQ_S_gt_s": S_gt_s,
                "check_sSbQ_s_gt_b": s_gt_b,
                "check_sSbQ_order_quantity_to_params": order_quantity_to_params,
            }
        )
