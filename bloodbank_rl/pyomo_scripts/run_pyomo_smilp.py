import numpy as np
import pandas as pd

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm

from bloodbank_rl.environments.platelet_bankSR import PoissonDemandProviderSR
import bloodbank_rl.pyomo_models.model_constructors_nonweekly as pyomo_mc


def scenario_creator(scenario_name, n_scenarios, t_max, a_max):
    prov = PoissonDemandProviderSR(seed=int(scenario_name))
    prov.reset()
    demand = {t: prov.generate_demand() for t in range(1, t_max + 1)}

    model = pyomo_mc.sS_PyomoModelConstructor(
        demand=demand, t_max=t_max, a_max=a_max
    ).build_model()
    # Telling it which decisions belong to first stage - for us this could be all our decison variable
    # because we can;t change them during a trajectory
    sputils.attach_root_node(model, 0, [model.S, model.s])
    # If we don't specify, assume that all equally likely
    model._mpisppy_probability = 1.0 / n_scenarios
    return model


if __name__ == "__main__":

    t_max = 30
    a_max = 3
    n_scenarios = 5

    options = {"solver": "gurobi_persistent"}
    all_scenario_names = [
        f"{i+310}" for i in range(1, n_scenarios + 1)
    ]  # seed 5 used for example run in Excel, so add const

    ef = ExtensiveForm(
        options=options,
        all_scenario_names=all_scenario_names,
        scenario_creator=scenario_creator,
        scenario_creator_kwargs={
            "t_max": t_max,
            "a_max": a_max,
            "n_scenarios": n_scenarios,
        },
    )

    results = ef.solve_extensive_form()
    objval = ef.get_objective_value()

    results_list = []
    for tup in ef.scenarios():
        scen = tup[0]
        print(f"Scenario {scen}")
        prov = PoissonDemandProviderSR(seed=int(scen))
        prov.reset()
        demand = {t: prov.generate_demand() for t in range(1, t_max + 1)}
        model = tup[1]
        res_dict = [
            {
                "opening_inventory": [round(model.IssB[t, a](), 0) for a in model.A],
                "received": [round(model.X[t, a](), 0) for a in model.A],
                "demand": demand[t],
                "DSSR": [model.DssR[t, a]() for a in model.A],
                "wastage": model.W[t](),
                "shortage": round(model.E[t](), 0),
                "closing inventory": [model.IssE[t, a]() for a in model.A],
                "inventory position": round(model.IP[t](), 0),
                "s": model.s[t](),
                "S": model.S[t](),
                "order quantity": round(model.OQ[t](), 0),
            }
            for t in model.T
        ]
        results_list.append(res_dict)
        print(f"Variable cost: {model.variable_cost()}")
        print(f"Holding cost: {model.holding_cost()}")
        print(f"Fixed cost: {model.fixed_cost()}")
        print(f"Wastage cost: {model.wastage_cost()}")
        print(f"Shortage cost: {model.shortage_cost()}")
        print("")

    for i, x in enumerate(results_list):
        pd.DataFrame(x).to_csv(f"scenario_{i}_output.csv")
