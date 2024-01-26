import numpy as np
import pandas as pd
from scipy.stats import poisson
from joblib import Parallel, delayed
import pickle
import functools
from pathlib import Path
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

weekdays = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}

# Enable logging
log = logging.getLogger(__name__)


def get_demand_probabilities(demand_min, demand_max, poisson_mu):
    # Demand max is inclusive
    demands = list(range(demand_min, demand_max + 1))
    probs = poisson.pmf(demands, mu=poisson_mu)
    return {k: v for k, v in zip(demands, probs)}


def get_inventory_received(action, shelf_life_at_arrival_dist):
    shelf_life_at_arrival_dist = np.array(shelf_life_at_arrival_dist)
    inv_received = np.floor(action * shelf_life_at_arrival_dist)

    # if action can be evenly split, just return it
    if np.sum(inv_received) == action:
        return inv_received.astype("int")
    # otherwise, allocation remainder to slot with biggest gap between current and desired proportion
    while np.sum(inv_received) != action:
        proportion_difference = shelf_life_at_arrival_dist - (inv_received / action)
        inv_received[np.argmax(proportion_difference)] += 1

    # Orders need to be integer units, flag up if there's a problem with rounding code so we're not receiving
    # the right number of units
    assert action == sum(
        inv_received
    ), "Sum of inventory received not the same as action, check rounding code"
    return inv_received.astype("int")


def determine_new_inventory_position(inventory, demand):
    remaining_demand = demand
    for idx, stock in enumerate(inventory):
        demand_filled = min(remaining_demand, stock)
        remaining_stock = stock - demand_filled
        remaining_demand = remaining_demand - demand_filled
        inventory[idx] = remaining_stock

    shortage = remaining_demand
    expiries = inventory[0]

    return inventory[1:], expiries, shortage


def calculate_reward(action, expiries, shortage, inventory_position, cost_dict):
    fixed = (action > 0) * cost_dict["fixed_order_cost"]
    variable = action * cost_dict["variable_order_cost"]
    holding = np.sum(inventory_position) * cost_dict["holding_cost"]
    wastage = expiries * cost_dict["wastage_cost"]
    shortage = shortage * cost_dict["emergency_procurement_cost"]

    cost = fixed + variable + holding + wastage + shortage

    return -cost


def play_step(starting_state, action, demand, age_at_arrival_dict, cost_dict):
    starting_state = eval(starting_state)
    starting_inventory = np.array([*starting_state[1:], 0])
    inventory_received = age_at_arrival_dict[action]
    starting_inventory = starting_inventory + inventory_received

    final_inventory, expiries, shortage = determine_new_inventory_position(
        starting_inventory, demand
    )

    reward = calculate_reward(action, expiries, shortage, final_inventory, cost_dict)
    new_weekday = (starting_state[0] + 1) % len(weekdays)
    new_state = [new_weekday, *final_inventory]

    return reward, str(new_state)


def create_transition_df(state, action, demand_df_dict, age_at_arrival_dict, cost_dict):
    weekday = eval(state)[0]
    demand_df = demand_df_dict[weekday]
    temp_df = pd.DataFrame(index=demand_df.index, columns=["reward", "new_state"])
    plays = [
        (play_step(state, action, d, age_at_arrival_dict, cost_dict))
        for d in demand_df.index
    ]
    temp_df["reward"] = [x[0] for x in plays]
    temp_df["new_state"] = [x[1] for x in plays]
    return temp_df


def get_demand_df(min_demand, max_demand, mean_demand):
    dem = get_demand_probabilities(min_demand, max_demand, mean_demand)
    demdf = pd.DataFrame(dem, index=[0])
    demdf = demdf.transpose()
    demdf.columns = ["prob"]
    return demdf


def q_iteration_step(
    state, actions, gamma, q_values_old, demand_df_dict, age_at_arrival_dict, cost_dict
):
    weekday = eval(state)[0]
    df_state_dict = {
        a: create_transition_df(
            state, a, demand_df_dict, age_at_arrival_dict, cost_dict
        )
        for a in actions
    }
    demand_df = demand_df_dict[weekday]
    new_qs = np.zeros(len(q_values_old.columns))
    for action in q_values_old.columns:
        temp_df = df_state_dict[action]
        new_states = temp_df.loc[:, "new_state"].values
        q_maxes = q_values_old.loc[new_states, :].max(axis=1).values
        new_qs[action] = (
            demand_df["prob"].values * (temp_df["reward"].values + gamma * q_maxes)
        ).sum()
    return new_qs


def period_convergence_test(
    q_values, iteration, period, epsilon, iteration_qvalues_path
):
    p = iteration_qvalues_path / f"q_values_{iteration-period}.pkl"

    with p.open(mode="rb") as fp:
        q_values_period_start = pickle.load(fp)

    max_period_diff = np.abs(q_values - q_values_period_start).max().max()
    min_period_diff = np.abs(q_values - q_values_period_start).min().min()
    print(f"max_period_diff: {max_period_diff}")
    print(f"min_period_diff: {min_period_diff}")

    return (max_period_diff - min_period_diff) < epsilon * min_period_diff


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    log.info("Starting setup")
    #  Setup
    age_at_arrival_dict = {}
    for a in range(0, cfg.max_order + 1):
        age_at_arrival_dict[a] = get_inventory_received(
            a, cfg.shelf_life_at_arrival_dist
        )

    # Roll demand so that, for example, action at end of Monday followed by Tues demand
    max_demands = np.roll(cfg.max_demands, -1)
    min_demands = np.roll(cfg.min_demands, -1)
    mean_demands = np.roll(cfg.mean_demands, -1)

    demand_df_dict = {}
    for i in range(len(weekdays)):
        demand_df_dict[i] = get_demand_df(
            min_demands[i], max_demands[i], mean_demands[i]
        )

    max_stock_rem_age_1 = np.ceil(
        np.sum(cfg.shelf_life_at_arrival_dist[1:]) * cfg.max_order
    ).astype(int)
    max_stock_rem_age_2 = np.ceil(
        np.sum(cfg.shelf_life_at_arrival_dist[-1]) * cfg.max_order
    ).astype(int)

    states = [
        [i, j, k]
        for i in weekdays.keys()
        for j in range(0, max_stock_rem_age_1 + 1)
        for k in range(0, max_stock_rem_age_2 + 1)
    ]
    actions = [*range(cfg.max_order + 1)]

    q_values = pd.DataFrame(
        np.zeros((len(states), len(actions))),
        index=[str(s) for s in states],
        columns=actions,
    )

    if cfg.save_qvalues_each_iteration:
        iteration_qvalues_path = Path("iteration_qvalues/")
        iteration_qvalues_path.mkdir(parents=True, exist_ok=True)

    # For now, only do periodic convergence test
    # for undiscounted case and if saving qvalues each iteration
    conv_test_periodic = cfg.convergence_test.periodic
    if conv_test_periodic:
        assert (
            cfg.gamma == 1
        ), "Periodic convergence test currently only implement for undiscounted case"
        assert (
            cfg.save_qvalues_each_iteration
        ), "Peridic convergence test requires saving Q-values each iteration"

    # Run Q-iteration
    period = len(cfg.mean_demands)
    q_delta_df = pd.DataFrame(columns=list(range(period)))

    log.info("Setup complete, starting Q-value iteration")
    
    with Parallel(n_jobs=-1) as parallel:
        for i in range(cfg.max_iterations):
            q_values_old = q_values.copy()
            q_new = parallel(
                delayed(q_iteration_step)(
                    state,
                    actions,
                    cfg.gamma,
                    q_values_old,
                    demand_df_dict,
                    age_at_arrival_dict,
                    cfg.cost_dict,
                )
                for state in q_values.index
            )
            q_values = pd.DataFrame(
                q_new, index=q_values.index, columns=q_values.columns
            )
            q_diffs = np.abs(q_values - q_values_old)
            q_max_abs_diff = q_diffs.max().max()

            best_action_old = q_values_old.idxmax(axis=1).values
            best_actions_new = q_values.idxmax(axis=1).values

            prop_actions_change = np.mean(best_action_old != best_actions_new)

            log.info(
                f"iteration {i} done, max diff {q_max_abs_diff}, % actions changed {prop_actions_change}"
            )

            # Periodic covergence test
            if (i >= period) and (cfg.convergence_test.periodic):

                conv_test = period_convergence_test(
                    q_values,
                    i,
                    period,
                    cfg.convergence_test.epsilon,
                    iteration_qvalues_path,
                )

            # If not using period convergence test, then just look at biggest change in a q-value
            else:
                conv_test = q_max_abs_diff < (
                    cfg.convergence_test.epsilon * q_values.min().min()
                )

            if conv_test:
                log.info(f"Period convergence test met on iteration {i}")
                break

            # If flag is True, save the q_values at the end of each iteration
            if cfg.save_qvalues_each_iteration:
                p = iteration_qvalues_path / f"q_values_{i}.pkl"

                with p.open(mode="wb") as fp:
                    pickle.dump(q_values, fp)

    # Process and save the results

    best_action = pd.DataFrame(q_values.idxmax(axis=1))

    best_action = best_action.reset_index()

    best_action["weekday"] = best_action["index"].apply(lambda x: eval(x)[0])
    best_action["1day"] = best_action["index"].apply(lambda x: eval(x)[1])
    best_action["2day"] = best_action["index"].apply(lambda x: eval(x)[2])

    for key, value in weekdays.items():
        output = best_action[best_action["weekday"] == key].pivot(
            index="1day", columns="2day", values=0
        )
        output.to_csv(f"best_action_{value}.csv")

    with open("q_values.pkl", "wb") as fp:
        pickle.dump(q_values, fp)

    log.info("Policy extracted and saved.")


if __name__ == "__main__":
    main()
