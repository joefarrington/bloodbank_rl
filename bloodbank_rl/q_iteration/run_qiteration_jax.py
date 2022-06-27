import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import pickle
import functools
from pathlib import Path
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

# This is a first go at using Jax for this problem
# Likely possible to make more efficient
# In particular, expect there is a more effective (and more principled) way of splitting
# up states, BUT this gives us a huge speed up versus old version : 4s per iteration vesus 20mins before.
# And that's on CPU. 


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

# In the old implementation, we let demand_min and demand_max
# be set separately for each weekday.
# Better for jax to keep same shape
def get_demand_probabilities(demand_min, demand_max, poisson_mu):
    # poisson_mu can be an single number or a list
    # if a list, output shape is days (len of poisson_mu) x demand_value

    # Get the potential demands to be evaluated
    quantiles = jnp.arange(demand_min, demand_max + 1)
    probs = jax.scipy.stats.poisson.pmf(quantiles, jnp.array(poisson_mu).reshape(-1, 1))
    return probs


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


def step_known_demand(state, units_received, demand):
    stock_from_prev_day = jnp.array([*state[1:], 0])
    stock = stock_from_prev_day + jnp.array(units_received)

    remaining_demand = demand
    closing_stock = []
    for idx, n_units in enumerate(stock):
        demand_filled = jnp.minimum(remaining_demand, n_units)
        remaining_stock = n_units - demand_filled
        remaining_demand = remaining_demand - demand_filled
        closing_stock.append(remaining_stock)

    shortage = remaining_demand
    expiries = closing_stock[0]

    return jnp.array(closing_stock[1:]), expiries, shortage


def calculate_reward(action, closing_stock, expiries, shortage, cost_dict):
    fixed = (action > 0) * cost_dict["fixed_order_cost"]
    variable = action * cost_dict["variable_order_cost"]
    holding = jnp.sum(closing_stock) * cost_dict["holding_cost"]
    wastage = expiries * cost_dict["wastage_cost"]
    shortage = shortage * cost_dict["emergency_procurement_cost"]

    cost = fixed + variable + holding + wastage + shortage

    return -cost


def calculate_reward_and_next_state_idx(
    state, action, units_received, cost_dict, demand, state_to_idx
):
    # Work out holding, shortage, expiries
    closing_stock, expiries, shortage = step_known_demand(state, units_received, demand)

    # Use holding to get next state
    weekday = state[0]
    next_weekday = jnp.remainder(weekday + 1, 7)  # Assume 7 weekdays as period
    next_state = tuple([next_weekday] + [x for x in closing_stock])

    # Work out the rewards, based on action, holding, shortage, expiries
    reward = calculate_reward(action, closing_stock, expiries, shortage, cost_dict)

    next_state_idx = state_to_idx[next_state]

    # Return the reward and the next state
    return reward, next_state_idx


# Vmap over demands
calculate_reward_and_next_state_idx_vmap_demands = jax.vmap(
    calculate_reward_and_next_state_idx, in_axes=[None, None, None, None, 0, None]
)


def update_q(
    state,
    action,
    q_values_old,
    cost_dict,
    age_at_arrival,
    demands,
    demand_probs,
    gamma,
    state_to_idx,
):
    # If put in state idx, want to get the actual state

    units_received = age_at_arrival[action, :]

    # vmap over demands
    # Work out holding, shortage, expiries
    reward, next_state_idx = calculate_reward_and_next_state_idx_vmap_demands(
        state, action, units_received, cost_dict, demands, state_to_idx
    )

    # Work out the update sum(prob x (reward + gamma * old q_value for next state)
    # Each of demand_probs, reward, q_old_next state should be of shape (demands,) so may need reshaping here
    q_old_next_state = q_values_old[next_state_idx.reshape(-1), :].max(axis=1)
    reward = reward.reshape(-1)

    weekday = state[0]
    demand_probs_day = demand_probs[weekday, :].reshape(-1)

    q = jnp.dot(demand_probs_day, (reward + (gamma * q_old_next_state)))
    return q


# Vmap over states and actions and use jit
update_q_vmap_states_actions = jax.jit(
    jax.vmap(
        jax.vmap(update_q, in_axes=(None, 0, None, None, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None, None, None),
    )
)


def period_convergence_test(
    q_values, iteration, period, epsilon, iteration_qvalues_path
):
    p = iteration_qvalues_path / f"q_values_{iteration-period}.pkl"

    with p.open(mode="rb") as fp:
        q_values_period_start = pickle.load(fp)

    max_period_diff = jnp.abs(q_values - q_values_period_start.values).max().max()
    min_period_diff = jnp.abs(q_values - q_values_period_start.values).min().min()
    print(f"max_period_diff: {max_period_diff}")
    print(f"min_period_diff: {min_period_diff}")

    return (max_period_diff - min_period_diff) < epsilon * min_period_diff


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Other version lets there by set by weekday
    # We'll probably amend that, better just to have min and max for Jax
    max_demands = OmegaConf.to_container(cfg.max_demands)
    min_demands = OmegaConf.to_container(cfg.min_demands)
    mean_demands = OmegaConf.to_container(cfg.mean_demands)

    cost_dict = OmegaConf.to_container(cfg.cost_dict)

    max_demand = np.array(max_demands).max()
    min_demand = np.array(min_demands).min()
    demands = jnp.arange(0, max_demand + 1)
    demand_probs = get_demand_probabilities(min_demand, max_demand, mean_demands)

    #  Setup
    max_useful_life = len(cfg.shelf_life_at_arrival_dist)
    age_at_arrival = np.zeros((cfg.max_order + 1, max_useful_life))
    for a in range(0, max_demand + 1):
        age_at_arrival[a, :] = get_inventory_received(a, cfg.shelf_life_at_arrival_dist)
    age_at_arrival = jnp.array(age_at_arrival, dtype=jnp.int32)

    max_stock_rem_age_1 = np.ceil(
        np.sum(cfg.shelf_life_at_arrival_dist[1:]) * cfg.max_order
    ).astype(int)
    max_stock_rem_age_2 = np.ceil(
        np.sum(cfg.shelf_life_at_arrival_dist[-1]) * cfg.max_order
    ).astype(int)

    # Generate states and way to map  state->idx
    # Convert to jnp array after creating state_to_idx to avoid indexing issues
    state_tuples = [
        (i, j, k)
        for i in weekdays.keys()
        for j in range(0, max_stock_rem_age_1 + 1)
        for k in range(0, max_stock_rem_age_2 + 1)
    ]

    state_to_idx = np.zeros(
        (len(weekdays.keys()), max_stock_rem_age_1 + 1, max_stock_rem_age_2 + 1)
    )
    for idx, state in enumerate(state_tuples):
        state_to_idx[state] = idx
    state_to_idx = jnp.array(state_to_idx, dtype=jnp.int32)

    states = jnp.array(state_tuples)

    actions = jnp.arange(0, cfg.max_order + 1)

    q_values = jnp.zeros((len(states), len(actions)))

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

    # With normal problem, got memory error
    # This is a hack, combined with the jnp.vstack
    # TODO: A more principled way of doing this
    s_1 = states[:13000, :]
    s_2 = states[13000:, :]

    for i in range(cfg.max_iterations):
        q_values_old = q_values.copy()
        q_values = jnp.vstack(
            [
                update_q_vmap_states_actions(
                    s,
                    actions,
                    q_values_old,
                    cost_dict,
                    age_at_arrival,
                    demands,
                    demand_probs,
                    cfg.gamma,
                    state_to_idx,
                )
                for s in [s_1, s_2]
            ]
        )

        q_diffs = jnp.abs(q_values - q_values_old)
        q_max_abs_diff = q_diffs.max().max()

        best_action_old = q_values_old.argmax(axis=1)
        best_actions_new = q_values.argmax(axis=1)

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
            q_values_df = pd.DataFrame(q_values, index=state_tuples, columns=actions)
            p = iteration_qvalues_path / f"q_values_{i}.pkl"

            with p.open(mode="wb") as fp:
                pickle.dump(q_values_df, fp)

    # Process and save the results

    q_values_df = pd.DataFrame(q_values, index=state_tuples, columns=actions)

    best_action = pd.DataFrame(q_values_df.idxmax(axis=1))

    best_action = best_action.reset_index()

    best_action["weekday"] = best_action["index"].apply(lambda x: x[0])
    best_action["1day"] = best_action["index"].apply(lambda x: x[1])
    best_action["2day"] = best_action["index"].apply(lambda x: x[2])

    for key, value in weekdays.items():
        output = best_action[best_action["weekday"] == key].pivot(
            index="1day", columns="2day", values=0
        )
        output.to_csv(f"best_action_{value}.csv")

    with open("q_values.pkl", "wb") as fp:
        pickle.dump(q_values_df, fp)


if __name__ == "__main__":
    main()
