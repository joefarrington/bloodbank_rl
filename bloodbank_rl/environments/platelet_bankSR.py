# TODO observation space should distinguish in some way between inventory slots, day of the week and (when added) hospital
# information. This can be achieved using gym.spaces.Tuple and setting the constraints for each individually.
# TODO: add in ability to consider multiple products, e.g. units of different blood types and have order
# of preference for allocation
# TODO: mis-match costs if using multiple types
# TODO: checking of arguments for legality

# This is simplified version that doesn't use SimPy because it's not need for what we're doing right now.
# Demand is Poisson with mean demand for each weekday

# Default parameters based on 'Stochastic inventory model for minimizing blood shortage and outdating in a blood supply chain under supply and demand uncertainty'
# by Shih and Rajendran (2020)

import numpy as np
import pandas as pd
from collections import namedtuple, Counter
import gym
import math
from pathlib import Path


class PlateletBankGym(gym.Env):
    def __init__(
        self,
        demand_provider,
        max_order,
        max_shelf_life,
        lead_time,
        shelf_life_at_arrival_dist,
        fixed_order_cost,
        variable_order_cost,
        holding_cost,
        emergency_procurement_cost,
        wastage_cost,
        render_env=False,
        seed=None,
        seed_demand=True,
        include_timelimit_info=True,
        stock_age_in_state=True,
    ):
        self.demand_provider = demand_provider
        self.max_order = max_order
        self.max_shelf_life = max_shelf_life
        self.lead_time = lead_time

        # Report stock by age, or just one number
        # Remember that if lead time is not 0 this won't be able to tell the difference
        # between items in stock and items ordered but not yet received
        self.stock_age_in_state = stock_age_in_state

        # Probability that newly arrived unit has shelf life between 1 and max_shelf_life
        # Should have number of elements equal to max_shelf_life
        assert (
            len(shelf_life_at_arrival_dist) == max_shelf_life
        ), "`shelf_life_at_arrival_dist` must have number of elements equal to `max_shelf_life`"
        assert (
            np.sum(shelf_life_at_arrival_dist) == 1
        ), "`shelf_life_at_arrival_dist` must sum to 1"
        self.shelf_life_at_arrival_dist = shelf_life_at_arrival_dist

        # Total number of different ages of inventory we need to track
        self.max_inv_slots = self.max_shelf_life + self.lead_time

        # Costs
        self.fixed_order_cost = fixed_order_cost
        self.variable_order_cost = variable_order_cost
        self.holding_cost = holding_cost
        self.emergency_procurement_cost = emergency_procurement_cost
        self.wastage_cost = wastage_cost

        self.render_env = render_env

        # If seed_demand is true, seed provided to env will be used to seed both the env and the demand provider
        # Print message to make this clear
        self.seed_demand = seed_demand
        if self.seed_demand:
            print(
                "Seed for environment will be used for demand provider instead of any seed provided to the demand provider because seed_demand is True."
            )

        # Set random seed value
        self.seed_value = self.seed(seed)

        # Set up the action and observation space
        self._setup_spaces()

        # Whether to include info dict element as in gym.wrapper.TimeLimit
        # This would indictate to libraries like Tianshou that termination
        # due to time limit which is the case because we've set an artificial
        # limit on the length of the simulations
        self.include_time_limit_info = include_timelimit_info

        self.reset()

    def reset(self):

        # Reset demand
        self.demand_provider.reset()

        self.daily_expiries = 0
        self.daily_emergency_orders = 0
        self.daily_demand = 0

        self.timestep = 0

        # Set up the inventory state
        self.inventory, self.in_transit = self.demand_provider.get_initial_stock(
            self.max_shelf_life, self.lead_time
        )

        # Compose the obervation
        observation = self._get_observation()

        return observation

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)

        if self.seed_demand:
            self.demand_provider.np_rng = np.random.default_rng(seed)

        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def _setup_spaces(self):
        # Create action and observation spaces
        # Action space is one slot per number up to the maximum, plus one slot for an order of zero
        self.action_space = gym.spaces.Discrete(n=self.max_order + 1)
        # Seed the action space for reproducibility when sampling from it (e.g. epsilon-greedy)
        self.action_space.seed(self.seed_value[0])

        # Observation space is slots for (max_inv_slots - 1) lots of inventory
        # and then additional slots that may be supplier by the demand provider
        # TODO: This could be more precise about min and max for different elements of obs
        if self.stock_age_in_state:
            obs_dim = (
                self.max_inv_slots - 1 + self.demand_provider.additional_observation_dim
            )
        else:
            obs_inv_dim = (
                1 if self.lead_time == 0 else 2
            )  # Will only observe stock in transit if lead time > 0
            obs_dim = obs_inv_dim + self.demand_provider.additional_observation_dim
        self.observation_space = gym.spaces.Box(low=0, high=500, shape=(obs_dim,))

    def step(self, action):
        # Check the action is legal
        self._islegal(action)

        # Update the timestep
        self.timestep += 1

        # Reset expiries and backorders for the current day
        self.daily_expiries = 0
        self.daily_backorders = 0

        # Add ordered stock to in transit
        self._order_units(action)

        # Add received stock to available inventory
        units_received_by_age = self._receive_ordered_units()

        # Get the demand
        daily_demand = self.demand_provider.generate_demand()

        # Fill demand or log shortage
        self._fill_demand(daily_demand)

        # Remove expired units
        self._remove_expired_units()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Get the new observation
        observation = self._get_observation()

        # Return relevant information for debugging, some duplication
        # for easier post-processing
        info = {
            "action": action,
            "demand": daily_demand,
            "daily_expiries": self.daily_expiries,
            "daily_backorders": self.daily_backorders,
            "units_in_stock": np.sum(self.inventory),
            "units_in_transit": np.sum(self.in_transit),
            "units_received_by_age": units_received_by_age,
            "observation": observation,
        }

        # Indicate that end of episode due to artificial timelimit if we are including this information
        if (
            self.include_time_limit_info
            and self.timestep >= self.demand_provider.sim_duration
        ):
            info["TimeLimit.truncated"] = True

        # Determine if simulation finised to report in output tuple as
        # expected for Gym environments
        terminal = True if self.timestep >= self.demand_provider.sim_duration else False

        if self.render_env:
            self.render()

        return (observation, reward, terminal, info)

    def _order_units(self, action):
        self.in_transit.append(action)

    def _receive_ordered_units(self):
        units_received = self.in_transit.pop(0)
        units_received_by_age = self.np_rng.multinomial(
            units_received, self.shelf_life_at_arrival_dist
        )
        self.inventory += units_received_by_age
        return units_received_by_age

    def _fill_demand(self, demand):
        # Assumes we use the stock with shortest remaining life first
        # FIFO if stock of uniform age on receipt
        remaining_demand = demand
        for idx, stock in enumerate(self.inventory):
            demand_filled = min(remaining_demand, stock)
            remaining_stock = stock - demand_filled
            remaining_demand = remaining_demand - demand_filled
            self.inventory[idx] = remaining_stock

        self.daily_backorders += remaining_demand

    def _remove_expired_units(self):
        self.daily_expiries += self.inventory[0]

        self.inventory[:-1] = self.inventory[1:]
        self.inventory[-1] = 0

    def _calculate_reward(self, action):
        fixed = (action > 0) * self.fixed_order_cost
        variable = action * self.variable_order_cost
        holding = np.sum(self.inventory) * self.holding_cost
        wastage = self.daily_expiries * self.wastage_cost
        shortage = self.daily_backorders * self.emergency_procurement_cost

        cost = fixed + variable + holding + wastage + shortage

        return -cost

    def _get_observation(self):
        # In default, this would just get the weekday
        # Could be more complex as part of more detailed simulation
        additional_obs = self.demand_provider.additional_observation()

        # If we don't want stock age in state, add it up and return single number in numpy array
        if self.stock_age_in_state:
            # At point where observation made, stock has been aged so none with 3 days of useful life
            stock_position = np.hstack((self.inventory[:-1], self.in_transit))
        elif (
            self.lead_time > 0
        ):  # If there's lead time, want obs to include stock on order not yet received
            stock_position = np.array(
                [np.sum(self.inventory[:-1]), np.sum(self.in_transit)]
            )
        else:
            stock_position = np.array(np.sum(self.inventory[:-1]))

        if additional_obs is None:
            return stock_position.astype(int)
        else:
            return np.hstack((additional_obs, stock_position)).astype(int)

    def _islegal(self, action):
        # Check if the supplied action is allowed

        if action < 0:
            raise ValueError("An order cannot be negative")
        elif action > self.max_order:
            raise ValueError(f"Order cannot exceed {self.max_order} units")
        elif not self.action_space.contains(action):
            raise ValueError("Requested action is not allowed")

    def render(self):
        pass


# Just for debugging
class SimpleProvider:
    def __init__(self, constant_demand=10, sim_duration=30):
        self.sim_duration = 30
        self.constant_demand = 10
        self.additional_observation_dim = 1
        self.reset()

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time

        return inventory, in_transit

    def generate_demand(self):
        self.weekday = (self.weekday + 1) % 7
        return self.constant_demand

    def additional_observation(self):
        return [self.weekday]

    def reset(self):
        # Initial state is Sunday
        self.weekday = 6
        pass


class PoissonDemandProviderSR:
    def __init__(
        self,
        mean_daily_demands=[37.5, 37.3, 39.2, 37.8, 40.5, 27.2, 28.4],
        sim_duration=365,
        initial_weekday=6,
        seed=None,
        one_hot_encode_weekday=False,
    ):

        self.mean_daily_demands = mean_daily_demands
        self.sim_duration = sim_duration

        # Set random seed and store value for potential future logging
        self.seed_value = self.seed(seed)

        # need to provide the weekday as state
        if one_hot_encode_weekday:
            self.additional_observation_dim = 7
        else:
            self.additional_observation_dim = 1
        self.one_hot_encode_weekday = one_hot_encode_weekday

        self.initial_weekday = initial_weekday

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time

        return inventory, in_transit

    def generate_demand(self):
        # Update the weekday - moving to morning after observation
        self.weekday = (self.weekday + 1) % 7

        demand = self.np_rng.poisson(self.mean_daily_demands[self.weekday])

        return demand

    def additional_observation(self):
        # Return the weekday
        if self.one_hot_encode_weekday:
            oh_weekday = [0] * 7
            oh_weekday[self.weekday] = 1
            return oh_weekday
        else:
            return [self.weekday]

    def reset(self):
        # Initial state is Sunday
        self.weekday = self.initial_weekday


### Demand provider to provde extracts from a pre-generated sequence of data


class DFDemandProvider:
    def __init__(
        self,
        filename,
        demand_col_name,
        additional_observation_col_names=None,
        sample_sim_duration=None,
        seed=None,
    ):

        filepath = Path(filename).resolve()
        self.df_all = pd.read_csv(filepath)
        # Reset the index of the df for easier slicing
        self.df_all = self.df_all.reset_index(drop=True)

        # If not sampling, simulation uses all rows of df
        # Otherwise, it is the length we sample
        # If sampling, calculate the highest index we can sample that still
        # gives us a full sim_duration before the end of the data
        self.sample_sim_duration = sample_sim_duration
        if self.sample_sim_duration is None:
            self.sim_duration = self.df_all.shape[0] - 1
        else:
            self.sim_duration = self.sample_sim_duration
            self.max_sample_index = self.df_all.shape[0] - self.sim_duration - 1

        self.seed_value = self.seed(seed)

        self.demand_col_name = demand_col_name

        self.additional_observation_col_names = additional_observation_col_names
        self.additional_observation_dim = len(self.additional_observation_col_names)

    def seed(self, seed):
        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time
        return inventory, in_transit

    def generate_demand(self):
        self.current_index += 1
        return self.df_all.loc[self.current_index, self.demand_col_name]

    def additional_observation(self):
        if self.additional_observation_col_names is None:
            return None
        else:
            return self.df_all.loc[
                self.current_index, self.additional_observation_col_names
            ].values

    def reset(self):

        if self.sample_sim_duration is None:
            # Just running all the rows in order
            self.current_index = 0
        else:
            self.current_index = self.np_rng.integers(
                low=0, high=self.max_sample_index + 1
            )


### Demand provider that provides non-overlapping extracts of pre-generated data for Pyomo


class DFPyomoDemandProvider:
    def __init__(
        self,
        filename,
        demand_col_name,
        weekday_col_name,
        sim_start_weekday=0,  # Monday by default
        sim_duration=None,
        seed=0,
    ):
        filepath = Path(filename).resolve()
        self.df_all = pd.read_csv(filepath)
        # Reset the index of the df for easier slicing
        self.df_all = self.df_all.reset_index(drop=True)[
            [weekday_col_name, demand_col_name]
        ]

        self.demand_col_name = demand_col_name
        self.weekday_col_name = weekday_col_name
        self.sim_start_weekday = sim_start_weekday
        self.sim_duration = sim_duration

        self.seed_value = self.seed(seed)

        self.additional_observation_dim = 1  # just weekday

        # Index of the first instance of starting weekday (e.g. first Monday)
        self.first_start_weekday_index = self._calculate_first_start_weekday_index()
        # Index to start scenario from
        self.initial_index = self._calculate_initial_index()
        self.current_index = self.initial_index

    def seed(self, seed):

        # Here seed is not used for random number generator, but
        # to work out the starting index of the period

        # Makes the results reproducible, and makes this class
        # compatible with the others

        return [seed]

    def _calculate_first_start_weekday_index(self):
        first_start_weekday_index = (
            self.df_all[self.df_all[self.weekday_col_name] == self.sim_start_weekday]
            .iloc[0:1, :]
            .index.values[0]
        )
        return first_start_weekday_index

    def _calculate_initial_index(self):
        # Remember to minus 1 because generate demand adds one before giving next demand
        initial_index = (
            self.seed_value[0] * (math.ceil(self.sim_duration / 7) * 7)
            + self.first_start_weekday_index
            - 1
        )
        if initial_index + self.sim_duration > self.df_all.shape[0] - 1:
            raise ValueError(
                "Data not available for full scenario, reduce seed or scenario length"
            )
        else:
            return initial_index

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock
        # Note that this isn't used in Pyomo modelling,
        # where initial inventory set elsewhere

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time
        return inventory, in_transit

    def generate_demand(self):
        self.current_index += 1
        return self.df_all.loc[self.current_index, self.demand_col_name]

    def additional_observation(self):
        return self.df_all.loc[self.current_index, self.weekday_col_name].values

    def reset(self):
        self.current_index = self.initial_index


# We can use this to change the coefficient of variation.
# But it can only do an overdispersed Poisson so min cv is
# say 0.2, so we can't currently do the setting where CV is 0.1
# as reported in the paper


class NegBinDemandProviderSR:
    def __init__(
        self,
        mean_daily_demands=[37.5, 37.3, 39.2, 37.8, 40.5, 27.2, 28.4],
        cv=0.2,
        sim_duration=365,
        seed=None,
    ):

        self.mean_daily_demands = np.array(mean_daily_demands)
        self.cv = cv
        self.sim_duration = sim_duration

        # Based on the means and CV, calculate the std dev and
        # then p and n to parameterise the numpy negative binomial dist
        self.daily_stddev = self._calculate_daily_stddev(
            self.mean_daily_demands, self.cv
        )
        self.daily_nb_p = self._calculate_daily_nb_p(
            self.mean_daily_demands, self.daily_stddev
        )
        self.daily_nb_n = self._calculate_daily_nb_n(
            self.mean_daily_demands, self.daily_stddev
        )

        # Set random seed and store value for potential future logging
        self.seed_value = self.seed(seed)

        # need to provide the weekday as state
        self.additional_observation_dim = 1

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time

        return inventory, in_transit

    def generate_demand(self):
        # Update the weekday - moving to morning after observation
        self.weekday = (self.weekday + 1) % 7

        demand = self.np_rng.negative_binomial(
            self.daily_nb_n[self.weekday], self.daily_nb_p[self.weekday]
        )

        return demand

    def additional_observation(self):
        # Return the weekday
        return [self.weekday]

    def reset(self):
        # Initial state is Sunday
        self.weekday = 6

    def _calculate_daily_stddev(self, mean_daily_demands, cv):
        return mean_daily_demands * cv

    def _calculate_daily_nb_p(self, mean_daily_demands, daily_stddev):
        p = (mean_daily_demands) / daily_stddev ** 2
        return p

    def _calculate_daily_nb_n(self, mean_daily_demands, daily_stddev):
        n = (mean_daily_demands ** 2) / (daily_stddev ** 2 - mean_daily_demands)
        return n


# This differ slightly, demand is generated one day ahead and stored so that it can be used to
# calculate a dummy forecast. Should end up with the same demand sequence if the rngs are
# set up correctly.

# Need to be a little bit careful, because this calls the RNG as part of reset, if we instantiate
# the env, which calls reset, and then manually reset, rng is 'one ahead' of same seeded value
# for normal Poisson demand provider so don;t get the same demand trajectory.
# Either don't mnaully reset at the start, or remove reset statement from init of the env
# But don't want to be doing both


class PoissonDemandProviderWithForecastSR:
    def __init__(
        self,
        mean_daily_demands=[37.5, 37.3, 39.2, 37.8, 40.5, 27.2, 28.4],
        sim_duration=365,
        seed=None,
    ):
        self.mean_daily_demands = mean_daily_demands
        self.sim_duration = sim_duration

        # Set random seed and store value for potential future logging
        self.seed_value = self.seed(seed)

        # need to provide the weekday as state
        # and also the forecast
        self.additional_observation_dim = 2

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy
        # Have an extra rng for the forecast noise, so
        # should get same demands using same seed for comparison
        self.np_rng_forecast_noise = np.random.default_rng(seed)
        forecast_seed_value = self.np_rng_forecast_noise.bit_generator._seed_seq.entropy

        return [seed_value, forecast_seed_value]

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time

        return inventory, in_transit

    def generate_demand(self):
        # Update the weekday - moving to morning after observation
        self.weekday = (self.weekday + 1) % 7

        # We will have generated and stored today's demand yesterday
        # so just store and then return
        demand = self.next_day_demand

        # Generate demand and forecast for the next day
        self.next_day_demand = self._generate_next_day_demand()
        self.next_day_forecast = self._generate_next_day_forecast()

        return demand

    def _generate_next_day_demand(self):
        next_weekday = (self.weekday + 1) % 7
        next_day_demand = self.np_rng.poisson(self.mean_daily_demands[next_weekday])
        return next_day_demand

    def _generate_next_day_forecast(self):
        # For now, generate forecast by just adding some noise to true demand
        # noise = self.np_rng_forecast_noise.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])

        # What happens with a perfect forecast?
        noise = 0

        return self.next_day_demand + noise

    def additional_observation(self):
        # Return the weekday
        return [self.weekday, self.next_day_forecast]

    def reset(self):
        # Initial state is Sunday
        self.weekday = 6

        # Generate the next day's demand and the forecast
        # based on it to include in the state
        self.next_day_demand = self._generate_next_day_demand()
        self.next_day_forecast = self._generate_next_day_forecast()


# This is a lightweight wrapper around the PoissonDemandProvider


class PoissonDemandProviderLimitedScenarios:
    def __init__(
        self,
        mean_daily_demands=[37.5, 37.3, 39.2, 37.8, 40.5, 27.2, 28.4],
        sim_duration=365,
        initial_weekday=6,
        seed=None,
        one_hot_encode_weekday=False,
        n_scenarios=None,
        scenario_seeds=None,
    ):
        self.mean_daily_demands = mean_daily_demands
        self.sim_duration = sim_duration
        self.initial_weekday = initial_weekday
        self.one_hot_encode_weekday = one_hot_encode_weekday

        # Set random seed and store value for potential future logging
        # This seed is used to generate scenario seeds if not provided
        #  and decide which scenario is used after
        # each reset
        self.seed_value = self.seed(seed)

        # Should specifiy either the number of scenarios or a set of seeds
        if n_scenarios is None and scenario_seeds is None:
            raise ValueError("Set n_scenarios or scenario_seeds")
        elif n_scenarios is not None and scenario_seeds is not None:
            raise ValueError("Set only one of n_scenarios and scenario_seeds")
        elif scenario_seeds is not None:
            self.scenario_seeds = scenario_seeds
            self.n_scenarios = len(scenario_seeds)
        elif n_scenarios is not None:
            self.n_scenarios = n_scenarios
            self.scenario_seeds = self.np_rng.integers(
                low=0, high=1e8, size=n_scenarios
            )

        # need to provide the weekday as state
        if one_hot_encode_weekday:
            self.additional_observation_dim = 7
        else:
            self.additional_observation_dim = 1
        self.one_hot_encode_weekday = one_hot_encode_weekday

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def get_initial_stock(self, max_age, transit_time):
        return self.current_demand_provider.get_initial_stock(max_age, transit_time)

    def generate_demand(self):
        demand = self.current_demand_provider.generate_demand()
        self.weekday = self.current_demand_provider.weekday
        return demand

    def additional_observation(self):
        # Return the additional obseration
        return self.current_demand_provider.additional_observation()

    def reset(self):
        self.current_scenario_seed = self.np_rng.choice(self.scenario_seeds)
        self.current_demand_provider = PoissonDemandProviderSR(
            mean_daily_demands=self.mean_daily_demands,
            sim_duration=self.sim_duration,
            initial_weekday=self.initial_weekday,
            seed=self.current_scenario_seed,
            one_hot_encode_weekday=self.one_hot_encode_weekday,
        )
        self.current_demand_provider.reset()


class NormalDemandProviderSR:
    def __init__(
        self,
        mean_daily_demands=[200] * 7,
        std_daily_demands=[32] * 7,
        sim_duration=365,
        seed=None,
    ):

        self.mean_daily_demands = mean_daily_demands
        self.std_daily_demands = std_daily_demands
        self.sim_duration = sim_duration

        # Set random seed and store value for potential future logging
        self.seed_value = self.seed(seed)

        # need to provide the weekday as state
        self.additional_observation_dim = 1

        self.reset()

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def get_initial_stock(self, max_age, transit_time):
        # Initial stock is supposed to be a parameter
        # Need to check exactly what it should be
        # For now, just assume no stock

        inventory = np.zeros((max_age))

        in_transit = [0] * transit_time

        return inventory, in_transit

    def generate_demand(self):
        # Update the weekday - moving to morning after observation
        self.weekday = (self.weekday + 1) % 7

        # Generate demand from normal dist, rounding to nearest integer
        # And ensuring that it can't be negative
        demand = self.np_rng.normal(
            self.mean_daily_demands[self.weekday], self.std_daily_demands[self.weekday],
        )
        demand = max(0, int(demand))

        return demand

    def additional_observation(self):
        # Return the weekday
        return [self.weekday]

    def reset(self):
        # Initial state is Sunday
        self.weekday = 6
