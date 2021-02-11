# Based on https://github.com/MichaelAllen1966/learninghospital/blob/master/simpy_envs/env_simple_hospital_bed_1.py

# TODO: full docstrings for functions once design finalised
# TODO: may want to consider case where time between order and delivery does not all take up useful life
# should be able to separately specify how long it takes
# from order to delivery and how much remaining useful life there is on arrival
# TODO: add ability to load historical demand from file, rather than generating randomly
# TODO: make sure if timestep different weekday still updates properly and everything else flows. Currently the logic
# assumes that the timestep will be one day
# TODO observation space should distinguish in some way between inventory slots, day of the week and (when added) hospital
# information. This can be achieved using gym.spaces.Tuple and setting the constraints for each individually.
# TODO: add in ability to consider multiple products, e.g. units of different blood types and have order
# of preference for allocation
# TODO: order, holding and mis-match costs


import simpy
import numpy as np
import pandas as pd
from collections import namedtuple, Counter
import gym

PlateletUnit = namedtuple("PlateletUnit", ["order_day"])


class PlateletBankGym(gym.Env):
    def __init__(
        self,
        mean_demand_per_day,
        max_order,
        max_age,
        transit_time,
        daily_demand_factors=[1] * 7,
        expiry_cost=1,
        backorder_cost=1,
        render_env=False,
        sim_duration=365,
        time_step=1,
        seed=None,
    ):
        self.mean_demand_per_day = mean_demand_per_day
        self.max_order = max_order
        self.max_age = max_age
        self.transit_time = transit_time
        self.expiry_cost = expiry_cost
        self.backorder_cost = backorder_cost
        self.render_env = render_env
        self.sim_duration = sim_duration
        self.time_step = time_step

        self.next_time_stop = 0

        if self.time_step != 1:
            raise NotImplementedError(
                "Current implementation requires timesteps of 1 day"
            )

        # Set random seed and store value for potential future logging
        self.seed_value = self.seed(seed)

        self.state = {}

        # Action space is one slot per number up to the maximum, plus one slot for an order of zero
        self.action_space = gym.spaces.Discrete(n=max_order + 1)
        # Seed the action space for reproducibility when sampling from it (e.g. epsilon-greedy)
        self.action_space.seed(self.seed_value[0])

        # Observation space is one slot of weekday and slots for (max_age - 1) lots of inventory
        self.observation_space = gym.spaces.Box(low=0, high=500, shape=(self.max_age,))

        # Ensure daily demand factors are valid, then create a dictionary
        # of mean orders per day
        assert sum(daily_demand_factors) == 7, "Sum of daily demand factors must be 7"

        self.daily_demand = {}
        for day_num in range(7):
            self.daily_demand[day_num] = (
                daily_demand_factors[day_num] * self.mean_demand_per_day
            )

    def _calculate_reward(self):
        # Reward is based on how many units expired and how many had
        # to be backordered

        cost = (self.daily_expiries * self.expiry_cost) + (
            self.daily_backorders * self.backorder_cost
        )
        return -cost

    def _create_initial_stock(self):
        # Assume mean amount has been ordered and requested
        # each previous day

        self.state["available_stock"].items = [
            PlateletUnit(t)
            for t in range(-self.max_age + 1, -self.transit_time + 1)
            for i in range(self.mean_demand_per_day)
        ]
        self.state["in_transit"].items = [
            PlateletUnit(t)
            for t in range(-self.transit_time + 1, 0)
            for i in range(self.mean_demand_per_day)
        ]

    def _create_stock_counter(self):
        # Utility for summarizing stock levels

        zero_counts = {}
        for i in range(1, self.max_age):
            zero_counts[i] = 0
        return Counter(zero_counts)

    def _fill_prescription(self):
        # Fill order or raise a backorder

        if len(self.state["available_stock"].items) > 0:
            self.state["available_stock"].get()
        else:
            self.daily_backorders += 1

    def _generate_demand(self):
        # Assume that demand is Poisson and therefore
        # wait between requests follows exponential distribution

        while True:

            self._fill_prescription()

            yield self.simpy_env.timeout(
                self.np_rng.exponential(1 / self.daily_demand[self.state["weekday"]])
            )

    def _get_observation(self):

        # Use counters to summarize how many units of stock
        # are held of each age

        counter = self._create_stock_counter()
        counter.update(
            [
                self.simpy_env.now - plt.order_day
                for plt in self.state["available_stock"].items
            ]
        )
        counter.update(
            [
                self.simpy_env.now - plt.order_day
                for plt in self.state["in_transit"].items
            ]
        )
        units = [v for k, v in counter.items()]

        return np.array([self.state["weekday"], *units])

    def _islegal(self, action):
        # Check if the supplied action is allowed

        if action < 0:
            raise ValueError("An order cannot be negative")
        elif action > self.max_order:
            raise ValueError(f"Order cannot exceed {self.max_order} units")
        elif not self.action_space.contains(action):
            raise ValueError("Requested action is not allowed")

    def _order_units(self, action):
        # Units ordered from blood service based on selected action

        self.state["in_transit"].items.extend(
            [PlateletUnit(self.simpy_env.now) for i in range(action)]
        )

    def _remove_expired_units(self):

        self.daily_expiries = len(
            [
                plt
                for plt in self.state["available_stock"].items
                if plt.order_day <= self.simpy_env.now - self.max_age
            ]
        )

        self.state["available_stock"].items = [
            plt
            for plt in self.state["available_stock"].items
            if plt.order_day > self.simpy_env.now - self.max_age
        ]

    def _receive_ordered_units(self):
        received = [
            plt
            for plt in self.state["in_transit"].items
            if plt.order_day <= self.simpy_env.now - self.transit_time
        ]
        self.state["available_stock"].items.extend(received)

        self.state["in_transit"].items = [
            plt
            for plt in self.state["in_transit"].items
            if plt.order_day > self.simpy_env.now - self.transit_time
        ]

    def render(self):

        print(f"Weekday: {self.state['weekday']}")

        available = self._create_stock_counter()
        available.update(
            [
                self.max_age - (self.simpy_env.now - plt.order_day)
                for plt in self.state["available_stock"].items
            ]
        )
        available_df = pd.DataFrame([available], index=["available units"])
        available_df.columns = pd.MultiIndex.from_product(
            [["remaining days"], available_df.columns]
        )

        print(available_df)

        transit = self._create_stock_counter()
        transit.update(
            [
                self.max_age - (self.simpy_env.now - plt.order_day)
                for plt in self.state["in_transit"].items
            ]
        )
        transit_df = pd.DataFrame([transit], index=["units in transit"])
        transit_df.columns = pd.MultiIndex.from_product(
            [["remaining days"], transit_df.columns]
        )

        print(transit_df)

    def reset(self):

        self.simpy_env = simpy.Environment()
        self.next_time_stop = 0

        self.daily_expiries = 0
        self.daily_backorders = 0

        # Set up the state
        self.state["weekday"] = 0
        self.state["available_stock"] = simpy.Store(self.simpy_env)
        self.state["in_transit"] = simpy.Store(self.simpy_env)
        self._create_initial_stock()

        self.simpy_env.process(self._generate_demand())

        observation = self._get_observation()

        return observation

    def seed(self, seed=None):

        self.np_rng = np.random.default_rng(seed)
        seed_value = self.np_rng.bit_generator._seed_seq.entropy

        return [seed_value]

    def step(self, action):

        # Check the action is legal
        self._islegal(action)

        # Reset expiries and backorders for the current day
        self.daily_expiries = 0
        self.daily_backorders = 0

        # Add ordered stock to in transit
        self._order_units(action)

        # Run the sumulation for one timestep
        self.next_time_stop += self.time_step
        self.simpy_env.run(until=self.next_time_stop)

        # Remove expired units
        self._remove_expired_units()

        # Add received stock to available inventory
        self._receive_ordered_units()

        # Update the weekday
        self.state["weekday"] = (self.state["weekday"] + 1) % 7

        # Get the new observation
        observation = self._get_observation()

        # Determine if simulation finised to report in output tuple as
        # expected for Gym environments
        terminal = True if self.simpy_env.now > self.sim_duration else False

        # Calculate reward based on expiries and backorders
        reward = self._calculate_reward()

        # Supply expiries and backorders to be tracked by agent/logs
        info = {
            "daily_expiries": self.daily_expiries,
            "daily_backorders": self.daily_backorders,
        }

        if self.render_env:
            self.render()

        return (observation, reward, terminal, info)