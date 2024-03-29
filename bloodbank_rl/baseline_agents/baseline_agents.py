import numpy as np
from scipy.stats import mode, poisson, norm
import torch
import tianshou as ts

# We want a general agent class that we can then subclass to create agents that will take
# steps using different heuristics

# Might be nice to have a way to specify the additional information we want to collect
# separately - maybe we just log everything in the info dict that comes back?

# The specific agents (e.g. sS) assume an environment where the first element of the observation
# is the weekday and the rest are the inventory.
#  Might want to tweak this to make it a bit more flexible because it's just hard-coded
# at the moment.


class Agent:
    def __init__(self, env, seed):
        self.env = env
        self.np_rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.info_list = []

    # Play a single step following the heuristic policy
    def play_step(self):
        done_reward = None

        action = self._select_action()
        new_state, reward, is_done, info = self.env.step(action)
        # Just for debugging
        info["reward"] = reward
        self.info_list.append(info)
        self.total_reward += reward
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            temp_info_list = self.info_list.copy()
            self._reset()
            return done_reward, temp_info_list
        else:
            return None, None

    def _select_action(self):
        # For this version, we just randomly select an action
        # Note this depends on the seed of the environment, not the Agent
        action = self.env.action_space.sample()
        return action


class Agent_sS(Agent):
    def __init__(self, s_low_parameters, S_high_parameters, env, seed):
        super().__init__(env, seed)
        self.s_low_parameters = s_low_parameters
        self.S_high_parameters = S_high_parameters

    def _select_action(self):
        # Observation has weekday as first element, rest is inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        s_low = self.s_low_parameters[weekday]
        S_high = self.S_high_parameters[weekday]
        if inventory < s_low:
            return S_high - inventory
        else:
            return 0


class Agent_S(Agent):
    def __init__(self, S_high_parameters, env, seed):
        super().__init__(env, seed)
        self.S_high_parameters = S_high_parameters

    def _select_action(self):
        # Observation has weekday as first element, rest is inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        S_high = self.S_high_parameters[weekday]
        return max(S_high - inventory, 0)


class Agent_sQ(Agent):
    def __init__(self, s_low_parameters, Quantity_parameters, env, seed):
        super().__init__(env, seed)
        self.s_low_parameters = s_low_parameters
        self.Quantity_parameters = Quantity_parameters

    def _select_action(self):
        # Observation has weekday as first element, rest as inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        s_low = self.s_low_parameters[weekday]
        Quantity = self.Quantity_parameters[weekday]

        if inventory < s_low:
            return Quantity
        else:
            return 0


class Agent_sSaQ(Agent):
    def __init__(
        self,
        s_low_parameters,
        S_high_parameters,
        alpha_parameters,
        Quantity_parameters,
        env,
        seed,
    ):
        super().__init__(env, seed)
        self.s_low_parameters = s_low_parameters
        self.S_high_parameters = S_high_parameters
        self.alpha_parameters = alpha_parameters
        self.Quantity_parameters = Quantity_parameters

    def _select_action(self):
        # Observation has weekday as first element, rest as inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        s_low = self.s_low_parameters[weekday]
        S_high = self.S_high_parameters[weekday]
        alpha = self.alpha_parameters[weekday]
        Quantity = self.Quantity_parameters[weekday]

        if (inventory >= alpha) and (inventory < s_low):
            return Quantity
        elif inventory < alpha:
            return S_high - inventory
        else:
            return 0


class Agent_sSbQ(Agent):
    def __init__(
        self,
        s_low_parameters,
        S_high_parameters,
        beta_parameters,
        Quantity_parameters,
        env,
        seed,
    ):
        super().__init__(env, seed)
        self.s_low_parameters = s_low_parameters
        self.S_high_parameters = S_high_parameters
        self.beta_parameters = beta_parameters
        self.Quantity_parameters = Quantity_parameters

    def _select_action(self):
        # Observation has weekday as first element, rest as inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        s_low = self.s_low_parameters[weekday]
        S_high = self.S_high_parameters[weekday]
        beta = self.beta_parameters[weekday]
        Quantity = self.Quantity_parameters[weekday]

        if (inventory >= beta) and (inventory < s_low):
            return S_high - inventory
        elif inventory < beta:
            return Quantity
        else:
            return 0


# Agent that takes a StableBaselines3 model
# Use for evaluation after training that is
# compatible with other methods
class SBAgent(Agent):
    def __init__(self, model, env, seed):
        super().__init__(env, seed)
        self.model = model

    def _select_action(self):
        action, _states = self.model.predict(self.state, deterministic=True)
        return action


# Agent that takes a Tianshou policy
# Use for evaluation after training
# in a way compatible with other models
class TSAgent(Agent):
    def __init__(self, policy, env, seed):
        super().__init__(env, seed)
        self.policy = policy

    def _select_action(self):
        with torch.no_grad():
            batch = ts.data.Batch(
                obs=self.state.reshape(1, -1),
                act={},
                done={},
                obs_next={},
                info={},
                policy={},
            )
            action = torch.argmax(self.policy(batch)["logits"]).item()

        return action


# For now, we assume one value per weekday
# and that Poisson distributed
# TODO: pull these from the env
# rather than providing and argument here
class Agent_servicelevel(Agent):
    def __init__(self, mean_daily_demands, minimum_service_level, env, seed):
        super().__init__(env, seed)

        # Mean daily demands are M,T,W,T,F,S,S
        # Need to rearrange, so order-up-to on
        # Sunday based on Monday demand
        next_day_mean_demand = np.roll(mean_daily_demands, -1)

        # Calculate order up to level based on PPF on next day's demand distribution
        # Convert to int as expected by action space of env
        self.S_high_parameters = poisson.ppf(
            minimum_service_level, next_day_mean_demand
        ).astype(int)

    def _select_action(self):
        # Observation has weekday as first element, rest as inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        S_high = self.S_high_parameters[weekday]

        return max(S_high - inventory, 0)


class Agent_servicelevelNormal(Agent):
    def __init__(
        self, mean_daily_demands, std_daily_demands, minimum_service_level, env, seed,
    ):
        super().__init__(env, seed)

        # Mean daily demands are M,T,W,T,F,S,S
        # Need to rearrange, so order-up-to on
        # Sunday based on Monday demand
        next_day_mean_demand = np.roll(mean_daily_demands, -1)
        next_day_std_demand = np.roll(std_daily_demands, -1)

        # Calculate order up to level based on PPF on next day's demand distribution
        # Take the ceiling when generating from continuous distribution
        # CRound  as expected by action space of env
        self.S_high_parameters = np.ceil(
            norm.ppf(minimum_service_level, next_day_mean_demand, next_day_std_demand,)
        ).astype(int)
        print(self.S_high_parameters)

    def _select_action(self):
        # Observation has weekday as first element, rest as inventory
        weekday = int(self.state[0])
        inventory = np.sum(self.state[1:])
        S_high = self.S_high_parameters[weekday]

        return max(S_high - inventory, 0)


class Agent_lookuptable(Agent):
    def __init__(self, best_action_dict, env, seed):
        super().__init__(env, seed)
        # Load in a dictionary where keys are the state as a string
        # and values are the action to take
        # This allows us to easily implement to polocy learned from
        # Q-iteration, or any arbitrary policy
        self.best_action_dict = best_action_dict

    def _select_action(self):
        # Select best action using dict
        state_string = str(list(self.state))
        return self.best_action_dict[state_string]
