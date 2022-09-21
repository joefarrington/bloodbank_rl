import numpy as np
import torch
import tianshou as ts

# The specific agents (e.g. sS) assume an environment where the first element of the observation
# is the weekday and the rest are the inventory - this is hardcoded.


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


# Agent that takes in a lookup table, e.g. from the output of value iteration
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
