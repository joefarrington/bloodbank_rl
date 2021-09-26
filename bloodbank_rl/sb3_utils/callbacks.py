from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class LogInfoCallback(BaseCallback):
    # Use during training to log specific elements not logged by default
    # This works with a single environment, but the local dictionary elements
    # are different for a vectorised environment
    # Also need to consider how you would want to log these things for
    # multiple environments being traversed at the same time
    def __init__(self, verbose=0):
        super(LogInfoCallback, self).__init__(verbose)
        self.reset()

    def reset(self):
        self.daily_expiries = []
        self.daily_backorders = []
        self.daily_units_in_stock = []

    def _on_step(self):
        # For expiries, backorders and stock we log the mean over each episode
        # so it's the mean daily value of each
        self.daily_expiries.append(self.locals["infos"][0]["daily_expiries"])
        self.daily_backorders.append(self.locals["infos"][0]["daily_backorders"])
        self.daily_units_in_stock.append(self.locals["infos"][0]["units_in_stock"])

        if self.locals["done"][0]:
            ep_expiries = np.mean(self.daily_expiries)
            ep_backorders = np.mean(self.daily_backorders)
            ep_units_in_stock = np.mean(self.daily_units_in_stock)

            ep_reward = self.locals["infos"][0]["episode"]["r"]
            self.logger.record("cost_components/mean_expiries", ep_expiries)
            self.logger.record("cost_components/mean_backorders", ep_backorders)
            self.logger.record("cost_components/mean_units_in_stock", ep_units_in_stock)
            self.logger.record("rollout/episode_reward", ep_reward)
            self.reset()
        return True