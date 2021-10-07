import os

import tianshou as ts
import numpy as np
import gym
import torch
import mlflow

from bloodbank_rl.environments.platelet_bankSR import PlateletBankGym, SimpleProvider, PoissonDemandProviderSR
from bloodbank_rl.baseline_agents.baseline_agents import Agent

from bloodbank_rl.tianshou_utils.logging import TianshouMLFlowLogger, InfoLogger
from bloodbank_rl.tianshou_utils.models import FCDQN

def main():

    # Define environments
    env = PlateletBankGym(PoissonDemandProviderSR, {"sim_duration": 365}, 60, 3, 0, [0, 0, 1], 225, 650, 130, 3250, 650)
    train_envs = ts.env.DummyVectorEnv([lambda: PlateletBankGym(PoissonDemandProviderSR, {"sim_duration": 365}, 60, 3, 0, [0, 0, 1], 225, 650, 130, 3250, 650) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: PlateletBankGym(PoissonDemandProviderSR, {"sim_duration": 365}, 60, 3, 0, [0, 0, 1], 225, 650, 130, 3250, 650) for _ in range(1)])

    np.random.seed(57)
    torch.manual_seed(57)
    train_envs.seed(11)
    test_envs.seed(12)

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = FCDQN(state_shape, action_shape, n_hidden=[128,128,128], device='cuda').to(device='cuda')
    optim = torch.optim.Adam(net.parameters(), lr=1e-5)

    # Declare the policy
    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.75, estimation_step=1, target_update_freq=1000, is_double=False)

    # Declare the collectors
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(1000000, 1), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # Declare parameters for epsilon greedy exploration
    step_per_epoch = 5000
    max_epoch= 110

    max_steps = step_per_epoch * max_epoch
    eps_max = 1
    eps_min = 0.01
    exploration_fraction = 0.8

    def decay_eps(epoch, env_step):
        eps = max(eps_min, eps_max - env_step / (max_steps*exploration_fraction))
        return eps

    # Run and log training
    logger = TianshouMLFlowLogger(filename='train_dqn.py', experiment_name='tianshou_dqn', info_logger=InfoLogger())

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(1000000, 1), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True, preprocess_fn=logger.info_logger.preprocess_fn)

    train_collector.collect(n_step=10000, random=True)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=max_epoch, step_per_epoch=step_per_epoch, step_per_collect=1,
        update_per_step=1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(decay_eps(epoch, env_step)),
        test_fn=lambda epoch, env_step: policy.set_eps(0.00),
        logger=logger, verbose=False)
    print(f'Finished training! Use {result["duration"]}')
    logger.close()

if __name__ == '__main__':
    main()


