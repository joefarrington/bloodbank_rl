import os

import tianshou as ts
import numpy as np
import gym
import torch
from torch import nn
import mlflow

from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from omegaconf import DictConfig, OmegaConf
import hydra

from bloodbank_rl.environments.platelet_bankSR import (
    PlateletBankGym,
    SimpleProvider,
    PoissonDemandProviderSR,
)
from bloodbank_rl.baseline_agents.baseline_agents import Agent

from bloodbank_rl.tianshou_utils.logging import TianshouMLFlowLogger, InfoLogger
from bloodbank_rl.tianshou_utils.policies import A2CPolicyforMLFlow


def dist_fn(probs):
    return torch.distributions.categorical.Categorical(probs=probs)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):

    # Single env just used to determine state and actions spaces
    env = hydra.utils.instantiate(cfg.env)
    train_envs = ts.env.DummyVectorEnv(
        [
            lambda: hydra.utils.instantiate(cfg.env)
            for _ in range(cfg.vec_env.n_train_envs)
        ]
    )

    test_envs = ts.env.DummyVectorEnv(
        [
            lambda: hydra.utils.instantiate(cfg.env)
            for _ in range(cfg.vec_env.n_test_envs)
        ]
    )

    # Seed everything for reproducibility
    # Train and test encs can either take int or list same length as n_envs
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_envs.seed(cfg.vec_env.train_env_seed)
    test_envs.seed(cfg.vec_env.test_env_seed)

    # Set up the actor and critic networks
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net_a = hydra.utils.instantiate(
        cfg.actor_network, state_shape=state_shape, activation=nn.Tanh
    )
    actor = Actor(net_a, action_shape=action_shape, device=cfg.device).to(cfg.device)

    net_c = hydra.utils.instantiate(
        cfg.critic_network, state_shape=state_shape, activation=nn.Tanh
    )
    critic = Critic(net_c, device=cfg.device).to(cfg.device)

    optim = hydra.utils.instantiate(
        cfg.optimiser, params=ActorCritic(actor, critic).parameters()
    )

    policy = hydra.utils.instantiate(
        cfg.policy,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
    )

    logger = hydra.utils.instantiate(cfg.logger, policy=policy)
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparameters(conf_dict)

    train_collector = hydra.utils.instantiate(
        cfg.train_collector, policy=policy, env=train_envs
    )
    test_collector = hydra.utils.instantiate(
        cfg.test_collector,
        policy=policy,
        env=test_envs,
        preprocess_fn=logger.info_logger.preprocess_fn,
    )

    result = hydra.utils.call(
        cfg.trainer,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        logger=logger,
    )

    # Additional logging
    logger.experiment.log_artifacts(logger.run_id, cfg.hydra_logdir)

    print(f'Finished training in {result["duration"]}')
    logger.close()


if __name__ == "__main__":
    main()
