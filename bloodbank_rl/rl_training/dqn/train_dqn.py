import os

import tianshou as ts
import numpy as np
import gym
import torch
import mlflow

from omegaconf import DictConfig, OmegaConf
import hydra

from bloodbank_rl.environments.platelet_bankSR import (
    PlateletBankGym,
    SimpleProvider,
    PoissonDemandProviderSR,
)
from bloodbank_rl.baseline_agents.baseline_agents import Agent

from bloodbank_rl.tianshou_utils.logging import TianshouMLFlowLogger, InfoLogger
from bloodbank_rl.tianshou_utils.models import FCDQN
from bloodbank_rl.tianshou_utils.exploration import EpsilonScheduler


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):

    logger = hydra.utils.instantiate(cfg.logger)
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparameters(conf_dict)

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

    # Set up the DQN network
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = hydra.utils.instantiate(
        cfg.model, state_shape=state_shape, action_shape=action_shape
    ).to(device=cfg.device)

    optim = hydra.utils.instantiate(cfg.optimiser, params=net.parameters())

    policy = hydra.utils.instantiate(cfg.policy, model=net, optim=optim)

    train_collector = hydra.utils.instantiate(
        cfg.train_collector, policy=policy, env=train_envs
    )
    test_collector = hydra.utils.instantiate(
        cfg.test_collector,
        policy=policy,
        env=test_envs,
        preprocess_fn=logger.info_logger.preprocess_fn,
    )

    # Collect samples to fill the buffer before training
    train_collector.collect(n_step=cfg.n_steps_before_learning, random=True)

    # Set the exploration schedule
    epsilon_scheduler = hydra.utils.instantiate(cfg.exploration)

    # Train the model
    result = hydra.utils.call(
        cfg.trainer,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        train_fn=lambda epoch, env_step: policy.set_eps(
            epsilon_scheduler.current_eps(epoch, env_step)
        ),
        test_fn=lambda epoch, env_step: policy.set_eps(0.00),
        logger=logger,
    )

    # Additional logging
    logger.experiment.log_artifacts(
        logger.run_id, cfg.hydra_logdir, artifact_path="hydra_logs"
    )

    print(f'Finished training in {result["duration"]}')
    logger.close()


if __name__ == "__main__":
    main()

