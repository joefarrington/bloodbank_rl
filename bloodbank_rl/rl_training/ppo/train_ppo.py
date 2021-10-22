import os

import tianshou as ts
import numpy as np
import gym
import torch
from torch import nn
import mlflow
from pathlib import Path
from datetime import datetime

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

    logger = hydra.utils.instantiate(cfg.logger)
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparameters(conf_dict)

    # Log Hydra config files as artifacts for easy replication
    logger.experiment.log_artifacts(logger.run_id, cfg.hydra_logdir)

    # Set up checkpointing if needed
    if cfg.checkpoints.save_checkpoints:
        cp_path = Path(cfg.checkpoints.path)
        cp_path.mkdir(parents=True, exist_ok=True)

        best_models_path = cp_path / "best_models"
        best_models_path.mkdir(parents=True, exist_ok=True)

        # Runs when we have a new best mean evaluation reward
        def save_fn(policy, cp_path=cp_path, mlflow_logger=logger):
            now_time = datetime.strftime(datetime.now(), "%H-%M-%S")
            model_checkpoint_path = cp_path / f"best_models/policy_{now_time}.pt"
            torch.save(policy.state_dict(), model_checkpoint_path)
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id, model_checkpoint_path, "best_models"
            )

        # Runs at frequency based on save_interval argument to logger
        def save_checkpoint_fn(epoch, env_step, gradient_step, cp_path=cp_path):
            checkpoint_dir = cp_path / f"epoch_{epoch:04}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_filepath = checkpoint_dir / "policy.pt"
            torch.save({"model": policy.state_dict()}, checkpoint_filepath)
            return checkpoint_filepath

    else:
        save_fn = None
        save_checkpoint_fn = None

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
        save_fn=save_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    print(f'Finished training in {result["duration"]}')
    logger.close()


if __name__ == "__main__":
    main()
