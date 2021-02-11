# Based on DQN model for Pong from Chapter 6 of Deep Reinforcement Learning Hands On 2nd Ed by Maxim Lapan
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

# TODO: docstrings when design finalised
# TODO: consider whether logged model should be loaded from best checkpoint
# TODO: consider what else could be logged so training can be continued

import collections
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import mode
import gym
import mlflow

from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_deterministic(True)

from bloodbank_rl.environments.simple_platelet_bank import PlateletBankGym
from bloodbank_rl.models.fc_dqn import FCDQN

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, capacity, seed):
        self.buffer = collections.deque(maxlen=capacity)
        self.np_rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        return self.buffer.append(experience)

    def sample(self, batch_size):
        indices = self.np_rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env, exp_buffer, seed):
        self.env = env
        self.exp_buffer = exp_buffer
        self.np_rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.expiries = []
        self.backorders = []
        self.actions = []

    # Play a single step following an epsilon greedy policy
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if self.np_rng.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        self.actions.append(action)
        new_state, reward, is_done, info = self.env.step(action)
        self.total_reward += reward

        # Log expirires and backorders for interest only
        self.expiries.append(info["daily_expiries"])
        self.backorders.append(info["daily_backorders"])

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            total_expiries = np.array(self.expiries).sum()
            total_backorders = np.array(self.backorders).sum()
            most_frequent_action = mode(np.array(self.actions))[0][0]
            self._reset()
            return done_reward, total_expiries, total_backorders, most_frequent_action
        else:
            return None, None, None, None


def calc_loss(batch, discount_rate, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.FloatTensor(np.array(states)).to(device)
    next_states_v = torch.FloatTensor(np.array(next_states)).to(device)
    actions_v = torch.LongTensor(np.array(actions)).to(device)
    rewards_v = torch.FloatTensor(np.array(rewards)).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # Get the estimated Q-values for the actions actually taken
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        # Get the maximum predicted Q-value for each example in the batch
        next_state_values = tgt_net(next_states_v).max(1)[0]

        # There is no next state if we are on the final timestep
        next_state_values[done_mask] = 0.0

        # Use `detach()` so the gradients don't flow from the loss to the target network
        # The target network should only be updated periodically by copying the weights from
        # the main network
        next_state_values = next_state_values.detach()

    # Compute the target Q-value
    expected_state_action_values = rewards_v + discount_rate * next_state_values

    return nn.MSELoss()(state_action_values, expected_state_action_values)


@hydra.main(config_name="config")
def main(cfg):

    print(OmegaConf.to_yaml(cfg))

    # Paths for logging
    now_day = datetime.strftime(datetime.now(), "%Y-%m-%d")
    now_time = datetime.strftime(datetime.now(), "%H-%M-%S")
    cp_path = Path(f"model_checkpoints/{now_day}/{now_time}/checkpoints")
    cp_path.mkdir(parents=True, exist_ok=True)

    # Seed PyTorch for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Log the training with MLFlow
    with mlflow.start_run():

        # Log hyperparameters
        for space in ["env", "model", "training"]:
            for key, value in cfg[space].items():
                mlflow.log_param(key, value)

        env = PlateletBankGym(
            cfg.env.mean_demand_per_day,
            cfg.env.max_order,
            cfg.env.max_age,
            cfg.env.transit_time,
            seed=cfg.env.env_seed,
        )
        net = FCDQN(
            env.observation_space.shape, env.action_space.n, cfg.model.n_hidden
        ).to(cfg.training.device)
        tgt_net = FCDQN(
            env.observation_space.shape, env.action_space.n, cfg.model.n_hidden
        ).to(cfg.training.device)

        buffer = ExperienceBuffer(cfg.training.replay_size, cfg.training.seed)
        agent = Agent(env, buffer, cfg.training.seed)
        epsilon = cfg.training.epsilon_start

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate)
        total_rewards = []
        day_index = 0
        best_m_reward = None
        n_simulations = 0

        while len(total_rewards) < cfg.training.total_years:
            day_index += 1

            # Decay epsilon linearly overtime to reduce exploration
            epsilon = max(
                cfg.training.epsilon_final,
                cfg.training.epsilon_start
                - day_index / cfg.training.epsilon_decay_last_timestep,
            )

            (
                reward,
                total_expiries,
                total_backorders,
                most_frequent_action,
            ) = agent.play_step(net, epsilon, device=cfg.training.device)

            # Log results at the end of each simulation
            if reward is not None:
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                print(
                    f"Done {len(total_rewards)} years, reward: {m_reward:.3f}, epsilon: {epsilon:.3f}"
                )

                mlflow.log_metric("epsilon", epsilon, day_index)
                mlflow.log_metric("reward_100", m_reward, day_index)
                mlflow.log_metric("reward", reward, day_index)
                mlflow.log_metric("expiries", total_expiries, day_index)
                mlflow.log_metric("backorders", total_backorders, day_index)
                mlflow.log_metric(
                    "most_frequent_action", most_frequent_action, day_index
                )

                # If the mean reward has improved, checkpoint the model
                if best_m_reward is None or best_m_reward < m_reward:
                    torch.save(
                        net.state_dict(),
                        cp_path.joinpath(f"{m_reward}.dat"),
                    )
                    if best_m_reward is not None:
                        print(
                            f"Best mean reward updated: {best_m_reward} -> {m_reward}"
                        )
                    best_m_reward = m_reward

            # Don't start training until the replay buffer is full
            if len(buffer) < cfg.training.replay_start_size:
                continue

            # Periodically update the DQN target network with current weights
            if day_index % cfg.training.sync_target_timesteps == 0:
                tgt_net.load_state_dict(net.state_dict())

            # Update parameters
            optimizer.zero_grad()
            batch = buffer.sample(cfg.training.batch_size)
            loss_t = calc_loss(
                batch, cfg.training.gamma, net, tgt_net, device=cfg.training.device
            )
            loss_t.backward()
            optimizer.step()

        mlflow.pytorch.log_model(net, "model")
        mlflow.log_artifact(cp_path)
        mlflow.log_artifacts(cfg.hydra_logdir, artifact_path="hydra_logs")


if __name__ == "__main__":
    main()
