import torch
import torch.nn as nn
import time
from torch.distributions import MultivariateNormal
from .network import CNNPolicy
import numpy as np


class PPOAgent:
    def __init__(self,
                 env,
                 learning_rate=3e-4,
                 gamma=0.99,
                 clip=0.2,
                 timesteps_per_batch=4800,
                 max_ep_len=1600,
                 train_epochs=5,):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip = clip
        self.timesteps_per_batch = timesteps_per_batch
        self.max_ep_len = max_ep_len
        self.n_updates_per_iteration = train_epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_env = env.num_envs

        self.action_dim = env.action_space.n

        self.policy = CNNPolicy(self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.ep_info_buffer = []

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
        }

    def learn(self, total_timesteps, log_interval=1):
        timesteps_so_far = 0
        iteration = 0

        while timesteps_so_far < total_timesteps:
            (batch_states, batch_actions, batch_log_probs,
             batch_rewards_to_go, batch_lens) = self.collect_trajectories()

            timesteps_so_far += sum(batch_lens)
            iteration += 1
            self.logger['t_so_far'] = timesteps_so_far
            self.logger['i_so_far'] = iteration

            V, _ = self.evaluate(batch_states, batch_actions)
            advantage = batch_rewards_to_go - V.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_states, batch_actions)

                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage
                actor_loss = (-torch.min(surr1, surr2)).mean()

                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)

                total_loss = actor_loss + 0.5 * critic_loss 

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.logger['actor_losses'].append(actor_loss.item())

            self._log_summary()

    def collect_trajectories(self):
        """
        Collect trajectories from multiple parallel environments.

        Returns:
            - batch_states: tensor (T, C, H, W)
            - batch_actions: tensor (T, action_dim) or (T,) depending on your policy
            - batch_log_probs: tensor (T,)
            - batch_rewards_to_go: tensor (T,)
            - batch_lens: list of episode lengths
        """

        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []     
        batch_lens = []

        ep_rewards_env = [[] for _ in range(self.n_env)]
        ep_lengths_env = [0 for _ in range(self.n_env)]

        t = 0
        states = self.env.reset() 
        if isinstance(states, tuple) or isinstance(states, list):
            states = states[0]

        while t < self.timesteps_per_batch:
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                means, _ = self.policy(states_tensor / 255.0) 
                dist = MultivariateNormal(means, self.cov_mat)
                actions_tensor = dist.sample()
                log_probs_tensor = dist.log_prob(actions_tensor)

            actions_np = actions_tensor.cpu().numpy()
            log_probs_np = log_probs_tensor.cpu().numpy()

            step_result = self.env.step(actions_np)

            if len(step_result) == 4:
                next_states, rewards, terminated, infos = step_result
                truncated = np.array([False] * self.n_env)
            else:
                next_states, rewards, terminated, truncated, infos = step_result

            rewards = np.array(rewards)
            terminated = np.array(terminated)
            truncated = np.array(truncated)
            dones = np.logical_or(terminated, truncated)

            for env_i in range(self.n_env):
                batch_states.append(states[env_i])            # raw observation (C,H,W)
                batch_actions.append(actions_np[env_i])       # could be vector
                batch_log_probs.append(log_probs_np[env_i])   # scalar per action-vector
                ep_rewards_env[env_i].append(float(rewards[env_i]))
                ep_lengths_env[env_i] += 1

                if dones[env_i]:
                    ep_rewards = ep_rewards_env[env_i]
                    batch_rewards.append(ep_rewards)

                    batch_lens.append(ep_lengths_env[env_i])

                    ep_rewards_env[env_i] = []
                    ep_lengths_env[env_i] = 0

            t += self.n_env  
            states = next_states
            if isinstance(states, tuple) or isinstance(states, list):
                states = states[0]

        
        for env_i in range(self.n_env):
            if ep_lengths_env[env_i] > 0:
                ep_rewards = ep_rewards_env[env_i]
                batch_rewards.append(ep_rewards)
                batch_lens.append(ep_lengths_env[env_i])
                # no need to reset buffers (we discard them now)

       
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32).to(self.device)

        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32).to(self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32).to(self.device)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        self.logger['batch_rews'] = [batch_rewards]
        self.logger['batch_lens'] = [batch_lens]

        return batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens

    def get_action(self, states):
        """
        (Not used by the rewritten collect_trajectories above, but kept for compatibility)
        Get actions for a batch of states from all environments.

        Args:
            states: numpy array of shape (n_env, C, H, W)

        Returns:
            - actions: numpy array of shape (n_env, action_dim)
            - log_probs: numpy array of shape (n_env,)
        """
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            means, _ = self.policy(states_tensor / 255.0)
            dist = MultivariateNormal(means, self.cov_mat)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def compute_rewards_to_go(self, episodes_rewards_list):
        """
        episodes_rewards_list: list where each element is a list of rewards for one episode.
        We return a 1-D tensor with rewards-to-go for every timestep in order of episodes given.
        """
        flat_returns = []
        for ep_rewards in episodes_rewards_list:
            discounted = 0.0
            for r in reversed(ep_rewards):
                discounted = r + self.gamma * discounted
                flat_returns.insert(0, discounted)
        return torch.tensor(np.array(flat_returns), dtype=torch.float32).to(self.device)

    def evaluate(self, batch_states, batch_actions):
        """
        Evaluate the critic and compute log_probs for batch actions.
        batch_states: tensor (T, C, H, W)
        batch_actions: tensor (T, action_dim) or (T,)
        """

        batch_states = batch_states.to(self.device)
        means, values = self.policy(batch_states / 255.0)   
        V = values.squeeze()

        dist = MultivariateNormal(means, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)

        return V, log_probs

    def _log_summary(self):
        avg_actor_loss = np.mean(self.logger['actor_losses']) if self.logger['actor_losses'] else 0.0
        flat_batch_rews = self.logger['batch_rews'][0] if self.logger['batch_rews'] else []
        mean_ep_return = np.mean([sum(ep) for ep in flat_batch_rews]) if flat_batch_rews else 0.0

        print(f"Iteration {self.logger['i_so_far']} \t "
              f"Avg Actor Loss: {avg_actor_loss:.3f} \t "
              f"Avg Ep Reward: {mean_ep_return:.3f} \t "
              f"Timesteps So Far: {self.logger['t_so_far']}")
