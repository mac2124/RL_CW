import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .network import FeedForwardNN
import numpy as np

class PPOAgent:
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env

        self.action_dim = env.action_space.n

        if hasattr(env.observation_space, 'shape'):
            # Calculate the flattened size of the observation space
            self.state_dim = int(np.prod(env.observation_space.shape))
        else:
            raise ValueError("Unsupported observation space type")

        self.actor = FeedForwardNN(self.state_dim, self.action_dim)
        self.critic = FeedForwardNN(self.state_dim, 1)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:
            batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens = self.collect_trajectories()
            timesteps_so_far += sum(batch_lens)

            V,_ = self.evaluate(batch_states, batch_actions)
            advantage = batch_rewards_to_go - V.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                _, curr_log_probs = self.evaluate(batch_states, batch_actions)

                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage
                actor_loss = (-torch.min(surr1, surr2)).mean()

                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def collect_trajectories(self):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rewards = []
            state = self.env.reset()
            done = False
            for ep_t in range(self.max_ep_len):
                t += 1.
                batch_states.append(state)
                action, log_prob = self.get_action(state)
                state, reward, done, _ = self.env.step(action)

                # Flatten the state if it is multidimensional
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state.flatten()
                
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                ep_rewards.append(reward)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)
        
        batch_states = torch.tensor(batch_states, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        return batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens
    
    def get_action(self, state):
        mean = self.actor(state)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rewards_to_go.insert(0, discounted_reward)
        return torch.tensor(batch_rewards_to_go, dtype=torch.float)
    
    def evaluate(self, batch_states, batch_actions):
        V = self.critic(batch_states).squeeze()

        mean = self.actor(batch_states)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs
    
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_ep_len = 1600
        self.gamma = 0.99
        self.clip = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.0003
        self.train_epochs = 80