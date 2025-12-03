import torch 
import torch.nn as nn
import time
from torch.distributions import MultivariateNormal
from .network import FeedForwardNN
import numpy as np

class PPOAgent:
    def __init__(self,
                 env,
                 learning_rate = 3e-4,
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

        self.episode_count = 0
        # Ensure the state dimension matches the flattened observation space
        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = env.action_space.n

        self.actor = FeedForwardNN(self.state_dim, self.action_dim).to(self.device)
        self.critic = FeedForwardNN(self.state_dim, 1).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.ep_info_buffer = []

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}


    def learn(self, total_timesteps, log_interval=1):
        timesteps_so_far = 0
        iteration = 0

        while timesteps_so_far < total_timesteps:
            batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens = self.collect_trajectories()
            timesteps_so_far += sum(batch_lens)
            iteration += 1

            self.logger['t_so_far'] = timesteps_so_far
            self.logger['i_so_far'] = iteration

            V, _ = self.evaluate(batch_states, batch_actions)
            advantage = batch_rewards_to_go - V.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            

            for _ in range(self.n_updates_per_iteration):
                # Linear learning rate decay
                
                V, curr_log_probs = self.evaluate(batch_states, batch_actions)

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

                self.logger['actor_losses'].append(actor_loss.item())
            
            self._log_summary()



    def collect_trajectories(self):
        """
        Collect trajectories from multiple parallel environments.
        
        Returns:
            - batch_states: tensor of shape (total_timesteps, state_dim)
            - batch_actions: tensor of shape (total_timesteps,)
            - batch_log_probs: tensor of shape (total_timesteps,)
            - batch_rewards_to_go: tensor of shape (total_timesteps,)
            - batch_lens: list of episode lengths
        """
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = []
        
        # Track per-environment episode rewards and steps
        ep_rewards = []

        t = 0 
                
        while t < self.timesteps_per_batch:
            ep_rewards = []

            states = self.env.reset()
            done = [False] 

            for ep_t in range(self.max_ep_len):
                t += 1
                batch_states.append(states)
                actions, log_probs = self.get_action(states)
                
                # Handle environments returning 4 or 5 values from `step()`
                step_result = self.env.step(actions)
                if len(step_result) == 4:
                    next_states, rewards, terminated, info = step_result
                    truncateds = False  # Default truncated to False if not returned
                else:
                    next_states, rewards, terminated, truncateds, info = step_result

                dones = terminated | truncateds
                
                batch_actions.append(actions)
                batch_log_probs.append(log_probs)
                ep_rewards.append(rewards)

                if all(dones):
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)
        
        # Convert to tensors
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).to(self.device)
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        self.logger['batch_rews'] = [batch_rewards]
        self.logger['batch_lens'] = [batch_lens]

        return batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens

    
    def get_action(self, states):
        """
        Get actions for a batch of states from all environments.
        
        Args:
            states: numpy array of shape (n_env, state_dim)
        
        Returns:
            - actions: numpy array of shape (n_env,)
            - log_probs: numpy array of shape (n_env,)
        """
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            means = self.actor(states_tensor)
        
        dist = MultivariateNormal(means, self.cov_mat)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions.cpu().detach().numpy(), log_probs.cpu().detach()
    
    def compute_rewards_to_go(self, batch_rewards):
        """
        Compute discounted cumulative rewards-to-go.
        
        Args:
            batch_rewards: list of episode reward lists
        
        Returns:
            torch tensor of discounted rewards
        """
        batch_rewards_to_go = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rewards_to_go.insert(0, discounted_reward)
        return torch.tensor(batch_rewards_to_go, dtype=torch.float).to(self.device)
    
    def evaluate(self, batch_states, batch_actions):
        """
        Evaluate the critic and actor for a batch of states and actions.
        
        Args:
            batch_states: tensor of shape (batch_size, state_dim)
            batch_actions: tensor of shape (batch_size,)
        
        Returns:
            - V: state values, shape (batch_size,)
            - log_probs: log probabilities of actions, shape (batch_size,)
        """
        V = self.critic(batch_states).squeeze()

        means = self.actor(batch_states)
        dist = MultivariateNormal(means, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)
        
        return V, log_probs
    
    def _log_summary(self):

        # Calculate and print summary statistics
        avg_actor_loss = np.mean(self.logger['actor_losses'])
        avg_ep_reward = np.mean(self.logger['batch_rews']) if self.logger['batch_rews'] else 0
       
        print(f"Iteration {self.logger['i_so_far']} \t "
                f"Avg Actor Loss: {avg_actor_loss:.3f} \t "
                f"Avg Ep Reward: {avg_ep_reward:.3f} \t "
                f"Timesteps So Far: {self.logger['t_so_far']}")


