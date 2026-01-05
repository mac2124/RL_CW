
import torch
import torch.nn as nn
import time
from torch.distributions import Categorical
from .network import CNNPolicy
import numpy as np
import csv
import matplotlib.pyplot as plt


class PPOAgent:
    def __init__(self,
                 env,
                 learning_rate=2.5e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip=0.2,
                 timesteps_per_batch=4096,
                 max_ep_len=1600,
                 n_updates_per_iteration=5,
                 batch_size=64,
                 ent_coef=0.01,
                 vf_coef=0.5):
        # Initialize PPO Agent with given hyperparameters
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.timesteps_per_batch = timesteps_per_batch
        self.max_ep_len = max_ep_len
        self.n_updates_per_iteration = n_updates_per_iteration
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_env = env.num_envs
        self.action_dim = env.action_space.n

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {self.device}")


        # policy returns (logits, value)
        self.policy = CNNPolicy(self.action_dim).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Logging
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'entropies': [],
        }

        # For saving trajectory info
        self.trajectory_info = []


    def save(self, model_name):
        """Saves the model parameters to the specified file."""
        torch.save(self.policy.state_dict(), model_name)


    def load(self, model_path):
        """Loads the model parameters from the specified file."""
        self.policy.load_state_dict(torch.load(model_path, weights_only=True))


    def learn(self, total_timesteps, log_interval=1):
        """Main learning loop for PPO."""
        timesteps_so_far = 0
        iteration = 0

        while timesteps_so_far < total_timesteps:
            # collect rollout
            (obs_batch, actions_batch, old_log_probs_batch,
             returns_batch, advantages_batch, batch_lens, ep_rews_list) = self.collect_trajectories_and_gae()

            # update timesteps so far
            timesteps_so_far += obs_batch.size(0)
            iteration += 1

            # Logging
            self.logger['t_so_far'] = timesteps_so_far
            self.logger['i_so_far'] = iteration
            self.logger['batch_lens'] = batch_lens
            self.logger['batch_rews'] = ep_rews_list

            # normalize advantages
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # PPO update
            dataset_size = obs_batch.size(0)
            indices = np.arange(dataset_size)

            # Perform multiple epochs of updates
            for _ in range(self.n_updates_per_iteration):
                np.random.shuffle(indices)
                # Mini-batch updates
                for start in range(0, dataset_size, self.batch_size):
                    # create mini-batch
                    end = start + self.batch_size
                    mb_idx = indices[start:end]

                    # extract mini-batch data
                    mb_obs = obs_batch[mb_idx].to(self.device)
                    mb_actions = actions_batch[mb_idx].to(self.device)
                    mb_old_log_probs = old_log_probs_batch[mb_idx].to(self.device)
                    mb_returns = returns_batch[mb_idx].to(self.device)
                    mb_advantages = advantages_batch[mb_idx].to(self.device)

                    # forward current policy (get logits and value)
                    logits, values = self.policy(mb_obs)
                    dist = Categorical(logits=logits)
                    mb_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    # policy loss (PPO clipped surrogate)
                    ratios = torch.exp(mb_log_probs - mb_old_log_probs)
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # value loss
                    values = values.squeeze(-1)
                    critic_loss = nn.MSELoss()(values, mb_returns)

                    # combined loss
                    loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                    # backpropagation step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

                # log the last minibatch metrics
                self.logger['actor_losses'].append(actor_loss.item())
                self.logger['entropies'].append(entropy.item())

            # End of iteration logging
            self._log_summary()
    
        # save trajectory data
        with open('model_learning_info.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.trajectory_info)

        # create learning plots
        avg_ep_reward = [t[3] for t in self.trajectory_info]
        iteration_num = [t[0] for t in self.trajectory_info]

        # Plot average episode reward over iterations
        plt.plot(iteration_num, avg_ep_reward, marker="o")
        plt.xlabel("Timestep (x8196)")
        plt.ylabel("Average Episode Reward")
        plt.grid(True)
        plt.savefig("learning_plot.png", dpi=300, bbox_inches="tight")
        plt.close()



    def collect_trajectories_and_gae(self):
        """
        Collects data from vectorized envs and computes GAE advantages + returns.
        """
        # Buffers for collected data
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        log_probs_buf = []
        ep_rews_list = []
        batch_lens = []

        # reset env once
        states = self.env.reset()
        if isinstance(states, (tuple, list)):
            states = states[0]

        # convert to numpy, shape (n_env, C, H, W)
        states = np.asarray(states)

        # per-environment episode reward trackers
        ep_rewards_env = [[] for _ in range(self.n_env)]

        # collect data until we have timesteps_per_batch
        t = 0
        while t < self.timesteps_per_batch:
            # convert states to tensor
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                # get action distribution and value estimates
                logits, values = self.policy(states_tensor)   # values shape (n_env, 1)
                values = values.squeeze(-1).cpu().numpy()            # shape (n_env,)

                # create action distribution
                dist = Categorical(logits=logits)
                actions_tensor = dist.sample()
                log_probs_tensor = dist.log_prob(actions_tensor)

            actions_np = actions_tensor.cpu().numpy()
            log_probs_np = log_probs_tensor.cpu().numpy()

            # step vectorized env
            step_result = self.env.step(actions_np)
            if len(step_result) == 4:
                next_states, rewards, terminated, infos = step_result
                truncated = np.array([False] * self.n_env)
            else:
                next_states, rewards, terminated, truncated, infos = step_result

            # normalize
            rewards = np.array(rewards, dtype=np.float32)
            terminated = np.array(terminated, dtype=np.bool_)
            truncated = np.array(truncated, dtype=np.bool_)
            dones = np.logical_or(terminated, truncated)

            # store per-environment transition data in the same chronological order
            for env_i in range(self.n_env):
                obs_buf.append(states[env_i].copy())
                actions_buf.append(int(actions_np[env_i]))
                rewards_buf.append(float(rewards[env_i]))
                dones_buf.append(bool(dones[env_i]))
                values_buf.append(float(values[env_i]))
                log_probs_buf.append(float(log_probs_np[env_i]))

                ep_rewards_env[env_i].append(float(rewards[env_i]))
                
                # check for episode termination
                if dones[env_i]:

                    maybe_info = infos[env_i]
                    
                    if "episode" in maybe_info:
                        # extract the real unnormalised game score
                        real_score = maybe_info["episode"]["r"]
                        ep_rews_list.append([real_score])
                        batch_lens.append(maybe_info["episode"]["l"])
                    else:
                        # episode finished in this env; record episode rewards and length
                        ep_rews_list.append(ep_rewards_env[env_i])
                        batch_lens.append(len(ep_rewards_env[env_i]))
                    ep_rewards_env[env_i] = []

            # increment timestep
            t += self.n_env

            # move to next states
            states = next_states
            if isinstance(states, (tuple, list)):
                states = states[0]
            states = np.asarray(states)

        # compute last values for GAE
        last_states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, last_values = self.policy(last_states_tensor)
            last_values = last_values.squeeze(-1).cpu().numpy()  # shape (n_env,)

        # Close out remaining partial episode buffers
        for env_i in range(self.n_env):
            if len(ep_rewards_env[env_i]) > 0:
                ep_rews_list.append(ep_rewards_env[env_i])
                batch_lens.append(len(ep_rewards_env[env_i]))

        # Compute GAE advantages and returns per environment by walking the sequence backwards per env.
        # To do that we need to reconstruct per-env sequences in the same chronological order they were appended.
        # We'll iterate over the collected buffers in reverse and use last_values appropriately.
        T = len(rewards_buf)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        env_indices = np.array([i % self.n_env for i in range(T)], dtype=np.int32)

        # GAE computation
        adv = 0.0
        for idx in reversed(range(T)):
            env_i = env_indices[idx]
            reward = rewards_buf[idx]
            done = dones_buf[idx]
            value = values_buf[idx]
            next_index = idx + self.n_env
            if next_index < T:
                next_value = values_buf[next_index]
                next_done = dones_buf[next_index]
            else:
                next_value = last_values[env_i]
                next_done = False  # if next doesn't exist, we use last_value and assume not-done for formula (bootstrap)
            delta = reward + self.gamma * next_value * (1.0 - float(done)) - value
            adv = delta + self.gamma * self.lam * (1.0 - float(done)) * adv
            advantages[idx] = adv
            returns[idx] = adv + value

        # Convert collected lists into tensors in the same order
        obs_batch = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        actions_batch = torch.tensor(np.array(actions_buf), dtype=torch.long)
        old_log_probs_batch = torch.tensor(np.array(log_probs_buf), dtype=torch.float32)
        returns_batch = torch.tensor(returns, dtype=torch.float32)
        advantages_batch = torch.tensor(advantages, dtype=torch.float32)

        return obs_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch, batch_lens, ep_rews_list

    def evaluate(self, batch_states, batch_actions):
        """
        Evaluate (log_probs, values, entropy) for a batch of states & actions (torch tensors).
        """
        batch_states = batch_states.to(self.device)
        logits, values = self.policy(batch_states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_actions.to(self.device))
        entropy = dist.entropy()
        return values.squeeze(-1), log_probs, entropy

    def _log_summary(self):
        """Logs a summary of the current iteration."""

        avg_actor_loss = np.mean(self.logger['actor_losses']) if self.logger['actor_losses'] else 0.0
        avg_entropy = np.mean(self.logger['entropies']) if self.logger['entropies'] else 0.0

        flat_batch_rews = self.logger['batch_rews'] if self.logger['batch_rews'] else []
        if flat_batch_rews and isinstance(flat_batch_rews[0], list):
            mean_ep_return = np.mean([sum(ep) for ep in flat_batch_rews])
        else:
            # fallback
            mean_ep_return = 0.0

        print(f"Iteration {self.logger['i_so_far']} \t "
              f"Avg Actor Loss: {avg_actor_loss:.3f} \t "
              f"Entropy: {avg_entropy:.3f} \t "
              f"Avg Ep Reward: {mean_ep_return:.3f} \t "
              f"Timesteps So Far: {self.logger['t_so_far']}")

        data = (self.logger['i_so_far'], avg_actor_loss, avg_entropy, mean_ep_return, self.logger['t_so_far'])
        self.trajectory_info.append(data)


        # reset per-iteration logs
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['entropies'] = []
