"""
Train an agent to play Mortal Kombat II using DQN from Stable Baselines 3
"""

import argparse
import gymnasium as gym
import numpy as np
from custom_ppo.ppo import PPOAgent
from gymnasium.wrappers import TimeLimit
import torch
from torch.distributions import Categorical
import time
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecNormalize
)
from stable_baselines3.common.monitor import Monitor

import stable_retro as retro

# --- CUSTOM WRAPPERS ---

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gymnasium environment and translate discrete actions into multi-binary actions.
    This is required for DQN to work with Retro environments.
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[button] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        # Debugging: Print the type of `act` to confirm its type
        if isinstance(act, np.ndarray):
            act = int(np.argmax(act))  # Example: Use argmax to map to a discrete action index
        return self._decode_discrete_action[act].copy()


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


# --- CONFIGURATION ---

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)

    # --- FIXED GENESIS BUTTON MAPPING ---
    # The Retro Genesis controller has 12 buttons.
    # Array Order: ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    # Indices:       0    1     2        3      4      5       6        7      8    9   10   11
    
    mk2_combos = [
        [],             # 0: No-Op
        
        # --- MOVEMENT ---
        [4],            # 1: Up (Jump)
        [5],            # 2: Down (Crouch)
        [6],            # 3: Left
        [7],            # 4: Right
        [6, 4],         # 5: Up + Left (Jump Back)
        [7, 4],         # 6: Up + Right (Jump Fwd)
        
        # --- ATTACKS (Single Buttons) ---
        [1],            # 7:  A (Low Punch)
        [10],           # 8:  X (High Punch)
        [8],            # 9:  C (Low Kick)
        [11],           # 10: Z (High Kick)
        
        # --- BLOCKING (Crucial) ---
        [9],            # 11: Y (Block)
        [5, 9],         # 12: Down + Block (Crouch Block) - ESSENTIAL
    ]

    
    env = Discretizer(env, combos=mk2_combos)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = Monitor(env)
    return env

def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari
    """
    env = WarpFrame(env) # Grayscale + Resize to 84x84
    env = MaxAndSkipEnv(env, skip=4) # Bin rewards to {-1, 0, 1} for stability
    return env


def watch_agent_play():
    parser = argparse.ArgumentParser()
    # Default changed to MKII
    parser.add_argument("--game", default="MortalKombatII-Genesis-v0") 
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env
    # 1. Recreate Environment (Usually just 1 env is enough for watching)
    env = SubprocVecEnv([make_env]) # Only 1 env needed to watch
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 2. Load Statistics
    # We still load them because the agent expects inputs/rewards to be scaled
    #env = VecNormalize.load("vec_normalise.pkl", env)

    # CRITICAL SETTINGS FOR WATCHING:
    env.training = False     # Do not update stats (freeze the "glasses")
    env.norm_reward = False  # Return RAW rewards (e.g. +100) so you can see the real score

    # 3. Initialize Agent
    model = PPOAgent(env=env) # Hyperparams don't matter for watching, only playing
    model.load("ppo_mk2.pt")

    # 4. Play Loop
    obs = env.reset()
    done = False
    i = 0
    win_count = 0
    avg_reward = 0
    while i < 100:
        # PPOAgent.policy returns (logits, values), we just need to sample the action
        # You might need to add a predict() method to your PPOAgent or manually call the policy:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).float().to(model.device)
            logits, _ = model.policy(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().cpu().numpy()

        obs, reward, done, info = env.step(action)
        if done:
            if reward > 0:
                win_count += 1
            avg_reward += reward
            i += 1
        # Optional: Render if your environment supports it, 
        # or relying on the emulator window if visible.
        # env.render() 
        
        time.sleep(0.01) # Slow down slightly to watch
    
    print(f"win rate: {win_count}")
    print(f"reward: {avg_reward/100}")


def resume_training():
    parser = argparse.ArgumentParser()
    # Default changed to MKII
    parser.add_argument("--game", default="MortalKombatII-Genesis-v0") 
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env

    # 1. Recreate the Base Environment (Must match exactly!)
    n_envs = 6
    env = SubprocVecEnv([make_env] * n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 2. Load the Statistics
    # Instead of creating a fresh VecNormalize, we load the old one
    print("Loading environment stats...")
    env = VecNormalize.load("vec_normalise.pkl", env)
    
    # Critical: Ensure it is in training mode so it keeps updating stats
    env.training = True 
    env.norm_reward = True

    # 3. Re-initialize the Agent
    # Note: You can change learning_rate here to lower it for fine-tuning
    model = PPOAgent(
        env=env,
        learning_rate=1.0e-4, # Lower LR for resuming
        gamma=0.99,
        clip=0.2,
        timesteps_per_batch=8192,
        max_ep_len=3000,
        n_updates_per_iteration=8,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # 4. Load the Agent Weights
    print("Loading agent weights...")
    model.load("ppo_mk3.pt")

    # 5. Continue Learning
    print("Resuming training...")
    model.learn(20_000, log_interval=1)
    
    # Save again when done
    model.save("ppo_mk3_continued.pt")
    env.save("vec_normalize_continued.pkl")


def main():
    parser = argparse.ArgumentParser()
    # Default changed to MKII
    parser.add_argument("--game", default="MortalKombatII-Genesis-v0") 
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env



    # DQN typically requires a buffer. 
    # Warning: Multiprocessing with Large Replay Buffers + FrameStack can consume massive RAM.
    # If you run out of RAM, reduce n_envs or buffer_size.
    n_envs = 6
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * n_envs), n_stack=4))
    env = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPOAgent(
        env=env,
        learning_rate=1.5e-4,
        gamma=0.99,
        clip=0.2,
        timesteps_per_batch=8192,
        max_ep_len=3000,
        n_updates_per_iteration=8,
        batch_size=64,
        ent_coef=0.001,
        vf_coef=0.5,
    )
    #model.load("ppo_mk2.pt")
    print(f"Training PPO on {args.game}...")
    model.learn(8_000_000, log_interval=1)
    
    # Save the model
    model.save("ppo_mk3.pt")
    env.save("vec_normalise.pkl")

if __name__ == "__main__":
    watch_agent_play()
    #resume_training()
    #main()