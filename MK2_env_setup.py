"""
Train an agent to play Mortal Kombat II using DQN from Stable Baselines 3
"""

import argparse
import gymnasium as gym
import numpy as np
from custom_ppo.ppo import PPOAgent
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor

import retro

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


def main():
    parser = argparse.ArgumentParser()
    # Default changed to MKII
    parser.add_argument("--game", default="MortalKombatII-Genesis") 
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

    model = PPOAgent(
        env=venv,
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
    print(f"Training PPO on {args.game}...")
    model.learn(10_000_000, log_interval=1)
    
    # Save the model
    model.save("dqn_mk2")

if __name__ == "__main__":
    main()