"""
Train an agent to play Mortal Kombat II using Distributional DQN (QR-DQN)
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

# We switch from standard DQN to QRDQN (Quantile Regression DQN)
from sb3_contrib import QRDQN
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
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
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for i in combo:
                arr[i] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
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
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
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
    
    # Optimized MKII Combo List (Indices 0-11 only)
    mk2_combos = [
        [],             # 0: No-Op
        
        # --- MOVEMENT ---
        [4],            # 1: Up (Jump)
        [5],            # 2: Down (Crouch)
        [6],            # 3: Left
        [7],            # 4: Right
        [6, 4],         # 5: Up + Left (Jump Back)
        [7, 4],         # 6: Up + Right (Jump Fwd)
        
        # --- ATTACKS ---
        [1],            # 7:  A (Low Punch)
        [10],           # 8:  X (High Punch)
        [8],            # 9:  C (Low Kick)
        [11],           # 10: Z (High Kick)
        
        # --- DEFENSE ---
        [9],            # 11: Y (Block)
        [5, 9],         # 12: Down + Block (Crouch Block)
        
        # --- COMBOS/DIRECTIONALS ---
        [5, 10],        # 13: Down + High Punch (Uppercut)
        [5, 8],         # 14: Down + Low Kick (Low Poke)
        [6, 8],         # 15: Left + Low Kick (Sweep)
        [7, 8],         # 16: Right + Low Kick (Sweep)
        [6, 11],        # 17: Left + High Kick (Roundhouse)
        [7, 11],        # 18: Right + High Kick (Roundhouse)
        
    ]
    
    env = Discretizer(env, combos=mk2_combos)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    env = Monitor(env)
    return env


def wrap_deepmind_retro(env):
    env = WarpFrame(env) 
    env = ClipRewardEnv(env) 
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="MortalKombatII-Genesis") 
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env

    # Create Vector Environment
    n_envs = 8
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * n_envs), n_stack=4))

    # --- QR-DQN SETUP ---
    model = QRDQN(
        policy="CnnPolicy",
        env=venv,
        learning_rate=1e-4,
        buffer_size=100_000, 
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1, 
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./mk2_qrdqn_tensorboard/",
        
        # QR-DQN Specific Parameters
        policy_kwargs=dict(n_quantiles=50), # 50 quantiles is standard (similar to C51's 51 atoms)
    )

    print(f"Training Distributional DQN (QR-DQN) on {args.game}...")
    model.learn(
        total_timesteps=10_000_000, 
        log_interval=4,
    )
    
    model.save("qrdqn_mk2")

if __name__ == "__main__":
    main()