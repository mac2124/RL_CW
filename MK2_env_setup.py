"""
Environment setup for Mortal Kombat II – Genesis
For use with any Actor–Critic model (PyTorch, SB3, custom A2C/PPO, etc.)
"""

import gymnasium as gym
import numpy as np
import retro

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------
# Discrete Action Wrapper (convert discrete → multi-binary)
# ---------------------------------------------------------
class Discretizer(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self._decode = []
        for combo in combos:
            arr = np.zeros(env.action_space.n, dtype=bool)
            for i in combo:
                arr[i] = True
            self._decode.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode))

    def action(self, act):
        return self._decode[act].copy()


# ---------------------------------------------------------
# Stochastic Frame Skip Wrapper
# ---------------------------------------------------------
class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = truncated = False
        total_reward = 0

        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac

            if self.supports_want_render and i < self.n - 1:
                obs, reward, terminated, truncated, info = self.env.step(
                    self.curac, want_render=False
                )
            else:
                obs, reward, terminated, truncated, info = self.env.step(self.curac)

            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# ---------------------------------------------------------
# Retro + MK2 Configuration
# ---------------------------------------------------------
def make_retro_env(game="MortalKombatII-Genesis", state=None, scenario=None, max_ep_len=4500):
    if state is None:
        state = retro.State.DEFAULT

    env = retro.make(game, state, scenario=scenario)

    # Genesis Button Layout
    mk2_combos = [
        [],             # 0: noop
        [4],            # jump
        [5],            # crouch
        [6],            # back
        [7],            # forward
        [6, 4],         # jump back
        [7, 4],         # jump forward
        [5, 6],         # down + back
        [5, 7],         # down + forward
        [0],            # B (HK)
        [1],            # A (LK)
        [8],            # C (LP)
        [9],            # Y (HP)
        [10],           # X (block)
        [11],           # Z (run/block)
        [5, 12],        # uppercut


        
    ]

    env = Discretizer(env, combos=mk2_combos)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)

    if max_ep_len is not None:
        env = TimeLimit(env, max_episode_steps=max_ep_len)

    return Monitor(env)


# ---------------------------------------------------------
# DeepMind-Style Processing (resize, grayscale, clip reward)
# ---------------------------------------------------------
def wrap_deepmind(env):
    env = WarpFrame(env)      # 84×84 grayscale
    env = ClipRewardEnv(env)  # reward ∈ {−1, 0, +1}
    return env


# ---------------------------------------------------------
# Final multi-env MK2 environment constructor
# Use this for Actor–Critic training loops
# ---------------------------------------------------------
def make_mk2_env(n_envs=8, scenario=None):
    def make_single():
        env = make_retro_env(
            game="MortalKombatII-Genesis",
            scenario=scenario,
        )
        env = wrap_deepmind(env)
        return env

    venv = SubprocVecEnv([make_single] * n_envs)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)

    return venv
