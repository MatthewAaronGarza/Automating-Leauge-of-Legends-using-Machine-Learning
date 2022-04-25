import numpy as np

import gym
from gym.spaces import Box, Discrete

from pylol.lib import actions, point

from lolgym.envs.lol_game import LoLGameEnv

_NO_OP = [actions.FUNCTIONS.no_op.id]

class Escape1DEnv(LoLGameEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def transform_obs(self, obs):
        obs = np.array(obs[0].observation["enemy_unit"].distance_to_me, dtype=np.float32)[None]
        return obs

    def reset(self):
        obs = self.transform_obs(super().reset())
        return obs

    def _safe_step(self, act):
        act_x = 8 if act else 0
        act_y = 4
        act = [[1, point.Point(act_x, act_y)],
                _NO_OP]

        obs_n, reward_n, done_n, _ = super()._safe_step(act)

        obs = self.transform_obs(obs_n) # obs_n[0].observation["enemy_unit"].distance_to_me
        reward = obs_n[0].observation["enemy_unit"].distance_to_me.item()
        reward = reward if reward else 0.0 # Ensures reward is something sensible
        done = all(done_n)
        return obs, reward, done, {}

    @property
    def action_space(self):
        if self._env is None:
            self._init_env()
        action_space = Discrete(2)
        return action_space

    @property
    def observation_space(self):
        if self._env is None:
            self._init_env()
        observation_space = Box(low=0, high=24000, shape=(1,), dtype=np.float32)
        return observation_space