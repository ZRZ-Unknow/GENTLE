import numpy as np
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True, max_episode_steps=200):
        super(HopperRandParamsWrappedEnv, self).__init__()
        self.randomize_tasks = randomize_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self._max_episode_steps = max_episode_steps
        self.env_step = 0

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset(self):
        self.env_step = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.env_step += 1
        if self.env_step >= self._max_episode_steps:
            done = True
        return obs, reward, done, info

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def load_params(self, n_tasks):
        param_sets = []
        for i in range(n_tasks):
            param = np.load(f'/data/zrz/gentle_data/hopper-rand-params/goal_idx{i}/log.npy', allow_pickle=True)
            param_sets.append(param.item())
        return param_sets

