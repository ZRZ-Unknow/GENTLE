import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

from . import register_env

def read_log_params(log_file):
    params_dict = {}
    with open(log_file) as f:
        lines = f.readlines()
    cur_key = None
    for line in lines:
        if "'" in line:
            if ")" in line:
                last_entry = line.split(")")[0].split("[")[1].split("]")[0].split(",")
                #print(last_entry)
                last_entry_float = [float(s) for s in last_entry]
                params_dict[cur_key].append(np.array(last_entry_float))
            key = line.split("'")[1]
            #print('key is %s' %key)
            cur_key = key
            params_dict[key] = []
            if "(" in line:
                first_entry = line.split("(")[1].split("[")[2].split("]")[0].split(",")
                #print(first_entry)
                first_entry_float = [float(s) for s in first_entry]
                params_dict[cur_key].append(np.array(first_entry_float))
        else:
            entry = line.split("[")[1].split("]")[0].split(",")
            entry_float = [float(s) for s in entry]
            params_dict[cur_key].append(entry_float)
    for key, value in params_dict.items():
        params_dict[key] = np.array(params_dict[key])
    return params_dict


@register_env('walker-rand-params')
class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True, max_episode_steps=200):
        super(WalkerRandParamsWrappedEnv, self).__init__()
        self.randomize_tasks = randomize_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self._max_episode_steps = max_episode_steps
        self.env_step = 0
    
    def load_params(self, n_tasks):
        param_sets = []
        for i in range(n_tasks):
            param = np.load(f'/data/zrz/gentle_data/walker-rand-params/goal_idx{i}/log.npy', allow_pickle=True)
            param_sets.append(param.item())
        return param_sets
    
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


@register_env('sparse-walker-rand-params')
class SparseWalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True, max_episode_steps=200, goal_radius=0.5):
        super(SparseWalkerRandParamsWrappedEnv, self).__init__(n_tasks, randomize_tasks, max_episode_steps)
        self.goal_radius = goal_radius

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        #if reward >= self.goal_radius:
        #    sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= self.goal_radius)
        r = r * mask
        return r