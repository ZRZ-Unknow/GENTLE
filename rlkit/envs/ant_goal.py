import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, max_episode_steps=200, randomize_tasks=True, **kwargs):
        super(AntGoalEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) + 4.0# make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        goals = [[3 * np.random.random() ** 0.5, np.random.random() * 2 * np.pi] for _ in range(num_tasks)]
        goals = np.stack([[r*np.cos(a), r*np.sin(a)] for (r,a) in goals])
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])