import numpy as np
import gym
from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


@register_env('ant-dir')
class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, max_episode_steps=200, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        super(AntDirEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)


    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )
    
    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            np.random.seed(1337)
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks
    
@register_env('ant-dir-v1')
class AntDirV1Env(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, max_episode_steps=200, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        super(AntDirV1Env, self).__init__(task, n_tasks, max_episode_steps, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )
    
    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        train_tasks = list(np.random.uniform(0., np.pi/2, size=(3,))) + list(np.random.uniform(np.pi/2, np.pi, size=(1,))) +\
                 list(np.random.uniform(np.pi, np.pi*3/2, size=(1,))) + list(np.random.uniform(np.pi*3/2, np.pi*2, size=(1,)))
        test_tasks = list(np.random.uniform(0., np.pi/2, size=(2,))) + list(np.random.uniform(np.pi/2, np.pi, size=(2,))) +\
                 list(np.random.uniform(np.pi, np.pi*3/2, size=(2,))) + list(np.random.uniform(np.pi*3/2, np.pi*2, size=(2,)))
        _tasks = train_tasks[:3] + train_tasks[4:5] + test_tasks
        tasks = [{'goal': velocity} for velocity in _tasks]
        return tasks

class VariBadWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 episodes_per_task
                 ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - normalized actions in case of continuous action space
        - adding the timestep / done info to the state (might be needed to make states markov)
        """

        super().__init__(env)

        # if continuous actions, make sure in [-1, 1]
        if isinstance(self.env.action_space, gym.spaces.Box):
            self._normalize_actions = True
        else:
            self._normalize_actions = False

        if episodes_per_task > 1:
            self.add_done_info = True
        else:
            self.add_done_info = False

        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box):
                if len(self.observation_space.shape) > 1:
                    raise ValueError  # can't add additional info for obs of more than 1D
                self.observation_space = spaces.Box(low=np.array([*self.observation_space.low, 0]),  # shape will be deduced from this
                                                    high=np.array([*self.observation_space.high, 1]),
                                                    dtype=np.float32)
            else:
                # TODO: add something simliar for the other possible spaces,
                # "Space", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"
                raise NotImplementedError

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = self.episodes_per_task * self.env.unwrapped._max_episode_steps

        # add dummy timesteps in-between episodes for resetting the MDP
        self.horizon_bamdp += self.episodes_per_task - 1

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    # def reset(self, task):
    def reset(self, task=None):

        # reset task -- this sets goal and state -- sets self.env._goal and self.env._state
        self.env.reset_task(task)

        self.episode_count = 0
        self.step_count_bamdp = 0

        # normal reset
        try:
            state = self.env.reset()
        except AttributeError:
            state = self.env.unwrapped.reset()

        if self.add_done_info:
            state = np.concatenate((state, [0.0]))

        self.done_mdp = False

        return state

    def reset_mdp(self):
        state = self.env.reset()
        # if self.add_timestep:
        #     state = np.concatenate((state, [self.step_count_bamdp / self.horizon_bamdp]))
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state

    def step(self, action):

        if self._normalize_actions:     # from [-1, 1] to [lb, ub]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        # do normal environment step in MDP
        state, reward, self.done_mdp, info = self.env.step(action)

        info['done_mdp'] = self.done_mdp

        # if self.add_timestep:
        #     state = np.concatenate((state, [self.step_count_bamdp / self.horizon_bamdp]))
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            info['start_state'] = self.reset_mdp()

        return state, reward, done_bamdp, info