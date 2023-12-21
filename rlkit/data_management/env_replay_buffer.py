import numpy as np
import torch
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple


class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
            goal_radius,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, SimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            goal_radius=goal_radius,
        )) for idx in tasks])


    def add_sample(self, task, observation, action, reward, terminal,
            next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
                observation, action, reward, terminal,
                next_observation, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def get_all_data(self, task):
        return self.task_buffers[task].get_all_data()


class SimpleContextBuffer(object):
    def __init__(
            self, max_replay_buffer_size, context_dim):
        self._context_dim = context_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._contexts = np.zeros((max_replay_buffer_size, context_dim))
        self.clear()

    def add_sample(self, context, **kwargs):
        self._contexts[self._top] = context
        self._advance()

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return self._contexts[indices]

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        assert self._size > 0
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size

    def get_all_data(self):
        return self.sample_data(range(self._size))
    

class MultiTaskContextBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
            context_dim,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self.task_buffers = dict([(idx, SimpleContextBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            context_dim=context_dim,
        )) for idx in tasks])

    def add_sample(self, task, context, **kwargs):
        if isinstance(task, int):
            for i in range(len(context)):
                self.task_buffers[task].add_sample(context[i], **kwargs)
        else:
            for i in range(len(task)):
                for j in range(context.shape[1]):
                    self.task_buffers[task[i]].add_sample(context[i][j], **kwargs)

    def random_batch(self, task, batch_size, sequence=False):
        batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def get_all_data(self, task):
        return self.task_buffers[task].get_all_data()
    
    def random_batch_task(self, batch_size, indices):
        batch = [self.random_batch(idx, batch_size) for idx in indices]
        batch = np.stack(batch)
        return batch
    
    def clear(self, idxs):
        for i in idxs:
            self.clear_buffer(i)
    
    def random_batch_task_ind(self, batch_size, indices):
        inds = np.random.randint(0, self.task_buffers[0]._size, batch_size)
        batch = [self.task_buffers[i].sample_data(inds) for i in indices]
        batch = np.stack(batch)
        return batch


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size
        else:
            raise TypeError("Unknown space: {}".format(space))
