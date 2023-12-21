import abc
from collections import OrderedDict
import time
import os
import glob
import gtimer as gt
import numpy as np
import torch, copy

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer, get_dim
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler, OfflineInPlacePathSampler
from rlkit.torch import pytorch_util as ptu
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


save_freq = 200

class OfflineMetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            goal_radius,
            obs_normalizer,
            eval_deterministic=True,
            render=False,
            render_eval_paths=False,
            plotter=None,
            **kwargs
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param goal_radius: reward threshold for defining sparse rewards

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env                             = env
        self.agent                           = agent[0]
        self.context_policy                  = agent[1]
        self.train_tasks                     = train_tasks
        self.eval_tasks                      = eval_tasks
        self.goal_radius                     = goal_radius

        self.meta_batch                      = kwargs['meta_batch']
        self.batch_size                      = kwargs['batch_size']
        self.num_iterations                  = kwargs['num_iterations']
        self.num_train_steps_per_itr         = kwargs['num_train_steps_per_itr']
        self.num_initial_steps               = kwargs['num_initial_steps']
        self.num_tasks_sample                = kwargs['num_tasks_sample']
        self.num_steps_prior                 = kwargs['num_steps_prior']
        self.num_steps_posterior             = kwargs['num_steps_posterior']
        self.num_extra_rl_steps_posterior    = kwargs['num_extra_rl_steps_posterior']
        self.num_evals                       = kwargs['num_evals']
        self.num_steps_per_eval              = kwargs['num_steps_per_eval']
        self.embedding_batch_size            = kwargs['embedding_batch_size']
        self.embedding_mini_batch_size       = kwargs['embedding_mini_batch_size']
        self.max_path_length                 = kwargs['max_path_length']
        self.discount                        = kwargs['discount']
        self.replay_buffer_size              = kwargs['replay_buffer_size']
        self.reward_scale                    = kwargs['reward_scale']
        self.update_post_train               = kwargs['update_post_train']
        self.num_exp_traj_eval               = kwargs['num_exp_traj_eval']
        self.save_replay_buffer              = kwargs['save_replay_buffer']
        self.save_algorithm                  = kwargs['save_algorithm']
        self.save_environment                = kwargs['save_environment']
        self.dump_eval_paths                 = kwargs['dump_eval_paths']
        self.data_dir                        = kwargs['data_dir']
        self.train_epoch                     = kwargs['train_epoch']
        self.eval_epoch                      = kwargs['eval_epoch']
        self.load_eval_buffer                = kwargs['load_eval_buffer']
        self.sample                          = kwargs['sample']
        self.n_trj                           = kwargs['n_trj']
        self.allow_eval                      = kwargs['allow_eval']
        self.online_sample_num               = kwargs['online_sample_num']


        self.eval_deterministic              = eval_deterministic
        self.render                          = render
        self.eval_statistics                 = None
        self.render_eval_paths               = render_eval_paths
        self.plotter                         = plotter
        
        self.train_buffer      = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.eval_buffer       = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.eval_tasks,  self.goal_radius)
        self.replay_buffer     = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.enc_replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.context_buffer = SimpleReplayBuffer(self.replay_buffer_size, get_dim(env.observation_space), get_dim(env.action_space), self.goal_radius)
        # offline sampler which samples from the train/eval buffer
        self.offline_sampler   = OfflineInPlacePathSampler(env=env, policy=self.agent, max_path_length=self.max_path_length)
        # online sampler for evaluation (if collect on-policy context, for offline context, use self.offline_sampler)
        self.sampler           = InPlacePathSampler(env=env, policy=self.agent, max_path_length=self.max_path_length)

        self.enc_train_buffer      = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.enc_eval_buffer       = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.eval_tasks,  self.goal_radius)

        self._n_env_steps_total     = 0
        self._n_train_steps_total   = 0
        self._n_rollouts_total      = 0
        self._do_train_time         = 0
        self._epoch_start_time      = None
        self._algo_start_time       = None
        self._old_table_keys        = None
        self._current_path_builder  = PathBuilder()
        self._exploration_paths     = []
        self.obs_normalizer         = obs_normalizer
        self.obs_dim = get_dim(env.observation_space)
        self.action_dim = get_dim(env.action_space)
        self.init_buffer(kwargs)
        self.enc_obs_normalizer = ptu.RunningMeanStd(shape=self.obs_dim)

    def init_buffer(self, kwargs):
        train_trj_paths = []
        eval_trj_paths = []
        # trj entry format: [obs, action, reward, new_obs]
        for n in range(self.n_trj):
            if self.train_epoch is None:
                train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" %(n)))
            else:
                train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, self.train_epoch)))
            if self.load_eval_buffer:
                if self.eval_epoch is None:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" %(n)))
                else:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, self.eval_epoch)))
        
        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        if self.load_eval_buffer:
            eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                          int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]
            eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                              int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]

        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []
        obs_eval_lst = []
        action_eval_lst = []
        reward_eval_lst = []
        next_obs_eval_lst = []
        terminal_eval_lst = []
        task_eval_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs_train_lst += list(trj_npy[:, 0])
            action_train_lst += list(trj_npy[:, 1])
            reward_train_lst += list(trj_npy[:, 2])
            next_obs_train_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train
        if self.load_eval_buffer:
            for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
                trj_npy = np.load(eval_path, allow_pickle=True)
                obs_eval_lst += list(trj_npy[:, 0])
                action_eval_lst += list(trj_npy[:, 1])
                reward_eval_lst += list(trj_npy[:, 2])
                next_obs_eval_lst += list(trj_npy[:, 3])
                terminal = [0 for _ in range(trj_npy.shape[0])]
                terminal[-1] = 1
                terminal_eval_lst += terminal
                task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
                task_eval_lst += task_eval
        

        self.obs_normalizer.update(obs_train_lst)
        self.env.update_obs_mean_var(self.obs_normalizer.mean, self.obs_normalizer.var)
        obs_train_lst = self.obs_normalizer.forward(obs_train_lst)
        next_obs_train_lst = self.obs_normalizer.forward(next_obs_train_lst)
        if self.load_eval_buffer:
            obs_eval_lst = self.obs_normalizer.forward(obs_eval_lst)
            next_obs_eval_lst = self.obs_normalizer.forward(next_obs_eval_lst)
        
        # load training buffer
        for i, (
                task_train,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
        )):
            self.train_buffer.add_sample(
                task_train,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )
        
        if self.load_eval_buffer:
        # load evaluation buffer
            for i, (
                task_eval,
                obs,
                action,
                reward,
                next_obs,
                terminal,
            ) in enumerate(zip(
                task_eval_lst,
                obs_eval_lst,
                action_eval_lst,
                reward_eval_lst,
                next_obs_eval_lst,
                terminal_eval_lst,
            )):
                self.eval_buffer.add_sample(
                    task_eval,
                    obs,
                    action,
                    reward,
                    terminal,
                    next_obs,
                    **{'env_info': {}},
                )

    def _try_to_eval(self, epoch):
        if self._can_evaluate():
            self.evaluate(epoch)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys
            logger.record_tabular("Number of train steps total", self._n_train_steps_total)
            logger.record_tabular("Number of env steps total",   self._n_env_steps_total)
            logger.record_tabular("Number of rollouts total",    self._n_rollouts_total)

            times_itrs  = gt.get_times().stamps.itrs
            train_time  = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time   = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time  = train_time + sample_time + eval_time
            total_time  = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer=self.tb_writer)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True
    
    def _can_train(self):
        return all([self.train_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _do_eval(self, indices, epoch, buffer):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, buffer)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean(all_rets))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def unpack_context_batch(self, batch):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        if self.use_next_obs_in_context:
            return np.concatenate([o, a, r, no], axis=-1)
        else:
            return np.concatenate([o, a, r], axis=-1)

    def _do_eval_with_online_context(self, indices, epoch, buffer):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            self.context_buffer.clear()
            self.collect_context_by_expl_policy(idx)
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, buffer, context_buffer=self.context_buffer)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean(all_rets))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        
        if epoch == (self.num_iterations - 1):
            context_zs = []
            n_points = 400
            for idx in indices:
                batchs = []
                self.context_buffer.clear()
                self.collect_context_by_expl_policy(idx, given_sample_num=self.online_sample_num*5)
                for i in range(n_points):
                    batch = self.unpack_context_batch(self.context_buffer.random_batch(self.online_sample_num))
                    batchs.append(batch)
                batchs = ptu.from_numpy(np.concatenate(batchs))
                context_z = ptu.get_numpy(self.agent.infer_z(batchs, b=1))
                context_zs.append(context_z[np.newaxis,:,:])
            context_zs = np.concatenate(context_zs)
        else:
            context_zs = None
        return final_returns, online_returns, context_zs

    def collect_context_by_expl_policy(self, idx, given_sample_num=None):
        self.env.reset_task(idx)
        sample_num = 0
        self.context_policy.set_task(idx)
        if given_sample_num is None:
            given_sample_num = self.online_sample_num
        while sample_num < given_sample_num:
            o = self.env.reset()
            path_length = 0
            while path_length < self.max_path_length:
                with torch.no_grad():
                    a, agent_info = self.context_policy.get_action(ptu.from_numpy(o[None]))
                next_o, r, d, env_info = self.env.step(a)
                self.context_buffer.add_sample(o, a, r, d, next_o, **{'env_info': {}}) 
                path_length += 1
                sample_num += 1
                o = next_o
                if d:
                    break
                if sample_num >= given_sample_num: 
                    break

    def train(self, tb_writer):
        '''
        meta-training loop
        '''
        self.tb_writer = tb_writer
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()
        
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf, buffer=self.train_buffer, add_to_enc_buffer=True)
            # Sample data from train tasks.
            self.itr = it_
            for i in range(self.num_tasks_sample):
                idx = np.random.choice(self.train_tasks, 1)[0]
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, buffer=self.train_buffer, add_to_enc_buffer=True)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer, add_to_enc_buffer=True)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer,
                                      add_to_enc_buffer=False)
            indices_lst = []
            z_means_lst = []
            z_vars_lst = []
            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                if len(self.train_tasks) < self.meta_batch:
                    indices = np.random.choice(self.train_tasks, self.meta_batch, replace=True)
                else:
                    indices = np.random.choice(self.train_tasks, self.meta_batch, replace=False)
                z_means, z_vars = self._do_training(indices)
                indices_lst.append(indices)
                z_means_lst.append(z_means)
                z_vars_lst.append(z_vars)
                self._n_train_steps_total += 1
            
            indices = np.concatenate(indices_lst)
            z_means = np.concatenate(z_means_lst)
            z_vars = np.concatenate(z_vars_lst)

            gt.stamp('train')
            self.training_mode(False)
            
            if it_ % save_freq == 0 or it_ == (self.num_iterations-1):
                params = self.get_epoch_snapshot(it_)
                logger.save_itr_params(it_, params)

            if self.allow_eval:
                if it_ % save_freq == 0 or it_ == (self.num_iterations-1):
                    logger.save_extra_data(self.get_extra_data_to_save(it_))
                self._try_to_eval(it_)

                gt.stamp('eval')
            self._end_epoch()

    def data_dict(self, indices, z_means, z_vars):
        data_dict = {}
        data_dict['task_idx'] = indices
        for i in range(z_means.shape[1]):
            data_dict['z_means%d' %i] = list(z_means[:, i])
        for i in range(z_vars.shape[1]):
            data_dict['z_vars%d' % i] = list(z_vars[:, i])
        return data_dict

    def evaluate(self, epoch):

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        # cheetah-dir env
        if len(self.train_tasks) == 2 and len(self.eval_tasks) == 2:
            indices = self.train_tasks

            eval_util.dprint('evaluating on {} test tasks with offline context'.format(len(self.eval_tasks)))
            test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.train_buffer)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)

            train_zs = None
            eval_util.dprint('evaluating on {} test tasks with online context'.format(len(self.eval_tasks)))
            test_final_returns_expl, test_online_returns_expl, test_zs = self._do_eval_with_online_context(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns expl')
            eval_util.dprint(test_online_returns_expl)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            avg_test_return = np.mean(test_final_returns)
            avg_test_return_expl = np.mean(test_final_returns_expl)
            
            self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
            for i in indices:
                self.eval_statistics[f'AverageReturn_all_train_tasks{i}'] = test_final_returns[i]
            self.eval_statistics['AverageReturn_all_test_tasks_expl'] = avg_test_return_expl
            for i in indices:
                self.eval_statistics[f'AverageReturn_all_train_tasks{i}_expl'] = test_final_returns_expl[i]

        # other envs
        else:
            eval_util.dprint('evaluating on {} train tasks with offline context'.format(len(self.train_tasks)))
            train_final_returns, train_online_returns = self._do_eval(self.train_tasks, epoch, buffer=self.train_buffer)
            eval_util.dprint('train online returns')
            eval_util.dprint(train_online_returns)
            
            if self.load_eval_buffer:
                eval_util.dprint('evaluating on {} test tasks with offline context'.format(len(self.eval_tasks)))
                test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
                eval_util.dprint('test online returns')
                eval_util.dprint(test_online_returns)
            else:
                test_final_returns, test_online_returns = np.zeros_like(train_final_returns), np.zeros_like(train_online_returns)

            eval_util.dprint('evaluating on {} train tasks with online context'.format(len(self.train_tasks)))
            train_final_returns_expl, train_online_returns_expl, train_zs = self._do_eval_with_online_context(self.train_tasks, epoch, buffer=self.train_buffer)
            eval_util.dprint('train online returns expl')
            eval_util.dprint(train_online_returns_expl)

            eval_util.dprint('evaluating on {} test tasks with online context'.format(len(self.eval_tasks)))
            test_final_returns_expl, test_online_returns_expl, test_zs = self._do_eval_with_online_context(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns expl')
            eval_util.dprint(test_online_returns_expl)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            avg_train_return = np.mean(train_final_returns)
            avg_test_return = np.mean(test_final_returns)
            avg_train_return_expl = np.mean(train_final_returns_expl)
            avg_test_return_expl = np.mean(test_final_returns_expl)

            self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
            self.eval_statistics['AverageReturn_all_train_tasks_expl'] = avg_train_return_expl
            self.eval_statistics['AverageReturn_all_test_tasks_expl'] = avg_test_return_expl
        
        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

        if epoch == (self.num_iterations - 1):
            assert(test_zs is not None)
            if train_zs is not None:
                logger.save_contexts(epoch, train_zs, f'online_z_train_itr_{epoch}.npy')
            logger.save_contexts(epoch, test_zs,  f'online_z_test_itr_{epoch}.npy')
            fig_save_dir = logger._snapshot_dir + '/figures'
            if train_zs is not None:
                self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_z_train_itr_{epoch}.png', zs=[train_zs], subplot_title_lst=[f'online_z_train_itr_{epoch}'])
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_z_test_itr_{epoch}.png', zs=[test_zs], subplot_title_lst=[f'online_z_test_itr_{epoch}'])

            offline_zs = []
            n_points = 400
            for idx in self.train_tasks:
                offline_batchs = []
                for _ in range(n_points):
                    offline_batch = self.unpack_context_batch(self.train_buffer.random_batch(task=idx, batch_size=self.online_sample_num))
                    offline_batchs.append(offline_batch)
                offline_batchs = ptu.from_numpy(np.concatenate(offline_batchs))
                offline_z = ptu.get_numpy(self.agent.infer_z(offline_batchs, b=1))
                offline_zs.append(offline_z[np.newaxis,:,:])
            offline_zs = np.concatenate(offline_zs)

            logger.save_contexts(epoch, offline_zs, f'offline_z_train_itr_{epoch}.npy')
            fig_save_dir = logger._snapshot_dir + '/figures'
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'offline_z_train_itr_{epoch}.png', zs=[offline_zs], subplot_title_lst=[f'offline_z_train_itr_{epoch}'])
            
    def collect_paths(self, idx, epoch, run, buffer, context_buffer=None):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        # num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            if context_buffer is not None:
                path, num = self.offline_sampler.obtain_samples_with_context(
                    context_buffer=context_buffer,
                    deterministic=self.eval_deterministic,
                    max_samples=self.num_steps_per_eval - num_transitions,
                    max_trajs=1,
                    accum_context=True,
                    rollout=True)
            else:
                path, num = self.offline_sampler.obtain_samples(
                    buffer=buffer,
                    deterministic=self.eval_deterministic,
                    max_samples=self.num_steps_per_eval - num_transitions,
                    max_trajs=1,
                    accum_context=True,
                    rollout=True)
            paths += path
            num_transitions += num

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            if epoch % save_freq == 0 or epoch == (self.num_iterations-1):
                logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, buffer, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.offline_sampler.obtain_samples(buffer=buffer,
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate,
                                                           rollout=False)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context, task_indices=np.array([self.task_idx]))
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_context_data(self, num_samples, resample_z_rate, update_posterior_rate, buffer, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.offline_sampler.obtain_samples(buffer=buffer,
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate,
                                                           rollout=False)
            num_transitions += n_samples
            self.enc_replay_buffer.add_paths(self.task_idx, paths)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    def vis_task_embeddings(self, save_dir, fig_name, zs, rows=1, cols=1, n_figs=1,
                subplot_title_lst = ["train_itr_0"],  goals_name_lst=None, figsize=[12, 6], fontsize=15):
        if figsize is None:
            fig = plt.figure(figsize=(8, 10))
        else:
            fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        if goals_name_lst is None:
            goals_name_lst = [None]*n_figs
            legend = False
        else:
            legend = True

        for z, n_fig, subplot_title, goals_name in zip(zs, range(1, n_figs+1), subplot_title_lst, goals_name_lst):
            n_tasks, n_points, _ = z.shape
            proj = TSNE(n_components=2).fit_transform(X=z.reshape(n_tasks*n_points, -1))
            ax = fig.add_subplot(rows, cols, n_fig)
            for task_idx in range(n_tasks):
                idxs = np.arange(task_idx*n_points, (task_idx+1)*n_points)
                if goals_name is None:
                    ax.scatter(proj[idxs, 0], proj[idxs, 1], s=1, alpha=0.3, cmap=plt.cm.Spectral)
                else:
                    ax.scatter(proj[idxs, 0], proj[idxs, 1], s=1, alpha=0.3, cmap=plt.cm.Spectral, label=goals_name[task_idx])

            ax.set_title(subplot_title, fontsize=fontsize)
            ax.set_xlabel('t-SNE dimension 1', fontsize=fontsize)
            ax.set_ylabel('t-SNE dimension 2', fontsize=fontsize)
            if legend:
                ax.legend(loc='best')
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fig_name), dpi=200)
        plt.close()