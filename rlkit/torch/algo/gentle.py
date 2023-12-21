import os
import torch
import torch.optim as optim
import numpy as np
import copy
import time
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F 
from collections import OrderedDict
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.data_management.env_replay_buffer import MultiTaskContextBuffer
from itertools import chain

class GENTLE(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            obs_normalizer=None,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            obs_normalizer=obs_normalizer,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.context_lr                     = kwargs['context_lr']

        self.policy_noise                   = kwargs['policy_noise']
        self.noise_clip                     = kwargs['noise_clip']
        self.policy_freq                    = kwargs['policy_freq']
        self.bc_weight                      = kwargs['bc_weight']
        self.recon_loss_weight              = kwargs['recon_loss_weight']
        self.relabel_data_ratio             = kwargs['relabel_data_ratio']
        self.relabel_buffer_size            = kwargs['relabel_buffer_size']
        self.num_aug_neg_tasks              = kwargs['num_aug_neg_tasks']
        self.max_action                     = 1.

        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths

        self.qf1, self.qf2 = nets[1:3]
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.context_decoder = nets[3]
        self.task_dynamics = nets[4]
        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.context_optimizer              = optimizer_class(chain(self.agent.context_encoder.parameters(), 
                                                                    self.context_decoder.parameters()), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10

        self.relabel_buffer     = MultiTaskContextBuffer(self.relabel_buffer_size, env, self.train_tasks, kwargs['context_dim'])

    ###### Torch stuff #####
    @property
    def networks(self):
        nets = self.agent.networks + [self.agent] + [self.context_policy, self.context_policy.policy] + [self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.context_decoder]
        return nets

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.task_dynamics is not None:
            self.task_dynamics.to(device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]
    
    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.train_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, b_size=None):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size if b_size is None else b_size)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context
    
    def get_relabel_output(self, obs, actions, task_indices):
        with torch.no_grad():
            relabel_output, relabel_std = self.task_dynamics.step(obs, actions, task_indices, return_std=True)
        return relabel_output, relabel_std

    def make_relabel(self, _n=10):
        sample_b_s = self.embedding_batch_size * _n
        indices = np.array(self.train_tasks)
        
        context_batch = np.concatenate([self.unpack_context_batch(self.train_buffer.random_batch(idx, sample_b_s)) for idx in indices])
        context_batch = ptu.from_numpy(context_batch)
        c_mb, c_b, _ = context_batch.shape

        relabel_context = copy.deepcopy(context_batch)
        relabel_obs = relabel_context[:,:,:self.obs_dim].view(c_mb * c_b, -1)
        
        with torch.no_grad():
            relabel_actions = self.agent.get_target_policy_action(relabel_context[:,:,:self.obs_dim], context_batch, task_indices=indices, reinfer=True)

        relabel_context[:,:,self.obs_dim:self.obs_dim+self.action_dim] = relabel_actions.reshape(c_mb, c_b, -1)
        relabel_output, relabel_std = self.get_relabel_output(relabel_obs, relabel_actions, task_indices=indices)
        
        relabel_context[:,:,self.obs_dim+self.action_dim:] = relabel_output.view(c_mb, c_b, -1)
        sorted_ind = torch.argsort(relabel_std, dim=-1)
        sorted_relabel = torch.cat([relabel_context[i, sorted_ind[i], :] for i in range(c_mb)]).reshape(c_mb, c_b, -1)

        self.relabel_buffer.add_sample(indices, ptu.get_numpy(sorted_relabel))

        num_aug = self.num_aug_neg_tasks
        all_neg_indices = np.array([np.random.choice(list(indices)[0:i] + list(indices)[i+1:], num_aug, replace=False) for i in range(len(indices))])
        for i in range(num_aug):
            neg_indices = all_neg_indices[:, i]
            
            relabel_output, relabel_std = self.get_relabel_output(relabel_obs, relabel_actions, task_indices=neg_indices)
            relabel_context[:,:,self.obs_dim+self.action_dim:] = relabel_output.view(c_mb, c_b, -1)
            sorted_ind = torch.argsort(relabel_std, dim=-1)
            sorted_relabel = torch.cat([relabel_context[i, sorted_ind[i], :] for i in range(c_mb)]).reshape(c_mb, c_b, -1)

            self.relabel_buffer.add_sample(neg_indices, ptu.get_numpy(sorted_relabel))


    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        if self._n_train_steps_total % self.num_train_steps_per_itr == 0:
            self.relabel_buffer.clear(self.train_tasks)
            self.make_relabel()

        # sample context batch
        relabel_data_size = int(self.embedding_batch_size * self.relabel_data_ratio)
        context_batch = self.sample_context(indices, b_size=self.embedding_batch_size-relabel_data_size)
        relabel_context = ptu.from_numpy(self.relabel_buffer.random_batch_task(relabel_data_size, indices))
        context_batch = torch.cat([context_batch, relabel_context], dim=1)

        z_means_lst = []
        z_vars_lst = []
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
            torch.cuda.empty_cache()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, t, b, obs, actions, task_z):
        q1 = self.qf1(t, b, obs, actions, task_z.detach())
        q2 = self.qf2(t, b, obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self, f, target_f):
        ptu.soft_update_from_to(f, target_f, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            target_policy=self.agent.target_policy.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            context_decoder=self.context_decoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.agent.target_policy.load_state_dict(torch.load(os.path.join(path, 'target_policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1_itr_{}.pth'.format(epoch))))
            self.target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False
    
    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        with torch.no_grad():
            next_actions = self.agent.get_target_policy_action(next_obs, context, task_indices=indices)	
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        next_actions = next_actions.view(t * b, -1)

        c_mb, c_b, _ = context.size()


        r_next_s = context[...,obs_dim+action_dim:]
        pred_r_next_s = self.context_decoder(context[...,:obs_dim], context[...,obs_dim:obs_dim+action_dim], task_z.reshape(c_mb,c_b,-1))
        recon_loss = torch.mean((r_next_s - pred_r_next_s)**2)
        context_loss = self.recon_loss_weight * recon_loss
        self.loss['recon_loss'] = recon_loss.item()
        
        self.context_optimizer.zero_grad()
        context_loss.backward(retain_graph=True)
        self.context_optimizer.step()
        
        q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
        q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
        with torch.no_grad():
            target_q1 = self.target_qf1(t, b, next_obs, next_actions, task_z)
            target_q2 = self.target_qf2(t, b, next_obs, next_actions, task_z)
            target_q = torch.min(target_q1, target_q2)
            rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            # scale rewards for Bellman update
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = terms.view(self.batch_size * num_tasks, -1)
            target_q = rewards_flat + (1. - terms_flat) * self.discount * target_q
        qf_loss = torch.mean((q1_pred - target_q) ** 2) + torch.mean((q2_pred - target_q) ** 2)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(target_q).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        if self._num_steps % self.policy_freq == 0:
            Q = self._min_q(t, b, obs, new_actions, task_z)
            lmbda = self.bc_weight/Q.abs().mean().detach()
            policy_loss = -lmbda * Q.mean()
            bc_loss = F.mse_loss(new_actions, actions)
            policy_loss = policy_loss + bc_loss
            self.loss["policy_loss"] = policy_loss.item()
            self.loss["bc_loss"] = bc_loss.item()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self._update_target_network(self.qf1, self.target_qf1)
            self._update_target_network(self.qf2, self.target_qf2)
            self._update_target_network(self.agent.policy, self.agent.target_policy)

        # save some statistics for eval
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean
            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            self.eval_statistics['Recon Loss'] = ptu.get_numpy(recon_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['BC Loss'] = np.mean(ptu.get_numpy(
                bc_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)
    