import numpy as np

import torch
import copy
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared



class Agent(nn.Module):
    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy
        self.target_policy = copy.deepcopy(self.policy)

        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        self.policy_noise                   = kwargs['policy_noise']
        self.noise_clip                     = kwargs['noise_clip']
        self.policy_freq                    = kwargs['policy_freq']
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        #self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
 
    def update_context_dict(self, batch_dict, env):
        ''' append context dictionary containing single/multiple transitions to the current context '''
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            r = batch_dict['rewards']
            sr = []
            for r_entry in r:
                sr.append(env.sparsify_rewards(r_entry))
            r = ptu.from_numpy(np.array(sr)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, next_o], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            # permutation invariant encoding
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1) # dim: task, batch, feature (latent dim)
            self.z_vars = torch.std(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, task_indices=None):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z

        # self.meta_batch * self.batch_size * dim(obs)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        if not self.use_ib:
            task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
            task_z_vars = torch.cat(task_z_vars, dim=0)
            return policy_outputs, task_z, task_z_vars
        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        # adds logging data about encodings to eval_statistics
        for i in range(len(self.z_means[0])):
            z_mean = ptu.get_numpy(self.z_means[0][i])
            name = 'Z mean eval' + str(i)
            eval_statistics[name] = z_mean

        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z variance eval'] = z_sig
        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        return [self.context_encoder, self.policy, self.target_policy]

    def get_target_policy_action(self, obs, context, task_indices=None, reinfer=False, given_z=None):
        if given_z is not None:
            task_z = given_z
        else:
            if reinfer:
                task_z = self.infer_z(context, task_indices, 1)
            else:    
                task_z = self.z
        # self.meta_batch * self.batch_size * dim(obs)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        if given_z is None:
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)
        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.target_policy(t, b, in_, deterministic=True, reparameterize=True, return_log_prob=True)
        return policy_outputs[0]

    def infer_z(self, context, task_indices=None, b=1):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if task_indices is None:
            task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            task_indices = np.array([task_indices])
        else:
            task_indices = np.array(task_indices)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            # permutation invariant encoding
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            z_means = torch.mean(params, dim=1) # dim: task, batch, feature (latent dim)
            z_vars = torch.std(params, dim=1)
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            task_z = torch.stack(z)
        else:
            task_z = z_means
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        return task_z

    @torch.no_grad()
    def infer_z_unpooling(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        return params

    def get_current_z(self, b, return_mean=False):
        if return_mean:
            task_z = self.z_means
        else:
            task_z = self.z
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        return task_z
