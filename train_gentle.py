import os
import pathlib
import numpy as np
import click
import json
import torch
import random
import multiprocessing as mp
from itertools import product
from tensorboardX import SummaryWriter
import ast

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.autoencoder import MlpEncoder, MlpDecoder
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
from rlkit.torch.sac.agent import Agent
from rlkit.torch.sac.policies import ContextPolicyWrapper
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng
from rlkit.torch.algo.gentle import GENTLE

rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def experiment(variant, seed=None):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    
    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    obs_normalizer = ptu.RunningMeanStd(shape=obs_dim)

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    use_next_obs_in_context = variant['algo_params']['use_next_obs_in_context']
    variant['algo_params']['context_dim'] = context_encoder_input_dim

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    context_encoder = MlpEncoder(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
            output_activation=torch.tanh,
            batch_attention=False,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )

    context_decoder = MlpDecoder(hidden_size=net_size,
                                    num_hidden_layers=3,
                                    z_dim=latent_dim,
                                    action_dim=action_dim,
                                    obs_dim=obs_dim,
                                    reward_dim=1,
                                    use_next_obs_in_context=use_next_obs_in_context)

    if use_next_obs_in_context:
        task_dynamics =  MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
                                     hidden_size=net_size, 
                                     num_hidden_layers=3, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5])
    else:
        task_dynamics = MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
                                     hidden_size=net_size, 
                                     num_hidden_layers=2, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])
    task_dynamics.load('/data/zrz/gentle_data/asset/dynamics/'+variant['env_name']+'/'+f'expert_seed{seed}')

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    context_policy = ContextPolicyWrapper(policy, latent_dim)

    agent = Agent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    agent = [agent, context_policy]

    algorithm = GENTLE(
                    env=env,
                    train_tasks=list(tasks[:variant['n_train_tasks']]),
                    eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
                    nets=[agent, qf1, qf2, context_decoder, task_dynamics],
                    latent_dim=latent_dim,
                    obs_normalizer=obs_normalizer,
                    **variant['algo_params']
    )
    # optional GPU mode
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed,
        snapshot_mode="all",
        algo_name=variant['output_prefix']+variant['algo_type']
    )

    tb_writer = SummaryWriter(experiment_log_dir)
    algorithm.train(tb_writer)

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--debug', default=0)
@click.option('--algo_type', default='gentle')  
@click.option('--seed_list', default=[0])
@click.option('--output_prefix', default='')
def main(config, gpu, debug, algo_type, seed_list, output_prefix):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['debug'] = debug
    variant['algo_type'] = algo_type
    variant['output_prefix'] = output_prefix
    variant['util_params']['base_log_dir'] = './logs'

    # multi-processing
    if len(seed_list) > 1:
        if isinstance(seed_list, str):
            seed_list = ast.literal_eval(seed_list)
        p = mp.Pool(len(seed_list))
        p.starmap(experiment, product([variant], seed_list))
    else:
        experiment(variant, seed=seed_list[0])

if __name__ == "__main__":
    main()



