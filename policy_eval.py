"""
Launcher for experiments with PEARL

"""

import click
import json
import os
from hydra.experimental import compose, initialize
import argparse
import multiprocessing as mp
from multiprocessing import Pool

from rlkit.torch.sac.pytorch_sac.train import Workspace
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from configs.default import default_config

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


# @hydra.main(config_path='rlkit/torch/sac/pytorch_sac/config/train.yaml', strict=True)
# c = []
# hydra.main(config_path='rlkit/torch/sac/pytorch_sac/config/train.yaml')(c.append)()
# print(c)
# cfg = c[0]
# print(cfg)
initialize(config_dir="rlkit/torch/sac/pytorch_sac/config/")
cfg = compose("train.yaml")
print(cfg.agent)
def experiment(cfg=cfg, env_name=None, env=None, goal_idx=0, checkpoint_step=1e6, num_episodes=10, action_noise=0.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    workspace = Workspace(cfg=cfg, env_name=env_name ,env=env, goal_idx=goal_idx, eval=True)
    workspace.collect_sample(checkpoint_step, num_episodes, action_noise)


# @click.command()
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./configs/ant-dir.json")
parser.add_argument("--checkpoint_step", type=int, default=int(1e6))
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--action_noise", type=float, default=0.0)
args = parser.parse_args()
def main(goal_idx=0):
    variant = default_config
    if args.config:
        with open(os.path.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    env.seed(1)
    env.reset_task(goal_idx)
    experiment(env_name=variant['env_name'] ,env=env, goal_idx=goal_idx, checkpoint_step=args.checkpoint_step, num_episodes=args.num_episodes, action_noise=args.action_noise)

if __name__ == '__main__':
    variant = default_config
    if args.config:
        with open(os.path.join(args.config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    for i in range(0, variant['env_params']['n_tasks']):
        main(i)
