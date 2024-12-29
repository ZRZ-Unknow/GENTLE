# Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations
Code for AAAI'24 paper "Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations".

## Installation

First install [MuJoCo](https://www.roboti.us/index.html). For tasks differ in reward functions (Cheetah, Ant), install MuJoCo150 or plus. Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers.

Then create conda environment by:

```bash
conda env create -f environment.yaml
```

**For Hopper and Walker environments**, MuJoCo131 is required. Simply install it the same way as MuJoCo200. To switch between different MuJoCo versions:

```bash
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
```

## Data Generation

Example of training behavior policies on multiple tasks:

```bash
python policy_train.py ./configs/ant-dir.json --gpu 0
```

It will run SAC to train a policy on each task, you can modify `self.work_dir` of `Workspace` in `rlkit/torch/sac/pytorch_sac/train.py` to specify the directory to save the trained policies.

Generate trajectories from trained policies:

```bash
python policy_eval.py --config ./configs/ant-dir.json
```

Data will be saved in `self.work_dir/gentle_data/$env_name/$goal_idx{i}`

## Training GENTLE 

The configration files to run GENTLE is in `./configs`. For example, to train GENTLE on Ant-Dir, first you need to pretrain the dynamics model:
```bash
python pretrain_dynamics.py ./configs/ant-dir.json 
```
Then run:
```bash
python train_gentle.py ./configs/ant-dir.json
```

Logs will be written to `./logs/ant-dir/gentle/`

## Reference

```bash
@inproceedings{gentle,
  author={Renzhe Zhou, Chen-Xiao Gao, Zongzhang Zhang, Yang Yu},
  title={Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```

