from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
import os
from rlkit.torch.networks import  EnsembleDynamicsModel

class MultiTaskDynamics(object):
    def __init__(self,
                 num_tasks,
                 hidden_size,
                 num_hidden_layers,
                 action_dim,
                 obs_dim,
                 reward_dim,
                 use_next_obs_in_context,
                 ensemble_size,
                 dynamics_weight_decay):
        super().__init__()
        self.num_tasks = num_tasks
        self.action_dim=action_dim
        self.obs_dim=obs_dim
        self.reward_dim=reward_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.ensemble_size = ensemble_size

        self.models = []
        self.optims = []
        for i in range(self.num_tasks):
            model = EnsembleDynamicsModel(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[hidden_size]*num_hidden_layers, 
                                          num_ensemble=ensemble_size, num_elites=ensemble_size, weight_decays=dynamics_weight_decay, 
                                          with_next_obs=use_next_obs_in_context)
            optim = torch.optim.Adam(model.parameters(), lr=1e-3) 
            self.models.append(model)
            self.optims.append(optim)

    def step(self, obs, action, task_indices, mean_output=True, return_std=False):
        "imagine single forward step"
        obs_act = torch.cat([obs, action], axis=-1)
        bs = int(len(obs_act)/len(task_indices))
        output = []
        output_std = []
        for i in range(len(task_indices)):
            curr_task = task_indices[i]
            curr_input = obs_act[i*bs : (i+1)*bs, :]
            curr_output = self.models[curr_task](curr_input)
            if mean_output:
                curr_std = torch.std(curr_output, 0, keepdim=True).sum(-1) 
                curr_output = torch.mean(curr_output, dim=0)
                output_std.append(curr_std)
            else:
                # random choose one
                num_models, batch_size, _ = curr_output.shape
                model_idxs = self.models[curr_task].random_idxs(batch_size)
                samples = curr_output[model_idxs, np.arange(batch_size)]
                curr_output = samples
            output.append(curr_output)
        if return_std:
            return torch.cat(output), torch.cat(output_std)
        else:
            return torch.cat(output)

    def set_task_idx(self, task_idx):
        self.curr_idx = task_idx
        self.model = self.models[task_idx]
        self.optim = self.optims[task_idx]

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        inputs = np.concatenate([obss, actions], axis=-1)
        if self.use_next_obs_in_context:
            targets = np.concatenate([rewards, next_obss], axis=-1)
        else:
            targets = rewards
        return inputs, targets
   
    def train(
        self,
        data: Dict,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, dynamics_train_loss {train_loss}, holdout_loss {holdout_loss}")
            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        self.model.load_save()
        self.model.eval()

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean = self.model(inputs_batch)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) ).mean(dim=(1, 2))
            loss = mse_loss_inv.sum()
            loss = loss + self.model.get_decay_loss()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
        
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

    def save(self, save_path: str) -> None:
        for i in range(self.num_tasks):
            torch.save(self.models[i].state_dict(), os.path.join(save_path, f"task{i}_dynamics.pth"))
    
    def load(self, load_path: str) -> None:
        for i in range(self.num_tasks):
            self.models[i].load_state_dict(torch.load(os.path.join(load_path, f"task{i}_dynamics.pth"), map_location=self.models[i].device))

    def to(self, device):
        for i in range(self.num_tasks):
            self.models[i].to(device)
            self.models[i].device = device
