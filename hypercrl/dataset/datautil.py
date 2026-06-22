from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, random_split
try:
    from gymnasium.envs.robotics.rotations import quat_mul, quat_conjugate
except Exception:
    def _ensure_2d_quat(quat):
        quat = np.asarray(quat)
        is_vector = quat.ndim == 1
        if is_vector:
            quat = quat.reshape(1, 4)
        return quat, is_vector

    def quat_mul(q0, q1):
        q0, q0_is_vector = _ensure_2d_quat(q0)
        q1, _ = _ensure_2d_quat(q1)
        w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        out = np.stack([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], axis=1)
        return out[0] if q0_is_vector else out

    def quat_conjugate(quat):
        quat, is_vector = _ensure_2d_quat(quat)
        out = np.concatenate([quat[:, :1], -quat[:, 1:]], axis=1)
        return out[0] if is_vector else out


def rotate_imgs(data, r=None):
    if r is None:
        r = random.choice([0, 1, 2, 3])
    if (type(data) is tuple):
        img, label = data
        img = np.rot90(data[0], k=r, axes=(1, 2))
        return img, label, r
    if (type(data) is list):
        data[0] = np.rot90(data[0], k=r, axes=(1, 2))
        data += [r]
        return data
    else:
        img = np.rot90(data, k=r, axes=(1, 2))
        return (img, r)


def train_val_split(dataset, val_size=0.25):
    val_size = int(len(dataset) * 0.25)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set


class DataCollector():
    def __init__(self, hparams) -> None:
        self.states: Dict[int, List[np.ndarray]] = {}
        self.actions: Dict[int, List[np.ndarray]] = {}
        self.nexts: Dict[int, List[np.ndarray]] = {}
        self.train_inds: Dict[int, List[int]] = {}
        self.val_inds: Dict[int, List[int]] = {}

        self.x_aggregate: Dict[int, Tuple] = {}
        self.a_aggregate: Dict[int, Tuple] = {}
        self.dx_aggregate: Dict[int, Tuple] = {}
        self.norms: Dict[int, Tuple] = {}

        # Safety certificate values stored per transition; None when no filter is active
        self.cbf_values: Dict[int, List[Optional[float]]] = {}
        self.clf_values: Dict[int, List[Optional[float]]] = {}

        self.fig = None
        self.next_mode: str = hparams.dnn_out
        self.normalize_xu: bool = hparams.normalize_xu
        # Normalize the diff target by its per-dimension std (diff mode only).
        # Essential for stiff systems (e.g. the satellite env) where per-step
        # changes in some dims (omega) are far smaller than in others, so the
        # joint MSE otherwise ignores the small-diff dims and the model learns
        # to predict ~0 for them.
        self.normalize_diff: bool = getattr(hparams, "normalize_diff", False)
        self.diff_norms: Dict[int, Tuple] = {}
        self.env_name: str = hparams.env

    def num_tasks(self):
        return len(self.states)

    def update(self, existingAggregate, newValue):
        """
        compute new count, new mean, and new second momentum
        """
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        return (count, mean, M2)

    def preprocess(self, x_t, u, x_tt):
        """
        Process the pre-/post- processing of states
        """
        if self.env_name.startswith("inverted_pendulum") or self.env_name.startswith('cartpole'):
            if self.next_mode == "diff":
                x_tt = x_tt - x_t
            x_t = np.vstack((x_t[0:1, :], np.cos(
                x_t[1:2, :]), np.sin(x_t[1:2, :]), x_t[2:, :]))
        elif self.env_name in ["half_cheetah_body", "half_cheetah_safe", "hopper"]:
            if self.next_mode == "diff":
                x_tt = np.vstack((x_tt[0:1, :], x_tt[1:, :] - x_t[1:, :]))
            x_t = np.vstack((x_t[1:2, :], np.cos(
                x_t[2:3, :]), np.sin(x_t[2:3, :]), x_t[3:, :]))
        elif self.env_name == "door":
            if self.next_mode == "diff":
                x_tt = x_tt - x_t
            x_t = np.vstack(
                (x_t[0:-1, :], np.cos(x_t[-1:, :]), np.sin(x_t[-1:, :])))
        elif self.env_name == "door_pose":
            if self.next_mode == "diff":
                quat_diff = quat_mul(
                    x_tt[3:7, :].T, quat_conjugate(x_t[3:7, :].T)).T
                x_tt = x_tt - x_t
                x_tt[3:7, :] = quat_diff

            x_t = np.vstack((x_t[0:-2, :], np.cos(x_t[-2:-1, :]), np.sin(x_t[-2:-1, :]),
                             np.cos(x_t[-1:, :]), np.sin(x_t[-1:, :])))
        else:
            if self.next_mode == "diff":
                x_tt = x_tt - x_t

        return x_t, u, x_tt

    def add(
        self,
        x_t: np.ndarray,
        u: np.ndarray,
        x_tt: np.ndarray,
        task_id: int,
        cbf_val: Optional[float] = None,
        clf_val: Optional[float] = None,
    ) -> None:
        # Convert Format
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if x_t.ndim == 1:
            x_t = x_t[:, None]
        if x_tt.ndim == 1:
            x_tt = x_tt[:, None]
        if u.ndim == 1:
            u = u[:, None]

        x_t, u, x_tt = self.preprocess(x_t, u, x_tt)

        if task_id in self.states:
            self.states[task_id].append(x_t)
            self.actions[task_id].append(u)
            self.nexts[task_id].append(x_tt)
            self.cbf_values[task_id].append(cbf_val)
            self.clf_values[task_id].append(clf_val)
            if self.normalize_xu:
                self.x_aggregate[task_id] = self.update(
                    self.x_aggregate[task_id], x_t)
                self.a_aggregate[task_id] = self.update(
                    self.a_aggregate[task_id], u)
            if getattr(self, "normalize_diff", False) and self.next_mode == "diff":
                self.dx_aggregate[task_id] = self.update(
                    self.dx_aggregate[task_id], x_tt)
        else:
            self.states[task_id] = [x_t]
            self.actions[task_id] = [u]
            self.nexts[task_id] = [x_tt]
            self.cbf_values[task_id] = [cbf_val]
            self.clf_values[task_id] = [clf_val]
            if self.normalize_xu:
                self.x_aggregate[task_id] = self.update((0, 0, 0), x_t)
                self.a_aggregate[task_id] = self.update((0, 0, 0), u)
            if getattr(self, "normalize_diff", False) and self.next_mode == "diff":
                self.dx_aggregate[task_id] = self.update((0, 0, 0), x_tt)

        # Train or val
        is_train = (random.random() <= 0.75)
        ind = len(self.states[task_id]) - 1
        if is_train:
            if task_id in self.train_inds:
                self.train_inds[task_id].append(ind)
            else:
                self.train_inds[task_id] = [ind]
        else:
            if task_id in self.val_inds:
                self.val_inds[task_id].append(ind)
            else:
                self.val_inds[task_id] = [ind]

    def finalize(self, task_id):
        def one(existingAggregate):
            (count, mean, M2) = existingAggregate
            if count < 2:
                return float('nan')
            else:
                sample_var = M2 / (count - 1)
                std = np.sqrt(sample_var)
                mean, std = torch.FloatTensor(mean).T, torch.FloatTensor(std).T
                std[std < 1e-9] = 1
                return mean, std

        x_mu, x_std = one(self.x_aggregate[task_id])
        a_mu, a_std = one(self.a_aggregate[task_id])

        self.norms[task_id] = (x_mu, x_std, a_mu, a_std)

        if getattr(self, "normalize_diff", False) and self.next_mode == "diff" \
                and task_id in self.dx_aggregate:
            dx_mu, dx_std = one(self.dx_aggregate[task_id])
            # Center diffs at 0 so x_{t+1} = x_t + pred maps back cleanly;
            # only rescale by the per-dim std.
            dx_mu = torch.zeros_like(dx_mu)
            self.diff_norms[task_id] = (dx_mu, dx_std)

        return self.norms[task_id]

    def norm(self, task_id: int) -> Tuple:
        return self.norms[task_id]

    def norm_diff(self, task_id: int) -> Tuple:
        return self.diff_norms[task_id]

    def get_safety_values(
        self, task_id: int
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Return (cbf_values, clf_values) for the given task.

        Each list is aligned with the stored transitions; entries are None
        when no safety filter was active for that step.
        """
        return self.cbf_values[task_id], self.clf_values[task_id]

    def get_dataset(self, task_id, ds_range=None):
        """
        Return a pytorch dataset of (state, actions, next_state)
        states, actions are normalized to N(0, 1)
        """

        states = torch.FloatTensor(np.hstack(self.states[task_id])).T
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).T

        # Get Norm and normalize
        if self.normalize_xu:
            x_mu, x_std, a_mu, a_std = self.finalize(task_id)
            states = (states - x_mu) / x_std
            actions = (actions - a_mu) / a_std
            if self.next_mode != "diff":
                nexts = (nexts - x_mu) / x_std
            elif getattr(self, "normalize_diff", False):
                dx_mu, dx_std = self.diff_norms[task_id]
                nexts = (nexts - dx_mu) / dx_std

        train_inds = self.train_inds[task_id]
        val_inds = self.val_inds[task_id]

        if ds_range == "second_half":
            train_inds = train_inds[len(train_inds) // 2:]
        train_set = TensorDataset(
            states[train_inds], actions[train_inds], nexts[train_inds])
        val_set = TensorDataset(
            states[val_inds], actions[val_inds], nexts[val_inds])

        return train_set, val_set

    def sizes(self):
        N = []
        for x in self.states:
            N.append(len(self.states[x]))
        return N


class Split_Class_Dataset(Dataset):
    """
    New Disjoint Class Setting

    Given a dataset (train or val or test),
    Return a dataset that can set new tasks
    """

    def __init__(self, data, num_task=1, rot=False):
        self.data = data
        self.rot = rot
        self.map = {}
        self.task_id = 0
        self.num_task = num_task
        self.compile()

    def compile(self):
        for i, (X, y) in enumerate(self.data):
            if y not in self.map:
                self.map[y] = [i]
            else:
                self.map[y].append(i)

        # Divide the all classes in the dataset
        # to a number of tasks
        total = len(self.map.keys())
        per_task = int(total // self.num_task)
        index = [i for i in range(total)]
        # TODO: randomize index selection per group

        # Map each task to a set of data indices
        # by joining the tasks
        task_inds = {}
        for id in range(self.num_task):
            base = id * per_task
            task_inds[id] = []
            for i in range(base, base+per_task):
                task_inds[id] += self.map[index[i]]

        task_inds[self.num_task] = np.arange(len(self.data))

        self.map = task_inds

    def set_task(self, id):
        if id == "all":
            id = self.num_task
        if id not in self.map:
            raise ValueError("Task ID not found")

        self.task_id = id

    def __getitem__(self, index):
        ind = self.map[self.task_id][index]
        data = self.data[ind]
        if self.rot:
            data = rotate_imgs(data)
        return data

    def __len__(self):
        return len(self.map[self.task_id])


class Mixed_Dataset(Dataset):

    def __init__(self, datas, mode):
        assert hasattr(datas, '__len__')
        assert hasattr(datas[0], '__len__')

        self.mode = mode
        self.datas = datas
        self.lens = [len(data) for data in self.datas]
        self.total = sum(self.lens)

        self.map = []
        ds_ind = 0
        ds_sum = 0
        for i in range(self.total):
            ind = i - ds_sum
            while ind >= self.lens[ds_ind]:
                ds_sum += self.lens[ds_ind]
                ds_ind += 1
                ind = i - ds_sum
            self.map.append((ds_ind, ind))

        if mode == "iid":
            np.random.shuffle(self.map)

    def __getitem__(self, index):
        if self.mode == "iid":
            ds, ind = self.map[index]
            return self.datas[ds][ind]
        elif self.mode == "sequential":
            ds, ind = self.map[index]
            return self.datas[ds][ind]

    def __len__(self):
        return self.total
