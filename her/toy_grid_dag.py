import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count, permutations
import wandb
import random
import time
from sklearn.preprocessing import LabelEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from math import log2
import pandas as pd


from itertools import chain

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
"""
python run_hydra.py ndim=3 horizon=16 R0=0.001 method=qm N=8 quantile_dim=256 seed=0
python run_hydra.py ndim=3 horizon=16 method=qm N=8 beta=wang eta=0.75 # or -0.75
python run_hydra.py ndim=3 horizon=16 method=qm N=8 beta=cpw eta=0.71
python run_hydra.py ndim=3 horizon=16 method=qm N=8 beta=cvar eta=0.25 or 0.1
python run_hydra.py ndim=5 horizon=20 method=fm_egfn n_train_steps=5000 replay_sample_size=16 R0=0.0001
"""

TOKEN_GAP = "-"
TOKENS_AA = list("ARNDCEQGHILKMFPSTWYV")
TOKENS_AHO = sorted([TOKEN_GAP, *TOKENS_AA])

ALPHABET_AHO = LabelEncoder().fit(TOKENS_AHO)

_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])



def set_device(dev):
    _dev[0] = dev

def token_string_from_tensor(
    tensor: torch.Tensor,
    alphabet: LabelEncoder,
    from_logits: bool = True,
) -> list[str]:
    """Convert tensor representation of sequence to list of string

    Parameters
    ----------
    tensor: torch.Tensor
        Input tensor
    from_logits: bool
        If True, elect to first compute the argmax in the final dimension of the tensor

    Returns
    -------
    List[str]
        The sequence version
    """
    # convert to shape (b, L, V) (if from_logits)  or (b, L) (if not from_logits) if necessary
    if (from_logits and tensor.dim() == 2) or (not from_logits and tensor.dim() == 1):
        tensor = tensor.unsqueeze(0)

    if from_logits:
        tensor = tensor.argmax(-1)

    tensor = tensor.cpu()
    return [
        "".join(alphabet.inverse_transform(tensor[i, ...].tolist()).tolist())
        for i in range(tensor.size(0))
    ]


def get_seq(s):
    base_seq = 'QVQLVQS-GTEVKKPGSSVKVSCKASG-GTFSS-----YAVSWVRQAPGQGLEWMGRFIPI---LNIKNYAQDFQGRVTITADKSTTTAYMELINLGPEDTAVYYCARGSLSGR-----------------EGLPLEYWGQGTLVSVSS' + 'EVVMTQSPATLSVSPGESATLYCRAS--QIVT------SDLAWYQQIPGQAPRLLIFA--------ASTRATGIPARFSGSGSE--TDFTLTISSLQSEDFAIYYCQQYFH-----------------------WPPTFGQGTKVEIK'
    ax = torch.from_numpy(s)

    alphabet = ALPHABET_AHO
    first_seq = token_string_from_tensor(ax, alphabet, from_logits=False)
    # now paste the first_seq at the beginning of the base_seq
    base_seq = first_seq[0] + base_seq[len(first_seq):]
    return base_seq

def get_func(arg):
    if arg.func == "corner":

        def func(x):
            base_seq = 'QVQLVQS-GTEVKKPGSSVKVSCKASG-GTFSS-----YAVSWVRQAPGQGLEWMGRFIPI---LNIKNYAQDFQGRVTITADKSTTTAYMELINLGPEDTAVYYCARGSLSGR-----------------EGLPLEYWGQGTLVSVSS' + 'EVVMTQSPATLSVSPGESATLYCRAS--QIVT------SDLAWYQQIPGQAPRLLIFA--------ASTRATGIPARFSGSGSE--TDFTLTISSLQSEDFAIYYCQQYFH-----------------------WPPTFGQGTKVEIK'
            ax = torch.from_numpy(x)

            alphabet = ALPHABET_AHO
            first_seq = token_string_from_tensor(ax, alphabet, from_logits=False)
            # now paste the first_seq at the beginning of the base_seq
            base_seq = first_seq[0] + base_seq[len(first_seq):]
            base_seq = base_seq.replace("-", "")
            r = 50 - ProteinAnalysis(str(base_seq)).instability_index()
            r = r / 10
            r = 2 ** r
            
            return torch.tensor(r)

        return func
    elif arg.func == "cosine":

        def func(x):
            return arg.R0 + ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1)

        return func
    else:
        raise NotImplementedError


def func_diff_corners(x, coes):
    r_lr = (x > 0.5).prod(-1) * coes[0] + ((x < 0.8) * (x > 0.6)).prod(-1) * coes[4]
    r_ll = ((x[0] > 0.6) * (x[1] < -0.6)) * coes[1] + (
        (x[0] > 0.6) * (x[0] < 0.8) * (x[1] > -0.8) * (x[1] < -0.6)
    ) * coes[5]
    r_ul = ((x[0] < -0.6) * (x[1] < -0.6)) * coes[2] + (
        (x[0] > -0.8) * (x[0] < -0.6) * (x[1] > -0.8) * (x[1] < -0.6)
    ) * coes[6]
    r_ur = ((x[0] < -0.6) * (x[1] > 0.6)) * coes[3] + (
        (x[0] > -0.8) * (x[0] < -0.6) * (x[1] > 0.6) * (x[1] < 0.8)
    ) * coes[7]
    r = r_lr + r_ll + r_ul + r_ur + +1e-1
    return r


class GridEnv:
    def __init__(
        self, horizon, ndim=2, xrange=[-1, 1], func=None, allow_backward=False
    ):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01)
            if func is None
            else func
        )
        self.xspace = np.linspace(*xrange, horizon)
        self.allow_backward = allow_backward  # If true then this is a
        # MCMC ergodic env,
        # otherwise a DAG
        self._true_density = None

    # s: [3, 4, 2]; x: [-0.3, 0.6, 0.12]
    # e.g., [0, 1] -> [1, 0, ..., 0; 0, 1, 0, ..., 0], mainly for NN to understand easily
    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        return s

    # [1, 6, 3] -> [1, -1, 0]
    # 1 or -1 means in mode, 0 means not in mode
    def s2mode(self, s):
        x = np.asarray(s)
        reward = 50 - log2(self.func(self.s2x(x))) * 10
        if reward <= 35:
            return True
        # TO-DO: return true if s is in mode
        return False

    def reset(self):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        return self.obs(), self.func(self.s2x(self._state)), self._state

    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.obs(s)], [self.ndim]
        parents = []
        actions = []
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                if sp.max() == self.horizon - 1:  # can't have a terminal parent
                    continue
                parents += [self.obs(sp)]
                actions += [i]
        return parents, actions

    def step(self, a, s=None):
        if self.allow_backward:
            return self.step_chain(a, s)
        return self.step_dag(a, s)

    def step_dag(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1
        # a == self.ndim means to stop
        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.func(self.s2x(s)), done, s

    def step_chain(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        sc = s + 0
        if a < self.ndim:
            s[a] = min(s[a] + 1, self.horizon - 1)
        if a >= self.ndim:  # go backward
            s[a - self.ndim] = max(s[a - self.ndim] - 1, 0)

        reverse_a = ((a + self.ndim) % (2 * self.ndim)) if any(sc != s) else a

        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), self.func(self.s2x(s)), s, reverse_a

    def true_density(self):
        return None

    def true_traj_density(self):
        td, end_states, true_r = self.true_density()
        true_density = tf(td)
        # trajectory length is defined by sum of each state in end_states
        traj_density = torch.zeros(sum(end_states[-1]) + 1, device=_dev[0])
        # now for state in end_states, add the corresponding true_r to traj_density[state.sum()]
        traj_density.index_add_(0, torch.tensor(end_states).sum(1), tf(true_r))
        return traj_density / traj_density.sum()

    def all_possible_states(self):
        """Compute quantities for debugging and analysis"""

        # all possible action sequences
        def step_fast(a, s):
            s = s + 0
            s[a] += 1
            return s

        f = lambda a, s: (
            [np.int32(a)]
            if np.max(s) == self.horizon - 1
            else [np.int32(a + [self.ndim])]
            + sum([f(a + [i], step_fast(i, s)) for i in range(self.ndim)], [])
        )
        all_act_seqs = f([], np.zeros(self.ndim, dtype="int32"))

        # all RL states / intermediary nodes
        all_int_states = list(
            itertools.product(*[list(range(self.horizon))] * self.ndim)
        )
        # Now we need to know for each partial action sequence what
        # the corresponding states are. Here we can just count how
        # many times we moved in each dimension:
        all_traj_states = np.int32(
            [
                np.bincount(i[:j], minlength=self.ndim + 1)[:-1]
                for i in all_act_seqs
                for j in range(len(i))
            ]
        )
        # all_int_states is ordered, so we can map a trajectory to its
        # index via a sum
        arr_mult = np.int32(
            [self.horizon ** (self.ndim - i - 1) for i in range(self.ndim)]
        )
        all_traj_states_idx = (all_traj_states * arr_mult[None, :]).sum(1)
        # For each partial trajectory, we want the index of which trajectory it belongs to
        all_traj_idxs = [[j] * len(i) for j, i in enumerate(all_act_seqs)]
        # For each partial trajectory, we want the index of which state it leads to
        all_traj_s_idxs = [
            (np.bincount(i, minlength=self.ndim + 1)[:-1] * arr_mult).sum()
            for i in all_act_seqs
        ]
        # Vectorized
        a = torch.cat(list(map(torch.LongTensor, all_act_seqs)))
        u = torch.LongTensor(all_traj_states_idx)
        v1 = torch.cat(list(map(torch.LongTensor, all_traj_idxs)))
        v2 = torch.LongTensor(all_traj_s_idxs)

        # With all this we can do an index_add, given
        # pi(all_int_states):
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            dev = policy_for_all_states.device
            pi_a_s = torch.log(policy_for_all_states[u, a])
            q = torch.exp(
                torch.zeros(len(all_act_seqs), device=dev).index_add_(0, v1, pi_a_s)
            )
            q_sum = torch.zeros((all_xs.shape[0],), device=dev).index_add_(0, v2, q)
            return q_sum[state_mask]

        # some states aren't actually reachable
        state_mask = np.bincount(all_traj_s_idxs, minlength=len(all_int_states)) > 0
        # Let's compute the reward as well
        all_xs = (
            np.float32(all_int_states)
            / (self.horizon - 1)
            * (self.xspace[-1] - self.xspace[0])
            + self.xspace[0]
        )
        traj_rewards = self.func(all_xs)[state_mask]
        # All the states as the agent sees them:
        all_int_obs = np.float32([self.obs(i) for i in all_int_states])
        return all_int_obs, traj_rewards, all_xs, compute_all_probs

    def getModeStates(self):
        _, states, rewards = self.true_density()
        modes = [(x, r) for x, r in zip(states, rewards) if r > 1]
        return modes


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(
        *(
            sum(
                [
                    [nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
                    for n, (i, o) in enumerate(zip(l, l[1:]))
                ],
                [],
            )
            + tail
        )
    )
    return net


class ReplayBuffer:
    def __init__(self, args, env):
        self.buf = []
        self.args = args
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, r_x):
        if self.strat == "top_k":
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, x)])[-self.bufsize :]

    def sample(self):
        if not len(self.buf):
            return []

        if self.args.prb:
            # sample priority reward buffer. Divide the buffer into two parts: high reward and low reward
            # sample from high reward part with probability 0.5, and sample from low reward part with probability 0.5
            # get the last 10% from self.buf
            sample_size = int(self.sample_size / 2)
            top10_percentile = self.buf[int(len(self.buf) * 0.9) :]
            idxs = np.random.randint(0, len(top10_percentile), sample_size)
            sample1 = sum(
                [self.generate_backward(*top10_percentile[i]) for i in idxs], []
            )
            other90_percentile = self.buf[: int(len(self.buf) * 0.9)]
            idxs = np.random.randint(0, len(other90_percentile), sample_size)
            sample2 = sum(
                [self.generate_backward(*other90_percentile[i]) for i in idxs], []
            )
            return sample1 + sample2
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return sum([self.generate_backward(*self.buf[i]) for i in idxs], [])

    def generate_backward(self, r, s0):
        s = np.int8(s0)
        os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            parents, actions = self.env.parent_transitions(s, used_stop_action)
            # add the transition
            traj.append(
                [tf(i) for i in (parents, actions, [r], [self.env.obs(s)], [done])]
            )
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj


class ReplayBufferTB:
    def __init__(self, args, env):
        self.buf = []
        self.args = args
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, a, r_x):
        if self.strat == "top_k":
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, a, x)], key=lambda x: x[0])[
                    -self.bufsize :
                ]

    def sample(self):
        # this has to give us [state, action, reward] state.shape is (self.sample_size, traj_length, dim)
        # action.shape is (self.sample_size, traj_length, 1) # reward.shape is (self.sample_size, 1)
        if not len(self.buf):
            return []

        if self.args.prb:
            # sample priority reward buffer. Divide the buffer into two parts: high reward and low reward
            # sample from high reward part with probability 0.5, and sample from low reward part with probability 0.5
            # get the last 10% from self.buf
            top_sample_size = int(self.sample_size * self.args.top_sample_perc)
            bottom_sample_size = self.sample_size - top_sample_size
            top10_percentile = self.buf[int(len(self.buf) * self.args.percentile) :]
            idxs = np.random.randint(0, len(top10_percentile), top_sample_size)
            sample1 = [top10_percentile[i] for i in idxs]
            other90_percentile = self.buf[: int(len(self.buf) * self.args.percentile)]
            idxs = np.random.randint(0, len(other90_percentile), bottom_sample_size)
            sample2 = [other90_percentile[i] for i in idxs]
            return sample1 + sample2
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return [self.buf[i] for i in idxs]

    def generate_backward(self, r, s0):
        s = np.int8(s0)
        os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            parents, actions = self.env.parent_transitions(s, used_stop_action)
            # add the transition
            traj.append(
                [tf(i) for i in (parents, actions, [r], [self.env.obs(s)], [done])]
            )
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj


class ReplayBufferDB:
    def __init__(self, args, env):
        self.buf = []
        self.args = args
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, a, r_x, steps):
        if self.strat == "top_k":
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, a, x, steps)], key=lambda x: x[0])[
                    -self.bufsize :
                ]

    def sample(self):
        # this has to give us [state, action, reward] state.shape is (self.sample_size, traj_length, dim)
        # action.shape is (self.sample_size, traj_length, 1) # reward.shape is (self.sample_size, 1)
        if not len(self.buf):
            return []
        if self.args.prb:
            # sample priority reward buffer. Divide the buffer into two parts: high reward and low reward
            # sample from high reward part with probability 0.5, and sample from low reward part with probability 0.5
            # get the last 10% from self.buf
            top_sample_size = int(self.sample_size * self.args.top_sample_perc)
            bottom_sample_size = self.sample_size - top_sample_size
            top10_percentile = self.buf[int(len(self.buf) * self.args.percentile) :]
            idxs = np.random.randint(0, len(top10_percentile), top_sample_size)
            sample1 = [top10_percentile[i] for i in idxs]
            other90_percentile = self.buf[: int(len(self.buf) * self.args.percentile)]
            idxs = np.random.randint(0, len(other90_percentile), bottom_sample_size)
            sample2 = [other90_percentile[i] for i in idxs]
            return sample1 + sample2
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return [self.buf[i] for i in idxs]


class FlowNetAgent:
    def __init__(self, args, envs, is_star=True):
        self.model = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.model.to(args.dev)
        self.target = copy.deepcopy(self.model)
        self.envs = envs
        self.ndim = args.ndim
        self.tau = args.bootstrap_tau
        self.replay = ReplayBuffer(args, envs[0])
        self.args = args
        self.is_star = is_star

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        batch = []
        if self.is_star:
            batch += self.replay.sample()  # only agent star samples from replay
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        while not all(done):
            # Note to self: this is ugly, ugly code
            with torch.no_grad():
                pred = self.model(s)
                acts = Categorical(logits=pred).sample()

            step = [
                e.step(a)
                for e, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            p_a = [
                self.envs[0].parent_transitions(sp_state, a == self.ndim)
                for a, (sp, r, done, sp_state) in zip(acts, step)
            ]
            batch += [
                [tf(i) for i in (p, a, [r], [sp], [d])]
                for (p, a), (sp, r, d, _) in zip(p_a, step)
            ]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for _, r, d, sp in step:
                if d:
                    all_visited.append(tuple(sp))
                    self.replay.add(tuple(sp), r)
        return batch

    def learn_from(self, it, batch):
        loginf = tf([1000])

        # batch_idxs.shape[0] = parents_Qsa.shape[0] > sp.shape[0] = in_flow.shape[0]
        batch_idxs = tl(
            sum(
                [[i] * len(parents) for i, (parents, _, _, _, _) in enumerate(batch)],
                [],
            )
        )
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        parents_Qsa = self.model(parents)[
            torch.arange(parents.shape[0]), actions.long()
        ]
        in_flow = torch.log(
            torch.zeros((sp.shape[0],))
            .to(self.args.dev)
            .index_add_(0, batch_idxs, torch.exp(parents_Qsa))
        )

        if self.tau > 0:
            with torch.no_grad():
                next_q = self.target(sp)
        else:
            next_q = self.model(sp)

        next_qd = next_q * (1 - done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        # because r > 0 only when done == 1, it is exactly same with
        # out_flow = torch.where(done.bool(), torch.log(r), next_qd.logsumexp(dim=-1))
        # loss = (in_flow - out_flow).pow(2).mean()

        losses = (in_flow - out_flow).pow(2)
        loss = (losses * done * self.args.leaf_coef + losses * (1 - done)).sum() / len(
            losses
        )

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (
                done.sum() + 1e-20
            )
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / (
                (1 - done).sum() + 1e-20
            )

        if self.tau > 0:
            for a, b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)

        return loss, term_loss.detach(), flow_loss.detach()

class RND(nn.Module):
    def __init__(self, state_dim, reward_scale=0.5, hidden_dim=256, s_latent_dim=128):
        super(RND, self).__init__()

        self.random_target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim)
        )

        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )
        
        self.reward_scale = reward_scale

    def forward(self, next_state):
        random_phi_s_next = self.random_target_network(next_state)
        predicted_phi_s_next = self.predictor_network(next_state)
        return random_phi_s_next, predicted_phi_s_next

    def compute_intrinsic_reward(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)

        intrinsic_reward = torch.norm(predicted_phi_s_next.detach() - random_phi_s_next.detach(), dim=-1, p=2)
        intrinsic_reward *= self.reward_scale

        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()

        return intrinsic_reward

    def compute_loss(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)
        rnd_loss = torch.norm(predicted_phi_s_next - random_phi_s_next.detach(), dim=-1, p=2)
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss

class TBFlowNetAgent:
    def __init__(self, args, envs, is_star=True):
        out_dim = 2 * args.ndim + 1
        if args.augmented:
            out_dim += 1
        self.model = make_mlp([args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [out_dim])

        self.model.to(args.dev)
        print(self.model)

        self.augmented = args.augmented
        if self.augmented:
            self.intrinsic_reward_model = RND(args.horizon * args.ndim, args.ri_eta)
            self.intrinsic_reward_model.to(args.dev)
            self.ri_loss_coe = 1.

        self.args = args
        self.replay = ReplayBufferTB(args, envs)
        self.Z = torch.zeros((1,)).to(args.dev)
        self.Z.requires_grad_()

        self.envs = envs
        self.ndim = args.ndim
        self.horizon = args.horizon
        self.tau = args.bootstrap_tau
        self.is_star = is_star

        self.exp_weight = args.exp_weight
        self.temp = args.temp
        self.uniform_pb = args.rand_pb
        self.iter_cnt = 0
        self.dev = args.dev

    def parameters(self):
        if self.augmented:
            return chain(self.model.parameters(), self.intrinsic_reward_model.parameters())
        else:
            return self.model.parameters()

    def sample_many(self, mbsize, all_visited, to_print=False):
        if self.augmented:
            return self.sample_many_augmented(mbsize, all_visited, to_print)
        else:
            return self.sample_many_tb(mbsize, all_visited, to_print)
        
    def sample_many_augmented(self, mbsize, all_visited, to_print=False):
        inf = 1000000000

        batch_s, batch_a, batch_next_s, batch_ri = [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        
        while not all(done):
            with torch.no_grad():
                pred = self.model(s)

                z = s.reshape(-1, self.ndim, self.horizon).argmax(-1)
           

                edge_mask = torch.cat([(z == self.horizon - 1).float(), torch.zeros((len(done) - sum(done), 1), device=self.dev)], 1)
                logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

                sample_ins_probs = logits.softmax(1)
                acts = sample_ins_probs.multinomial(1).squeeze(-1)

            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]

            if self.augmented:
                next_s = tf([i[0] for i in step])
                intrinsic_rewards = self.intrinsic_reward_model.compute_intrinsic_reward(next_s)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = curr_s.reshape(self.ndim, self.horizon).argmax(-1)


                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

                if self.augmented:
                    batch_next_s[env_idx].append(next_s[dat_idx])
                    batch_ri[env_idx].append(intrinsic_rewards[dat_idx])

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()

                    formatted_ns = ns.reshape(self.ndim, self.horizon).argmax(-1)

                    batch_s[env_idx].append(tl(formatted_ns.tolist()))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
                    
        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1
            if self.augmented:
                batch_next_s[i] = torch.stack(batch_next_s[i])
                batch_ri[i] = torch.tensor(batch_ri[i]).unsqueeze(-1).float().to(self.dev)

        batch_R = [env_idx_return_map[i] + batch_ri[i][-1].item() if self.augmented else env_idx_return_map[i] for i in range(len(batch_s))]


   
        return [batch_s, batch_a, batch_R, batch_steps, batch_next_s, batch_ri]
    
    def sample_many_tb(self, mbsize, all_visited, to_print=False):
        self.iter_cnt += 1
        if self.augmented:
            batch_s, batch_a, batch_next_s, batch_ri = [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        else:
            batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs])[:mbsize, ...]
        done = [False] * mbsize

        terminals = []
        while not all(done):
            with torch.no_grad():
                pred = self.model(s)
                z = s.reshape(-1, self.ndim, self.horizon).argmax(-1)
                # mask unavailable actions
                edge_mask = torch.cat(
                    [
                        (z == self.horizon - 1).float(),
                        torch.zeros((len(done) - sum(done), 1), device=self.args.dev),
                    ],
                    1,
                )
                logits = (
                    pred[..., : self.args.ndim + 1] - 1000000000 * edge_mask
                ).log_softmax(1)
                sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(
                    1
                ) + self.exp_weight * (1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(
                    1
                ).unsqueeze(
                    1
                )
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            # observation, reward, done, state
            step = [
                i.step(a)
                for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]

            if self.augmented:
                next_s = tf([i[0] for i in step])
                intrinsic_rewards = self.intrinsic_reward_model.compute_intrinsic_reward(next_s)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]
                curr_formatted_s = curr_s.reshape(self.ndim, self.horizon).argmax(-1)
                batch_s[env_idx].append(curr_formatted_s)  # save this for training
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

                if self.augmented:
                    batch_next_s[env_idx].append(next_s[dat_idx])
                    batch_ri[env_idx].append(intrinsic_rewards[dat_idx])

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()
                    formatted_ns = ns.reshape(self.ndim, self.horizon).argmax(-1)
                    batch_s[env_idx].append(tl(formatted_ns.tolist()))


            not_done_envs = [
                env_idx for env_idx, env_d in env_idx_done_map.items() if not env_d
            ]

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            for _, r, d, sp in step:
                if d:
                    all_visited.append(tuple(sp))
                    terminals.append(list(sp))
        # batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]
        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])
            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1
            if self.augmented:
                batch_next_s[i] = torch.stack(batch_next_s[i])
                batch_ri[i] = torch.tensor(batch_ri[i]).unsqueeze(-1).float().to(self.dev)
        replay_s, replay_a, replay_R = [], [], []
        if self.is_star:  # only agent star samples from replay
            for r, a, x in self.replay.sample():
                replay_s.append(x)
                replay_a.append(a)
                replay_R.append(r)
        # batch_R = [env_idx_return_map[i] for i in range(len(batch_s))]
        batch_R = [env_idx_return_map[i] + batch_ri[i][-1].item() if self.augmented else env_idx_return_map[i] for i in range(len(batch_s))]
        for s, a, r in zip(batch_s, batch_a, batch_R):
            self.replay.add(s, a, r)
        return [
            batch_s + replay_s,
            batch_a + replay_a,
            batch_R + replay_R,
        ]  # , np.mean(batch_R) this gives us trajectory

    def convert_states_to_onehot(self, states):
        # convert to onehot format
        try:
            return (
                torch.nn.functional.one_hot(states, self.horizon)
                .view(states.shape[0], -1)
                .float()
            )
        except RuntimeError:
            print(states)

    def learn_from(self, it, batch):
        if self.augmented:
            return self.learn_from_augmented(it, batch)
        else:
            return self.learn_from_normal(it, batch)
        
    def learn_from_augmented(self, it, batch):
        inf = 1000000000

        states, actions, returns, episode_lens, next_states, intrinsic_rewards = batch
        returns = torch.tensor(returns).to(self.dev)
        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :]
            curr_actions = actions[data_idx][:curr_episode_len - 1, :]
            curr_return = returns[data_idx]

            curr_states_onehot = self.convert_states_to_onehot(curr_states)
            pred = self.model(curr_states_onehot)
            
            edge_mask = torch.cat([(curr_states == self.horizon - 1).float(), torch.zeros((curr_states.shape[0], 1), device=self.dev)], 1)
            logits = (pred[..., :self.ndim + 1] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float()
            back_logits_end_pos = -1 if self.augmented else pred.shape[-1]
            # print(f"init mask shape: {init_edge_mask.shape}, back_logits_end_pos: {back_logits_end_pos}")
            # print(f"pred shape: {pred[..., self.ndim + 1:back_logits_end_pos].shape}")
            
            back_logits = (pred[..., self.ndim + 1:back_logits_end_pos] - inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            back_logits = back_logits[1:-1, :].gather(1, curr_actions[:-1, :]).squeeze(1) if curr_actions[-1] == self.ndim else back_logits[1:, :].gather(1, curr_actions).squeeze(1)

            sum_logits = torch.sum(logits)
            if self.augmented:
                curr_intrinsic_rewards = intrinsic_rewards[data_idx].squeeze(-1)[:-1] 
                flow = (pred[..., -1][1:-1]).exp()
                augmented_r_f = curr_intrinsic_rewards / flow    
                sum_back_logits = torch.sum((back_logits.exp() + augmented_r_f).log()) if curr_actions[-1] == self.ndim else torch.sum((back_logits[:-1].exp() + augmented_r_f).log()) + back_logits[-1]
            else:
                sum_back_logits = torch.sum(back_logits)

            curr_return = curr_return.float() + 1e-8

            curr_ll_diff = self.Z + sum_logits - curr_return.log() - sum_back_logits
            ll_diff.append(curr_ll_diff ** 2)

        loss = torch.cat(ll_diff).sum() / len(states)
        if self.augmented:
            rnd_loss = torch.stack([self.intrinsic_reward_model.compute_loss(next_states[data_idx]) for data_idx in range(len(states))]).sum() / len(states)
            loss += self.ri_loss_coe * rnd_loss

        return [loss]

    def learn_from_normal(self, it, batch):
        inf = 1000000000
        states, actions, returns = batch
        returns = torch.tensor(returns).to(self.dev)

        ll_diff = []
        for data_idx in range(len(states)):
            curr_states = states[data_idx]
            curr_actions = actions[data_idx]
            curr_return = returns[data_idx]

            # convert state into one-hot format: steps, ndim x horizon
            curr_states_onehot = self.convert_states_to_onehot(curr_states)
            # get predicted forward (from 0 to n_dim) and backward logits (from n_dim to last): steps, 2 x ndim + 1
            pred = self.model(curr_states_onehot)
            edge_mask = torch.cat(
                [
                    (curr_states == self.horizon - 1).float(),
                    torch.zeros((curr_states.shape[0], 1), device=self.dev),
                ],
                1,
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(
                1
            )  # steps, n_dim + 1

            init_edge_mask = (
                curr_states == 0
            ).float()  # whether it is at an initial position
            back_logits = (
                (0 if self.uniform_pb else 1) * pred[..., self.ndim + 1 :]
                - inf * init_edge_mask
            ).log_softmax(
                -1
            )  # steps, n_dim
            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, curr_actions[:-1, :]).squeeze(1)
            )

            curr_ll_diff = (
                self.Z
                + torch.sum(logits)
                - curr_return.float().log()
                - torch.sum(back_logits)
            )
            ll_diff.append(curr_ll_diff**2)

        loss = torch.cat(ll_diff).sum() / len(states)
        return [loss]


class DBFlowNetAgent:
    def __init__(self, args, envs, is_star=True):
        out_dim = 2 * args.ndim + 2
        self.model = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [out_dim]
        )
        self.model.to(args.dev)
        print(self.model)

        self.device = args.dev
        self.args = args

        self.envs = envs
        self.ndim = args.ndim
        self.horizon = args.horizon
        self.tau = args.bootstrap_tau

        self.exp_weight = args.exp_weight
        self.temp = 1
        self.uniform_pb = args.rand_pb
        self.dev = args.dev

        self.iter_cnt = 0
        self.is_star = is_star
        self.replay = ReplayBufferDB(args, envs)

    def parameters(self):
        # return parameters' of the model
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited, to_print=False):
        self.iter_cnt += 1

        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs]).to(self.device)
        done = [False] * mbsize
        s = s[:mbsize, ...]
        terminals = []
        while not all(done):
            with torch.no_grad():
                pred = self.model(s)
                z = s.reshape(-1, self.ndim, self.horizon).argmax(-1)

                # mask unavailable actions
                edge_mask = torch.cat(
                    [
                        (z == self.horizon - 1).float(),
                        torch.zeros((len(done) - sum(done), 1), device=self.args.dev),
                    ],
                    1,
                )
                logits = (
                    pred[..., : self.args.ndim + 1] - 1000000000 * edge_mask
                ).log_softmax(1)

                sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(
                    1
                ) + self.exp_weight * (1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(
                    1
                ).unsqueeze(
                    1
                )
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            # observation, reward, done, state
            step = [
                i.step(a)
                for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]
                curr_formatted_s = curr_s.reshape(self.ndim, self.horizon).argmax(-1)
                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()
                    formatted_ns = ns.reshape(self.ndim, self.horizon).argmax(-1)
                    batch_s[env_idx].append(tl(formatted_ns.tolist()))

            not_done_envs = [
                env_idx for env_idx, env_d in env_idx_done_map.items() if not env_d
            ]

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            for _, r, d, sp in step:
                if d:
                    all_visited.append(tuple(sp))
                    terminals.append(list(sp))

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])
            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1

        batch_R = [env_idx_return_map[i] for i in range(len(batch_s))]
        mean_return = 1.0 * sum(batch_R) / len(batch_R)
        replay_s, replay_a, replay_R, replay_steps = [], [], [], []
        if self.is_star:  # only agent star samples from replay
            for r, a, x, steps in self.replay.sample():
                replay_s.append(x)
                replay_a.append(a)
                replay_R.append(r)
                replay_steps.append(steps)
        for s, a, r, steps in zip(batch_s, batch_a, batch_R, batch_steps):
            self.replay.add(s, a, r, steps)
        return [
            batch_s + replay_s,
            batch_a + replay_a,
            batch_R + replay_R,
            batch_steps + replay_steps,
        ]  # mean_return

    def convert_states_to_onehot(self, states):
        # convert to onehot format
        return (
            torch.nn.functional.one_hot(states, self.horizon)
            .view(states.shape[0], -1)
            .float()
        )

    def learn_from(self, it, batch):
        inf = 1000000000

        states, actions, returns, episode_lens = batch
        returns = torch.tensor(returns).to(self.dev)

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][
                :curr_episode_len, :
            ]  # episode_len + 1, state_dim
            curr_actions = actions[data_idx][
                : curr_episode_len - 1, :
            ]  # episode_len, action_dim
            curr_return = returns[data_idx].float()

            # convert state into one-hot format: steps, ndim x horizon
            curr_states_onehot = self.convert_states_to_onehot(curr_states)

            # get predicted forward (from 0 to n_dim) and backward logits (from n_dim to last): steps, 2 x ndim + 1
            pred = self.model(curr_states_onehot)

            edge_mask = torch.cat(
                [
                    (curr_states == self.horizon - 1).float(),
                    torch.zeros((curr_states.shape[0], 1), device=self.dev),
                ],
                1,
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(
                1
            )  # steps, n_dim + 1

            init_edge_mask = (
                curr_states == 0
            ).float()  # whether it is at an initial position
            back_logits = (
                (0 if self.uniform_pb else 1) * pred[..., self.ndim + 1 : -1]
                - inf * init_edge_mask
            ).log_softmax(
                1
            )  # steps, n_dim

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, curr_actions[:-1, :]).squeeze(1)
            )

            log_flow = pred[..., -1]  # F(s) (the last dimension)
            log_flow = log_flow[:-1]

            curr_ll_diff = torch.zeros(curr_states.shape[0] - 1).to(self.dev)
            curr_ll_diff += log_flow
            curr_ll_diff += logits
            curr_ll_diff[:-1] -= log_flow[1:]
            curr_ll_diff[:-1] -= back_logits
            curr_ll_diff[-1] -= curr_return.log()

            ll_diff.append(curr_ll_diff**2)

        ll_diff = torch.cat(ll_diff)
        loss = ll_diff.sum() / len(ll_diff)
        return [loss]


class SplitCategorical:
    def __init__(self, n, logits):
        """Two mutually exclusive categoricals, stored in logits[..., :n] and
        logits[..., n:], that have probability 1/2 each."""
        self.cats = Categorical(logits=logits[..., :n]), Categorical(
            logits=logits[..., n:]
        )
        self.n = n
        self.logits = logits
        self.dev = logits.device

    def sample(self):
        split = torch.rand(self.logits.shape[:-1]) < 0.5
        return self.cats[0].sample() * split + (self.n + self.cats[1].sample()) * (
            ~split
        )

    def log_prob(self, a):
        split = (a < self.n).to(self.dev)
        log_one_half = -0.693147
        return (
            log_one_half
            + self.cats[  # We need to multiply the prob by 0.5, so add log(0.5) to logprob
                0
            ].log_prob(
                torch.minimum(a, torch.tensor(self.n - 1)).to(self.dev)
            )
            * split
            + self.cats[1].log_prob(
                torch.maximum(a - self.n, torch.tensor(0).to(self.dev))
            )
            * (~split)
        )

    def entropy(self):
        return Categorical(
            probs=torch.cat([self.cats[0].probs, self.cats[1].probs], -1) * 0.5
        ).entropy()


class MARSAgent:
    def __init__(self, args, envs):
        self.model = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim * 2]
        )
        self.model.to(args.dev)
        self.dataset = []
        self.dataset_max = 10000000  # args.n_dataset_pts
        self.mbsize = args.mbsize
        self.envs = envs
        self.batch = [i.reset() for i in envs]  # The N MCMC chains
        self.ndim = args.ndim
        self.bufsize = args.bufsize
        self.args = args

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        s = torch.cat([tf([i[0]]) for i in self.batch])
        r = torch.cat([tf([i[1]]) for i in self.batch])
        with torch.no_grad():
            logits = self.model(s)
        pi = SplitCategorical(self.ndim, logits=logits)
        a = pi.sample()
        steps = [
            self.envs[j].step(a[j].item(), s=self.batch[j][2])
            for j in range(len(self.envs))
        ]
        sp = torch.cat([tf([i[0]]) for i in steps])
        rp = torch.cat([tf([i[1]]) for i in steps])
        with torch.no_grad():
            logits_sp = self.model(sp)
        reverse_a = tl([i[3] for i in steps])
        pi_sp = SplitCategorical(self.ndim, logits=logits_sp)
        q_xxp = torch.exp(pi.log_prob(reverse_a))
        # This is the correct MH acceptance ratio:
        # A = (rp * q_xxp) / (r * q_xpx + 1e-6)

        # But the paper suggests to use this ratio, for reasons poorly
        # explained... it does seem to actually work better? but still
        # diverges sometimes. Idk
        A = rp / r
        U = torch.rand(self.bufsize)
        for j in range(self.bufsize):
            if A[j] > U[j]:  # Accept
                self.batch[j] = (sp[j].numpy(), rp[j].item(), steps[j][2])
                all_visited.append(tuple(steps[j][2]))
            # Added `or U[j] < 0.05` for stability in these toy settings
            if rp[j] > r[j] or U[j] < 0.05:  # Add to dataset
                self.dataset.append((s[j].unsqueeze(0), a[j].unsqueeze(0)))
        return []  # agent is stateful, no need to return minibatch data

    def learn_from(self, i, data):
        if not i % 20 and len(self.dataset) > self.dataset_max:
            self.dataset = self.dataset[-self.dataset_max :]
        if len(self.dataset) < self.mbsize:
            return None
        idxs = np.random.randint(0, len(self.dataset), self.mbsize)
        s, a = map(torch.cat, zip(*[self.dataset[i] for i in idxs]))
        logits = self.model(s)
        pi = SplitCategorical(self.ndim, logits=logits)
        q_xxp = pi.log_prob(a)
        loss = -q_xxp.mean() + np.log(0.5)
        return loss, pi.entropy().mean()


class MHAgent:
    def __init__(self, args, envs):
        self.envs = envs
        self.batch = [i.reset() for i in envs]  # The N MCMC chains
        self.bufsize = args.bufsize
        self.nactions = args.ndim * 2
        self.model = None

    def parameters(self):
        return []

    def sample_many(self, mbsize, all_visited):
        r = np.float32([i[1] for i in self.batch])
        a = np.random.randint(0, self.nactions, self.bufsize)
        # step: obs(s), r, s, reverse_a
        steps = [
            self.envs[j].step(a[j], s=self.batch[j][2]) for j in range(self.bufsize)
        ]
        rp = np.float32([i[1] for i in steps])
        A = rp / r
        U = np.random.uniform(0, 1, self.bufsize)
        for j in range(self.bufsize):
            if A[j] > U[j]:  # Accept
                self.batch[j] = (None, rp[j], steps[j][2])
                all_visited.append(tuple(steps[j][2]))
        return []

    def learn_from(self, *a):
        return None


class PPOAgent:
    def __init__(self, args, envs):
        self.model = make_mlp(
            [args.horizon * args.ndim]
            + [args.n_hid] * args.n_layers
            + [args.ndim + 1 + 1]
        )  # +1 for stop action, +1 for V
        self.model.to(args.dev)
        self.envs = envs
        self.mbsize = args.mbsize
        self.clip_param = args.ppo_clip
        self.entropy_coef = args.ppo_entropy_coef

        self.horizon = args.horizon
        self.gamma = 0.99
        self.device = args.dev

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        trajs = defaultdict(list)
        while not all(done):
            # Note to self: this is ugly, ugly code as well
            with torch.no_grad():
                pol = Categorical(logits=self.model(s)[:, :-1])
                acts = pol.sample()
            step = [
                i.step(a)
                for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            log_probs = pol.log_prob(acts)
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            for si, a, (sp, r, d, _), (traj_idx, _), lp in zip(
                s, acts, step, sorted(m.items()), log_probs
            ):
                trajs[traj_idx].append(
                    [si[None, :]] + [tf([i]) for i in (a, r, sp, d, lp)]
                )
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for _, r, d, sp in step:
                if d:
                    all_visited.append(tuple(sp))
        # Compute advantages
        for tau in trajs.values():
            s, a, r, sp, d, lp = [torch.cat(i, 0) for i in zip(*tau)]
            with torch.no_grad():
                vs = self.model(s)[:, -1]
                vsp = self.model(sp)[:, -1]
            adv = r + vsp * (1 - d) - vs
            for i, A in zip(tau, adv):
                i.append(
                    r[-1].unsqueeze(0)
                )  # The return is always just the last reward, gamma is 1
                i.append(A.unsqueeze(0))
        return sum(trajs.values(), [])

    def learn_from(self, it, batch):
        idxs = np.random.randint(0, len(batch), self.mbsize)
        s, a, r, sp, d, lp, G, A = [
            torch.cat(i, 0) for i in zip(*[batch[i] for i in idxs])
        ]
        o = self.model(s)
        logits, values = o[:, :-1], o[:, -1]

        new_pol = Categorical(logits=logits)
        new_logprob = new_pol.log_prob(a)
        ratio = torch.exp(new_logprob - lp)

        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * A
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (G - values).pow(2).mean()
        entropy = new_pol.entropy().mean()
        if not it % 100:
            print(G.mean())
        return (
            action_loss + value_loss - entropy * self.entropy_coef,
            action_loss,
            value_loss,
            entropy,
        )


class RandomTrajAgent:
    def __init__(self, args, envs):
        self.mbsize = args.mbsize
        self.envs = envs
        self.nact = args.ndim + 1
        self.model = None

    def parameters(self):
        return []

    def sample_many(self, mbsize, all_visited):
        [i.reset()[0] for i in self.envs]
        done = [False] * mbsize
        while not all(done):
            acts = np.random.randint(0, self.nact, mbsize)
            step = [
                e.step(a)
                for e, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            c = count(0)
            m = {
                j: next(c) for j in range(mbsize) if not done[j]
            }  # skip the finished envs
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            for _, r, d, sp in step:
                if d:
                    # tuple(): np.array -> tuple
                    all_visited.append(tuple(sp))
        return []

    def learn_from(self, it, batch):
        return None


class SACAgent:
    def __init__(self, args, envs):
        self.pol = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1],
            tail=[nn.Softmax(1)],
        )
        self.Q_1 = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.Q_2 = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.Q_t1 = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.Q_t2 = make_mlp(
            [args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [args.ndim + 1]
        )
        self.pol.to(args.dev)
        self.Q_1.to(args.dev)
        self.Q_2.to(args.dev)
        self.Q_t1.to(args.dev)
        self.Q_t2.to(args.dev)

        self.envs = envs
        self.mbsize = args.mbsize
        self.tau = args.bootstrap_tau
        self.alpha = torch.tensor([args.sac_alpha], requires_grad=True, device=_dev[0])
        self.alpha_target = args.sac_alpha

    def parameters(self):
        return (
            list(self.pol.parameters())
            + list(self.Q_1.parameters())
            + list(self.Q_2.parameters())
            + [self.alpha]
        )

    def sample_many(self, mbsize, all_visited):
        # batch = []
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        trajs = defaultdict(list)  # dict of list of (s, a, r, s', d) tuple
        while not all(done):
            with torch.no_grad():
                pol = Categorical(probs=self.pol(s))
                acts = pol.sample()
            step = [
                e.step(a)
                for e, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            for si, a, (sp, r, d, _), (traj_idx, _) in zip(
                s, acts, step, sorted(m.items())
            ):
                trajs[traj_idx].append([si[None, :]] + [tf([i]) for i in (a, r, sp, d)])
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for _, r, d, sp in step:
                if d:
                    all_visited.append(tuple(sp))
        return sum(trajs.values(), [])  # list of (s, a, r, s', d) tuple

    def learn_from(self, it, batch):
        s, a, r, sp, d = [torch.cat(i, 0) for i in zip(*batch)]
        ar = torch.arange(s.shape[0])
        a = a.long()
        d = d.unsqueeze(1)
        q1 = self.Q_1(s)
        q1a = q1[ar, a]
        q2 = self.Q_2(s)
        q2a = q2[ar, a]
        ps = self.pol(s)
        with torch.no_grad():
            qt1 = self.Q_t1(sp)
            qt2 = self.Q_t2(sp)
            psp = self.pol(sp)
        vsp1 = ((1 - d) * psp * (qt1 - self.alpha * torch.log(psp))).sum(1)
        vsp2 = ((1 - d) * psp * (qt2 - self.alpha * torch.log(psp))).sum(1)
        J_Q = (0.5 * (q1a - r - vsp1).pow(2) + 0.5 * (q2a - r - vsp2).pow(2)).mean()
        minq = torch.min(q1, q2).detach()
        J_pi = (
            (ps * (self.alpha * torch.log(ps) - minq)).sum(1).mean()
        )  # no need for re-param
        J_alpha = (
            (ps.detach() * (-self.alpha * torch.log(ps.detach()) + self.alpha_target))
            .sum(1)
            .mean()
        )
        if not it % 100:
            print(ps[0].data, ps[-1].data, (ps * torch.log(ps)).sum(1).mean())
        for A, B in [(self.Q_1, self.Q_t1), (self.Q_2, self.Q_t2)]:
            for a, b in zip(A.parameters(), B.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)
        return J_Q + J_pi + J_alpha, J_Q, J_pi, J_alpha, self.alpha


def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == "adam":
        opt = torch.optim.Adam(
            params, args.lr, betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.opt == "msgd":
        opt = torch.optim.SGD(params, args.lr, momentum=args.momentum)
    return opt


def compute_empirical_distribution_error(env, visited):
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for i in visited:
        hist[i] += 1
    td, end_states, true_r = env.true_density()
    true_density = tf(td)
    Z = sum([hist[i] for i in end_states])
    estimated_density = tf([hist[i] / Z for i in end_states])
    k1 = abs(estimated_density - true_density).mean().item()

    # KL divergence
    kl = (
        (true_density * torch.log(true_density / estimated_density)).sum().item()
    )  # could be inf
    # many places in estimated density are zero => KL = inf
    return k1, kl


def seed_torch(seed, verbose=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))


def main(args):
    torch.set_num_threads(1)
    print("Args:", vars(args))

    if args.wandb:
        wandb.init(project="GFN-Grid", config=args, save_code=True)

    if args.augmented:
        print("doing augmentation")
        ri_eta_map = {8: 0.005, 21: args.ri, 32: 0.001, 64: 0.005, 128: 0.001}
        args.ri_eta = ri_eta_map[args.horizon]

    seed_torch(args.seed)
    args.dev = torch.device(args.device)
    set_device(args.dev)
    args.is_mcmc = args.method in ["mars", "mcmc"]

    f = get_func(args)
    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
    envs = [
        GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
        for _ in range(args.mbsize)
    ]


    # GFN methods
    if args.method in ["fm"]:
        agent = FlowNetAgent(args, envs)
    elif args.method in ["tb", "tb_gfn"]:
        agent = TBFlowNetAgent(args, envs)
    elif args.method in ["db", "db_gfn"]:
        agent = DBFlowNetAgent(args, envs)
    elif args.method in ["fm_egfn", "tb_egfn", "db_egfn"]:
        from egfn import EvolutionGFNAgent

        evo_agent = EvolutionGFNAgent(args, envs)
        agent = evo_agent.agent_star
    elif args.method in ["qm"]:
        from distributional import DistFlowNetAgentIQN

        agent = DistFlowNetAgentIQN(args, envs)

    # non-GFN methods
    elif args.method == "mars":
        agent = MARSAgent(args, envs)
    elif args.method == "mcmc":
        agent = MHAgent(args, envs)
    elif args.method == "ppo":
        agent = PPOAgent(args, envs)
    elif args.method == "sac":
        agent = SACAgent(args, envs)
    elif args.method in ["random_traj", "rand", "random"]:
        agent = RandomTrajAgent(args, envs)

    if args.method in ["tb", "tb_gfn", "tb_egfn", "ftb"]:
        opt = torch.optim.Adam(
            [
                {"params": agent.parameters(), "lr": args.tlr},
                {"params": [agent.Z], "lr": args.zlr},
            ]
        )
    elif args.method in ["db", "db_gfn", "fdb", "db_egfn"]:
        opt = torch.optim.Adam([{"params": agent.parameters(), "lr": args.tlr}])
    else:
        opt = make_opt(agent.parameters(), args)

    # metrics
    all_losses = []
    all_times = [] # useful for runtime analysis
    all_visited = []
    all_visited_eval = []
    empirical_distrib_losses = []
    error_dict = {
        "L1": {},
        "KL": {},
        "top10perc_reward": {},
        "bottom10perc_reward": {},
        "state_visited": {},
    }
    modes_set = set()
    modes_dict = {}
    replay_dict = {}
    sample_dict = {}
    last_idx = 0

    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1 / args.train_to_sample_ratio), 1)  # sample to train ratio

    if args.method == "ppo":
        ttsr = args.ppo_num_epochs
        sttr = args.ppo_epoch_size

    
    start_time = time.time()
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        if args.method in ["fm_egfn", "tb_egfn", "db_egfn"]:
            evo_agent.evolve()
        data = []
        for j in range(sttr):
            data += agent.sample_many(args.mbsize, all_visited)
        for j in range(ttsr):
            with torch.autograd.set_detect_anomaly(False):
                losses = agent.learn_from(
                    i * ttsr + j, data
                )  # returns (opt loss, *metrics)
                if losses is not None:
                    losses[0].backward()
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            agent.parameters(), args.clip_grad_norm
                        )
                    opt.step()
                    opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
                    time_elapsed = time.time() - start_time
                    all_times.append(time_elapsed)

        eval_every = 100
        if i % eval_every == 0 or i == args.n_train_steps:
            l1, kl = (0, 0)
            empirical_distrib_losses.append((l1, kl))

            if args.progress:
                recent = min(len(all_visited), args.num_empirical_loss)  # 1000
                ten_perc = int(0.1 * recent)
            
                rewards = np.sort(
                    [(50 - log2(env.func(env.s2x(np.asarray(s)))) * 10) for s in all_visited[-recent:]]
                )  # ascending
                top_reward = rewards[-ten_perc:].mean()
                bottom_reward = rewards[:ten_perc].mean()

                if args.method in ["qm"]:
                    pbar = range(eval_every)
                    for _ in pbar:
                        agent.sample_many(args.mbsize, all_visited_eval, eval=True)
                    l1, kl = compute_empirical_distribution_error(
                        env, all_visited_eval[-args.num_empirical_loss :]
                    )
                else:
                    l1, kl = empirical_distrib_losses[-1]

                print(
                    f"empirical L1 distance={l1:.4e}",
                    f"KL={kl:.4e};   "
                    f"top10%-reward={top_reward:.4f} bottom10%-reward={bottom_reward:.4f}",
                )

            if args.func == "corner":
                smode_ls = [s for s in all_visited[last_idx:] if env.s2mode(s)]
                # add smodes_ls items to the mode_set
                modes_set.update(smode_ls)
                num_modes = len(modes_set)
                modes_dict[i] = num_modes
                # replay_dict[i] = agent.replay.buf
                sample_dict[i] = data
                last_idx = len(all_visited)

            if args.progress:
                if args.wandb:
                    wandb_dict = {
                        "Error/L1": l1,
                        "Error/KL": kl,
                        f"top10% reward": top_reward,
                        f"bottom10% reward": bottom_reward,
                        "state_visited": len(all_visited),
                    }
                    if len(all_losses):
                        wandb_dict["Loss"] = all_losses[-1][0]
                    wandb.log(wandb_dict, step=i)
                    if args.func == "corner":
                        wandb.log({"Modes": num_modes}, step=i)

                if len(all_losses):
                    print(
                        "Loss:",
                        *[
                            f"{np.mean([i[j] for i in all_losses[-100:]]):.4e}"
                            for j in range(len(all_losses[0]))
                        ],
                        (f"Num_mode={num_modes}" if args.func == "corner" else ""),
                    )

            error_dict["L1"][i] = l1
            error_dict["KL"][i] = kl
            error_dict["top10perc_reward"][i] = top_reward
            error_dict["bottom10perc_reward"][i] = bottom_reward
            error_dict["state_visited"][i] = len(all_visited)

            save_dict = {
                "run_time": all_times,
                "losses": np.float32(all_losses),
                "visited": np.int8(all_visited),
                "emp_dist_loss": empirical_distrib_losses,
                "error_dict": error_dict,
                "modes_dict": modes_dict,
                "replay_dict": replay_dict,
                "sample_dict": sample_dict,
            }
            pickle.dump(save_dict, gzip.open("./result.json", "wb"))
        
    states = all_visited[-1000:]
    seqs = [get_seq(np.asarray(s)) for s in states]
    df = pd.DataFrame(
            {
                "AASeq": seqs
            }
        )
    df.to_csv("samples.csv", index=False)

    if args.wandb:
        wandb.finish()
