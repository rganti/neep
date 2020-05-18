from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from misc.sampler import CartesianSampler
from toy.bead_spring import del_medium_etpy, del_shannon_etpy, simulation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class NEEP(nn.Module):
    def __init__(self, opt):
        super(NEEP, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(2*opt.n_input, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, opt.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(opt.n_hidden, 1)
        )

    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=-1)
        _x = torch.cat([s2, s1], dim=-1)
        # Equation (1) in the paper
        return self.h(x) - self.h(_x)


class Parameters(object):
    def __init__(self):
        # self.device = 'cpu'
        self.device = 'cuda:0'
        self.batch_size = 4096
        self.test_batch_size = 50000
        self.n_input = 2
        self.n_hidden = 512

        self.lr = 0.0001
        self.wd = 5e-5

        self.record_freq = 1000
        self.seed = 398

        self.M = 1000  # number of trajectories
        self.L = 10000  # length of a trajectory
        self.n_beads = self.n_input  # number of beads
        self.Tc = 1  # cold temperature
        self.Th = 10  # hot temperature
        self.time_step = 0.01  # time step size for Langevin simulation


class NeepModel(object):
    def __init__(self, parameters=None):
        if parameters:
            self.opt = parameters
        else:
            self.opt = Parameters()

        # Generate training trajectories
        self.trajs = simulation(self.opt.M, self.opt.L, self.opt.n_beads, self.opt.Tc,
                                self.opt.Th, self.opt.time_step, seed=0)
        self.trajs_t = torch.from_numpy(self.trajs).float()

        # Generate test trajectories
        self.test_trajs = simulation(self.opt.M, self.opt.L, self.opt.n_beads, self.opt.Tc,
                                     self.opt.Th, self.opt.time_step, seed=1)
        self.test_trajs_t = torch.from_numpy(self.test_trajs).float()

        self.model = NEEP(self.opt)
        self.model = self.model.to(self.opt.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.opt.lr, weight_decay=self.opt.wd)

        self.train_sampler = CartesianSampler(self.opt.M, self.opt.L, self.opt.batch_size,
                                              device=self.opt.device)
        self.test_sampler = CartesianSampler(self.opt.M, self.opt.L, self.opt.test_batch_size,
                                             device=self.opt.device, train=False)

    def train_model(self):
        self.model.train()
        batch, next_batch = next(self.train_sampler)

        s_prev = self.trajs_t[batch].to(self.opt.device)
        s_next = self.trajs_t[next_batch].to(self.opt.device)
        ent_production = self.model(s_prev, s_next)
        self.optim.zero_grad()

        # The objective function J. Equation (2)
        loss = (-ent_production + torch.exp(-ent_production)).mean()
        loss.backward()
        self.optim.step()
        return loss.item()

    def validate_model(self):
        self.model.eval()

        ret = []
        loss = 0
        with torch.no_grad():
            for batch, next_batch in self.test_sampler:
                s_prev = self.test_trajs_t[batch].to(self.opt.device)
                s_next = self.test_trajs_t[next_batch].to(self.opt.device)

                ent_production = self.model(s_prev, s_next)
                entropy = ent_production.cpu().squeeze().numpy()
                ret.append(entropy)
                loss += (- ent_production + torch.exp(-ent_production)).sum().cpu().item()
        loss = loss / self.test_sampler.size
        ret = np.concatenate(ret)
        ret = ret.reshape(self.test_trajs_t.shape[0], self.test_trajs_t.shape[1] - 1)
        return ret, loss


