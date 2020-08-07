import gsd.hoomd
import numpy as np
import torch
from tqdm import tqdm

num_particles = 49

train_traj = np.zeros((1000, 10000, num_particles*2), dtype=float)

for i in tqdm(range(0, 1000)):
    traj = gsd.hoomd.open('Active_output_p100/trajectory_{0}.gsd'.format(i), 'rb')
    full = [traj[i].particles.position[:, 0:2] for i in range(len(traj))]
    train_traj[i] = np.reshape(full, (10000, num_particles*2))

train_traj_t = torch.from_numpy(train_traj).float()
torch.save(train_traj_t, 'active_tensor_{0}.pt'.format(num_particles))