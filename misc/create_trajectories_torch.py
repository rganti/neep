import pickle

import hoomd
import hoomd.md
from tqdm import tqdm
import gsd.hoomd
import numpy as np
import torch
import datetime
import os
import argparse
import subprocess


class SlurmHeader(object):
    def __init__(self, simulation_name, simulation_time=1, nodes=1, ppn=1):
        self.simulation_name = simulation_name
        self.simulation_time = simulation_time
        self.nodes = nodes
        self.ppn = ppn

    def set_header(self, q):
        q.write("#!/bin/bash\n")
        q.write("#SBATCH --job-name={0}\n".format(self.simulation_name))
        q.write("#SBATCH --nodes {0}\n".format(self.nodes))
        q.write("#SBATCH --ntasks-per-node {0}\n".format(self.ppn))
        q.write("#SBATCH --time={0}\n".format(datetime.timedelta(hours=self.simulation_time)))
        q.write("#SBATCH --partition=sched_mit_arupc\n")
        # q.write("#SBATCH --mem-per-cpu=10gb\n\n")


class SlurmSimulation(object):
    def __init__(self, active=False, simulation_time=5, nodes=1, ppn=20):
        self.active = active
        if self.active:
            self.simulation_name = "active_traj"
        else:
            self.simulation_name = "passive_traj"

        self.header = SlurmHeader(simulation_name=self.simulation_name, simulation_time=simulation_time, nodes=nodes,
                                      ppn=ppn)
        self.script_name = "create_trajectories_torch.py"

    def set_python_script(self, q):
        if self.active:
            q.write("python {0}/{1} --run --active \n".format(os.path.dirname(__file__), self.script_name))
        else:
            q.write("python {0}/{1} --run \n".format(os.path.dirname(__file__), self.script_name))

    def generate_sbatch(self):
        q = open("sbatch.sh", "w")
        self.header.set_header(q)
        self.set_python_script(q)
        q.close()


def pickle_out_data(data, pickle_name):
    pickle_out = open(pickle_name + ".pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_reshape(particle_type, N):

    train_traj = np.zeros((1000, 10000, N * 2), dtype=float)

    for i in tqdm(range(0, 1000)):
        traj = gsd.hoomd.open("{0}_output_n{1}/trajectory_{2}.gsd".format(particle_type, N, i), 'rb')
        full = [traj[i].particles.position[:, 0:2] for i in range(len(traj))]
        train_traj[i] = np.reshape(full, (10000, N * 2))

    # pickle_out_data(train_traj, "{0}_tensor_{1}".format(type, N))

    train_traj_t = torch.from_numpy(train_traj).float()
    torch.save(train_traj_t, '{0}_tensor_{1}.pt'.format(particle_type, N))


def run_simulation(lattice_size=7, active=False):
    hoomd.context.initialize("");
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=1.4), n=lattice_size);
    all_particles = hoomd.group.all();
    N = len(all_particles);

    nl = hoomd.md.nlist.cell();

    lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl);
    lj.set_params(mode='shift');

    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);

    hoomd.md.integrate.mode_standard(dt=0.001);

    if active:
        activity = [(((np.random.rand(1)[0] - 0.5) * 2.0), ((np.random.rand(1)[0] - 0.5) * 2.0), 0)
                    for i in range(N)];

        hoomd.md.force.active(group=all_particles,
                              seed=123,
                              f_lst=activity,
                              rotation_diff=0.005,
                              orientation_link=False);

        hoomd.md.integrate.brownian(group=all_particles, kT=0.0, seed=123);
        particle_type = "active"

    else:
        hoomd.md.integrate.brownian(group=all_particles, kT=0.2, seed=123);
        particle_type = "passive"

    d = hoomd.dump.gsd("trajectory_{0}_{1}.gsd".format(particle_type, N), period=100,
                       group=hoomd.group.all(), overwrite=True);
    hoomd.run(2000);

    d.disable();

    make_dir("{0}_output_n{1}".format(particle_type, N))

    for i in tqdm(range(0, 1000)):
        hoomd.run(1000, quiet=True);

        d = hoomd.dump.gsd("{0}_output_n{1}/trajectory_{2}.gsd".format(particle_type, N, i), period=10,
                           group=hoomd.group.all(), overwrite=True);
        hoomd.run(100000, quiet=True);

        d.disable();

    load_reshape(particle_type, N)


def make_dir(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print("Made " + directory_name)


def run_sbatch():
    (stdout, stderr) = subprocess.Popen(["sbatch {0}".format("sbatch.sh")], shell=True, stdout=subprocess.PIPE,
                                        cwd=os.getcwd()).communicate()
    f = open("error_log", "w")
    f.write("stdout = " + str(stdout) + "\n")
    f.write("stderr = " + str(stderr) + "\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submitting active or passive calculations.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--active', dest='active', action='store_true', default=False,
                        help="Flag for setting particle type in simulations.")
    parser.add_argument('--run', dest='run', action='store_true', default=False,
                        help="Flag for running simulations.")
    args = parser.parse_args()

    if args.run:
        run_simulation(lattice_size=10, active=args.active)

    else:
        sbatch = SlurmSimulation(active=args.active)
        sbatch.generate_sbatch()
        run_sbatch()







