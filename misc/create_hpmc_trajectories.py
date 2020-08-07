import hoomd.hpmc
from tqdm import tqdm

# Initialize hoomd
hoomd.context.initialize("--mode=cpu")

# Setup simulation
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=1.05), n=10)
mc = hoomd.hpmc.integrate.sphere(d=0.2, seed=1)
mc.shape_param.set('A', diameter=1.0)

d = hoomd.dump.gsd("trajectory_hpmc.gsd", period=10, group=hoomd.group.all(), overwrite=True)
hoomd.run(100)

d.disable()
hoomd.run(10000)

for i in tqdm(range(0, 1000)):
    hoomd.run(1000, quiet=True)

    d = hoomd.dump.gsd("Hard_Disk_MC_output/trajectory_{0}.gsd".format(i), period=1,
                       group=hoomd.group.all(), overwrite=True)
    hoomd.run(10000, quiet=True)

    d.disable()

