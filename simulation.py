from . import Grains
from SPPARKS.python.Apps import Potts_AGG
from SPPARKS.python import SPPARKS, open
import numpy as np
import torch
import math
from mpi4py import MPI

import traceback

class _MCP(Potts_AGG):
    def __init__(self, spparks: SPPARKS, *args):
        super().__init__(spparks, args)

        self.__shm_comm = self._comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.__shape = None

    @property
    def shape(self) -> list[int]:
        xlo, xhi, ylo, yhi, zlo, zhi = self.box
        x = int(xhi - xlo)
        y = int(yhi - ylo)
        z = int(zhi - zlo)
        value = [x,y,z][:self.dimension]
        return tuple(value)
    
    @property
    def grains(self) -> Grains:


        # global_sites = self._comm.gather(self.local_sites)
        # global_id = self._comm.gather(self.local_id)

        local_sites = self.local_sites

        if self._rank == 0 and self.__shape == None:
            self.__shape = self.shape

        win = MPI.Win.Allocate_shared(math.prod(self.__shape) * local_sites.itemsize if self._rank == 0 else 0,
                                      local_sites.itemsize if self._rank == 0 else 0,
                                      comm=self.__shm_comm)
        win.Fence()
        buf, itemsize = win.Shared_query(0)
        shared_image = np.frombuffer(buf, dtype=np.int32)
        shared_image[self.local_id - 1] = local_sites

        win.Fence()
        if self._rank == 0:
            
            image = shared_image.copy().reshape(self.__shape)

        win.Fence()
        win.Free()
        
        if self._rank == 0:
            grains = Grains(image = torch.from_numpy(image), euler_angle = torch.from_numpy(self.euler_angle.copy()))
            return grains

        return None
    
    @grains.setter
    def grains(self, value: Grains):
        shape = self._comm.bcast(value.shape if self._rank == 0 else None)

        if self.__shape == None:
            self.__shape = shape

            dim = len(self.__shape)
            size = np.array([*self.__shape] + [1] * (3 - dim))

            self.region("box", "block", 0, size[0], 0, size[1], 0, size[2])

            self.create_box("box")
            self.create_sites("box")

        elif shape != self.__shape:
            raise RuntimeError("You provide grains map whose shape is different from the previous one. This is not possible")


        # TODO: Broadcast the image between shared groups
        if self._rank == 0:
            self.spins = value.ngrains
            flat_image = value.image.flatten().detach().cpu().numpy()
        else:
            self.spins = 0

        win = MPI.Win.Allocate_shared(flat_image.nbytes if self._rank == 0 else 0,
                                      flat_image.itemsize if self._rank == 0 else 0,
                                      comm=self.__shm_comm)
        win.Fence()
        buf, itemsize = win.Shared_query(0)
        shared_image = np.frombuffer(buf, dtype=np.int32)

        if self._rank == 0:
            shared_image[:] = flat_image

        win.Fence()

        self.local_sites = shared_image[self.local_id - 1] + 1

        win.Fence()
        win.Free()

        if self._rank == 0:
            self.euler_angle = value.euler_angle.detach().cpu().numpy()
        self._comm.barrier()


class MCPSimulator:
    def __init__(self, 
                 init_grains: Grains, 
                 lattice = "sq/8n", 
                 cutoff = 0.0, 
                 temperature = 0.66, 
                 osym = 24, 
                 rseed = 10000,
                 nsteps = None,
                 method = "normal"):
        
        self.__nsteps = nsteps
        self.__comm = MPI.COMM_WORLD
        self.__spk = open(self.__comm)
        self.__spk.__enter__()

        self.__app = _MCP(self.__spk, osym)
        self.__spk.command("diag_style", "energy")
        self.__spk.command("sweep", "random")
        self.__spk.command("sector", "yes")
        self.__spk.command("energy_scaling", 1)
        self.__spk.command("stats", 10)

        self.__app.dimension = len(init_grains.shape) if self.__comm.rank == 0 else None
        self.__app.boundary('p', 'p', 'p')
        self.__app.lattice(lattice, 1.0)
        self.__app.cutoff = cutoff
        self.__app.temperature = temperature
        self.__app.seed(rseed)
        self.__app.method = method

        self.__app.grains = init_grains if self.__comm.rank == 0 else None

    def __enter__(self):
        def grains_generator():
            i = 0
            while self.__nsteps == None or i < self.__nsteps:
                self.__app.run(1, post=False, pre=(i==0))
                i += 1
                yield self.__app.grains
        
        return grains_generator()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"[Rank {self.__comm.rank}] Exception occurred in with block: {exc_type.__name__}: {exc_val}")
            traceback.print_exception(exc_type, exc_val, exc_tb)

        self.__spk.__exit__(exc_type, exc_val, exc_tb)