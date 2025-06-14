from . import Grains
from SPPARKS.python.Apps import Potts_AGG
from SPPARKS.python import SPPARKS
import numpy as np
import torch
import os
from mpi4py import MPI

class Simulator(Potts_AGG):
    def __init__(self, spparks: SPPARKS, *args):
        super().__init__(spparks, args)

        self.command("diag_style", "energy")
        self.command("sweep", "random")
        self.command("sector", "yes")
        self.command("energy_scaling", 1)
        self.command("stats", 10)

        self.__shape = None
    
    @property
    def grains(self) -> Grains:
        global_sites = self._comm.gather(self.local_sites)
        global_id = self._comm.gather(self.local_id)
        
        if self._rank == 0:
            image = np.concatenate(global_sites)
            id = np.concatenate(global_id)

            sort_id = np.argsort(id)
            image = image[sort_id]

            image = image.reshape(self.__shape)
            # print(len(np.unique(euler_angle, axis=0)), len(np.unique(image)))
            grains = Grains(image = torch.from_numpy(image), euler_angle = torch.from_numpy(self.euler_angle.copy()))
            return grains

        return None
    
    @grains.setter
    def grains(self, value: Grains):
        shape = self._comm.bcast(value.shape if self._rank == 0 else None)

        if self.__shape == None:
            self.__shape = shape

            bcs = np.array(['p', 'p', 'p'])
            dim = len(self.__shape)
            size = np.array([*self.__shape] + [1] * (3 - dim))
            size = size - 0.5*(bcs=='n')*(size!=1)

            self.dimension = dim
            self.command("boundary", *bcs)
            self.command("lattice", 'sq/8n' if dim==2 else 'sc/26n', 1.0)
            self.command("region", "box", "block", 0, size[0], 0, size[1], 0, size[2])

            self.create_box("box")
            self.create_sites("box")

        elif shape != self.__shape:
            raise RuntimeError("You provide grains map whose shape is different from the previous one. This is not possible")

        shm_comm = self._comm.Split_type(MPI.COMM_TYPE_SHARED)

        # TODO: Broadcast the image between shared groups
        if self._rank == 0:
            self.spins = value.ngrains
            flat_image = value.image.flatten().detach()
        else:
            self.spins = 0

        win = MPI.Win.Allocate_shared(flat_image.nbytes if self._rank == 0 else 0,
                                      flat_image.itemsize if self._rank == 0 else 0,
                                      comm=shm_comm)
        win.Fence()
        buf, itemsize = win.Shared_query(0)
        shared_image = torch.frombuffer(buf, dtype=torch.int32)

        if self._rank == 0:
            shared_image.copy_(flat_image.to('cpu'))

        win.Fence()

        self.local_sites = shared_image[self.local_id - 1] + 1

        win.Fence()
        win.Free()
        shm_comm.Free()

        self.euler_angle = self._comm.bcast(value.euler_angle.detach().cpu().numpy() if self._rank == 0 else None)