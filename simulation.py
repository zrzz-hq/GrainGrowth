from . import Grains
from SPPARKS.python.Apps import Potts_AGG
from SPPARKS.python import SPPARKS
import numpy as np
import torch
import os

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

            dim = len(self.__shape)
            size = np.array([*self.__shape] + [1] * (3 - dim))

            self.command("region", "box", "block", 0, size[0], 0, size[1], 0, size[2])

            self.create_box("box")
            self.create_sites("box")

        elif shape != self.__shape:
            raise RuntimeError("You provide grains map whose shape is different from the previous one. This is not possible")

        global_id = self._comm.gather(self.local_id)
        if self._rank == 0:
            self.spins = value.ngrains
            for i, local_id in enumerate(global_id):
                image_chunk = value.image.flatten()[local_id-1].detach().cpu().numpy()
                if i == 0:
                    local_image = image_chunk
                else:
                    self._comm.send(image_chunk, dest=i, tag=77)
        else:
            self.spins = None
            local_image = self._comm.recv(source=0, tag=77)

        self.euler_angle = self._comm.bcast(value.euler_angle.detach().cpu().numpy() if self._rank == 0 else None)

        self.local_sites = local_image + 1
        self._comm.barrier()