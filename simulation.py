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

    # def __generate_initial_condition(self, image, euler_angle, filename):
    #     '''
    #     Takes an image of grain IDs (and euler angles assigned to each ID) and writes it to an init file for a SPPARKS simulation
    #     The initial condition file is written to the 2D or 3D file based on the dimension of 'img'
        
    #     Inputs:
    #         img (numpy, integers): pixels indicate the grain ID of the grain it belongs to
    #         euler_angke (numpy): number of grains by three Euler angles
    #     '''

    #     # Set local variables
    #     size = image.shape
    #     dim = len(image.shape)
    #     with open(filename, 'w') as file:
    #         file.write(" # This line is ignored\nValues\n\n")
    #         # Write the information in the SPPARKS format and save the file
            
    #         k=0
        
    #         if dim==3: 
    #             for i in range(0,size[2]):
    #                 for j in range(0,size[1]):
    #                     for h in range(0,size[0]):
    #                         site_id = int(image[h,j,i])
    #                         file.write(f"{k+1} {site_id+1} {euler_angle[i,j,h,0]} {euler_angle[i,j,h,1]} {euler_angle[i,j,h,2]}\n")
    #                         k = k + 1
            
    #         else:
    #             for i in range(0,size[1]):
    #                 for j in range(0,size[0]):
    #                     site_id = int(image[j,i])
    #                     file.write(f"{k+1} {site_id+1} {euler_angle[i,j,0]} {euler_angle[i,j,1]} {euler_angle[i,j,2]}\n")
    #                     k = k + 1
    #         os.fsync(file.fileno())
    
    @property
    def grains(self) -> Grains:
        global_sites = self._comm.gather(self.local_sites)
        global_darray = self._comm.gather(self.local_darray)
        global_id = self._comm.gather(self.local_id)
        
        if self._rank == 0:
            image = np.concatenate(global_sites)
            euler_angle = np.hstack(global_darray).transpose()
            id = np.concatenate(global_id)

            sort_id = np.argsort(id)
            image = image[sort_id]
            euler_angle = euler_angle[sort_id]

            image = image.reshape(self.__shape)
            euler_angle = euler_angle.reshape((*self.__shape, 3))
            grains = Grains(image = torch.from_numpy(image), euler_angle = torch.from_numpy(euler_angle))
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

        global_id = self._comm.gather(self.local_id)
        if self._rank == 0:
            self.spins = value.ngrains
            for i, local_id in enumerate(global_id):
                image_chunk = value.image.flatten()[local_id-1].detach().cpu().numpy()
                euler_angle_chunk = value.euler_angle.reshape(-1, 3)[local_id-1].detach().cpu().numpy()
                if i == 0:
                    local_image = image_chunk
                    local_euler_angle = euler_angle_chunk
                else:
                    self._comm.send(image_chunk, dest=i, tag=77)
                    self._comm.send(euler_angle_chunk, dest=i, tag=78)
        else:
            self.spins = None
            local_image = self._comm.recv(source=0, tag=77)
            local_euler_angle = self._comm.recv(source=0, tag=78)

        self.local_sites = local_image + 1
        self.local_darray = local_euler_angle.transpose()
        self._comm.barrier()