import torch.torch_version
from . import generation as _gen
from . import utils as _utils
import hashlib
from mpi4py import MPI
import numpy as np
import torch
from typing import Sequence
    
class GrainsSeq:
    def __init__(self, **kwargs):
        self._image_list: list[torch.Tensor] = kwargs["image_list"] if "image_list" in kwargs else []
        self._euler_angle_list: list[torch.Tensor] = kwargs["euler_angle_list"] if "euler_angle_list" in kwargs else []
        self.__images = None

    @property
    def images(self) -> torch.Tensor:
        if len(self._image_list) > 0:
            return torch.stack(self._image_list)
        
        return None
    
    @property
    def euler_angles(self) -> torch.Tensor:
        if len(self._euler_angle_list) > 0:
            return torch.stack(self._euler_angle_list)
        
        return None
    
    @property
    def energies(self) -> torch.Tensor:
        images = self.images
        if images == None:
            return None
        
        unfold_image = _utils.pad_unfoldNd(images)
        return _utils.ndifferent_neighbors(unfold_image).reshape(images.shape)

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        return Grains(image = self._image_list[index], 
                      euler_angle = self._euler_angle_list[index])
    
    def __add__(self, grains: "GrainsSeq"):
        image_list = self._image_list + grains._image_list
        euler_angle_list = self._euler_angle_list + grains._euler_angle_list
        return GrainsSeq(image_list = image_list, 
                         euler_angle_list = euler_angle_list)

class Grains(GrainsSeq):
    def __init__(self, **kwargs):
        image_list = [kwargs["image"]] if "image" in kwargs else []
        euler_angle_list = [kwargs["euler_angle"]] if "euler_angle" in kwargs else []

        super().__init__(image_list = image_list, 
                         euler_angle_list = euler_angle_list)

    @property
    def id(self) -> str:
        if self.__id == None:
            self.__id = hashlib.sha256(self._image_list[0].detach().cpu().numpy().tobytes()).hexdigest()

        return self.__id

    @property
    def image(self) -> torch.Tensor:
        return self._image_list[0] if len(self._image_list) > 0 else None
    
    @property
    def euler_angle(self) -> torch.Tensor:
        return self._euler_angle_list[0] if len(self._euler_angle_list) > 0 else None
    
    @property
    def ngrains(self) -> int:
        return self._euler_angle_list[0].shape[0] if len(self._image_list) > 0 else 0
    
    @property
    def shape(self) -> tuple:
        return tuple(self._image_list[0].shape) if len(self._image_list) > 0 else ()
    
    @property
    def energy(self) -> torch.Tensor:
        return self.energies[0]

    def grain_area(self, id: int) -> int:
        return (self._image_list[0] == id).sum().item()
    
    @property
    def all_grains_area(self) -> torch.Tensor:
        return torch.bincount(self._image_list[0].flatten(), minlength=self.ngrains)
    

    
    
def from_centers(shape: Sequence[int], grain_centers: torch.Tensor, p=2, device="cpu", nchunks = 1) -> Grains:
    image = _gen.generate_grains(shape, grain_centers, p, device, nchunks)
    euler_angle = _gen.generate_random_euler_angle(grain_centers.shape[0], device)
    return Grains(image = image, euler_angle = euler_angle)

def concatenate(grains_seqs: list[GrainsSeq]):
    image_list = []
    euler_angle_list = []
    
    for grains_seq in grains_seqs:
        image_list += grains_seq._image_list
        euler_angle_list += grains_seq._euler_angle_list

    return GrainsSeq(image_list = image_list,
                     euler_angle_list = euler_angle_list)

def random(shape: Sequence[int], 
           ngrains: int, 
           p: int = 2, 
           device: torch.device = "cpu",
           nchunks = 1
           ) -> Grains:
    
    grain_centers = _gen._generate_random_grain_centers(shape, ngrains, device)
    return from_centers(shape, grain_centers, p, device, nchunks)

def circle(shape: Sequence[int], radius: float, device: torch.device = "cpu") -> Grains:
    image = _gen.generate_circle_grain(shape, radius)
    euler_angle = _gen.generate_random_euler_angle(2, device)
    return Grains(image = image, euler_angle = euler_angle)

def square(shape: Sequence[int], side_length: float, device: torch.device = "cpu") -> Grains:
    image = _gen.generate_square_grain(shape, side_length)
    euler_angle = _gen.generate_random_euler_angle(2, device)
    return Grains(image = image, euler_angle = euler_angle)

def hexagons(ngrains: int, side_length: float, device: torch.device = "cpu", nchunks = 1) -> Grains:
    grain_centers, size = _gen._generate_hex_grain_centers(ngrains, side_length)
    return from_centers(size, grain_centers, 2, device, nchunks)

