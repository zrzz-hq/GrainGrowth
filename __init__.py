import torch.torch_version
from . import generation as _gen
from . import utils as _utils
import hashlib
from mpi4py import MPI
import numpy as np
import torch
from typing import Sequence
from tqdm import tqdm

import torch
from typing import List, Tuple

def find_grain_num_neighbors(im: torch.Tensor, ngrains: int) -> torch.Tensor:
    """
    Compute the number of unique neighbors for each grain ID in a 2D or 3D label image.

    Args:
        im (torch.Tensor): 2D or 3D tensor of shape (H, W) or (D, H, W), each value is a grain ID.
        ngrains (int): Number of distinct grain IDs (IDs assumed 0 to ngrains-1).

    Returns:
        torch.Tensor: 1D tensor of length ngrains, where output[i] is the number of unique neighboring grains for ID i.
    """
    assert im.dim() in (2, 3), "Only supports 2D or 3D tensors"
    dims = im.dim()
    device = im.device

    # Create shift vectors for ±1 along each axis (4-connectivity in 2D, 6-connectivity in 3D)
    eye = torch.eye(dims, dtype=torch.int64, device=device)
    shifts: List[Tuple[int, ...]] = []
    for i in range(dims):
        vec = tuple(eye[i].tolist())
        shifts.append(vec)
        shifts.append(tuple((-eye[i]).tolist()))

    neighbor_pairs: List[torch.Tensor] = []

    # Generate neighbor pairs via slicing
    for shift in shifts:
        slices_src, slices_dst = [], []
        for s in shift:
            if s == -1:
                slices_src.append(slice(1, None)); slices_dst.append(slice(0, -1))
            elif s == 1:
                slices_src.append(slice(0, -1)); slices_dst.append(slice(1, None))
            else:
                slices_src.append(slice(None));   slices_dst.append(slice(None))
        id_src = im[tuple(slices_src)]
        id_dst = im[tuple(slices_dst)]
        mask = id_src != id_dst
        pairs = torch.stack([id_src[mask], id_dst[mask]], dim=1)
        pairs = torch.sort(pairs, dim=1)[0]  # sort to ensure (a, b) == (b, a)
        neighbor_pairs.append(pairs)

    # Concatenate and deduplicate neighbor pairs
    all_pairs = torch.cat(neighbor_pairs, dim=0)
    unique_pairs = torch.unique(all_pairs, dim=0)

    # Count occurrences for each grain ID
    flat_ids = unique_pairs.flatten()
    counts = torch.bincount(flat_ids, minlength=ngrains)

    return counts

    
class GrainsSeq:
    def __init__(self, **kwargs):
        self._image_list: list[torch.Tensor] = kwargs["image_list"] if "image_list" in kwargs else []
        self._euler_angle_list: list[torch.Tensor] = kwargs["euler_angle_list"] if "euler_angle_list" in kwargs else []

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
        selected_images = self._image_list[index]
        selected_euler_angles = self._euler_angle_list[index]
        if isinstance(selected_images, list):
            return GrainsSeq(image_list = selected_images,
                             euler_angle_list = selected_euler_angles)

        return Grains(image = selected_images, 
                      euler_angle = selected_euler_angles)
    
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
        return len(torch.unique(self._image_list[0])) if len(self._image_list) > 0 else 0
    
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
    
    @property
    def average_area(self) -> float:
        area = self.all_grains_area
        return torch.sum(area).item() / torch.sum(area != 0).item()

    @property
    def all_grains_sides(self):
        return find_grain_num_neighbors(self._image_list[0], self.ngrains)
    
    @property
    def average_sides(self) -> float:
        sides = self.all_grains_sides
        return torch.sum(sides).item() / torch.sum(sides != 0).item()

    
    
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

