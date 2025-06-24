import torch
import numpy as np
from tqdm import tqdm
from typing import Sequence
import torch.distributed as dist
from mpi4py import MPI
from threading import Thread
import math

def _generate_random_grain_centers(size: Sequence[int], ngrains: int, device: torch.device = "cpu"):
    with torch.device(device):
        size = torch.tensor(size, dtype=torch.int32)
        total_points = torch.prod(size).item()

        if ngrains > total_points:
            raise ValueError(f"ngrains ({ngrains}) exceeds total number of grid points ({total_points})")

        # Randomly sample linear indices
        flat_indices = torch.randperm(total_points)[:ngrains]

        # Convert flat indices to multi-dimensional indices
        coords = torch.empty((ngrains, len(size)), dtype=torch.int32)
        for dim in reversed(range(len(size))):
            coords[:, dim] = flat_indices % size[dim]
            flat_indices = flat_indices // size[dim]

        return coords

def generate_random_euler_angle(ngrains, device: torch.device = "cpu"):
    with torch.device(device):
        return torch.pi*torch.rand(ngrains, 3, dtype=torch.float32)*torch.tensor([2.0,0.5,2.0], dtype=torch.float32)

def generate_circle_grain(size: Sequence[int], radius: float, device: torch.device = "cpu"):
    with torch.device(device):
        c = (torch.tensor(size, dtype=torch.float32)-1)/2
        a1 = torch.arange(size[0], dtype=torch.float32).view(-1, 1)
        a2 = torch.arange(size[1], dtype=torch.float32).view(1, -1)
        image = (torch.sqrt((c[0]-a1)**2+(c[1]-a2)**2) < radius).to(dtype=torch.int32) 
    return image

def generate_square_grain(size: Sequence[int], r: float, device: torch.device = "cpu"):
    with torch.device(device):
        c = (torch.tensor(size, dtype=torch.int32)-1)/2
        a1 = torch.arange(size[0], dtype=torch.int32).view(-1, 1)
        a2 = torch.arange(size[1], dtype=torch.int32).view(1, -1)
        image = ((torch.abs(c[0]-a1) < r) & (torch.abs(c[1]-a2) <r)).to(dtype=torch.int32)
    return image

def _generate_hex_grain_centers(ngrains, side_length, device = "cpu"):
    #Generates grain centers that can be used to generate a voronoi tesselation of hexagonal grains
    #"dim" is the dimension of one side length, the other is calculated to fit the same number of grains in that direction
    #"dim_ngrain" is the number of grains along one one dimension, it is the same for both dimensions
    # mid_length = dim/ngrains #length between two flat sides of the hexagon
     #side length of hexagon
    mid_length = side_length * np.sqrt(3)
    ngrains_oneside = int(np.sqrt(ngrains))
    height = mid_length * ngrains_oneside
    width = height * np.sqrt(3) / 2
    size = (int(width), int(height)) #image size
    
    with torch.device(device):
    
        r1 = torch.arange(1.5*side_length, width, 3*side_length, dtype=torch.float32) #row coordinates of first column
        r2 = torch.arange(0, width, 3*side_length, dtype=torch.float32) #row coordinates of second column
        c1 = torch.arange(0, height, mid_length, dtype=torch.float32) #column coordinates of first row
        c2 = torch.arange(mid_length/2, height, mid_length, dtype=torch.float32) #column coordinates of second row
        
        centers1 = torch.cartesian_prod(r1, c1) #take all combinations of first row and column coordinates
        centers2 = torch.cartesian_prod(r2, c2) #take all combinations of second row and column coordinates
        grain_centers = torch.concatenate([centers1,centers2], dim=0)[torch.randperm(ngrains_oneside**2)].to(dtype=torch.int32)
    
    return grain_centers, size

# def generate_random_grains(size, ngrains):
#     grian_centers = _generate_random_grain_centers(size, ngrains)
#     image, euler_angle = generate_grains(size, grian_centers)
#     return image, euler_angle, grian_centers

# def generate_hex_grains(ngrains, size_length):
#     grain_centers, size = _generate_hex_grain_centers(ngrains, size_length)
#     image, euler_angle = generate_grains(size, grain_centers)
#     return image, euler_angle, grain_centers

from mpi4py import MPI
import numpy as np
import torch
from threading import Thread
from tqdm import tqdm
from typing import Sequence

def _generate_grains_cpu(size: Sequence[int],
                         grain_centers: torch.Tensor,
                         p: int,
                         nchunks: int):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # ─── build shifts & grid_coords on rank 0 ────────────────────────────────
    if rank == 0:
        dim = len(size)
        ngrains = grain_centers.shape[0]
        size_tensor = torch.tensor(size, dtype=torch.float32)

        shifts = torch.cartesian_prod(
            *[torch.tensor([-1,0,1],dtype=torch.float32) for _ in range(dim)]
        ).reshape(dim, -1).T              # (3^D, D)

        shifted_centers = (
            grain_centers[None] +
            shifts[:,None,:] * size_tensor
        ).reshape(-1, dim)               # (3^D·N, D)

        ref = [torch.arange(n,dtype=torch.float32) for n in size]
        grid_coords = torch.cartesian_prod(*ref)  # (∏size, D)
    else:
        grid_coords     = None
        shifted_centers = None
        ngrains         = None
        dim             = None
        size            = None

    # ─── share parameters ────────────────────────────────────────────────────
    shifted_centers = comm.bcast(shifted_centers, root=0)
    ngrains         = comm.bcast(ngrains,         root=0)
    nchunks         = comm.bcast(nchunks,         root=0)
    size            = comm.bcast(size,            root=0)
    dim             = comm.bcast(dim,             root=0)

    # ─── divide work for grid_coords ─────────────────────────────────────────
    # we’ll reuse the same divmod logic you had; start‐offset in “elements”
    npts = math.prod(size)
    quot, rem      = divmod(npts, nprocs)
    nlocal_pts  = quot + (1 if rank < rem else 0)
    pts_disp    = quot * rank + min(rank, rem)  # in “rows”

    # print(f"rank {rank} grains {nlocal_pts} disp {pts_disp}")

    # ─── window for grid_coords ─────────────────────────────────────────────
    # root exposes torch → MPI buffer
    win_gc = MPI.Win.Create(
        grid_coords if rank==0 else None,
        grid_coords.element_size() if rank==0 else 1,
        MPI.INFO_NULL, comm
    )
    win_gc.Fence()
    if rank == 0:
        grid_coords_block = grid_coords[:nlocal_pts]
    else:
        grid_coords_block = torch.empty((nlocal_pts, dim), dtype=torch.float32)

        win_gc.Get(
            [grid_coords_block, MPI.FLOAT],
            0,
            pts_disp * dim,
        )
    win_gc.Fence()
    win_gc.Free()

    # ─── build image_block locally ──────────────────────────────────────────
    grid_chunks = torch.chunk(grid_coords_block, nchunks)
    image_block = torch.empty((nlocal_pts,), dtype=torch.int32)
    offset = 0

    if rank == 0:
        def prog():
            pbar = tqdm(total=nprocs*nchunks, desc="Generating")
            done = 0
            while done < nprocs*nchunks:
                comm.recv(tag=111)
                pbar.update(1)
                done += 1
        th = Thread(target=prog)
        th.start()

    for chunk in grid_chunks:
        d = torch.cdist(shifted_centers, chunk, p=p)
        sel = torch.argmin(d, dim=0) % ngrains
        image_block[offset:offset+chunk.shape[0]] = sel
        offset += chunk.shape[0]
        comm.send(None, dest=0, tag=111)

    if rank == 0:
        th.join()

    # ─── window for the final image ──────────────────────────────────────────
    if rank == 0:
        # allocate the full buffer on root
        image = torch.empty((npts,), dtype=torch.int32)
    else:
        image = None

    win_img = MPI.Win.Create(
        image if rank==0 else None,
        image_block.element_size() if rank==0 else 1,
        MPI.INFO_NULL, comm
    )
    win_img.Fence()
    
    if rank == 0:
        image[:nlocal_pts] = image_block
    else:
        win_img.Put(
            [image_block, MPI.INT],
            0,
            pts_disp,         # rows offset
        )

    win_img.Fence()
    win_img.Free()

    # ─── only root returns the final reshaped image ─────────────────────────
    if rank == 0:
        return image.reshape(*size)
    else:
        return None


def generate_grains(size: Sequence[int], grain_centers: torch.Tensor, p: int = 2, device: torch.device = "cpu", nchunks = 1):          
    
    if device == "cpu":
        return _generate_grains_cpu(size, grain_centers, p, nchunks)
        
                
        # else:
        #     #SETUP AND EDIT LOCAL VARIABLES
        #     # print("generating..")
        #     dim = len(size)
        #     ngrains = grain_centers.shape[0]
        #     size_tensor = torch.tensor(size, dtype=torch.float32)

        #     # Generate shift grid: all combinations of [-1, 0, 1] in D dimensions
        #     shifts = torch.cartesian_prod(*[torch.tensor([-1, 0, 1], dtype=torch.float32) for _ in range(dim)])
        #     shifts = shifts.reshape(dim, -1).T  # shape (3^D, D)

        #     # Broadcast and compute shifted centers
        #     shifted_centers = grain_centers[None, :, :] + shifts[:, None, :] * size_tensor  # (3^D, N, D)
        #     shifted_centers = shifted_centers.reshape(-1, shifted_centers.shape[-1])

        #     # Create a grid of all coordinates in the domain
        #     ref = [torch.arange(n, dtype=torch.float32) for n in size]
        #     grid_coords = torch.cartesian_prod(*ref)

        #     grid_coords_chunks = torch.chunk(grid_coords, nchunks)

        #     image = torch.empty((grid_coords.shape[0],), dtype=torch.int32)
        #     offset = 0
        #     for grid_coords_chunk in tqdm(grid_coords_chunks, desc="Generating"):
        #         dist_chunk = torch.cdist(shifted_centers, grid_coords_chunk, p=p)
        #         image[offset : offset + grid_coords_chunk.shape[0]] = torch.argmin(dist_chunk, dim=0) % ngrains
        #         offset += grid_coords_chunk.shape[0]

            # image = image.reshape(*size)

        # Assign each point to its nearest center, then reduce ID modulo to map to original grains