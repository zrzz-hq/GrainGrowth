from . import Grains, GrainsSeq

from pathlib import Path
import h5py
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class GrainsDataSet(Dataset):
    def __init__(self, grains_seq_list: list[GrainsSeq] = []):
        
        self._grains_seq_list = grains_seq_list

    def __len__(self) -> int:
        return len(self._grains_seq_list)
    
    def __getitem__(self, index) -> GrainsSeq:
        return self._grains_seq_list[index]
    
    def append(self, grains_seq: GrainsSeq):
        self._grains_seq_list.append(grains_seq)

def save(file: str, dataset: GrainsDataSet):
    filepath = Path(file)
    images_list = []
    euler_angles_list = []

    for grain_seq in tqdm(dataset, desc="Saving dataset"):
        images_list.append(grain_seq.images)
        euler_angles_list.append(grain_seq.euler_angles)

    suffix = filepath.suffix
    if suffix == '.h5':
        with h5py.File(file, 'w') as f:
            f.create_dataset("images", data=np.array(images_list))
            f.create_dataset("euler_angles", data=np.array(euler_angles_list))
    elif suffix == '.pickle':
        with open(file, 'wb') as f:
            pickle.dump([images_list, euler_angles_list], file)
    else:
        raise RuntimeError(f"Unsupported file format {suffix}")

def _load_grains_seq(images_set: h5py.Dataset, euler_angles_set: h5py.Dataset, index: slice):
    for i in range(len(images_set))[index]:
        images = torch.tensor(images_set[i])
        euler_angles = torch.tensor(euler_angles_set[i])
        yield GrainsSeq(image_list = list(images), euler_angle_list = list(euler_angles))

def load_grains_seq(file: h5py.File):
    images_set = file["images"]
    euler_angles_set = file["euler_angles"]
    
    return _load_grains_seq(images_set, euler_angles_set, slice(None, None, None))

def load(file: str):

    with h5py.File(file, 'r') as f:
        images_set = f["images"]
        euler_angles_set = f["euler_angles"]

        pbar = tqdm(total=len(images_set), desc="Loading")
        grains_seq_list = []
        for grains_seq in _load_grains_seq(images_set, euler_angles_set, slice(None, None, None)):
            grains_seq_list.append(grains_seq)
            pbar.update(1)

        return GrainsDataSet(grains_seq_list)

    
def scatter_load(file: str):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    
    with h5py.File(file, 'r', 'mpio', comm=comm) as f:
        images_set = f['images']
        euler_angles_set = f['euler_angles']

        nsets = len(images_set)
        quot, rem = divmod(nsets, comm.size)
        nlocal_sets = quot + (1 if comm.rank < rem else 0)
        disp = comm.rank * quot + min(comm.rank, rem)

        grains_seq_list = []
        for grains_seq in _load_grains_seq(images_set, euler_angles_set, slice(disp, disp + nlocal_sets, None)):
            grains_seq_list.append(grains_seq)
        comm.barrier()

        return GrainsDataSet(grains_seq_list)