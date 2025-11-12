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

def _load_grains_seq(images_seq: np.ndarray, euler_angles_seq: np.ndarray):
    # Load images
    images_list = [torch.from_numpy(images).squeeze() for images in images_seq]

    # Load euler angles
    if euler_angles_seq is not None:
        euler_angles_list = [torch.from_numpy(euler_angles) for euler_angles in euler_angles_seq]
    else:
        euler_angles_list = []
    
    return GrainsSeq(image_list = images_list, euler_angle_list = euler_angles_list)

def _get_sets(file: h5py.File):
    if 'images' in file:
        images_set = file["images"]
    elif 'ims_id' in file:
        images_set = file['ims_id']
    else:
        raise KeyError('No image found in the dataset')

    images_set = np.array([images_seq for images_seq in tqdm(images_set, desc="Reading images")])
    print(images_set.shape)

    nsets = images_set.shape[0]
    seq_len = images_set.shape[1]
    image_shape = tuple(s for s in images_set.shape[2:] if s != 1)
    images_set = images_set.reshape((nsets, seq_len, *image_shape))

    print("=======Dataset Info========")
    print(f"Number of sets {nsets}")
    print(f"Length of sequence {seq_len}")
    print(f"Image shape {image_shape}")
    print("===========================")

    if 'euler_angles' in file:
        euler_angles_set = np.array([euler_angles_seq for euler_angles_seq in tqdm(file["euler_angles"], desc="Reading euler angles")])
        ngrains = euler_angles_set.shape[-2]
        euler_angles_set = np.broadcast_to(euler_angles_set, (nsets, seq_len, ngrains, 3))
    else:
        euler_angles_set = None
    
    return images_set, euler_angles_set

def load_grains_seq(file: h5py.File):
    images_set, euler_angles_set = _get_sets(file)
    for i, images_seq in enumerate(images_set):
        yield _load_grains_seq(images_seq, euler_angles_set[i] if euler_angles_set is not None else None)

def load(file: str):

    f = h5py.File(file, 'r')

    images_set, euler_angles_set = _get_sets(f)
    f.close()
        
    pbar = tqdm(total=len(images_set), desc="Processing data")
    grains_seq_list = []
    for i, images_seq in enumerate(images_set):
        grains_seq_list.append(_load_grains_seq(images_seq, euler_angles_set[i] if euler_angles_set is not None else None))
        pbar.update(1)

    return GrainsDataSet(grains_seq_list)

    
# def scatter_load(file: str):
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
    
#     with h5py.File(file, 'r', 'mpio', comm=comm) as f:
#         images_set = f['images']
#         euler_angles_set = f['euler_angles']

#         nsets = len(images_set)
#         quot, rem = divmod(nsets, comm.size)
#         nlocal_sets = quot + (1 if comm.rank < rem else 0)
#         disp = comm.rank * quot + min(comm.rank, rem)

#         grains_seq_list = []
#         for images_seq, euler_angles_seq in zip(images_set, euler_angles_set):
#             grains_seq_list.append(_load_grains_seq(images_seq, euler_angles_seq))
#         comm.barrier()

#         return GrainsDataSet(grains_seq_list)