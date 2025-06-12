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

    for grain_seq in dataset:
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

def load(file: str, device: torch.device = "cpu"):
    euler_angles_data = None
    images_data = None

    filepath = Path(file)
    suffix = filepath.suffix
    if suffix == '.h5':
        with h5py.File(file, 'r') as f:
            images_data = np.array(f["images"])
            euler_angles_data = np.array(f["euler_angles"])
    elif suffix == '.pickle':
        with open(file, 'rb') as f:
            data = pickle.load(file)
            images_data = data[0]
            euler_angles_data = data[1]
    else:
        raise RuntimeError(f"Unsupported file format {suffix}")
    
    images = torch.from_numpy(images_data).to(device)
    euler_angles = torch.from_numpy(euler_angles_data).to(device)

    grains_seq_list = []
    for images, euler_angles in zip(list(images), list(euler_angles)):
        grains_seq_list.append(GrainsSeq(image_list = list(images), euler_angle_list = list(euler_angles)))
    
    return GrainsDataSet(grains_seq_list)