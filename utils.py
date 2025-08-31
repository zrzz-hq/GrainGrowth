import torch
from unfoldNd import unfoldNd
import torch.nn.functional as f
from typing import Sequence

def pad_unfoldNd(images: torch.Tensor, kernel_size=3, pad_mode='circular') -> torch.Tensor:
    #Pads "ims" before unfolding
    dims = images.dim() - 1

    if not isinstance(kernel_size, list): 
        kernel_size = [kernel_size] * dims #convert to "list" if it isn't

    pad = tuple(k // 2 for k in kernel_size for _ in range(2)) #calculate padding needed based on kernel_size

    if pad_mode == "circular":
        padded_images = f.pad(images.to(dtype=torch.float32), pad, pad_mode) #if "pad_mode"!=list, pad dimensions simultaneously
    elif pad_mode == 'constant':
        padded_images = f.pad(images.to(dtype=torch.float32), pad, pad_mode, value=-1) 
    

    unfold_images = unfoldNd(padded_images.unsqueeze(1), kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]

    return unfold_images.to(dtype=images.dtype)

def ndifferent_neighbors(unfold_images: torch.Tensor, center_pixels = None) -> torch.Tensor:  
    #ims_unfold - torch tensor of shape = [N, product(kernel_size), dim1*dim2] from [N, 1, dim1, dim2] using "torch.nn.Unfold" object
    #Addtiional dimensions to ims_unfold could be included at the end
    center_index = unfold_images.shape[1] // 2
    if center_pixels is None:
        center_pixels = unfold_images[:, center_index:center_index+1, :]
    
    diff_mask = (unfold_images != center_pixels) & (center_pixels != -1)
    diff_mask[:, center_index, :] = False
    # Count how many are different (along kernel dimension)
    ndiff_neighbors = diff_mask.sum(dim=1)  # [N, L]
    return ndiff_neighbors 
    
