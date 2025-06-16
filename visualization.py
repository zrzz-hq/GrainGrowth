from . import GrainsSeq

import numpy as np
from tqdm import tqdm
import matplotlib
import imageio

def make_video(grains_seq: GrainsSeq, filename: str, fps: int = 10, cmap: str='viridis'):
    images = grains_seq.images  # torch.Tensor [N, H, W]

    ims_merge = []
    for img in tqdm(images, desc="Making video"):
        # Convert to numpy
        np_img = img.detach().cpu().numpy()

        # Optional: normalize to [0, 1]
        np_img = (np_img - np_img.min()) / (np_img.ptp() + 1e-8)

        # Apply colormap and convert to RGB uint8
        colored = matplotlib.colormaps[cmap](np_img)[:, :, :3]  # Drop alpha channel
        colored = (colored * 255).astype(np.uint8)

        ims_merge.append(colored)

    # Save as mp4 and gif
    imageio.mimsave(filename, ims_merge, fps=fps)
