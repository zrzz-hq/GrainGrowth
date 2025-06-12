from . import GrainsSeq

import cv2
import numpy as np
from pathlib import Path
from typing import Sequence, Union
from tqdm import tqdm

import matplotlib.pyplot as plt

def make_video(
    grains_seq: GrainsSeq,
    output_path: Union[str, Path],
    fps: int = 10,
    codec: str = 'mp4v',
    cmap = 'viridis'
):
    # Load first frame to determine resolution
    first = grains_seq[0].image
    height, width = first.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter.fourcc(*codec)
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for image in grains_seq.images:
        fig, ax = plt.subplots()
        cax = ax.matshow(image, cmap=cmap)
        fig.canvas.draw()
        
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        video.write(frame)

        plt.close(fig)

    video.release()
    print(f"Video saved to {output_path}")
