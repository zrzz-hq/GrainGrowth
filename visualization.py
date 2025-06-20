from . import GrainsSeq

import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from threading import Thread

from mpi4py import MPI
import torch
from tqdm import tqdm
import itertools

def build_label_cmap(max_id, base_cmap='viridis'):
    """
    Create a discrete colormap & norm that maps integer labels [0..max_id]
    to distinct colors.
    """
    cmap = plt.get_cmap(base_cmap, max_id + 1)
    norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(-0.5, max_id + 1.5, 1.0),
        ncolors=max_id + 1
    )
    return cmap, norm

def grains_frame_to_rgb_labels(img_tensor, cmap, norm):
    """
    Convert a [H, W] integer-label torch.Tensor to a [H, W, 3] uint8 RGB array.
    """
    labels = img_tensor
    rgba = cmap(norm(labels))        # floats in [0..1], shape (H, W, 4)
    rgb  = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

def extract_cube_faces(volume_tensor):
    """
    Given a 3D torch.Tensor of shape [D, H, W], return a list of its six
    boundary faces (each a [H, W] or [D, W] or [D, H] tensor) in order:
    [front(z=0), back(z=D-1), bottom(y=0), top(y=H-1), left(x=0), right(x=W-1)].
    """
    D, H, W = volume_tensor.shape
    return [
        volume_tensor[0,   :, :],   # front
        volume_tensor[-1,  :, :],   # back
        volume_tensor[:,   0, :],   # bottom
        volume_tensor[:,  -1, :],   # top
        volume_tensor[:, :,   0],   # left
        volume_tensor[:, :,  -1],   # right
    ]

def draw_textured_cube_to_array_face_rgbs(face_rgbs):
    """
    Given 6 face RGB arrays (uint8) in the order:
    [front, back, bottom, top, left, right],
    render them onto the 6 faces of a unit cube and return the RGB image.
    """
    # assume each face_rgbs[i] has shape (H, W, 3)
    H, W, _ = face_rgbs[0].shape
    u = np.linspace(0, 1, W)
    v = np.linspace(0, 1, H)
    X, Y = np.meshgrid(u, v)

    fig = plt.figure(figsize=(4,4))
    ax  = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvas(fig)

    def _plot(face_img, xs, ys, zs):
        ax.plot_surface(
            xs, ys, zs,
            rstride=1, cstride=1,
            facecolors=face_img/255,
            shade=False
        )

    # unpack in order
    front, back, bottom, top, left, right = face_rgbs

    # front/back
    _plot(front,  X,       Y,       np.zeros_like(X))
    _plot(back,   X,       Y,       np.ones_like(X))
    # bottom/top
    _plot(bottom, X,       np.zeros_like(X), Y)
    _plot(top,    X,       np.ones_like(X),  Y)
    # left/right
    _plot(left,   np.zeros_like(X), X, Y)
    _plot(right,  np.ones_like(X),  X, Y)

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.axis('off')
    plt.tight_layout()

    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    plt.close(fig)
    return arr[..., :3]  # drop alpha

def make_video(grains_seq: GrainsSeq, filename: str, base_cmap='viridis', fps: int = 10):
    """
    Build a consistent label-colormap over all frames, then render each
    frame (2D label map or 3D volume) to RGB and save as video.
    """

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        # 1) find global max label
        ngrains = grains_seq._euler_angle_list[0].shape[0]
        # 2) build shared cmap & norm
        cmap, norm = build_label_cmap(ngrains - 1, base_cmap)
        images = torch.stack(grains_seq._image_list).cpu().numpy()
        images_chunks = np.array_split(images, comm.size)
    else:
        cmap = norm = None
        images_chunks = None

    cmap = comm.bcast(cmap)
    norm = comm.bcast(norm) 
    local_images = comm.scatter(images_chunks)

    if comm.rank == 0:
        def update_progress():
            pbar = tqdm(total=len(grains_seq), desc="Making video")
            total = 0
            while total < len(grains_seq):
                pbar.update(comm.recv(tag=112))
                total += 1

        thread = Thread(target=update_progress)
        thread.start()

    local_frames = []
    for img in local_images:
        if img.ndim == 2:
            # 2D: simple label → rgb
            rgb = grains_frame_to_rgb_labels(img, cmap, norm)
            local_frames.append(rgb)
        elif img.ndim == 3:
            # 3D: extract faces, convert each, then cube‐map
            faces = extract_cube_faces(img)
            face_rgbs = [grains_frame_to_rgb_labels(f, cmap, norm) for f in faces]
            cube_img = draw_textured_cube_to_array_face_rgbs(face_rgbs)
            local_frames.append(cube_img)
        else:
            raise ValueError(f"Unsupported ndim={img.ndim}")
        
        comm.send(1,0,112)
        
    frames = comm.gather(local_frames)

    if comm.rank == 0:   
        imageio.mimsave(filename, list(itertools.chain.from_iterable(frames)), fps=fps)
