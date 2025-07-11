from . import GrainsSeq, Grains

import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import itertools

def _build_cmap(max_id, base_cmap='viridis'):
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

def _draw_2d_grains(image, cmap, norm):
    """
    Convert a [H, W] integer-label torch.Tensor to a [H, W, 3] uint8 RGB array.
    """
    rgba = cmap(norm(image))        # floats in [0..1], shape (H, W, 4)
    rgb  = (rgba * 255).astype(np.uint8)
    return rgb

def _extract_cube_faces(volume_tensor):
    """
    Given a 3D torch.Tensor of shape [D, H, W], return a list of its six
    boundary faces (each a [H, W] or [D, W] or [D, H] tensor) in order:
    [front(z=0), back(z=D-1), bottom(y=0), top(y=H-1), left(x=0), right(x=W-1)].
    """
    D, H, W = volume_tensor.shape
    return [
        volume_tensor[0,   :, :],   # front
        volume_tensor[:,  -1, :],   # top
        volume_tensor[:, :,  -1],   # right
    ]

def _draw_3d_grains(image, cmap, norm):
    """
    Given 6 face RGB arrays (uint8) in the order:
    [front, back, bottom, top, left, right],
    render them onto the 6 faces of a unit cube and return the RGB image.
    """
    faces = _extract_cube_faces(image)
    face_rgbs = [_draw_2d_grains(f, cmap, norm) for f in faces]

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
    front, top, right = face_rgbs

    _plot(front, X, np.zeros_like(X), Y)
    _plot(top, X, Y, np.ones_like(X))
    _plot(right, np.ones_like(X), Y, X)

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.axis('off')
    plt.tight_layout()

    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    plt.close(fig)
    return arr  # drop alpha

def _draw(image, cmap, norm):
    if image.ndim == 2:
        # 2D: simple label → rgb
        return _draw_2d_grains(image, cmap, norm)
    elif image.ndim == 3:
        # 3D: extract faces, convert each, then cube‐map
        return _draw_3d_grains(image, cmap, norm)
    
    return None

def draw(grains: Grains, color_map='viridis'):
    ngrains = grains.ngrains
    cmap, norm = _build_cmap(grains.ngrains - 1, color_map)
    return _draw(grains.image.cpu().numpy(), cmap, norm)

def draw_sequence(grains_seq: GrainsSeq, color_map='viridis', comm = MPI.COMM_WORLD):

    if comm.rank == 0:
        ngrains = grains_seq._euler_angle_list[0].shape[0]
        cmap, norm = _build_cmap(ngrains - 1, color_map)

        if comm.size == 1:
            for image in grains_seq._image_list:
                yield _draw(image.cpu().numpy(), cmap, norm)
        else:
            # rank 0 acts as the “client”/manager
            with MPICommExecutor(comm) as executor:
                # submit a single job
                results = executor.map(_draw, 
                                        [image.cpu().numpy() for image in grains_seq._image_list],
                                        itertools.repeat(cmap),
                                        itertools.repeat(norm))
                for r in results:
                    yield r
    else:
        # non-root ranks simply block on incoming tasks
        with MPICommExecutor(comm):
            pass

    comm.barrier()


def make_video(grains_seq: GrainsSeq, filename: str, color_map='viridis', fps: int = 10):
    """
    Build a consistent label-colormap over all frames, then render each
    frame (2D label map or 3D volume) to RGB and save as video.
    """
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        with imageio.get_writer(filename, mode='I', fps=fps) as writer:
            for frame in tqdm(draw_sequence(grains_seq, color_map, comm), total=len(grains_seq), desc="Making video"):
                writer.append_data(frame)
    else:
        for _ in draw_sequence(None, comm=comm):
            pass
    
        
