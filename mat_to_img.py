import os
import re
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import tqdm


def get_image_stack(folder_path, length=700, step=0.01):
    """
    Load .mat files from the specified folder, interpolate each frame onto a regular grid,
    and return a 3D NumPy array of shape (n_frames, nx, ny). Frames that fail to load
    will be filled with NaNs.

    Filenames are sorted numerically by their final digit sequence before ".mat"
    to ensure correct frame order even without leading zeros.

    Parameters:
        folder_path (str): Path to the folder containing .mat files.
        length (int): Number of rows to keep in Y dimension (default 700).
        step (float): Grid spacing for interpolation (default 0.01).

    Returns:
        np.ndarray: 3D array with shape (n_frames, nx, ny) of interpolated frames.
    """
    # List valid .mat files
    all_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".mat") and not f.startswith("._")
    ]
    if not all_files:
        raise FileNotFoundError(f"No .mat files found in '{folder_path}'")

    # Sort files by numeric index extracted from filename
    def extract_index(fname):
        """Return integer index of the final digit group before .mat"""
        nums = re.findall(r"(\d+)(?=\.mat$)", fname)
        return int(nums[-1]) if nums else 0

    mat_files = sorted(all_files, key=extract_index)

    # Load first frame for grid definition
    first_path = os.path.join(folder_path, mat_files[0])
    try:
        first = loadmat(first_path)["ImgData"][0, 0]
    except Exception as e:
        raise RuntimeError(f"Failed to load initial .mat file '{mat_files[0]}': {e}")

    # Extract coordinate vectors
    x_full = np.squeeze(first["x"]).flatten()
    z_full = np.squeeze(first["z"]).flatten()[:length]

    # Build regular interpolation grid
    xi = np.arange(x_full.min(), x_full.max(), step)
    yi = np.arange(z_full.min(), z_full.max(), step)
    grid_x, grid_y = np.meshgrid(xi, yi, indexing="xy")  # (ny, nx)
    nx, ny = xi.size, yi.size

    # Prepare output stack
    n_frames = len(mat_files)
    stack = np.full((n_frames, nx, ny), np.nan, dtype=np.float32)

    # Process each file in numeric order
    for i, fname in tqdm.tqdm(enumerate(mat_files), total=len(mat_files)):
        path = os.path.join(folder_path, fname)
        try:
            data = loadmat(path)["ImgData"][0, 0]
            Im = data["Im"][:length, :]
            x = np.squeeze(data["x"]).flatten()
            z = np.squeeze(data["z"]).flatten()[:length]
        except Exception as e:
            tqdm.tqdm.write(f"Warning: could not read '{fname}': {e}")
            continue

        # Interpolate using regular grid
        interp = RegularGridInterpolator(
            (z, x), Im, method="linear", bounds_error=False, fill_value=np.nan
        )
        pts = np.vstack((grid_y.ravel(), grid_x.ravel())).T
        img = interp(pts).reshape(yi.size, xi.size)

        # Store transposed slice
        stack[i] = img.T

    return stack
