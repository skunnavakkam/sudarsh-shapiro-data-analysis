import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import warnings
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter1d
from skimage.measure import regionprops
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from typing import TypedDict, List, Dict, Any, Tuple
import os
from scipy.signal import savgol_filter
import argparse
from mat_to_img import get_image_stack
import json

warnings.filterwarnings("ignore")


class CellStats(TypedDict):
    tau_rise_mean: float
    tau_rise_std: float
    tau_decay_mean: float
    tau_decay_std: float
    amplitude_mean: float
    amplitude_std: float
    mean_intensity: float
    std_intensity: float
    n_spikes: int
    spike_times: list[int]


def load_stack(path: str) -> np.ndarray:
    return imread(path)


def get_cells(
    stack: np.ndarray,
    percentile: float = 90,
    corr_thresh: float = 0.999,
    log_txt: bool = False,
    log_img: bool = False,
    log_path: str = None,
) -> np.ndarray:
    """
    1) Threshold on the max-projection.
    2) Build a graph: nodes = thresholded pixels;
       edges between 8-neighbors only if corr(ts_i, ts_j) > corr_thresh.
    3) Find connected components in that graph => one "cell" per component.
    4) Split components at narrow isthmuses using morphological operations
    5) Overlay each cell in a distinct color on the binary mask.

    Returns
    -------
    label_image : np.ndarray (H, W)
        0 = background; 1..N = individual detected cells.
    """
    # 1) threshold
    max_img = np.max(stack, axis=0)
    thresh_val = np.percentile(max_img, percentile)
    mask = max_img > thresh_val

    # 2) extract coords & timeseries
    ys, xs = np.nonzero(mask)
    P = len(xs)
    if P == 0:
        return np.zeros_like(mask, dtype=int)

    # Make a copy of the timeseries data and normalize it
    ts_raw = stack[:, ys, xs]  # shape (T, P)
    ts = ts_raw.copy()
    # Normalize each timeseries by subtracting mean and dividing by std
    # Smooth each timeseries with a gaussian filter along the time axis
    sigma = 1.0  # Adjust smoothing amount as needed
    for p in range(P):
        ts[:, p] = gaussian_filter1d(ts_raw[:, p], sigma)

    # build a quick lookup from (y,x) -> pixel index
    H, W = mask.shape
    index_map = -np.ones((H, W), dtype=int)
    index_map[ys, xs] = np.arange(P)

    # 3) build adjacency lists based on spatial+temporal criterion
    neighbors = {i: [] for i in range(P)}
    for i in range(P):
        y, x = ys[i], xs[i]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    j = index_map[ny, nx]
                    if j >= 0:
                        # Check for narrow connection by looking at a larger neighborhood
                        neighbor_count = 0
                        for dy2 in (-2, -1, 0, 1, 2):
                            for dx2 in (-2, -1, 0, 1, 2):
                                if dy2 == 0 and dx2 == 0:
                                    continue
                                ny2, nx2 = y + dy2, x + dx2
                                if 0 <= ny2 < H and 0 <= nx2 < W and mask[ny2, nx2]:
                                    neighbor_count += 1

                        # Only connect if there's a substantial neighborhood (not a narrow connection)
                        if (
                            neighbor_count >= 15
                        ):  # Require more filled pixels in 5x5 neighborhood
                            # Compute correlation
                            c = np.dot(ts[:, i], ts[:, j]) / (
                                np.linalg.norm(ts[:, i]) * np.linalg.norm(ts[:, j])
                            )
                            if c > corr_thresh:
                                neighbors[i].append(j)

    # 4) graph connected‐component labeling
    visited = np.zeros(P, bool)
    label_image = np.zeros((H, W), int)
    current_label = 1

    for i in range(P):
        if visited[i]:
            continue
        # flood-fill / BFS
        stack_ids = [i]
        visited[i] = True
        while stack_ids:
            u = stack_ids.pop()
            y, x = ys[u], xs[u]
            label_image[y, x] = current_label
            for v in neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    stack_ids.append(v)
        current_label += 1

    # Filter clusters based on size and eccentricity
    # Get properties of each labeled region
    regions = regionprops(label_image)

    # Filter out small or highly eccentric regions
    min_diameter = 5
    max_eccentricity = 0.95  # Adjust this threshold as needed

    valid_labels = []
    for region in regions:
        # Check if diameter is large enough (using equivalent diameter)
        if region.equivalent_diameter >= min_diameter:
            # Check if region is not too eccentric
            if region.eccentricity < max_eccentricity:
                valid_labels.append(region.label)

    # Create new label image with only valid regions
    filtered_label_image = np.zeros_like(label_image)
    for label in valid_labels:
        filtered_label_image[label_image == label] = label

    # Update the label image
    label_image = filtered_label_image

    # 5) overlay in color
    overlay = label2rgb(
        label_image,
        image=mask.astype(float),
        bg_label=0,
        alpha=0.6,
        kind="overlay",
    )

    if log_txt:
        print(len(np.unique(label_image)))

    if log_img:
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        plt.title("Spatial‐Temporal Cells Overlaid on Thresholded Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(log_path)
        plt.close()

    num_timestamps = stack.shape[0]

    # 2) Initialize the array:
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]
    cell_time_series = np.zeros((len(unique_labels), num_timestamps))

    # 3) Loop over each cell and each time‐point,
    #    pulling from `stack` not from `mask`:
    for i, label in enumerate(unique_labels):
        # boolean mask for this cell (H, W)
        cell_mask = label_image == label
        # for each frame t, average the raw intensity over that cell's pixels
        for t in range(num_timestamps):
            # stack[t] is the H×W image at time t
            cell_time_series[i, t] = stack[t][cell_mask].mean()

    return cell_time_series


def generate_time_series_plot(cell_intensities: np.ndarray, fraction=0.2):
    """Generate subplots of a subset of cell time series.

    Args:
        cell_intensities: Array of shape (n_cells, n_timepoints) containing intensity values
        fraction: Fraction of cells to plot (default 0.2)
    """
    n_cells = cell_intensities.shape[0]
    n_timepoints = cell_intensities.shape[1]

    # Calculate number of cells to plot
    n_cells_to_plot = int(n_cells * fraction)

    # Randomly select cells to plot
    cells_to_plot = np.random.choice(n_cells, n_cells_to_plot, replace=False)

    # Calculate grid dimensions - aim for roughly square layout
    grid_size = int(np.ceil(np.sqrt(n_cells_to_plot)))
    n_rows = grid_size
    n_cols = grid_size

    # Create figure with subplots
    plt.figure(figsize=(15, 15))

    # Plot selected cells' time series
    for i, cell_idx in enumerate(cells_to_plot):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(range(n_timepoints), cell_intensities[cell_idx])
        plt.title(f"Cell {cell_idx + 1}")
        plt.xticks([])  # Remove x ticks to reduce clutter

    plt.tight_layout()
    plt.show()


def _rise_func(t, A, tau, baseline):
    return baseline + A * (1 - np.exp(-t / tau))


def _decay_func(t, A, tau, baseline):
    return baseline + A * np.exp(-t / tau)


def analyze_spikes(
    cell_intensities: np.ndarray,
    path: str,
):
    """Analyze calcium spikes by separately fitting the rise (on) from onset to peak,
    and the decay (off) from peak to the next onset point or end of trace.

    Returns:
        spiking_idxs: list of cell indices with consistent peaks
        stats: list of stats dict per cell
    """
    n_cells, _ = cell_intensities.shape
    unfiltered = cell_intensities.copy()
    cell_intensities = gaussian_filter1d(cell_intensities, sigma=6, axis=1)
    os.makedirs(path, exist_ok=True)

    spiking_idxs = []
    stats = []

    for idx in range(n_cells):
        y = cell_intensities[idx]
        t = np.arange(len(y))
        # Remove baseline
        coeffs = np.polyfit(t, y, 3)
        baseline = np.polyval(coeffs, t)
        y = y - baseline
        y -= np.min(y)

        # Detect peaks
        clipped = np.clip(y, np.min(y), np.percentile(y, 95))
        noise = np.std(clipped)
        height_thresh = max(1.5 * noise, 30)
        peaks, _ = find_peaks(
            y, prominence=height_thresh, height=np.mean(y) + height_thresh, distance=20
        )

        # Uniformity check
        if len(peaks) > 2:
            widths, _, _, _ = peak_widths(y, peaks, rel_height=0.5)
            heights = y[peaks]
            if np.all(np.abs(widths - widths.mean()) / widths.mean() <= 0.3) and np.all(
                np.abs(heights - heights.mean()) / heights.mean() <= 0.3
            ):
                spiking_idxs.append(idx)

        # Find onset points for each peak
        onset_pts = []
        if len(peaks) > 2:
            for p in peaks:
                start = max(0, p - 70)
                seg = y[start : p + 1]
                onset_pts.append(start + np.argmin(seg))

        # Fit kinetics: rise and decay
        rise_taus, decay_taus, amps = [], [], []
        rise_fits, decay_fits = [], []
        for i, p in enumerate(peaks):
            # Rise fit
            if onset_pts and i < len(onset_pts):
                o = onset_pts[i]
                t_r = np.arange(p - o)
                y_r = y[o:p]
                if len(t_r) > 2:
                    w = np.linspace(0.3, 1.0, len(t_r)) ** 2
                    try:
                        amp0 = y_r[-1] - y_r[0]
                        popt_r, _ = curve_fit(
                            _rise_func,
                            t_r,
                            y_r,
                            p0=[amp0, len(t_r) / 3, y_r[0]],
                            bounds=(
                                [amp0 * 0.8, 1, y_r[0] - 2],
                                [amp0 * 1.2, len(t_r) * 2, y_r[0] + 2],
                            ),
                            sigma=1 / w,
                            absolute_sigma=False,
                        )
                        rise_taus.append(popt_r[1])
                        amps.append(popt_r[0])
                        rise_fits.append((o, p, _rise_func(t_r, *popt_r)))
                    except:
                        rise_fits.append(None)
                else:
                    rise_fits.append(None)
            else:
                rise_fits.append(None)

            # Decay fit: until next onset or end of trace
            if onset_pts and i < len(peaks):
                # define end of decay segment
                end_pt = onset_pts[i + 1] if (i + 1 < len(onset_pts)) else len(y)
                max_len = end_pt - p
                if max_len > 2:
                    t_d = np.arange(max_len)
                    y_d = y[p : p + max_len]
                    try:
                        popt_d, _ = curve_fit(
                            _decay_func,
                            t_d,
                            y_d,
                            p0=[y_d[0] - y_d[-1], max_len / 3, y_d[-1]],
                            bounds=(
                                [0, 0.5, -np.inf],
                                [(y_d[0] - y_d[-1]) * 3, max_len * 2, np.inf],
                            ),
                        )
                        decay_taus.append(popt_d[1])
                        decay_fits.append(
                            (p, p + max_len, _decay_func(np.arange(max_len), *popt_d))
                        )
                    except:
                        decay_fits.append(None)
                else:
                    decay_fits.append(None)

        # Compile stats
        if rise_taus and decay_taus and amps:
            # Remove outliers using IQR method
            rise_q1, rise_q3 = np.percentile(rise_taus, [25, 75])
            rise_iqr = rise_q3 - rise_q1
            rise_mask = (rise_taus >= rise_q1 - 1.5 * rise_iqr) & (
                rise_taus <= rise_q3 + 1.5 * rise_iqr
            )
            filtered_rise_taus = np.array(rise_taus)[rise_mask]

            decay_q1, decay_q3 = np.percentile(decay_taus, [25, 75])
            decay_iqr = decay_q3 - decay_q1
            decay_mask = (decay_taus >= decay_q1 - 1.5 * decay_iqr) & (
                decay_taus <= decay_q3 + 1.5 * decay_iqr
            )
            filtered_decay_taus = np.array(decay_taus)[decay_mask]

            stats.append(
                {
                    "tau_rise_mean": float(np.mean(filtered_rise_taus)),
                    "tau_rise_std": float(np.std(filtered_rise_taus))
                    if len(filtered_rise_taus) > 1
                    else 0.0,
                    "tau_decay_mean": float(np.mean(filtered_decay_taus)),
                    "tau_decay_std": float(np.std(filtered_decay_taus))
                    if len(filtered_decay_taus) > 1
                    else 0.0,
                    "amplitude_mean": float(np.mean(amps)),
                    "amplitude_std": float(np.std(amps)) if len(amps) > 1 else 0.0,
                    "n_spikes": len(peaks),
                    "spike_times": peaks.tolist(),
                    "mean_intensity": float(np.mean(unfiltered[idx])),
                    "std_intensity": float(np.std(unfiltered[idx])),
                }
            )
        else:
            stats.append(
                {
                    **{
                        k: None
                        for k in [
                            "tau_rise_mean",
                            "tau_rise_std",
                            "tau_decay_mean",
                            "tau_decay_std",
                            "amplitude_mean",
                            "amplitude_std",
                        ]
                    },
                    **{
                        "n_spikes": None,
                        "spike_times": None,
                        "mean_intensity": None,
                        "std_intensity": None,
                    },
                }
            )

        # Plot
        plt.figure()
        plt.plot(y, label="Raw Data", alpha=0.7)
        if len(peaks) > 0:
            plt.plot(peaks, y[peaks], "rx", label="Peaks")
        if onset_pts:
            onset_arr = np.array(onset_pts)
            plt.plot(onset_arr, y[onset_arr], "go", label="Onset Points")
        # rise fits
        for fit in rise_fits:
            if fit is not None:
                o, p, f = fit
                plt.plot(np.arange(o, p), f, linewidth=2, alpha=0.8)
        # decay fits
        for fit in decay_fits:
            if fit is not None:
                p, end, f = fit
                plt.plot(np.arange(p, end), f, linewidth=2, alpha=0.8)

        plt.text(
            0.5,
            0.95,
            f"Height thresh: {height_thresh:.2f}",
            transform=plt.gca().transAxes,
        )
        plt.legend()
        plt.title(f"Cell {idx} - Spike Analysis")
        plt.xlabel("Time (frames)")
        plt.ylabel("Intensity (baseline corrected)")
        plt.savefig(os.path.join(path, f"cell_{idx}_analysis.png"), dpi=150)
        plt.close()

    return spiking_idxs, stats


def make_boxplot(
    rise_times: list[float], decay_times: list[float], title: str, path: str
):
    """Generate violin plots with overlaid data points for rise and decay times.

    Args:
        rise_times: List of float values for rise times
        decay_times: List of float values for decay times
        path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))

    # Create violin plots to show distributions
    plt.violinplot([rise_times, decay_times], showmeans=True)

    # Add individual points with slight horizontal jitter
    x_jitter1 = np.random.normal(1, 0.05, size=len(rise_times))
    x_jitter2 = np.random.normal(2, 0.05, size=len(decay_times))
    plt.scatter(x_jitter1, rise_times, alpha=0.4, c="black", s=20)
    plt.scatter(x_jitter2, decay_times, alpha=0.4, c="black", s=20)

    plt.title(title)
    plt.ylabel("Time (frames)")
    plt.xticks([1, 2], ["Rise Times", "Decay Times"])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    CACHE_DIR = "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the image stack, either a single tif or a folder of .mat files",
    )
    args = parser.parse_args()

    # Create results directory with filename
    results_dir = os.path.join("results", os.path.basename(args.path).replace(".", "+"))
    os.makedirs(results_dir, exist_ok=True)

    # Check if cached version exists
    cache_path = os.path.join(
        CACHE_DIR, os.path.basename(args.path).replace(".", "+") + ".npy"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        cell_intensities = np.load(cache_path)

    else:
        if os.path.isdir(args.path):
            image_stack = get_image_stack(args.path)
            cell_intensities = get_cells(
                image_stack,
                log_img=True,
                log_path=os.path.join(results_dir, "cells.png"),
            )
            cell_intensities = gaussian_filter1d(cell_intensities, sigma=2, axis=1)
            np.save(cache_path, cell_intensities)
        else:
            image_stack = load_stack(args.path)
            cell_intensities = get_cells(
                image_stack,
                log_img=True,
                log_path=os.path.join(results_dir, "cells.png"),
            )
            cell_intensities = gaussian_filter1d(cell_intensities, sigma=2, axis=1)
            np.save(cache_path, cell_intensities)

    spiking_idxs, stats = analyze_spikes(
        cell_intensities,
        path=os.path.join(results_dir, "spikes"),
    )

    cell_rises = [
        cell_stats["tau_rise_mean"]
        for cell_stats in stats
        if cell_stats["tau_rise_mean"] is not None
    ]

    cell_decays = [
        cell_stats["tau_decay_mean"]
        for cell_stats in stats
        if cell_stats["tau_decay_mean"] is not None
    ]

    try:
        make_boxplot(
            cell_rises,
            cell_decays,
            f"Rise and Decay Times for {os.path.basename(args.path.replace('.tif', ''))}",
            os.path.join(results_dir, "rise_decay_times.png"),
        )
    except Exception as e:
        print(f"Error making boxplot: {e}")

    with open(os.path.join(results_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
