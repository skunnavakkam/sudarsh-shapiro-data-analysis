import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import warnings
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter1d
from skimage.measure import regionprops
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from typing import TypedDict


warnings.filterwarnings("ignore")


class CellStats(TypedDict):
    a_mean: float | None
    a_std: float | None
    tau_mean: float | None
    tau_std: float | None
    n_spikes: int | None


def load_stack(path: str) -> np.ndarray:
    return imread(path)


def get_cells(
    stack: np.ndarray,
    percentile: float = 90,
    corr_thresh: float = 0.999,
    log_txt: bool = False,
    log_img: bool = False,
) -> np.ndarray:
    """
    1) Threshold on the max-projection.
    2) Build a graph: nodes = thresholded pixels;
       edges between 8-neighbors only if corr(ts_i, ts_j) > corr_thresh.
    3) Find connected components in that graph => one “cell” per component.
    4) Overlay each cell in a distinct color on the binary mask.

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
                        # only compute corr for this neighbor pair

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
        plt.show()

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


def analyze_spikes(
    cell_intensities: np.ndarray,
    sampling_rate: float = 1.0,
    smooth_sigma: float = 2.0,
    prominence: float = 10,
    amplitude_thresh: float = 0.1,
    fit_thresh: float = 0.9,
    plot: bool = True,
    plot_fraction: float = 0.2,
) -> tuple[list[list[tuple[list[float], list[float], list[float]]]], list[CellStats]]:
    """Analyze calcium spikes in cell intensity traces.

    Parameters
    ----------
    cell_intensities : np.ndarray
        Array of shape (n_cells, n_timepoints) containing intensity values
    sampling_rate : float
        Samples per second
    smooth_sigma : float
        Gaussian σ for pre-smoothing
    prominence : float
        Minimum peak prominence
    amplitude_thresh : float
        Absolute threshold for peaks
    fit_thresh : float
        Minimum R² value for accepting a fit
    plot : bool
        Whether to generate plots

    Returns
    -------
    all_fits : list
        List of fits for each cell
    all_stats : list
        List of statistics for each cell
    """

    def _kernel(x: np.ndarray, a: float, tau: float, c: float, d: float) -> np.ndarray:
        """a · (x-d) · exp(-(x-d)/τ) + c, with x ≥ d assumed."""
        return a * (x - d) * np.exp(-(x - d) / tau) + c

    all_fits = []
    all_stats = []

    if plot:
        n_cells_to_plot = int(cell_intensities.shape[0] * plot_fraction)
        n_rows = int(np.ceil(n_cells_to_plot / 3))  # 3 columns
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

    for cell_idx in range(cell_intensities.shape[0]):
        trace = cell_intensities[cell_idx]
        t = np.arange(len(trace)) / sampling_rate
        smooth = gaussian_filter1d(trace, sigma=smooth_sigma)

        # Find peaks
        peaks, _ = find_peaks(
            smooth, prominence=prominence, distance=5, height=amplitude_thresh
        )

        # Find onset points
        onset_points = []
        for peak in peaks:
            search_start = max(0, peak - 30)
            window = smooth[search_start : peak + 1]
            derivatives = np.diff(window)
            max_deriv_idx = np.argmax(derivatives)
            onset_idx = search_start + max_deriv_idx
            onset_points.append(onset_idx)

        onset_points = np.array(onset_points)

        # Analyze segments
        segments = []
        segment_fits = []

        for i in range(len(onset_points)):
            start = onset_points[i]
            if i < len(onset_points) - 1:
                end = min(onset_points[i + 1], start + 50)
            else:
                end = min(len(smooth), start + 50)

            t_segment = t[start:end]
            y_segment = smooth[start:end]

            if len(t_segment) < 10:
                continue

            try:
                peak_val = np.max(y_segment)
                baseline = np.min(y_segment)
                a0 = (peak_val - baseline) / (t_segment[1] - t_segment[0])
                tau0 = (t_segment[-1] - t_segment[0]) / 3
                d0 = t_segment[0]

                p0 = [a0, tau0, baseline, d0]
                bounds = (
                    [0, 0.001, -np.inf, t_segment[0] - 1],
                    [np.inf, 1000, np.inf, t_segment[0] + 1],
                )

                popt, _ = curve_fit(
                    _kernel, t_segment, y_segment, p0=p0, bounds=bounds, maxfev=2000
                )

                y_fit = _kernel(t_segment, *popt)
                residuals = y_segment - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_segment - np.mean(y_segment)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                if r_squared >= fit_thresh and 0 < popt[1] < 1000:
                    segments.append((t_segment, y_segment))
                    segment_fits.append((popt, t_segment, y_fit))

            except RuntimeError:
                continue

        # Calculate statistics
        if segment_fits:
            a_values = [fit[0][0] for fit in segment_fits]
            tau_values = [fit[0][1] for fit in segment_fits]
            stats = {
                "a_mean": np.mean(a_values),
                "a_std": np.std(a_values),
                "tau_mean": np.mean(tau_values),
                "tau_std": np.std(tau_values),
                "n_spikes": len(segment_fits),
            }
        else:
            stats = {
                "a_mean": np.nan,
                "a_std": np.nan,
                "tau_mean": np.nan,
                "tau_std": np.nan,
                "n_spikes": 0,
            }

        all_fits.append(segment_fits)
        all_stats.append(stats)

        if plot and cell_idx < n_cells_to_plot:
            ax = axes[cell_idx]
            ax.plot(t, smooth, "b-", label="Smoothed trace")
            if len(onset_points):
                ax.plot(
                    t[onset_points], smooth[onset_points], "g.", label="Onset points"
                )
                ax.plot(t[peaks], smooth[peaks], "r.", label="Peaks")
                # Plot fits
                for fit_params, t_seg, y_fit in segment_fits:
                    ax.plot(t_seg, y_fit, "m-", alpha=0.5)
            else:
                # Color background light red if no spikes detected
                ax.set_facecolor("#ffeded")
            ax.set_title(f"Cell {cell_idx + 1} - {stats['n_spikes']} spikes detected")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Intensity")
            ax.legend()

    if plot:
        # Hide any unused subplots
        for idx in range(n_cells_to_plot, len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.show()

    return all_fits, all_stats


if __name__ == "__main__":
    image_stack = load_stack(
        "images/xAM_data_processing_Sudarsh/mT89_re2_DMEM_2025-04-18@20-27-57.tif"
    )

    cell_intensities = get_cells(image_stack)
    cell_intensities = gaussian_filter1d(cell_intensities, sigma=1, axis=1)

    generate_time_series_plot(cell_intensities)
    fits, stats = analyze_spikes(cell_intensities)

    # Check for cells with no spikes detected
    cells_with_no_spikes = sum(1 for stat in stats if stat["n_spikes"] == 0)
    print(f"\nNumber of cells with no spikes detected: {cells_with_no_spikes}")

    tau_mean = [stats["tau_mean"] for stats in stats]
    tau_std = [stats["tau_std"] for stats in stats]
    a_mean = [stats["a_mean"] for stats in stats]
    a_std = [stats["a_std"] for stats in stats]

    # Remove outliers using IQR method
    def remove_outliers(data):
        if not all(np.isnan(x) for x in data):
            q1 = np.nanpercentile(data, 25)
            q3 = np.nanpercentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [
                x if (not np.isnan(x) and lower_bound <= x <= upper_bound) else np.nan
                for x in data
            ]
        return data

    # Apply outlier removal to each parameter
    tau_mean = remove_outliers(tau_mean)
    tau_std = remove_outliers(tau_std)
    a_mean = remove_outliers(a_mean)
    a_std = remove_outliers(a_std)

    # Create figure with 2x2 subplots for parameter distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histograms of fitted parameters
    ax1.hist(tau_mean, bins=20)
    ax1.set_title("Distribution of Mean τ")
    ax1.set_xlabel("τ (seconds)")
    ax1.set_ylabel("Count")

    ax2.hist(tau_std, bins=20)
    ax2.set_title("Distribution of τ Standard Deviation")
    ax2.set_xlabel("τ (seconds)")
    ax2.set_ylabel("Count")

    ax3.hist(a_mean, bins=20)
    ax3.set_title("Distribution of Mean Amplitude")
    ax3.set_xlabel("Amplitude")
    ax3.set_ylabel("Count")

    ax4.hist(a_std, bins=20)
    ax4.set_title("Distribution of Amplitude Standard Deviation")
    ax4.set_xlabel("Amplitude")
    ax4.set_ylabel("Count")

    plt.tight_layout()
    plt.show()
