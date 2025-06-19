import argparse
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def load_image_stack(path):
    image_stack = io.imread(path)
    return np.array(image_stack)


def run_cellpose_on_stack(stack, channels=[0, 0], diameter=30):
    img = stack[0]
    img = (img - img.min()) / (img.max() - img.min())

    model = models.Cellpose(model_type="cyto", gpu=True)

    mask, flows, styles, diams = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )

    return mask, flows, styles, diams


def get_average_cell_frequencies(window_size, image_stack, mask):
    # Get number of unique cells (excluding background which is 0)
    n_cells = len(np.unique(mask)) - 1

    # Calculate number of windows
    n_windows = len(image_stack) - window_size + 1

    # Initialize arrays to store frequencies and std for each cell
    cell_frequencies = np.zeros((n_cells, n_windows))
    total_spikes = np.zeros(n_cells)

    # First pass - calculate total spikes per cell for weights
    for cell_id in range(1, n_cells + 1):
        # Get the trace for current cell
        cell_trace = image_stack[:, mask == cell_id].mean(axis=1)

        # Apply smoothing to reduce noise
        cell_trace = gaussian_filter1d(cell_trace, sigma=0.25)

        # Find peaks in the trace
        peaks, _ = find_peaks(cell_trace, prominence=25)

        # Store total spikes for this cell
        total_spikes[cell_id - 1] = len(peaks)

        # Calculate spike frequencies for this cell
        for i in range(n_windows):
            window_start = i
            window_end = i + window_size
            # Count peaks that fall within this window
            peaks_in_window = np.sum((peaks >= window_start) & (peaks < window_end))
            cell_frequencies[cell_id - 1, i] = peaks_in_window

    # Calculate weights
    weights = total_spikes / np.sum(total_spikes)

    # Calculate weighted average per window
    weighted_avg_frequencies = np.average(cell_frequencies, axis=0, weights=weights)

    # --- begin modified error‐bar computation ---

    # 1) Unbiased weighted variance (Bessel’s correction):
    den = 1.0 - np.sum(weights**2)  # since sum(weights) == 1
    var_unbiased = (
        np.sum(
            weights[:, None] * (cell_frequencies - weighted_avg_frequencies) ** 2,
            axis=0,
        )
        / den
    )
    sigma_unbiased = np.sqrt(var_unbiased)

    # 2) Effective sample size:
    n_eff = 1.0 / np.sum(weights**2)

    # 3) Standard error of the weighted mean:
    stderr = sigma_unbiased / np.sqrt(n_eff)

    # 4) Normalize to starting frequency
    starting_frequency = weighted_avg_frequencies[0]
    if starting_frequency == 0:
        raise ValueError("No spikes in initial window; cannot normalize.")

    norm_mean = weighted_avg_frequencies / starting_frequency
    norm_stderr = stderr / starting_frequency

    return norm_mean, norm_stderr


def get_cell_frequency_distribution(image_stack, mask):
    n_cells = len(np.unique(mask)) - 1
    cell_frequencies = []

    for cell_id in range(1, n_cells + 1):
        # Get the trace for current cell
        cell_trace = image_stack[:, mask == cell_id].mean(axis=1)

        # Apply smoothing to reduce noise
        cell_trace = gaussian_filter1d(cell_trace, sigma=0.25)

        # Find peaks in the trace
        peaks, _ = find_peaks(cell_trace, prominence=25)

        # Calculate frequency (spikes per frame)
        frequency = len(peaks) / len(image_stack)
        cell_frequencies.append(frequency)

    return np.array(cell_frequencies)


def plot_cell_frequencies(mean, stderr):
    plt.figure(figsize=(10, 6))
    plt.plot(mean, "b-", label="Weighted mean (norm.)")
    plt.fill_between(
        np.arange(len(mean)),
        mean - stderr,
        mean + stderr,
        color="b",
        alpha=0.2,
        label="±1 SEM",
    )
    plt.legend()
    plt.show()


def plot_frequency_distribution(frequencies):
    plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=30, color="b", alpha=0.7)
    plt.title("Distribution of Cell Firing Frequencies")
    plt.xlabel("Frequency (spikes/frame)")
    plt.ylabel("Number of Cells")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process ultrasound images")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the ultrasound image stack"
    )
    args = parser.parse_args()

    image_stack = load_image_stack(args.path)
    mask, _, _, _ = run_cellpose_on_stack(image_stack)

    mean, stderr = get_average_cell_frequencies(100, image_stack, mask)

    frequencies = get_cell_frequency_distribution(image_stack, mask)

    plot_cell_frequencies(mean, stderr)
    plot_frequency_distribution(frequencies)
