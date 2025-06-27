from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks
import jaxtyping
from scipy.optimize import curve_fit


def load_stack(path):
    stack = io.imread(path)
    stack = stack[:, 300:700, :]
    return stack


def extract_cells(stack):
    max_img = stack.max(axis=0)
    threshold = np.percentile(max_img, 90)
    threshold_image = (max_img > threshold).astype(np.uint8) * 255

    lines = cv2.HoughLinesP(
        threshold_image,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=150,
        maxLineGap=40,
    )

    mask = np.zeros_like(max_img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 1, 5)

    threshold_image[mask > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        threshold_image, connectivity=8
    )

    eccentricities = []
    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            if len(contours[0]) >= 5:
                ellipse = cv2.fitEllipse(contours[0])
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0
        else:
            eccentricity = 0
        eccentricities.append(eccentricity)

    min_cluster_size = 10
    max_eccentricity = 0.95
    valid_clusters = np.logical_and(
        stats[1:, cv2.CC_STAT_AREA] > min_cluster_size,
        np.array(eccentricities) < max_eccentricity,
    )

    smooth_stack = gaussian_filter1d(stack, sigma=1, axis=0)
    cell_intensities = np.zeros((np.sum(valid_clusters), smooth_stack.shape[0]))

    cluster_idx = 0

    # For each valid cluster, calculate its mean intensity over time
    for i in range(1, num_labels):
        if valid_clusters[i - 1]:
            # Create a mask for this cluster
            cluster_mask = labels == i

            # Calculate mean intensity over time for this cluster
            for t in range(len(smooth_stack)):
                mean_intensity = np.mean(smooth_stack[t][cluster_mask])
                cell_intensities[cluster_idx, t] = mean_intensity

            # Calculate baseline using a very wide window moving average
            window_size = len(smooth_stack) // 4  # Use 1/4 of signal length
            if window_size % 2 == 0:  # Make window size odd
                window_size += 1
            baseline = savgol_filter(cell_intensities[cluster_idx], window_size, 1)

            # Subtract baseline
            cell_intensities[cluster_idx] -= baseline

            cluster_idx += 1

    return cell_intensities


def exp_decay_func(x, a, b, c, d):
    """Function to fit: a*(x-c)*exp(-b*(x-c)) + d"""
    # Clip extremely large values to prevent overflow
    exp_term = np.clip(-b * (x - c), -100, 100)  # Limit the exponent range
    return a * (x - c) * np.exp(exp_term) + d


def find_kinetics(cell_intensities: jaxtyping.Float[np.ndarray, "n_cells t"]):
    per_cell_params = []

    for cell_intensity in cell_intensities:
        slopes = np.zeros(len(cell_intensity) - 4)

        for i in range(len(cell_intensity) - 4):
            window = cell_intensity[i : i + 5]
            slope = np.polyfit(np.arange(5), window, 1)[0]
            slopes[i] = slope  # type: ignore

        peaks, _ = find_peaks(slopes, prominence=20, height=2)
        # Split cell_intensity into segments based on peaks
        segments = []
        start_idx = 0

        for peak in peaks:
            # Add segment from start to peak
            segments.append(cell_intensity[start_idx:peak])
            start_idx = peak

        # Add final segment from last peak to end
        if start_idx < len(cell_intensity):
            segments.append(cell_intensity[start_idx:])

        params = []
        # Fit exponential decay to each segment
        for segment in segments:
            # Need enough points to fit meaningfully
            if len(segment) < 10:
                continue

            x = np.arange(len(segment))
            y = segment

            try:
                # Initial parameter guesses
                p0 = [
                    np.max(y) - np.min(y),  # a: amplitude
                    0.1,  # b: decay rate
                    0,  # c: x offset
                    np.min(y),  # d: y offset
                ]

                # Fit the curve
                popt, _ = curve_fit(exp_decay_func, x, y, p0=p0, maxfev=2000)

                # Get fitted values
                y_fit = exp_decay_func(x, *popt)

                # Could store or return the parameters and fitted curves here
                # For now just printing the decay rate
                print(f"Decay rate: {popt[1]:.3f}")

                params.append(popt)

            except RuntimeError:
                # Curve fitting failed to converge
                print("Failed to fit curve to segment")
                continue

        # Calculate average parameters if any fits were successful
        if params:
            avg_params = np.mean(params, axis=0)
            per_cell_params.append(avg_params)
        else:
            per_cell_params.append(None)

    return per_cell_params


if __name__ == "__main__":
    # load stack
    stack = load_stack(
        "images/xAM_data_processing_Sudarsh/mT89_re2_DMEM_2025-04-18@20-27-57.tif"
    )

    cell_intensities = extract_cells(stack)

    del stack

    find_kinetics(cell_intensities)
