import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt  # Import Butterworth filter functions
from scipy.interpolate import interp1d
import time
import random

# Parameters for FFT
fft_size = 256  # Number of points to perform FFT on
sample_rate = 250  # Unicorn data typically streams at 250Hz, adjust as needed
buffer_size = 500  # Buffer size to store recent samples for FFT
n_channels = 2  # Now only 2 channels (channels 2 and 4)
fft_interval = 1  # Perform FFT every 1 second
selected_channels = [1, 3]  # Channel indices for channels 2 and 4
alpha_band = (8, 12)  # Alpha waves frequency range (8-12 Hz)

# Define a Butterworth band-pass filter for alpha waves (8-12 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = filtfilt(b, a, data)  # Apply the filter
    return y

def compute_alpha_power(fft_result, freqs):
    # Find the indices for the alpha range
    alpha_freqs_mask = (freqs >= 8) & (freqs <= 12)
    alpha_power = np.sum(np.abs(fft_result[alpha_freqs_mask])**2)  # Sum of power in alpha range
    return alpha_power

# Function to generate random colors, adjusted for alpha ratio
def random_color(alpha_ratio):
    return np.clip(np.random.rand(3,) * (1 - alpha_ratio) + alpha_ratio * 0.3, 0, 1)

# Function to generate random geometric shapes based on alpha ratio
def random_shapes(ax, num_shapes, alpha_ratio):
    for _ in range(num_shapes):
        shape_type = random.choices(
            ['circle', 'rectangle', 'triangle', 'line', 'polygon'],
            weights=[
                2 if alpha_ratio < 0.3 else 1,
                2 if 0.3 <= alpha_ratio < 0.6 else 1,
                1 if 0.6 <= alpha_ratio < 0.9 else 2,
                3 if alpha_ratio >= 1.5 else 1,
                3 if alpha_ratio >= 1.5 else 1])[0]

        color = random_color(alpha_ratio)

        if shape_type == 'circle':
            radius = random.uniform(0.05, 0.2)
            x, y = random.uniform(0, 1), random.uniform(0, 1)
            circle = plt.Circle((x, y), radius, color=color, fill=True)
            ax.add_patch(circle)
        elif shape_type == 'rectangle':
            width, height = random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)
            x, y = random.uniform(0, 1 - width), random.uniform(0, 1 - height)
            rectangle = plt.Rectangle((x, y), width, height, color=color, fill=True)
            ax.add_patch(rectangle)
        elif shape_type == 'triangle':
            points = np.random.rand(3, 2)
            triangle = plt.Polygon(points, color=color, fill=True)
            ax.add_patch(triangle)
        elif shape_type == 'line':
            x1, y1 = random.uniform(0, 1), random.uniform(0, 1)
            x2, y2 = random.uniform(0, 1), random.uniform(0, 1)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        elif shape_type == 'polygon':
            num_sides = random.randint(3, 6)
            points = np.random.rand(num_sides, 2)
            polygon = plt.Polygon(points, color=color, fill=True)
            ax.add_patch(polygon)

# Function to generate random painting based on alpha ratio
def generate_painting(ax, alpha_ratio):
    # Clear the axis for the new drawing
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Set number of shapes based on alpha ratio
    if (0 < alpha_ratio < 0.3):
        num_shapes = 5  # Very calm
    elif (0.3 < alpha_ratio < 0.6):
        num_shapes = 10  # Moderately calm
    elif (0.6 < alpha_ratio < 0.9):
        num_shapes = 15  # Slight stress
    elif (0.9 < alpha_ratio < 1.2):
        num_shapes = 20  # Neutral/Medium stress
    elif (1.2 < alpha_ratio < 1.5):
        num_shapes = 25  # High stress
    elif (1.5 < alpha_ratio < 1.8):
        num_shapes = 30  # Very high stress
    else:
        num_shapes = 40  # Extreme stress

    # Generate shapes based on alpha ratio
    random_shapes(ax, num_shapes, alpha_ratio)

    # Adjust the background color based on alpha ratio (lighter for calmer, darker for stressed)
    if alpha_ratio < 1:
        bg_color = np.clip(1 - alpha_ratio / 2, 0, 1)
        ax.figure.patch.set_facecolor((bg_color, bg_color, bg_color))
    else:
        ax.figure.patch.set_facecolor((1, 1, 1))

    # Redraw the canvas
    plt.draw()
    plt.pause(0.1)  # Small pause to allow plot updates

def main():
    print("Looking for a Unicorn stream...")
    streams = resolve_stream("name='Unicorn'")

    print("Creating a new inlet to read from the stream...")
    inlet = StreamInlet(streams[0])

    # Initialize buffers for the selected channels
    data_buffers = [[] for _ in selected_channels]
    time_buffer = []
    last_fft_time = time.time()

    # Set up interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    while True:
        # Get a new sample
        sample, timestamp = inlet.pull_sample()

        # Append to the buffer for each selected channel (channels 2 and 4)
        for idx, channel in enumerate(selected_channels):
            data_buffers[idx].append(sample[channel])
        time_buffer.append(timestamp)

        # Keep buffer size fixed
        if len(time_buffer) > buffer_size:
            for i in range(n_channels):
                data_buffers[i].pop(0)
            time_buffer.pop(0)

        # Perform FFT only at longer intervals (every `fft_interval` seconds)
        current_time = time.time()
        if current_time - last_fft_time >= fft_interval:
            last_fft_time = current_time

            # Resample the data to an evenly spaced timeline
            resampled_times = np.linspace(time_buffer[0], time_buffer[-1], fft_size)

            alpha_powers = []  # To store alpha powers for both channels

            for i in range(n_channels):
                # Interpolate to resample the data for the current channel
                interpolator = interp1d(time_buffer, data_buffers[i], kind='linear')
                resampled_data = interpolator(resampled_times)

                # Apply the band-pass filter to keep only alpha waves (8-12 Hz)
                filtered_data = apply_filter(resampled_data, alpha_band[0], alpha_band[1], sample_rate)

                # Perform FFT on the filtered data
                fft_result = fft(filtered_data)
                freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)

                # Compute the total alpha power for the current channel
                alpha_power = compute_alpha_power(fft_result, freqs)
                alpha_powers.append(alpha_power)

            # Calculate the ratio of alpha power between the two channels
            alpha_ratio = alpha_powers[0] / alpha_powers[1] if alpha_powers[1] != 0 else 1

            print(f"Alpha Ratio: {alpha_ratio}")

            # Update the existing plot with new alpha ratio
            generate_painting(ax, alpha_ratio)

if __name__ == "__main__":
    main()
