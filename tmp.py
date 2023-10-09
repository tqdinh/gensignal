import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# Parameters
num_chirps = 100                 # Number of chirps
num_samples_per_chirp = 1024    # Number of samples per chirp
bin_of_interest = 10            # Bin of interest for vital signal extraction

# Generate sample radar data (complex-valued)
radar_data = np.random.randn(num_chirps, num_samples_per_chirp) + \
    1j * np.random.randn(num_chirps, num_samples_per_chirp)

# Perform range FFT
range_fft_result = np.fft.fft(radar_data, axis=1)

print(np.array(range_fft_result).shape)
# Extract phase change at the specific bin over each chirp
phase_change = np.angle(range_fft_result[:, bin_of_interest])


# Perform further analysis on the phase change to extract vital signal information
# For example, compute heartbeat or respiration rate from the phase change values
# ...


# Define the bandpass filter parameters
fs = 10.0  # Sampling frequency (assuming each chirp has a duration of 1 unit)
lowcut = 0.4  # Lower cutoff frequency of the bandpass filter
highcut = 5.0  # Upper cutoff frequency of the bandpass filter
order = 4  # Filter order

# Normalize the cutoff frequencies
nyquist_freq = 0.5 * fs
low = lowcut / nyquist_freq
high = highcut / nyquist_freq

# Apply the bandpass filter
b, a = butter(order, [low, high], btype='band', fs=fs)
filtered_phase_change = filtfilt(b, a, phase_change)
peaks, _ = find_peaks(filtered_phase_change)


plt.figure(figsize=(10, 6))


plt.plot(filtered_phase_change, label='Filtered Phase Change')
plt.plot(peaks, filtered_phase_change[peaks], 'ro', label='Peaks')
plt.xlabel('Chirp Index')
plt.ylabel('Phase Change')
plt.title('Filtered Phase Change with Peaks')
plt.legend()
plt.grid(True)
plt.show()
