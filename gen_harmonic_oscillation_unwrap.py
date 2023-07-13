import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import mplcursors

# Signal parameters
sampling_rate = 1000  # Sample rate (Hz)
duration = 10  # Signal duration (seconds)
carrier_frequency = 100  # Frequency of the carrier wave (Hz)
harmonic_frequency = 10  # Frequency of the harmonic component (Hz)
amplitude_carrier = 1  # Amplitude of the carrier wave
amplitude_harmonic = 0.5  # Amplitude of the harmonic component
noise_amplitude = amplitude_harmonic/5

num_labels = 5  # Number of top peaks to label

LINEWIDTH = 0.5

# Time axis
t = np.linspace(0, duration, int(sampling_rate * duration))

# Generate the carrier wave
carrier = amplitude_carrier * np.sin(2 * np.pi * carrier_frequency * t)

# Generate the harmonic component
harmonic = amplitude_harmonic * np.sin(2 * np.pi * harmonic_frequency * t)
harmonic_tmp = amplitude_harmonic * \
    np.sin(2 * np.pi * harmonic_frequency/3 * t+np.pi/3)
random_freq = (harmonic_frequency - (-1)*harmonic_frequency) * \
    np.random.randn(len(t)) + (-1)*harmonic_frequency

#noise = noise_amplitude * np.sin(2 * np.pi * random_freq * t)

noise = noise_amplitude * np.sin(2 * np.pi * harmonic_frequency/2 * t)

# harmonic = harmonic+noise
harmonic_with_noise = harmonic + noise + harmonic_tmp

# Combine the carrier and harmonic components
signal = carrier + harmonic_with_noise

# Compute the phase of the signal
phase = np.unwrap(np.angle(signal))

# Compute the differences between consecutive elements in the unwrapped phase
phase_diff = np.diff(phase)


phase_fft = np.fft.fft(phase)

# Frequency axis
freq = np.fft.fftfreq(len(phase), d=1/sampling_rate)


# Plot the original signal and the unwrapped phase differences
plt.figure(figsize=(1920/160, 1080/160))

plt.subplot(4, 1, 1)
plt.plot(t, signal, label='Signal', linewidth=LINEWIDTH)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal')
plt.legend()
plt.grid(True)


plt.subplot(4, 1, 2)
plt.plot(t, harmonic, label='Harmonic', color='blue', linewidth=LINEWIDTH)
plt.plot(t, harmonic_tmp, label='Harmonic TMP',
         color='green', linewidth=LINEWIDTH)
plt.plot(t, noise, label='Noise', color='orange', linewidth=LINEWIDTH)
plt.plot(t, harmonic_with_noise, label='Mixed HN',
         color='red', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Harmonic')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t[:len(t)//5], phase[:len(t)//5], linewidth=LINEWIDTH)
plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')
plt.title('Phase ')
plt.grid(True)


plt.subplot(4, 1, 4)
plt.plot(freq, np.abs(phase_fft), linewidth=LINEWIDTH)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
# plt.title('Frequency Spectrum of Phase')
peaks, _ = find_peaks(np.abs(phase_fft), prominence=0.05)
sorted_peaks = sorted(peaks, key=lambda x: np.abs(phase_fft)[x], reverse=True)
for peak in sorted_peaks[:num_labels]:
    # show top 3
    plt.annotate(f'{freq[peak]:.2f}', xy=(freq[peak], np.abs(phase_fft)[peak]),
                 xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, rotation='vertical',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5))
    plt.plot(freq[peak], np.abs(phase_fft)[peak], 'ro')

plt.grid(True)


plt.tight_layout()
plt.show()
