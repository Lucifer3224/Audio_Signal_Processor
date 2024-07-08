# Numerical operations library
import numpy as np
# .pyplot is a module from matplotlib to make plots -It's a visualization library-
import matplotlib.pyplot as plt
# .display is a module from librosa library to display the plot -It's an audio analyser library-
import librosa.display

""".signal is a module for the signal processing
butter is a function for designing a filter and returning the coefficients for the transfer func.
lfilter is a fn. to apply the designed filter for the signal"""
from scipy.signal import butter, lfilter
# .fft is used for fourier transform operations
import scipy.fft as fft
# used to save the filtered sound
import soundfile as sf


# Function to plot the time domain signal
def plot_time_domain(signal, sampling_rate, title="Time Domain"):
    """ arange is a fn. creates an array of a given range
        the range 0 --> signal length is the number of the samples
        dividing the no. of samples by the sampling rate (no. of samples per sec) gives the time """
    time = np.arange(0, len(signal)) / sampling_rate
    plt.figure(figsize=(12, 4))
    # the first parameter is the x-axis and the second is the y-axis
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# Function to plot the frequency domain signal "X axis represents F"
def plot_frequency_domain(signal, title="Frequency Domain"):
    plt.figure(figsize=(12, 4))
    # transformed_signal is a complex array representing the Fourier coefficients -amplitude and phase of frequencies-
    transformed_signal = fft.fft(signal)
    """fftfreq is a function calculates the frequencies of the signal
       F(k) -freq_axis- = k/ (n*d) -- n is the no. of data points
       d is the time difference between samples in the original time-domain signal -the inverse of the sampling rate-"""
    freq_axis = fft.fftfreq(len(transformed_signal), d=1 / sampling_rate)
    # np.abs calculate the magnitude of each frequency
    plt.plot(freq_axis, np.abs(transformed_signal))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()


# Function to apply a filter to the signal
def apply_filter(signal, cutoff_freq, filter_type, sampling_rate):
    nyquist = 0.5 * sampling_rate
    # cutoff frequency must be expressed relative to the Nyquist frequency.
    normal_cutoff = cutoff_freq / nyquist
    # b and a are the numerator and denominator coefficients of the transfer function used in designing the filter
    b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# Read the audio file
file_path = "D:/college/Junior/Fall 23/Signals/audio/audio.wav"
original_signal, sampling_rate = librosa.load(file_path, sr=None)

# Plot the original signal in time domain
plot_time_domain(original_signal, sampling_rate, "Original Signal (Time Domain)")

# Plot the original signal in frequency domain
plot_frequency_domain(original_signal, "Original Signal (Frequency Domain)")

# Changing the speed
#speedup_signal = librosa.effects.time_stretch(original_signal, rate=2.0)

# Apply a high-pass filter
cutoff_frequency = 400
filter_type = 'highpass'
filtered_signal = apply_filter(original_signal, cutoff_frequency, filter_type, sampling_rate)

# Changing the sampling rate for the saved audio -tone changing-
#faster_sampling_rate = 2 * sampling_rate

# Plot the filtered signal in frequency domain
plot_frequency_domain(filtered_signal, "Filtered Signal (Frequency Domain)")

# Find the corresponding signal in time domain for the filtered signal and plot it
plot_time_domain(filtered_signal, sampling_rate, "Filtered Signal (Time Domain)")

# Save the filtered signal as an audio file
filtered_file_path = "D:/college/Junior/Fall 23/Signals/audio/filtered_audio.wav"
sf.write(filtered_file_path, filtered_signal, sampling_rate)

print(f"Filtered audio saved at: {filtered_file_path}")
