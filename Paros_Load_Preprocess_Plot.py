"""
Script Name: EQ_Paros_Load_Preprocess_Plot.py

Purpose:
    Fetches Paros infrasound/pressure sensor data from InfluxDB, preprocesses it,
    and generates key diagnostic plots:
        1. Preprocessed time series
        2. Power Spectral Density (Welch method)
        3. Spectrogram (short-time Fourier analysis)

Preprocessing steps include:
    - DC offset removal
    - Optional wavelet denoising
    - High-pass filtering at 0.1 Hz

Inputs:
    - Paros sensor data via InfluxDB query (Unix epoch timestamps, raw pressure in mBar)

Outputs:
    - Time series plot
    - PSD plot (dB/Hz)
    - Spectrogram plot (64-sample window, 8-sample shift)

Author: Ethan Gelfand
Date:   2025-08-17
"""

from pathlib import Path
from paros_data_grabber import query_influx_data, save_data
from Preprocessing_fun import preprocess, welch_psd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal.windows import hann
from datetime import datetime, timezone

# -------------------------------
# --- Configuration / Settings ---
# -------------------------------
password = "*****"  # InfluxDB password
output_dir = Path(r"C:\Data\PAROS\Exported_Paros_Data")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "Example_Earthquake.mat"
fs = 20  # Sampling frequency (Hz)

# -------------------------------
# --- Fetch Sensor Data ---
# -------------------------------
data = query_influx_data(
    start_time="2025-05-05T03:03:00",
    end_time="2025-05-05T03:04:00",
    box_id="parost2",
    sensor_id="141929",
    password=password
)

if not data:
    raise RuntimeError("No data returned from InfluxDB.")
else:
    # Convert to NumPy array
    data_array = np.array(data['parost2_141929'])
    # save_data(data, str(output_file))
    # print(f"Data saved to: {output_file.resolve()}")

# -------------------------------
# --- Prepare Time & Signal ---
# -------------------------------
unix_times = data_array[:, 0]
dt_utc = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_times])
raw_waveform = np.asarray(data_array[:, 1]).flatten()

# Preprocess the waveform
signal = preprocess(raw_waveform, fs)

# Compute PSD
pxx, f_psd= welch_psd(signal, fs)

# -------------------------------
# --- Plotting Functions ---
# -------------------------------
def plot_signal(signal, dt):
    plt.figure()
    plt.plot(dt, signal, linewidth=1.5)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Pressure (mB)')
    plt.title('Preprocessed Time Series Data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_psd(frequencies, power_spectral_density):
    plt.figure()
    plt.plot(frequencies, 10*np.log10(power_spectral_density), linewidth=1.2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Welch Power Spectral Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(signal, fs, window_length=64, shift=8, nfft=1024):
    overlap = window_length - shift
    f_spect, t_spect, Sxx = spectrogram(signal, fs, window=hann(window_length),
                                        nperseg=window_length, noverlap=overlap, nfft=nfft)
    plt.figure()
    plt.pcolormesh(t_spect, f_spect, 10*np.log10(Sxx), shading='gouraud', cmap='jet').set_clim(-110, -50)
    plt.ylim([0, 10])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram (64-sample window, 8-sample shift)')
    plt.colorbar(label='dB/Hz')
    plt.tight_layout()
    plt.show()

# -------------------------------
# --- Execute Plots ---
# -------------------------------
plot_signal(signal, dt_utc)
plot_psd(f_psd, pxx)
plot_spectrogram(signal, fs)
