% ========================================================================
% Script Name: Paros_Preprocess_PSD_Spectrogram.m
%
% Purpose:
%   This script loads Paros infrasound/pressure sensor data from a .mat file, 
%   preprocesses the raw signal, and generates key diagnostic plots:
%     1. Preprocessed time series
%     2. Power Spectral Density (Welch method)
%     3. Spectrogram (short-time Fourier analysis)
%
%   Preprocessing steps include:
%     - DC offset removal
%     - Optional wavelet denoising
%     - High-pass filtering at 0.1 Hz
%
% Inputs:
%   - Earthquake .mat file containing Paros sensor data 
%     (time in Unix epoch, raw signal in mBar)
%
% Outputs:
%   - Time series plot
%   - PSD plot (dB/Hz up to Nyquist, fs = 20 Hz)
%   - Spectrogram plot (64-sample window, 8-sample shift)
%
% Author: Ethan Gelfand
% Date:   2025-08-17
% ========================================================================

clc; clear; clf; close all;

%% --- Load & Preprocess Signal ---
output_dir = fullfile('C:', 'Data', 'PAROS', 'Exported_Paros_Data');
output_file = fullfile(output_dir, 'Example_Earthquake.mat');
data = load(output_file).parost2_141929;

unixTimes = data(:,1);               % Time vector
dtUTC = datetime(unixTimes, 'ConvertFrom', 'posixtime', 'TimeZone', 'UTC'); % Convert to datetime in UTC timezone
signal_raw = data(:,2);      % Raw signal
fs = 20;                     % Sampling frequency (Hz)

signal = preprocess(signal_raw, fs);
[pxx, f] = WelchPSD(signal, fs);

%% --- plot time series ---
figure;
plot(dtUTC,signal,'LineWidth',1.5)
xlabel('time (UTC)');
ylabel('pressure (mB)');
title('Preprocessed time series data');
xlim([dtUTC(1), dtUTC(end)])
grid on;


%% --- Plot PSD ---
figure;
plot(f, 10*log10(pxx), 'LineWidth', 1.2);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Welch Power Spectral Density');
grid on;

%% --- Plot Spectrogram ---
windowLength = 64;              % 64 samples window (~3.2 sec)
shift = 8;                     % 8 samples shift (~0.4 sec)
overlap = windowLength - shift; % 56 samples overlap
nfft = 1024;

figure;
spectrogram(signal, hann(windowLength), overlap, nfft, fs, 'yaxis');
title('Spectrogram (64-sample window, 8-sample shift)');
ylim([0 10]);        % Nyquist limit at fs=20 Hz
clim([-110 -50]);   % From -100 dB (min) to -50 dB (max)
colorbar;
colormap(jet);

%% FUNCTIONS

function y = preprocess(x, fs)
    % --- DC Block ---
    x = DCBlock(x);

    % --- Wavelet Denoising ---
    %x = safe_wdenoise(x, fs);

    % --- highpass Filter (0.1 Hz) ---
    cutoff = 0.1;
    wn = cutoff / (fs/2);
    [b, a] = butter(4, wn, 'high');
    y = filtfilt(b, a, x);
end

function [pxx, f] = WelchPSD(signal, fs)
    nperseg = round(5 * fs);  % 5 second window
    noverlap = round(nperseg * 0.75);
    nfft = 2^nextpow2(nperseg);
    window = hann(nperseg);
    [pxx, f] = pwelch(signal, window, noverlap, nfft, fs);
    
    % Keep only 0–20 Hz part
    keep = f <= 10;
    pxx = pxx(keep);
    f = f(keep);
end

function y = safe_resample(x, fs_in, fs_out)
    % Resamples signal safely with anti-imaging protection
    x = DCBlock(x);
    
    nyquist_in = fs_in / 2;
    fc = 0.9 * min(fs_in, fs_out) / 2;
    [b_lp, a_lp] = butter(4, fc / nyquist_in, 'low');
    x = filtfilt(b_lp, a_lp, x);
    
    y = resample(x, fs_out, fs_in);
end

function y = safe_wdenoise(x, fs)
    waveletType = 'sym8';
    level = wmaxlev(length(x), waveletType);
    level = min(level, 6);

    y = wdenoise(x, level, ...
        'Wavelet', waveletType, ...
        'DenoisingMethod', 'Bayes', ...
        'ThresholdRule', 'Soft', ...
        'NoiseEstimate', 'LevelDependent');

    fc = 0.9 * (fs/2);
    [b, a] = butter(4, fc/(fs/2), 'low');
    y = filtfilt(b, a, y);
end

function y = DCBlock(x)
    a = 0.999;
    b = [1 -1];
    a_coeffs = [1 -a];
    y = filtfilt(b, a_coeffs, x);
end
