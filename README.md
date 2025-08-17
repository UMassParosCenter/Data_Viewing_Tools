# Paros Infrasound Data Processing Tools

This repository provides utilities for working with **Paros infrasound sensor data**. The scripts allow data acquisition, preprocessing, and visualization, and were used in the development of deep learning earthquake detection models.  

## Repository Structure

- **`Preprocessing_functions.py`**  
  Contains Python signal preprocessing functions used in the training of deep learning earthquake models (from the Earthquake_DeepLearning_Tools repository).  
  Features include:
  - DC blocking  
  - High-pass filtering  
  - Welch Power Spectral Density (PSD) computation  
  - Safe resampling  

- **`GrabParosData.py`**  
  A simple Python script for extracting Paros infrasound sensor data from InfluxDB and saving it to `.mat` files for later analysis.  

- **`Paros_Load_Preprocess_Plot.py`**  
  A complete Python workflow for:  
  1. Loading Paros infrasound data  
  2. Preprocessing the waveform  
  3. Visualizing:  
     - Time series  
     - Power Spectral Density (PSD)  
     - Spectrogram  

- **`Paros_Preprocess_PSD_Spectrogram.m`**  
  MATLAB version of the preprocessing pipeline:  
  - Loads `.mat` files saved by `GrabParosData.py`  
  - Applies the same preprocessing steps as in Python  
  - Generates plots for time series, PSD, and spectrogram  
  
------
Ethan Gelfand, 8/17/2025
