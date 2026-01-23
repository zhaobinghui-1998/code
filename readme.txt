Seismic-Event-Classifier: Dual-Branch Adaptive Framework for Seismic Event Identification
A dual-branch deep learning model based on single-component seismic waveform data, designed to distinguish between tectonic earthquakes and artificial blasts, with cross-regional generalization and adaptive decision-making capabilities.
Project Overview
This project implements a dual-branch adaptive joint decision-making framework that processes single-channel seismic waveforms and their time-frequency matrices to enable intelligent classification of seismic event types. The model requires no complex feature engineering, directly processes raw waveform data, and performs excellently in low signal-to-noise ratio (SNR) and heterogeneous data scenarios—making it suitable for distributed seismic monitoring systems.
Key Features
Dual-Branch Architecture: Independently processes raw waveforms and time-frequency matrices, with dynamic decision fusion via an adaptive weight network.
Single-Component Adaptability: Operates solely on single-channel seismic data (e.g., vertical component), supporting low-cost sensing devices.
Cross-Regional Generalization: Maintains high accuracy across datasets from different geological regions without transfer learning.
Lightweight Preprocessing: Only requires low-frequency response filtering and sampling rate unification—no manual feature extraction needed.
Adaptive Decision-Making: Dynamic weight adjustment enhances classification reliability in complex environments.
Technical Architecture
Framework: TensorFlow 2.8 + Keras
Core Network:
Branch 1: 1D Convolutional Neural Network (processes raw waveforms)
Branch 2: 2D Convolutional Neural Network (processes time-frequency matrices)
Fusion Layer: Adaptive weight learning sub-network
Data Processing: Librosa (time-frequency conversion), Obspy (seismic data processing)
Installation Guide
Prerequisites
Python 3.8+
TensorFlow 2.8+
numpy, scipy, pandas
librosa, obspy
matplotlib, seaborn