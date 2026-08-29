# Copyright (c) 2026 Timur Togochiev

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal as sp
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
from scipy.ndimage import gaussian_filter
import pywt
import gc
import os
import shutil
import zipfile
import io
import functions as func

st.set_page_config(
    page_title="EpiAnalyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "detect_path" not in st.session_state:
    st.session_state.detect_path = None
if "data_dir" not in st.session_state:
    st.session_state.data_dir = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "df_ictal" not in st.session_state:
    st.session_state.df_ictal = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "sweeps_raw" not in st.session_state:
    st.session_state.sweeps_raw = None
if "time" not in st.session_state:
    st.session_state.time = None
if "fs" not in st.session_state:
    st.session_state.fs = None
if "n_sweeps" not in st.session_state:
    st.session_state.n_sweeps = 0
if "n_channels" not in st.session_state:
    st.session_state.n_channels = 0
if "labels" not in st.session_state:
    st.session_state.labels = []
if "colors" not in st.session_state:
    st.session_state.colors = []
if "sweep_duration" not in st.session_state:
    st.session_state.sweep_duration = 5
if "sweep_selector" not in st.session_state:
    st.session_state.sweep_selector = None
if "active_preset" not in st.session_state:
    st.session_state.active_preset = "EC-CA1"
if "prominence_factor" not in st.session_state:
    st.session_state.prominence_factor = 15.0
if "gap" not in st.session_state:
    st.session_state.gap = 2.0
if "min_duration" not in st.session_state:
    st.session_state.min_duration = 10.0
if "min_density" not in st.session_state:
    st.session_state.min_density = 1.0
if "max_density" not in st.session_state:
    st.session_state.max_density = 40.0
if "min_peak_freq" not in st.session_state:
    st.session_state.min_peak_freq = 3.0
if "color_ch0" not in st.session_state:
    st.session_state.color_ch0 = "#1f77b4"
if "color_ch1" not in st.session_state:
    st.session_state.color_ch1 = "#2ca02c"
if "color_interictal_bars" not in st.session_state:
    st.session_state.color_interictal_bars = "#4E79A7"
if "color_interictal_amp" not in st.session_state:
    st.session_state.color_interictal_amp = "#2B6E7A"
if "color_ictal_bars" not in st.session_state:
    st.session_state.color_ictal_bars = "#8b3a3a"
if "color_ictal_amp" not in st.session_state:
    st.session_state.color_ictal_amp = "#D97A5C"
if "color_ictal_duration" not in st.session_state:
    st.session_state.color_ictal_duration = "#8c7a6b"
if "ch0_name" not in st.session_state:
    st.session_state.ch0_name = ""
if "ch1_name" not in st.session_state:
    st.session_state.ch1_name = ""

st.markdown("<h1 style='text-align: center;'>🧠 EpiAnalyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Epileptiform Activity Analysis Pipeline</p>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("Description"):
    
    st.markdown("<h1 style='text-align: center;'>Welcome to EpiAnalyzer app!</h1>", unsafe_allow_html=True)
    st.markdown("""
    This app provides comprehensive analysis of your .abf file. 
    Pipeline was made in order to ease analysis of big files.
    All the parameters and settings for corresponding
    type of recording you will find in the sidebar.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Key Features**
        - Supports 2-channel .abf recordings
        - Developed for EC and CA1 brain regions
        - Compatible with Frontal lobe, Thalamus and other recordings
        - Low-pass, notch filtering and detrending
        - Spike detection (ictal and interictal)
        - Spectral analysis (FFT + wavelet)
        - Cross-correlation between channels
        - Intra-sweep dynamics
        - Cumulative dynamics
        - Full overview of results
        - Export: CSV and PNG
        """)
    with col2:
        st.markdown("""
        **How to use**
        1. Upload your .abf file in the sidebar
        2. Select your brain region preset
        3. Adjust parameters if needed
        4. Click Run Analysis
        5. Explore results in the tabs:
            - Sweep Inspector — quick view of any sweep
            - Time Analysis — spike counts, amplitudes, distributions
            - Frequency Analysis — FFT and wavelet spectrograms
            - Cross-correlation — channel coupling
            - Summary — overview and export
        
        Tip: Use presets below or adjust parameters manually.
        """)
    
    st.markdown("---")
    st.markdown("**Definitions**")
    st.markdown("""
        - **EC** — entorhinal cortex, a key brain region involved in generating ictal activity.
        - **CA1** — hippocampal area Cornu Ammonis 1, the main output region of the hippocampus.
        - **Sweep** — a single continuous recording block, analyzed independently
        - **Ictal event** — a long, high-frequency burst of spikes (seizure-like discharge)
        - **Interictal spikes** — short, isolated spikes between ictal events (background hyperexcitability)
        - **Spike** — a sharp, transient deflection above the noise threshold
        """)
    
    st.markdown("---")
    st.markdown("""
Author: Timur Togochiev  

Laboratory of Molecular Mechanisms of Neural Interactions, IEPhB RAS\n

2026
""")
    
st.markdown("---")

st.sidebar.header("Loading and settings")
st.sidebar.warning("Large files require 8+ GB RAM. Close other applications before running.")

uploaded_file = st.sidebar.file_uploader("Choose .abf file", type=["abf"])

with st.sidebar.expander("Settings", expanded=False):
    st.markdown("\n")

    st.warning("Default values are optimized for EC-CA1 recordings. Adjust if needed.")
    
    st.markdown("\n")
    
    st.markdown("**Quick Presets**")
    st.caption("Approximate values for different brain regions. Fine-tune manually if needed.")
    
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("EC-CA1", key="preset_ec_ca1", use_container_width=True, 
                     help="Default: sharp high-amplitude spikes",
                     type="primary" if st.session_state.active_preset == "EC-CA1" else "secondary"):
            st.session_state.prominence_factor = 15.0
            st.session_state.gap = 2.0
            st.session_state.min_duration = 10.0
            st.session_state.min_density = 1.0
            st.session_state.max_density = 40.0
            st.session_state.min_peak_freq = 3.0
            st.session_state.active_preset = "EC-CA1"
            st.rerun()
        
        if st.button("Frontal", key="preset_frontal", use_container_width=True, 
                     help="Smaller, more frequent spikes",
                     type="primary" if st.session_state.active_preset == "Frontal" else "secondary"):
            st.session_state.prominence_factor = 8.0
            st.session_state.gap = 1.5
            st.session_state.min_duration = 6.0
            st.session_state.min_density = 2.0
            st.session_state.max_density = 45.0
            st.session_state.min_peak_freq = 4.0
            st.session_state.active_preset = "Frontal"
            st.rerun()
    
    with preset_col2:
        if st.button("Occipital", key="preset_occipital", use_container_width=True, 
                     help="Shorter bursts, higher density",
                     type="primary" if st.session_state.active_preset == "Occipital" else "secondary"):
            st.session_state.prominence_factor = 10.0
            st.session_state.gap = 1.0
            st.session_state.min_duration = 5.0
            st.session_state.min_density = 3.0
            st.session_state.max_density = 50.0
            st.session_state.min_peak_freq = 4.5
            st.session_state.active_preset = "Occipital"
            st.rerun()
        
        if st.button("Thalamus", key="preset_thalamus", use_container_width=True, 
                     help="Very dense, short bursts",
                     type="primary" if st.session_state.active_preset == "Thalamus" else "secondary"):
            st.session_state.prominence_factor = 7.0
            st.session_state.gap = 1.0
            st.session_state.min_duration = 4.0
            st.session_state.min_density = 4.0
            st.session_state.max_density = 55.0
            st.session_state.min_peak_freq = 5.0
            st.session_state.active_preset = "Thalamus"
            st.rerun()

    if st.button("Temporal (other)", key="preset_temporal_other", use_container_width=True, 
                 help="Similar to EC-CA1 but slightly different kinetics",
                 type="primary" if st.session_state.active_preset == "Temporal (other)" else "secondary"):
        st.session_state.prominence_factor = 13.0
        st.session_state.gap = 2.5
        st.session_state.min_duration = 8.0
        st.session_state.min_density = 1.5
        st.session_state.max_density = 35.0
        st.session_state.min_peak_freq = 3.5
        st.session_state.active_preset = "Temporal (other)"
        st.rerun()

    st.markdown("---")

    st.markdown("**Channel Setup**")
    ch0_name_input = st.text_input("Channel 1 name", value=st.session_state.ch0_name, placeholder="EC")
    ch1_name_input = st.text_input("Channel 2 name", value=st.session_state.ch1_name, placeholder="CA1")
        
    if ch0_name_input != st.session_state.ch0_name:
        st.session_state.ch0_name = ch0_name_input
    if ch1_name_input != st.session_state.ch1_name:
        st.session_state.ch1_name = ch1_name_input

    st.markdown("---")
    st.markdown("**Plot Colors**")
    st.caption("Channel colors")

    new_color_ch0 = st.color_picker(
        "Channel 1", 
        st.session_state.color_ch0, 
        key="color_ch0_sidebar"
    )
    if new_color_ch0 != st.session_state.color_ch0:
        st.session_state.color_ch0 = new_color_ch0
        st.rerun()

    new_color_ch1 = st.color_picker(
        "Channel 2", 
        st.session_state.color_ch1, 
        key="color_ch1_sidebar"
    )
    if new_color_ch1 != st.session_state.color_ch1:
        st.session_state.color_ch1 = new_color_ch1
        st.rerun()

    if st.button("Reset Colors", key="reset_colors_sidebar"):
        st.session_state.color_ch0 = "#1f77b4"
        st.session_state.color_ch1 = "#2ca02c"
        st.rerun()

        
    st.markdown("---")

    st.markdown("# Analysis parameters")
    st.caption("Parameters below are adjusted with appropriate presets. Change if needed, but carefully.")

    st.markdown("---")

    st.markdown("**Signal Processing**")
    st.caption("Changing these may affect detection quality.")
    
    lowcut = st.slider("Low-pass cutoff (Hz)", 50, 500, 250, step=10, key="slider_lowcut",
                       help="Default: 250 Hz. Lower = smoother but may lose fast spikes.")
    order = st.slider("Filter order", 2, 10, 6, step=2, key="slider_order",
                      help="Default: 6. Higher = steeper roll-off but may cause ringing.")
    notch_freq = st.slider("Notch frequency (Hz)", 30, 100, 50, step=5, key="slider_notch",
                          help="Default: 50 Hz (mains). Change if your power line frequency differs.")
    q = st.slider("Notch quality factor", 50, 300, 150, step=10, key="slider_q",
                  help="Default: 150. Higher = narrower notch.")
    
    st.markdown("---")
    st.markdown("**Spike Detection**")
    st.caption("Higher values = fewer, more selective detections.")
    
    new_prominence = st.slider("Spike threshold (x noise)", 5.0, 25.0, 
                               st.session_state.prominence_factor, 
                               step=0.5, key="slider_prominence",
                               help="Default: 15.0 (EC-CA1). Higher = fewer spikes (cleaner), Lower = more spikes (more sensitive)")
    if new_prominence != st.session_state.prominence_factor:
        st.session_state.prominence_factor = new_prominence
    
    st.markdown("---")
    st.markdown("**Ictal Event Grouping**")
    st.caption("Parameters for seizure-like event detection.")
    
    new_gap = st.slider("Gap (s)", 0.5, 10.0, 
                        st.session_state.gap, step=0.5, key="slider_gap",
                        help="Default: 2.0s. Higher = groups more distant spikes, Lower = splits events more")
    if new_gap != st.session_state.gap:
        st.session_state.gap = new_gap
    
    new_min_duration = st.slider("Minimum event duration (s)", 2.0, 30.0, 
                                 st.session_state.min_duration, step=1.0, key="slider_min_duration",
                                 help="Default: 10.0s. Higher = longer events only, Lower = captures shorter bursts")
    if new_min_duration != st.session_state.min_duration:
        st.session_state.min_duration = new_min_duration
    
    new_min_density = st.slider("Minimum average density (spikes/s)", 0.1, 5.0, 
                                st.session_state.min_density, step=0.1, key="slider_min_density",
                                help="Default: 1.0 (EC-CA1). Higher = more active events only")
    if new_min_density != st.session_state.min_density:
        st.session_state.min_density = new_min_density
    
    new_max_density = st.slider("Maximum average density (spikes/s)", 10.0, 80.0, 
                                st.session_state.max_density, step=1.0, key="slider_max_density",
                                help="Default: 40.0. Higher = allows denser bursts, Lower = excludes noise")
    if new_max_density != st.session_state.max_density:
        st.session_state.max_density = new_max_density
    
    new_min_peak_freq = st.slider("Minimum peak frequency (spikes/s)", 1.0, 10.0, 
                                  st.session_state.min_peak_freq, step=0.5, key="slider_min_peak_freq",
                                  help="Default: 3.0 spikes/s. Events must reach this frequency in at least 2s window to be considered ictal. Higher = more selective.")
    if new_min_peak_freq != st.session_state.min_peak_freq:
        st.session_state.min_peak_freq = new_min_peak_freq

def load_abf_file(tmp_path, ch0_name, ch1_name):
    try:
        time, sweeps, fs, n_sweeps, n_channels = func.load_abf_sweeps(tmp_path)
        if ch0_name.strip():
            labels = [ch0_name.strip(), ch1_name.strip()]
        else:
            labels = func.get_channel_labels(n_channels)
        return time, sweeps, fs, n_sweeps, n_channels, labels
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")
        st.info("Please ensure the file is a valid .abf recording.")
        st.stop()

def process_and_save_to_disk(sweeps, fs, lowcut, notch, order, q, 
                              prominence_factor,
                              gap, min_duration, min_density, max_density,
                              min_peak_freq,
                              labels, time, n_sweeps, n_channels):
    
    data_dir = tempfile.mkdtemp()
    detect_path = os.path.join(data_dir, 'detect_sweeps.npy')
    
    np.save(detect_path, np.zeros((n_sweeps, n_channels, len(sweeps[0][0])), dtype=np.float32))
    
    filter_both = func.create_filters(fs, lowcut, notch, order, q)
    
    all_results = []
    ictal_rows = []
    
    detect_array = np.load(detect_path, mmap_mode='r+')
    
    for sweep_idx in range(n_sweeps):
        for ch_idx in range(n_channels):
            raw = sweeps[sweep_idx][ch_idx].astype(np.float32)
            detect, measure = filter_both(raw)
            detect_array[sweep_idx, ch_idx, :] = detect.astype(np.float32)
            
            channel_name = labels[ch_idx]
            
            peaks, noise_level = func.detect_spikes_adaptive(
                detect, fs,
                noise_level=None,
                prominence_factor=prominence_factor,
                min_distance_ms=75.0,
                max_amplitude=1.5,
                relative_height_threshold=0.9
            )
            peak_times = time[peaks]
            
            events, interictal = func.detect_ictal_events(
                peak_times,
                gap=gap, 
                min_duration=min_duration,
                min_density=min_density, 
                max_density=max_density, 
                min_peak_frequency=min_peak_freq
            )
            
            n_spikes = len(peaks)
            n_ictal = len(events)
            n_interictal = len(interictal)
            
            mean_amplitude = np.mean(np.abs(measure[peaks])) if n_spikes > 0 else 0
            max_amplitude_all = np.max(np.abs(measure[peaks])) if n_spikes > 0 else 0
            
            if n_ictal > 0:
                ictal_peaks_flat = [p for event in events for p in event]
                ictal_amplitude = np.mean(np.abs(measure[peaks][ictal_peaks_flat]))
                ictal_spike_count = len(ictal_peaks_flat)
            else:
                ictal_amplitude = 0
                ictal_spike_count = 0
            
            interictal_amplitude = np.mean(np.abs(measure[peaks][interictal])) if n_interictal > 0 else 0
            
            ictal_starts, ictal_durations, ictal_peaks_counts, ictal_freqs, ictal_freq_max_list, ictal_freq_min_list = func.compute_ictal_stats(events, peak_times)
            
            if n_interictal > 0 and len(interictal) > 1:
                interictal_dur = peak_times[interictal[-1]] - peak_times[interictal[0]]
                mean_interictal_freq = n_interictal / interictal_dur if interictal_dur > 0 else 0
            else:
                mean_interictal_freq = 0
            
            all_results.append({
                'sweep': sweep_idx + 1, 'channel': channel_name,
                'n_spikes': n_spikes, 'n_ictal': n_ictal, 'n_interictal': n_interictal,
                'mean_amplitude': round(mean_amplitude, 2),
                'max_amplitude': round(max_amplitude_all, 2),
                'ictal_spike_count': ictal_spike_count,
                'interictal_amplitude': round(interictal_amplitude, 2),
                'mean_interictal_freq': round(mean_interictal_freq, 2),
            })
            
            for i in range(len(ictal_starts)):
                ictal_rows.append({
                    'sweep': sweep_idx + 1, 'channel': channel_name,
                    'ictal_start': ictal_starts[i], 'duration': ictal_durations[i],
                    'n_peaks': ictal_peaks_counts[i], 'mean_freq': ictal_freqs[i],
                    'freq_max': ictal_freq_max_list[i], 'freq_min': ictal_freq_min_list[i],
                    'ictal_amplitude': round(ictal_amplitude, 2) if n_ictal > 0 else 0,
                })
            
            del detect, measure, raw, peaks
            gc.collect()
    
    del detect_array
    gc.collect()
    
    df_summary = pd.DataFrame(all_results)
    df_ictal = pd.DataFrame(ictal_rows)
    
    if os.path.exists(detect_path):
        st.session_state.data_dir = data_dir
        st.session_state.detect_path = detect_path
        st.session_state.n_sweeps = n_sweeps
        st.session_state.n_channels = n_channels
        st.session_state.fs = fs
    else:
        st.error("Failed to create detect file!")
    
    return df_summary, df_ictal

if uploaded_file is not None:
    st.success(f"Loaded: {uploaded_file.name}")

if uploaded_file is not None and st.sidebar.button("Run Analysis", type="primary"):
    colors = [st.session_state.color_ch0, st.session_state.color_ch1]
    
    with st.spinner("Loading... (~5 sec)"):
        tmp_path = f"temp_uploaded_{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if st.session_state.get("current_file") != uploaded_file.name:
            if 'data_dir' in st.session_state and st.session_state.data_dir is not None:
                try:
                    shutil.rmtree(st.session_state.data_dir, ignore_errors=True)
                except Exception as e:
                    st.warning(f"Could not remove old temporary directory: {e}")
            st.session_state.current_file = uploaded_file.name
            st.session_state.analysis_done = False
            
        time, sweeps, fs, n_sweeps, n_channels, labels = load_abf_file(
            tmp_path, 
            st.session_state.ch0_name, 
            st.session_state.ch1_name
        )
        
        sweep_duration_min = len(time) / fs / 60
        
        st.session_state.sweeps_raw = [[sweeps[i][ch].astype(np.float32) 
                                        for ch in range(n_channels)] for i in range(n_sweeps)]
        st.session_state.time = time
        st.session_state.fs = fs
        st.session_state.n_sweeps = n_sweeps
        st.session_state.n_channels = n_channels
        st.session_state.labels = labels
        st.session_state.colors = colors
        st.session_state.sweep_duration = sweep_duration_min
    
    st.success(f"Loaded: {n_sweeps} sweeps, {n_channels} channels, {fs} Hz")
    
    with st.spinner("Processing and filtering... (~60-90 sec)"):
        df_summary, df_ictal = process_and_save_to_disk(
            sweeps, fs, lowcut, notch_freq, order, q,
            st.session_state.prominence_factor,
            st.session_state.gap, st.session_state.min_duration, 
            st.session_state.min_density, st.session_state.max_density,
            st.session_state.min_peak_freq,
            labels, time, n_sweeps, n_channels
        )
    
    st.success("Processing complete.")
    
    st.session_state.df_summary = df_summary
    st.session_state.df_ictal = df_ictal
    st.session_state.labels = labels
    st.session_state.colors = colors
    st.session_state.analysis_done = True
    st.session_state.sweep_selector = None
    
    del sweeps
    gc.collect()
    
    st.rerun()

if st.session_state.get("analysis_done", False) and st.session_state.detect_path is not None:
    df_summary = st.session_state.df_summary
    df_ictal = st.session_state.df_ictal
    time = st.session_state.time
    fs = st.session_state.fs
    n_sweeps = st.session_state.n_sweeps
    n_channels = st.session_state.n_channels
    labels = st.session_state.labels
    colors = st.session_state.colors
    prominence_factor = st.session_state.prominence_factor
    sweep_duration = st.session_state.get("sweep_duration", 5)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sweep Inspector", "Time Analysis", "Frequency Analysis", "Cross-correlation", "Summary"])
    
    with tab1:
        st.header("Sweep Inspector")
        st.caption("Quick inspection of any single sweep: filtered signals, detected spikes, and key metrics for both channels.")
        
        sweep_labels = []
        for i in range(1, n_sweeps + 1):
            start_min = int(round((i - 1) * sweep_duration))
            end_min = int(round(i * sweep_duration))
            sweep_labels.append(f"{i} ({start_min}-{end_min} min)")
        
        sweep_to_view = st.selectbox("Sweep number", range(1, n_sweeps + 1), 
                                     index=None, placeholder="Choose a sweep...",
                                     format_func=lambda x: sweep_labels[x-1],
                                     key="sweep_selector")
    
        if sweep_to_view is not None:
            with st.spinner("Plotting..."):
                sweep_idx = sweep_to_view - 1
                fig, axes = plt.subplots(n_channels, 1, figsize=(16, 6 * n_channels), constrained_layout=True)
                
                start_min = int(round(sweep_idx * sweep_duration))
                end_min = int(round((sweep_idx + 1) * sweep_duration))
                fig.suptitle(f'Sweep {sweep_to_view} ({start_min}-{end_min} min)', fontsize=18, fontweight='bold')
                
                if n_channels == 1:
                    axes = [axes]
            
                detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
            
                for ch_idx in range(n_channels):
                    filt = detect_array[sweep_idx, ch_idx, :]
                    channel_name = labels[ch_idx]
                    
                    peaks, _ = func.detect_spikes_adaptive(
                        filt, fs,
                        noise_level=None,
                        prominence_factor=prominence_factor,
                        min_distance_ms=75.0,
                        max_amplitude=1.5,
                        relative_height_threshold=0.9
                    )
                    
                    summary_row = df_summary[(df_summary['sweep'] == sweep_to_view) & (df_summary['channel'] == channel_name)]
                    ictal_ch = df_ictal[(df_ictal['sweep'] == sweep_to_view) & (df_ictal['channel'] == channel_name)] if not df_ictal.empty else pd.DataFrame()
                    
                    n_spikes = int(summary_row['n_spikes'].values[0]) if not summary_row.empty else 0
                    
                    ax = axes[ch_idx]
                    ax.plot(time, filt, linewidth=0.7, color=colors[ch_idx])
                    if n_spikes > 0:
                        ax.plot(time[peaks], filt[peaks], 'x', color='red', markersize=4, label='Detected spikes')
                    ax.set_title(f'{channel_name}', fontsize=13)
                    ax.set_ylabel('mV', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    if ch_idx == n_channels - 1:
                        ax.set_xlabel('Time (s)', fontsize=10)
                    
                    if not ictal_ch.empty:
                        for _, row in ictal_ch.iterrows():
                            ax.axvspan(row['ictal_start'], row['ictal_start'] + row['duration'], 
                                       color='red', alpha=0.15, label='Ictal zone' if _ == ictal_ch.index[0] else "")
                        ax.legend(loc='best', fontsize=8)
                    elif n_spikes > 0:
                        ax.legend(loc='best', fontsize=8)
                
                del detect_array
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
        
            st.markdown("---")
            st.markdown(f"<h3 style='font-weight: bold;'>Sweep {sweep_to_view} Analytics</h3>", unsafe_allow_html=True)

            for ch_idx in range(n_channels):
                channel_name = labels[ch_idx]
                summary_row = df_summary[(df_summary['sweep'] == sweep_to_view) & (df_summary['channel'] == channel_name)]
                ictal_ch = df_ictal[(df_ictal['sweep'] == sweep_to_view) & (df_ictal['channel'] == channel_name)] if not df_ictal.empty else pd.DataFrame()

                n_spikes = int(summary_row['n_spikes'].values[0]) if not summary_row.empty else 0
                n_ictal = int(summary_row['n_ictal'].values[0]) if not summary_row.empty else 0
                n_interictal = int(summary_row['n_interictal'].values[0]) if not summary_row.empty else 0
                mean_amp = float(summary_row['mean_amplitude'].values[0]) if not summary_row.empty else 0
                max_amp = float(summary_row['max_amplitude'].values[0]) if not summary_row.empty else 0
                ictal_spike_count = int(summary_row['ictal_spike_count'].values[0]) if not summary_row.empty else 0

                st.markdown(f"<h4 style='color: {colors[ch_idx]};'>● {channel_name}</h4>", unsafe_allow_html=True)

                if n_spikes == 0:
                    st.info(f"No spikes detected in {channel_name}.")
                else:
                    if n_ictal > 0 and not ictal_ch.empty:
                        total_ictal_dur = ictal_ch['duration'].sum()
                        mean_ictal_freq = ictal_ch['mean_freq'].mean()
                        
                        metrics = [
                            ("Total spikes", str(n_spikes)),
                            ("Ictal events", str(n_ictal)),
                            ("Interictal spikes", str(n_interictal)),
                            ("Mean amplitude", f"{mean_amp:.3f} mV"),
                            ("Total ictal duration", f"{total_ictal_dur:.1f} s"),
                            ("Mean ictal frequency", f"{mean_ictal_freq:.1f} Hz"),
                            ("Ictal spike count", str(ictal_spike_count)),
                            ("Max amplitude", f"{max_amp:.3f} mV")
                        ]
                        
                        cols = st.columns(4)
                        for i, (label, value) in enumerate(metrics):
                            with cols[i % 4]:
                                st.metric(label, value)
                    else:
                        metrics = [
                            ("Total spikes", str(n_spikes)),
                            ("Interictal spikes", str(n_interictal)),
                            ("Mean amplitude", f"{mean_amp:.3f} mV"),
                            ("Max amplitude", f"{max_amp:.3f} mV")
                        ]
                        cols = st.columns(5)
                        for i, (label, value) in enumerate(metrics):
                            with cols[i]:
                                st.metric(label, value)

                st.markdown("---")
    
    with tab2:
        st.header("Time Analysis")
        st.caption("Sweep-by-sweep and cumulative dynamics: signal comparison, ictal and interictal analysis, and amplitude distributions. Sweep 1 (0-5 min) supposed to be non-active")
        st.info("Time Analysis tab content here - full implementation needed")
    
    with tab3:
        st.header("Frequency Analysis")
        st.caption("Frequency-domain and time-frequency representations: FFT power spectra and wavelet spectrograms.")
        st.info("Frequency Analysis tab content here - full implementation needed")
    
    with tab4:
        st.header("Cross-correlation")
        st.caption("Channel coupling analysis based on interictal activity. Negative lag = first channel leads; positive lag = second channel leads.")
        st.info("Cross-correlation tab content here - full implementation needed")
    
    with tab5:
        st.header("Summary")
        st.caption("Full overview of the analysis results and data export.")
        st.info("Summary tab content here - full implementation needed")