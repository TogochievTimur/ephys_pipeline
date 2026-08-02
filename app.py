# Copyright (c) 2026 Timur Togochiev

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
import gc
import os
import shutil
from functions import (
    load_abf_sweeps, 
    get_channel_labels, 
    create_filters, 
    detect_spikes, 
    find_best_segment_60s, 
    spectrum_db, 
    lowpass, 
    plot_raw_vs_filtered, 
    detect_ictal_events, 
    compute_ictal_stats
)

st.set_page_config(
    page_title="Epileptiform Activity Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>🧠 Epileptiform Activity Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Pipeline of linear analysis. Author: Timur Togochiev</p>", unsafe_allow_html=True)

with st.expander("Description"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Key Features**
        - Supports 2-channel .abf recordings
        - Low-pass, notch filtering & detrending
        - Spike detection (ictal & interictal)
        - Spectral analysis (FFT + wavelet)
        - Cross-correlation between channels
        - Intra-sweep dynamics
        - Cumulative dynamics
        - Full overview of results
        - Export: .CSV & .PNG
        """)
    with col2:
        st.markdown("""
        **How to use**
        1. Upload your .abf file in the sidebar
        2. Adjust settings if needed (defaults work for most cases)
        3. Click **Run Analysis**
        4. Explore results in the tabs:
            - **Sweep Inspector** — quick view of any sweep
            - **Time Analysis** — spike counts, amplitudes, distributions
            - **Frequency Analysis** — FFT & wavelet spectrograms
            - **Cross-correlation** — channel coupling
            - **Summary** — overview & export
        
        **Tip:** Use the sidebar to customize channel names, colors, and detection parameters.
        """)
    
    st.markdown("---")
    st.markdown("**Definitions**")
    st.markdown("""
        - **EC** — entorhinal cortex, a key brain region involved in generating ictal activity.
        - **CA1** — hippocampal area Cornu Ammonis 1, the main output region of the hippocampus.
        - **Sweep** — a single continuous recording block (e.g., 5 minutes), analyzed independently
        - **Ictal event** — a long, high-frequency burst of spikes (seizure-like discharge)
        - **Interictal spikes** — short, isolated spikes between ictal events (background hyperexcitability)
        - **Spike** — a sharp, transient deflection above the noise threshold
        """)

st.markdown("---")

st.sidebar.header("Settings")
st.sidebar.warning("Large files require 8+ GB RAM. Close other applications before running.")

uploaded_file = st.sidebar.file_uploader("Choose .abf file", type=["abf"])

st.sidebar.subheader("Channel setup")
ch0_name = st.sidebar.text_input("Channel 1 name (optional)", value="", placeholder="EC")
ch1_name = st.sidebar.text_input("Channel 2 name (optional)", value="", placeholder="CA1")

st.sidebar.subheader("Plot Colors")
st.sidebar.caption("Channel colors")

if st.session_state.get("_reset_colors"):
    st.session_state.color_ch0 = "#1f77b4"
    st.session_state.color_ch1 = "#2ca02c"
    st.session_state.color_interictal_bars = "#4E79A7"
    st.session_state.color_interictal_amp = "#2B6E7A"
    st.session_state.color_ictal_bars = "#8b3a3a"
    st.session_state.color_ictal_amp = "#D97A5C"
    st.session_state.color_ictal_duration = "#8c7a6b"
    st.session_state._reset_colors = False
      
col1, col2 = st.sidebar.columns(2)
with col1:
    color_ch0 = st.color_picker("ch0", 
        st.session_state.get("color_ch0", "#1f77b4"), key="color_ch0")
with col2:
    color_ch1 = st.color_picker("ch1", 
        st.session_state.get("color_ch1", "#2ca02c"), key="color_ch1")

if st.sidebar.button("Reset colors to defaults"):
    st.session_state._reset_colors = True
    st.rerun()

with st.sidebar.expander("Advanced settings"):
    st.warning("⚠️ These settings are pre-tuned for typical recordings. Change only if detection fails or you know what you're doing.")
    
    st.markdown("**Plot Colors (ictal/interictal markers)**")
    st.caption("Change only if default colors are hard to distinguish.")
    col_a, col_b = st.columns(2)
    with col_a:
        color_interictal_bars = st.color_picker("Interictal bars", 
            st.session_state.get("color_interictal_bars", "#4E79A7"), key="color_interictal_bars")
        color_interictal_amp = st.color_picker("Interictal amp", 
            st.session_state.get("color_interictal_amp", "#2B6E7A"), key="color_interictal_amp")
    with col_b:
        color_ictal_bars = st.color_picker("Ictal bars", 
            st.session_state.get("color_ictal_bars", "#8b3a3a"), key="color_ictal_bars")
        color_ictal_amp = st.color_picker("Ictal amp", 
            st.session_state.get("color_ictal_amp", "#D97A5C"), key="color_ictal_amp")
    color_ictal_duration = st.color_picker("Ictal duration bars", 
        st.session_state.get("color_ictal_duration", "#8c7a6b"), key="color_ictal_duration")
    
    st.markdown("---")
    st.markdown("**Signal Processing**")
    lowcut = st.slider("Low-pass cutoff (Hz)", 50, 500, 250, step=10)
    order = st.slider("Filter order", 2, 10, 6, step=2)
    q = st.slider("Notch quality factor", 50, 300, 150, step=10)
    
    st.markdown("**Spike Detection**")
    threshold_sigma = st.slider("Threshold (sigma)", 2.0, 20.0, 12.0, step=0.5)
    prominence_ch0 = st.slider("Prominence — ch0", 0.1, 2.0, 0.6, step=0.1)
    prominence_ch1 = st.slider("Prominence — ch1", 0.1, 2.0, 1.0, step=0.1)
    
    st.markdown("**Ictal Event Grouping**")
    st.caption("These define what counts as an ictal (seizure-like) event.")
    gap_ch0 = st.slider("Gap — ch0 (s)", 0.5, 10.0, 3.0, step=0.5)
    gap_ch1 = st.slider("Gap — ch1 (s)", 0.5, 10.0, 1.5, step=0.5)
    min_duration_ch0 = st.slider("Min duration - ch0 (s)", 2.0, 30.0, 10.0, step=1.0)
    min_duration_ch1 = st.slider("Min duration - ch1 (s)", 2.0, 30.0, 15.0, step=1.0)
    min_density_ch0 = st.slider("Min density — ch0 (spikes/s)", 0.5, 10.0, 2.0, step=0.5)
    min_density_ch1 = st.slider("Min density — ch1 (spikes/s)", 0.5, 10.0, 4.0, step=0.5)
    max_density = st.slider("Max density (spikes/s)", 10.0, 50.0, 20.0, step=1.0)

NOTCH = 50
sweep_duration = 5

@st.cache_resource(show_spinner=False)
def load_cached(tmp_path, ch0_name, ch1_name):
    time, sweeps, fs, n_sweeps, n_channels = load_abf_sweeps(tmp_path)
    if ch0_name.strip():
        labels = [ch0_name.strip(), ch1_name.strip()]
    else:
        labels = get_channel_labels(n_channels)
    return time, sweeps, fs, n_sweeps, n_channels, labels

def process_and_save_to_disk(sweeps, fs, lowcut, order, q, threshold_sigma, 
                              gap_ch0, gap_ch1, min_duration_ch0, min_duration_ch1,
                              min_density_ch0, min_density_ch1, max_density, 
                              prominence_ch0, prominence_ch1, labels, time, n_sweeps, n_channels):
    
    data_dir = tempfile.mkdtemp()
    detect_path = os.path.join(data_dir, 'detect_sweeps.npy')
    
    np.save(detect_path, np.zeros((n_sweeps, n_channels, len(sweeps[0][0])), dtype=np.float32))
    
    filter_both = create_filters(fs, lowcut, 50, order, q)
    
    CHANNEL_PARAMS = {
        labels[0]: {"gap": gap_ch0, "min_duration": min_duration_ch0, "min_density": min_density_ch0,
                    "max_density": max_density, "prominence": prominence_ch0},
        labels[1]: {"gap": gap_ch1, "min_duration": min_duration_ch1, "min_density": min_density_ch1,
                    "max_density": max_density, "prominence": prominence_ch1}
    }
    
    all_results = []
    ictal_rows = []
    
    detect_array = np.load(detect_path, mmap_mode='r+')
    
    for sweep_idx in range(n_sweeps):
        for ch_idx in range(n_channels):
            raw = sweeps[sweep_idx][ch_idx].astype(np.float32)
            detect, measure = filter_both(raw)
            detect_array[sweep_idx, ch_idx, :] = detect.astype(np.float32)
            
            channel_name = labels[ch_idx]
            params = CHANNEL_PARAMS[channel_name]
            
            peaks, height, noise = detect_spikes(detect, fs, threshold_sigma,
                                                  prominence_factor=params["prominence"])
            peak_times = time[peaks]
            
            events, interictal = detect_ictal_events(peak_times,
                gap=params["gap"], min_duration=params["min_duration"],
                min_density=params["min_density"], max_density=params["max_density"])
            
            n_spikes = len(peaks)
            n_ictal = len(events)
            n_interictal = len(interictal)
            
            mean_amplitude = np.mean(np.abs(measure[peaks])) if n_spikes > 0 else 0
            
            if n_ictal > 0:
                ictal_peaks_flat = [p for event in events for p in event]
                ictal_amplitude = np.mean(np.abs(measure[peaks][ictal_peaks_flat]))
            else:
                ictal_amplitude = 0
            
            interictal_amplitude = np.mean(np.abs(measure[peaks][interictal])) if n_interictal > 0 else 0
            
            ictal_starts, ictal_durations, ictal_peaks_counts, ictal_freqs, ictal_freq_max_list, ictal_freq_min_list = compute_ictal_stats(events, peak_times)
            
            if n_interictal > 0 and len(interictal) > 1:
                interictal_dur = peak_times[interictal[-1]] - peak_times[interictal[0]]
                mean_interictal_freq = n_interictal / interictal_dur if interictal_dur > 0 else 0
            else:
                mean_interictal_freq = 0
            
            all_results.append({
                'sweep': sweep_idx + 1, 'channel': channel_name,
                'n_spikes': n_spikes, 'n_ictal': n_ictal, 'n_interictal': n_interictal,
                'mean_amplitude': round(mean_amplitude, 2),
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
            
            del detect, measure, raw, peaks, height
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
    colors = [color_ch0, color_ch1]
    
    with st.spinner("Loading... (~5 sec)"):
        tmp_path = f"temp_uploaded_{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if st.session_state.get("current_file") != uploaded_file.name:
            if 'data_dir' in st.session_state:
                shutil.rmtree(st.session_state.data_dir, ignore_errors=True)
            st.session_state.current_file = uploaded_file.name
            st.session_state.analysis_done = False
            
        time, sweeps, fs, n_sweeps, n_channels, labels = load_cached(tmp_path, ch0_name, ch1_name)
        
        st.session_state.sweeps_raw = [[sweeps[i][ch].astype(np.float32) 
                                        for ch in range(n_channels)] for i in range(n_sweeps)]
        st.session_state.time = time
        st.session_state.fs = fs
        st.session_state.n_sweeps = n_sweeps
        st.session_state.n_channels = n_channels
        st.session_state.labels = labels
        st.session_state.colors = colors
    
    st.success(f"Loaded: {n_sweeps} sweeps, {n_channels} channels, {fs} Hz")
    
    CHANNEL_PARAMS = {
        labels[0]: {"gap": gap_ch0, "min_duration": min_duration_ch0, "min_density": min_density_ch0,
                    "max_density": max_density, "prominence": prominence_ch0},
        labels[1]: {"gap": gap_ch1, "min_duration": min_duration_ch1, "min_density": min_density_ch1,
                    "max_density": max_density, "prominence": prominence_ch1}
    }
    
    with st.spinner("Processing & filtering... (~60-90 sec)"):
        df_summary, df_ictal = process_and_save_to_disk(
            sweeps, fs, lowcut, order, q, threshold_sigma,
            gap_ch0, gap_ch1, min_duration_ch0, min_duration_ch1,
            min_density_ch0, min_density_ch1, max_density,
            prominence_ch0, prominence_ch1, labels, time, n_sweeps, n_channels
        )
    
    st.success("Processing complete.")
    
    st.session_state.df_summary = df_summary
    st.session_state.df_ictal = df_ictal
    st.session_state.labels = labels
    st.session_state.colors = colors
    st.session_state.CHANNEL_PARAMS = CHANNEL_PARAMS
    st.session_state.threshold_sigma = threshold_sigma
    st.session_state.analysis_done = True
    st.session_state.sweep_selector = None
    
    del sweeps
    gc.collect()

if st.session_state.get("analysis_done", False):
    df_summary = st.session_state.df_summary
    df_ictal = st.session_state.df_ictal
    time = st.session_state.time
    fs = st.session_state.fs
    n_sweeps = st.session_state.n_sweeps
    n_channels = st.session_state.n_channels
    labels = st.session_state.labels
    colors = st.session_state.colors
    CHANNEL_PARAMS = st.session_state.CHANNEL_PARAMS
    threshold_sigma = st.session_state.threshold_sigma
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sweep Inspector", "Time Analysis", "Frequency Analysis", "Cross-correlation", "Summary"])
    
    with tab1:
        st.header("Sweep Inspector")
        st.caption("Quick inspection of any single sweep: filtered signals, detected spikes, and key metrics for both channels.")
    
        sweep_labels = [f"{i} ({int((i-1)*sweep_duration)}–{int(i*sweep_duration)} min)" for i in range(1, n_sweeps + 1)]
        sweep_to_view = st.selectbox("Sweep number", range(1, n_sweeps + 1), 
                                     index=None, placeholder="Choose a sweep...",
                                     format_func=lambda x: sweep_labels[x-1],
                                     key="sweep_selector")
    
        if sweep_to_view is not None:
            with st.spinner("Plotting..."):
                sweep_idx = sweep_to_view - 1
                fig, axes = plt.subplots(n_channels, 1, figsize=(16, 6 * n_channels), constrained_layout=True)
                fig.suptitle(f'Sweep {sweep_to_view}', fontsize=25, fontweight='bold')
                if n_channels == 1:
                    axes = [axes]
            
                detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
            
                for ch_idx in range(n_channels):
                    filt = detect_array[sweep_idx, ch_idx, :]
                    channel_name = labels[ch_idx]
                    params = CHANNEL_PARAMS[channel_name]
                    
                    peaks, _, _ = detect_spikes(filt, fs, threshold_sigma, 
                                                 prominence_factor=params["prominence"])
                    
                    summary_row = df_summary[(df_summary['sweep'] == sweep_to_view) & (df_summary['channel'] == channel_name)]
                    ictal_ch = df_ictal[(df_ictal['sweep'] == sweep_to_view) & (df_ictal['channel'] == channel_name)] if not df_ictal.empty else pd.DataFrame()
                    
                    n_spikes = int(summary_row['n_spikes'].values[0]) if not summary_row.empty else 0
                    
                    ax = axes[ch_idx]
                    ax.plot(time, filt, linewidth=0.7, color=colors[ch_idx])
                    if n_spikes > 0:
                        ax.plot(time[peaks], filt[peaks], 'x', color='red', markersize=4, label='Detected spikes')
                    ax.set_title(f'{channel_name}', fontsize=16)
                    ax.set_ylabel('mV')
                    ax.grid(True, alpha=0.3)
                    if ch_idx == n_channels - 1:
                        ax.set_xlabel('Time (s)')
                    
                    if not ictal_ch.empty:
                        for _, row in ictal_ch.iterrows():
                            ax.axvspan(row['ictal_start'], row['ictal_start'] + row['duration'], 
                                       color='red', alpha=0.15, label='Ictal zone' if _ == ictal_ch.index[0] else "")
                        ax.legend(loc='upper right')
                    elif n_spikes > 0:
                        ax.legend(loc='upper right')
                
                del detect_array
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
        
            st.markdown("---")
            st.markdown(f"**Sweep {sweep_to_view} analytics**")
        
            total_spikes = 0
            for ch_idx in range(n_channels):
                channel_name = labels[ch_idx]
                summary_row = df_summary[(df_summary['sweep'] == sweep_to_view) & (df_summary['channel'] == channel_name)]
                ictal_ch = df_ictal[(df_ictal['sweep'] == sweep_to_view) & (df_ictal['channel'] == channel_name)] if not df_ictal.empty else pd.DataFrame()
            
                n_spikes = int(summary_row['n_spikes'].values[0]) if not summary_row.empty else 0
                n_ictal = int(summary_row['n_ictal'].values[0]) if not summary_row.empty else 0
                n_interictal = int(summary_row['n_interictal'].values[0]) if not summary_row.empty else 0
                mean_amp = float(summary_row['mean_amplitude'].values[0]) if not summary_row.empty else 0
                total_spikes += n_spikes
            
                st.markdown(f"**{channel_name}**")
            
                if n_spikes == 0:
                    st.info(f"No spikes detected in {channel_name}.")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total spikes", n_spikes)
                    col2.metric("Ictal events", n_ictal)
                    col3.metric("Interictal spikes", n_interictal)
                    col4.metric("Mean amplitude", f"{mean_amp:.3f} mV")
                
                    if n_ictal > 0 and not ictal_ch.empty:
                        total_ictal_dur = ictal_ch['duration'].sum()
                        mean_ictal_freq = ictal_ch['mean_freq'].mean()
                        col5, col6 = st.columns(2)
                        col5.metric("Total ictal duration", f"{total_ictal_dur:.1f} s")
                        col6.metric("Mean ictal frequency", f"{mean_ictal_freq:.1f} Hz")
        
            if total_spikes == 0:
                st.info("This sweep has no spikes at all.")
    
    with tab2:
        st.header("Time Analysis")
        st.caption("Sweep-by-sweep and cumulative dynamics: signal comparison, ictal & interictal analysis, and amplitude distributions. Sweep 1 (0-5 min) supposed to be non-active")
        
        subtab0, subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs([
            "Raw vs Filtered", "Spike Count", "Ictal Duration", 
            "Ictal vs Interictal", "Interictal Comparison", "Amplitudes", "Distribution"
        ])
        
        with subtab0:
            st.caption("Raw and filtered signals overlaid for both channels, with a 10-second zoom into the most prominent ictal event (if any).")
            with st.spinner("Plotting... (~10 sec)"):
                from functions import plot_raw_vs_filtered as plot_rf
                ictal_channels = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
                sweeps_raw = st.session_state.get('sweeps_raw', None)
        
                if sweeps_raw is not None:
                    detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
                    if len(ictal_channels) > 0 and not df_ictal.empty:
                        best_row = df_ictal.loc[df_ictal['duration'].idxmax()]
                        sweep_idx = int(best_row['sweep']) - 1
                        ictal_start = best_row['ictal_start']
                
                        fig1 = plot_rf(sweeps_raw, detect_array, time, sweep_idx, n_channels, labels,
                                       'Comparison of raw and filtered signals', colors)
                        st.pyplot(fig1)
                        plt.close(fig1)
                
                        fig2 = plot_rf(sweeps_raw, detect_array, time, sweep_idx, n_channels, labels,
                                       'Comparison of raw and filtered signals (zoomed)', colors,
                                       zoom_start=ictal_start, zoom_duration=10, fs=fs)
                        st.pyplot(fig2)
                        plt.close(fig2)
                    else:
                        sweep_idx = int(df_summary.loc[df_summary['n_interictal'].idxmax()]['sweep']) - 1
                        fig = plot_rf(sweeps_raw, detect_array, time, sweep_idx, n_channels, labels,
                                      'Comparison of raw and filtered signals', colors)
                        st.pyplot(fig)
                        plt.close(fig)
                    del detect_array
                else:
                    st.warning("Raw sweeps data not available")
                gc.collect()
        
        with subtab1:
            st.caption("Number of detected spikes per sweep for each channel, showing trends over time.")
            with st.spinner("Plotting..."):
                sweep_labels = [f"{int((i-1)*sweep_duration)}–{int(i*sweep_duration)}" for i in range(1, n_sweeps + 1)]
                data = df_summary[df_summary['sweep'] >= 2]
                
                fig, ax = plt.subplots(figsize=(16, 6))
                fig.suptitle('Spike count per sweep', fontsize=25, fontweight='bold')
                for i, ch_name in enumerate(labels):
                    channel_data = data[data['channel'] == ch_name]
                    ax.plot(channel_data['sweep'], channel_data['n_spikes'], 
                             marker='o', linestyle='-', linewidth=2, markersize=5, 
                             label=ch_name, color=colors[i])
                ax.set_xticks(range(2, n_sweeps + 1))
                ax.set_xticklabels(sweep_labels[1:], rotation=45, fontsize=14)
                ax.set_xlabel('Time interval (min)', labelpad=20)
                ax.set_ylabel('Number of spikes', labelpad=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1.5, n_sweeps + 0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
        
        with subtab2:
            st.caption("Total ictal duration per sweep for each channel. Skipped if no ictal events were detected.")
            with st.spinner("Plotting..."):
                ictal_channels = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
                if len(ictal_channels) > 0 and not df_ictal.empty:
                    n_plots = len(ictal_channels)
                    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))
                    fig.suptitle('Ictal duration per sweep', fontsize=25, fontweight='bold')
                    if n_plots == 1:
                        axes = [axes]
                    for ax, channel in zip(axes, ictal_channels):
                        ictal_ch_data = df_ictal[(df_ictal['channel'] == channel) & (df_ictal['sweep'] >= 2)]
                        dur_by_sweep = ictal_ch_data.groupby('sweep')['duration'].sum()
                        ax.bar(dur_by_sweep.index.values, dur_by_sweep.values, color=color_ictal_duration, alpha=1)
                        ax.set_xticks(range(2, n_sweeps + 1))
                        ax.set_xticklabels(sweep_labels[1:], rotation=45, fontsize=14)
                        ax.set_xlabel('Time interval (min)', labelpad=20)
                        ax.set_ylabel('Ictal duration (s)', labelpad=10)
                        ax.set_title(f'{channel}')
                        ax.set_xlim(1.5, n_sweeps + 0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    gc.collect()
        
        with subtab3:
            st.caption("Side-by-side comparison of interictal and ictal spike counts per sweep for each channel. Skipped if no ictal events were detected.")
            with st.spinner("Plotting..."):
                channels_with_ictal = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
                channels_to_plot = channels_with_ictal if len(channels_with_ictal) > 0 else []
                if len(channels_to_plot) > 0:
                    n_plots = len(channels_to_plot)
                    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))
                    fig.suptitle('Interictal vs ictal spike count', fontsize=25, fontweight='bold')
                    if n_plots == 1:
                        axes = [axes]
                    for ax, channel in zip(axes, channels_to_plot):
                        summary_ch = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                        ictal_ch = df_ictal[(df_ictal['channel'] == channel) & (df_ictal['sweep'] >= 2)]
                        ictal_peaks_by_sweep = ictal_ch.groupby('sweep')['n_peaks'].sum()
                        sweeps_all = summary_ch['sweep'].values
                        ictal_values = [ictal_peaks_by_sweep.get(sw, 0) for sw in sweeps_all]
                        ax.bar(sweeps_all - 0.2, summary_ch['n_interictal'].values, width=0.4, color=color_interictal_bars, label='Interictal')
                        ax.bar(sweeps_all + 0.2, ictal_values, width=0.4, color=color_ictal_bars, label='Ictal')
                        ax.set_xticks(range(2, n_sweeps + 1))
                        ax.set_xticklabels(sweep_labels[1:], rotation=45, fontsize=14)
                        ax.set_xlabel('Time interval (min)', labelpad=20)
                        ax.set_ylabel('Spike count', labelpad=10)
                        ax.set_title(f'{channel}')
                        ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    gc.collect()
        
        with subtab4:
            st.caption("Interictal spike counts per sweep, shown together for both channels to compare activity levels.")
            with st.spinner("Plotting..."):
                channels = df_summary['channel'].unique()
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.set_title('Interictal spikes count', fontsize=25, fontweight='bold')
                for i, channel in enumerate(channels):
                    data = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                    ax.bar(data['sweep'] + (i - (len(channels) - 1) / 2) * 0.25,
                           data['n_interictal'], width=0.25, label=channel, color=colors[i], alpha=1)
                ax.set_xticks(range(2, n_sweeps + 1))
                ax.set_xticklabels(sweep_labels[1:], rotation=45, fontsize=14)
                ax.set_xlabel('Time interval (min)', labelpad=20)
                ax.set_ylabel('Interictal spikes', labelpad=10)
                ax.legend()
                ax.set_xlim(1.5, n_sweeps + 0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
        
        with subtab5:
            st.caption("Mean spike amplitude per sweep for interictal (line) and ictal (square markers) spikes. Ictal values are shown only for sweeps containing ictal events.")
            with st.spinner("Plotting..."):
                channels = df_summary['channel'].unique()
                fig, axes = plt.subplots(len(channels), 1, figsize=(16, 6 * len(channels)))
                fig.suptitle('Spike amplitudes', fontsize=25, fontweight='bold')
                if len(channels) == 1:
                    axes = [axes]
                for ax, channel in zip(axes, channels):
                    data = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                    ictal_ch = df_ictal[df_ictal['channel'] == channel] if not df_ictal.empty else pd.DataFrame()
                    ax.plot(data['sweep'], data['interictal_amplitude'],
                            color=color_interictal_amp, linestyle='-', linewidth=1.2, alpha=0.6)
                    ax.scatter(data['sweep'], data['interictal_amplitude'],
                               label='Interictal', color=color_interictal_amp, s=50, alpha=0.7)
                    if not ictal_ch.empty:
                        ictal_amp_by_sweep = ictal_ch.groupby('sweep')['ictal_amplitude'].mean()
                        ax.scatter(ictal_amp_by_sweep.index, ictal_amp_by_sweep.values,
                                   label='Ictal', color=color_ictal_amp, s=80, marker='s', alpha=0.8)
                    ax.set_xticks(range(2, n_sweeps + 1))
                    ax.set_xticklabels(sweep_labels[1:], rotation=45, fontsize=14)
                    ax.set_xlabel('Time interval (min)', labelpad=20)
                    ax.set_ylabel('Amplitude (mV)', labelpad=10)
                    ax.set_title(f'{channel}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
        
        with subtab6:
            st.caption("Distribution of mean spike amplitudes across all sweeps for each channel. Boxes represent the interquartile range; dots show individual sweeps.")
            with st.spinner("Plotting..."):
                channels = df_summary['channel'].unique()
                data_for_box = df_summary[df_summary['mean_amplitude'] > 0]
                palette = {ch: colors[i] for i, ch in enumerate(channels)}
                fig, ax = plt.subplots(figsize=(16, 6))
                sns.boxplot(data=data_for_box, x='channel', y='mean_amplitude', palette=palette, showfliers=False)
                sns.stripplot(data=data_for_box, x='channel', y='mean_amplitude', color='k', alpha=1, size=5)
                ax.set_title('Amplitude distribution', fontweight='bold', fontsize=25)
                ax.set_ylabel('Mean amplitude (mV)')
                ax.set_xlabel('Channel')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()
    
    with tab3:
        st.header("Frequency Analysis")
        st.caption("Frequency-domain and time–frequency representations: FFT power spectra and wavelet spectrograms.")
        freq_subtab1, freq_subtab2 = st.tabs(["Spectral Power", "Wavelet Analysis"])
        
        with freq_subtab1:
            st.caption("Power spectral density (FFT) comparing background activity and an ictal segment (if available) for each channel.")
            
            with st.spinner("Plotting..."):
                from functions import spectrum_db
                
                duration = 10
                segment_len = int(duration * fs)
                
                channels = df_summary['channel'].unique()
                n_channels_freq = len(channels)
                
                fig, axes = plt.subplots(n_channels_freq, 1, figsize=(16, 6 * n_channels_freq))
                fig.suptitle('Spectral power', fontsize=25, fontweight='bold')
                if n_channels_freq == 1:
                    axes = [axes]
                
                detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
                
                for ax, channel in zip(axes, channels):
                    ch_idx = list(channels).index(channel)
                    
                    font_segment = detect_array[0, ch_idx, :segment_len]
                    
                    ictal_segment = None
                    if not df_ictal.empty:
                        ictal_ch = df_ictal[df_ictal['channel'] == channel]
                        if not ictal_ch.empty:
                            ictal_row = ictal_ch.iloc[0]
                            ictal_sweep_idx = int(ictal_row['sweep']) - 1
                            ictal_start = ictal_row['ictal_start']
                            ictal_start_idx = int((ictal_start + 2) * fs)
                            ictal_segment = detect_array[ictal_sweep_idx, ch_idx, ictal_start_idx:ictal_start_idx + segment_len]
                    
                    xf, font_db = spectrum_db(font_segment, fs)
                    
                    ax.plot(xf, font_db, label='Background', color='#8c9aa6')
                    
                    if ictal_segment is not None:
                        _, ictal_db = spectrum_db(ictal_segment, fs)
                        ax.plot(xf, ictal_db, label='Ictal', color='#2B6E7A' if ch_idx == 0 else '#D97A5C', linewidth=1.5)
                    
                    ax.set_xlim(0, 25)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power (dB)')
                    ax.set_title(f'{channel}')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower left')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                del detect_array
                gc.collect()
        
        with freq_subtab2:
            st.subheader("Wavelet Analysis of Ictal Discharge")
            st.caption("Morlet wavelet spectrogram showing how frequency content evolves before and during an ictal event. Power is normalized to the pre-ictal baseline and displayed in dB (cold = low, hot = high).")
            channels_with_ictal = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
            
            if len(channels_with_ictal) == 0 or df_ictal.empty:
                st.info("No ictal events found. Wavelet analysis requires at least one ictal event.")
            else:
                with st.spinner("Plotting..."):
                    from functions import lowpass
                    from scipy.ndimage import gaussian_filter
                    import pywt
                    from scipy import signal as sp
                    
                    duration_pre = 5
                    duration_post = 15
                    downsample_factor = 40
                    fs_down = fs / downsample_factor
                    freqs_target = np.linspace(1, 15, 100)
                    
                    fig, axes = plt.subplots(1, len(channels_with_ictal), figsize=(8 * len(channels_with_ictal), 6), squeeze=False)
                    fig.suptitle('Wavelet transform', fontsize=25, fontweight='bold')
                    axes = axes[0]
                    
                    detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
                    
                    for ax, channel in zip(axes, channels_with_ictal):
                        ictal_ch = df_ictal[df_ictal['channel'] == channel]
                        best_row = ictal_ch.loc[ictal_ch['duration'].idxmax()]
                        
                        sweep_idx = int(best_row['sweep']) - 1
                        ictal_start = best_row['ictal_start']
                        ch_idx = list(labels).index(channel)
                        
                        start_idx = int((ictal_start - duration_pre) * fs)
                        end_idx = int((ictal_start + duration_post) * fs)
                        signal_raw = detect_array[sweep_idx, ch_idx, start_idx:end_idx]
                        
                        signal = sp.detrend(signal_raw)
                        signal = lowpass(signal, fs, cutoff=40, order=4)
                        signal_down = sp.decimate(signal, downsample_factor, ftype='fir', zero_phase=True)
                        t = np.arange(len(signal_down)) / fs_down - duration_pre
                        
                        central_freq = pywt.central_frequency('morl')
                        scales = central_freq * fs_down / freqs_target
                        coef, freqs = pywt.cwt(signal_down, scales, 'morl', sampling_period=1/fs_down)
                        
                        power = np.abs(coef) ** 2
                        baseline_mask = (t >= -duration_pre) & (t < 0)
                        baseline = np.median(power[:, baseline_mask], axis=1, keepdims=True)
                        power_norm = power / (baseline + 1e-12)
                        power_db = 10 * np.log10(power_norm + 1e-12)
                        power_db = gaussian_filter(power_db, sigma=(0.8, 1.0))
                        
                        vmin, vmax = np.percentile(power_db, [10, 90])
                        
                        im = ax.imshow(power_db, aspect='auto', origin='lower',
                                       extent=[t[0], t[-1], freqs[0], freqs[-1]],
                                       cmap='turbo', vmin=vmin, vmax=vmax)
                        
                        ax.axvspan(-duration_pre, 0, color='white', alpha=0.2, zorder=2)
                        onset_line = ax.axvline(0, color='k', lw=2, ls='--', alpha=0.9, label='Ictal onset')
                        
                        ax.set_xlim(-duration_pre, duration_post)
                        ax.set_ylim(1, 15)
                        
                        xticks = [-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(['−5', '−2.5', '0', '2.5', '5', '7.5', '10', '12.5', '15'])
                        
                        ax.set_xlabel('Time relative to ictal onset (s)')
                        ax.set_ylabel('Frequency (Hz)')
                        ax.grid(False)
                        ax.set_title(f'{channel} — sweep {sweep_idx + 1}, {sweep_idx * 5}-{(sweep_idx+1)*5} min', fontsize=17)
                        ax.legend(handles=[onset_line], loc='upper right', frameon=True,
                                  framealpha=0.9, facecolor='white', edgecolor='0.3', fontsize=10)
                        
                        cbar = plt.colorbar(im, ax=ax, pad=0.02)
                        cbar.set_label('Power (dB)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    del detect_array
                    gc.collect()
    
    with tab4:
        st.header("Cross-correlation")
        st.caption("Channel coupling analysis based on interictal activity. Negative lag = first channel leads; positive lag = second channel leads.")
        
        with st.spinner("Computing cross-correlation..."):
            from scipy import signal as sp
            
            ch0_data = df_summary[df_summary['channel'] == labels[0]]
            ch1_data = df_summary[df_summary['channel'] == labels[1]]
            
            merged = pd.merge(ch0_data, ch1_data, on='sweep', suffixes=(f'_{labels[0]}', f'_{labels[1]}'))
            
            min_spikes_ch0 = 15
            min_spikes_ch1 = 30
            
            valid_sweeps = merged[(merged[f'n_interictal_{labels[0]}'] >= min_spikes_ch0) &
                                  (merged[f'n_interictal_{labels[1]}'] >= min_spikes_ch1)]
            
            all_corr_norm = []
            all_lags = None
            good_sweeps = 0
            
            detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
            
            for sweep_idx in valid_sweeps['sweep'].values:
                sweep_idx = int(sweep_idx) - 1
                
                sig0_full = detect_array[sweep_idx, 0, :]
                sig1_full = detect_array[sweep_idx, 1, :]
                
                start_idx, end_idx = find_best_segment_60s(sig0_full, sig1_full, fs, duration=60, step=10)
                
                step = 10
                sig0_segment = sig0_full[start_idx:end_idx:step]
                sig1_segment = sig1_full[start_idx:end_idx:step]
                
                spikes0 = np.sum(np.abs(sig0_segment) > 0.02)
                spikes1 = np.sum(np.abs(sig1_segment) > 0.02)
                
                if spikes0 < 50 or spikes1 < 50:
                    continue
                
                fs_down = fs / step
                
                sig0 = sp.detrend(sig0_segment)
                sig1 = sp.detrend(sig1_segment)
                
                sig0 = sig0 - np.mean(sig0)
                sig1 = sig1 - np.mean(sig1)
                
                corr = sp.correlate(sig0, sig1, mode='same', method='direct')
                corr_norm = corr / (np.std(sig0) * np.std(sig1) * len(sig0))
                lags = sp.correlation_lags(len(sig0), len(sig1), mode='same') / fs_down
                
                max_lag = 0.1
                lag_range = (lags >= -max_lag) & (lags <= max_lag)
                lags_near_zero = lags[lag_range]
                corr_near_zero = corr_norm[lag_range]
                
                max_corr = np.max(corr_near_zero)
                lag_max = lags_near_zero[np.argmax(corr_near_zero)]
                
                if max_corr > 0.5:
                    continue
                
                if abs(lag_max) < 0.005:
                    continue
                
                if max_corr < 0.05:
                    continue
                
                all_corr_norm.append(corr_norm)
                if all_lags is None:
                    all_lags = lags
                good_sweeps += 1
            
            del detect_array
            gc.collect()
            
            if good_sweeps < 2:
                st.warning(f"Insufficient data for cross-correlation averaging. Only {good_sweeps} sweep(s) passed all quality criteria (minimum 2 required).")
                
                with st.expander("Quality criteria for cross-correlation"):
                    st.markdown(f"""
                    Each sweep must pass all of the following to be included in the average:
                    
                    1. **≥50 spikes** detected in the selected 60-second segment for both channels
                    2. **Peak correlation <0.5** — excludes sweeps with suspiciously high correlation (likely artifact)
                    3. **Peak lag ≥5 ms** — excludes sweeps where the correlation peak is at or near zero
                    4. **Peak correlation >0.05** — excludes sweeps with negligible correlation
                    
                    *{good_sweeps} of {len(valid_sweeps)} valid sweeps passed these criteria.*
                    """)
            else:
                mean_corr_norm = np.mean(all_corr_norm, axis=0)
                
                lag_range = (all_lags >= -0.1) & (all_lags <= 0.1)
                lags_near = all_lags[lag_range]
                corr_near = mean_corr_norm[lag_range]
                
                max_corr = np.max(corr_near)
                lag_max = lags_near[np.argmax(corr_near)]
                
                fig, ax = plt.subplots(figsize=(16, 6))
                fig.suptitle('Cross-correlation', fontsize=25, fontweight='bold')
                
                ax.axvspan(-0.1, 0, color=colors[0], alpha=0.2, label=f'{labels[0]} leads')
                ax.axvspan(0, 0.1, color=colors[1], alpha=0.2, label=f'{labels[1]} leads')
                
                for corr_ind in all_corr_norm:
                    ax.plot(all_lags, corr_ind, linewidth=0.5, color='gray', alpha=0.6)
                
                ax.plot(all_lags, mean_corr_norm, linewidth=2.5, color='black', alpha=0.9, label='Mean')
                
                ax.scatter([lag_max], [max_corr], color='lime' if lag_max > 0 else 'aquamarine', 
                          s=100, edgecolor='black', zorder=5,
                          label=f'Peak at {lag_max*1000:.1f} ms')
                
                ax.axvline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.2)
                ax.axvline(lag_max, color='darkred', linestyle='--', alpha=1, linewidth=2)
                ax.set_xlim(-0.1, 0.1)
                ax.set_xlabel('Lag (s)')
                ax.set_ylabel('Normalized correlation')
                ax.legend()
                ax.grid(True, alpha=0.2)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                gc.collect()

                st.session_state.cross_corr_result = {
                    'good_sweeps': good_sweeps,
                    'max_corr': max_corr,
                    'lag_max': lag_max
                }
                st.session_state.cross_corr_fig = fig
            
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Averaged sweeps", good_sweeps)
                col2.metric("Max correlation", f"{max_corr:.3f}")
                col3.metric("Lag at peak", f"{lag_max*1000:.1f} ms")
                
                st.markdown("---")
                
                if max_corr > 0.15 and abs(lag_max) < 0.05:
                    st.success(f"**Significant correlation detected**  \n{labels[0]} and {labels[1]} show synchronized activity (r = {max_corr:.2f}, lag = {lag_max*1000:.1f} ms)")
                elif max_corr > 0.10 and abs(lag_max) < 0.1:
                    st.warning(f"**Weak correlation detected**  \nPossible weak connection (r = {max_corr:.2f}, lag = {lag_max*1000:.1f} ms)")
                else:
                    st.info(f"**No significant correlation detected**  \n{labels[0]} and {labels[1]} appear to be independent in this recording (r = {max_corr:.2f}, lag = {lag_max*1000:.1f} ms)")
    
    with tab5:
        st.header("Summary")
        st.caption("Full overview of the analysis results and data export.")
        
        st.markdown("---")
        
        total_spikes_ch0 = int(df_summary[df_summary['channel'] == labels[0]]['n_spikes'].sum())
        total_spikes_ch1 = int(df_summary[df_summary['channel'] == labels[1]]['n_spikes'].sum())
        total_ictal_ch0 = int(df_summary[df_summary['channel'] == labels[0]]['n_ictal'].sum())
        total_ictal_ch1 = int(df_summary[df_summary['channel'] == labels[1]]['n_ictal'].sum())
        total_interictal_ch0 = int(df_summary[df_summary['channel'] == labels[0]]['n_interictal'].sum())
        total_interictal_ch1 = int(df_summary[df_summary['channel'] == labels[1]]['n_interictal'].sum())
        
        ictal_sweeps = df_summary[df_summary['n_ictal'] > 0]['sweep'].unique()
        ictal_sweeps_list = sorted(ictal_sweeps) if len(ictal_sweeps) > 0 else []
        
        mean_amp_ch0 = df_summary[df_summary['channel'] == labels[0]]['mean_amplitude'].mean()
        mean_amp_ch1 = df_summary[df_summary['channel'] == labels[1]]['mean_amplitude'].mean()
        
        has_ictal = len(ictal_sweeps_list) > 0
        sweeps_raw = st.session_state.get('sweeps_raw', None)
        has_artifact = False
        if sweeps_raw is not None:
            has_artifact = any(np.max(np.abs(sweeps_raw[i][ch])) > 1.5 for i in range(n_sweeps) for ch in range(n_channels))
        
        st.markdown(f"""
        **Analysis of `{uploaded_file.name if uploaded_file else 'your file'}` completed successfully.**
        
        The recording contains **{n_sweeps} sweeps** ({n_sweeps * sweep_duration} minutes total).
        
        {f'Sweeps **{", ".join(map(str, ictal_sweeps_list))}** contain ictal events — long, high-frequency discharges characteristic of seizure-like activity. ' if has_ictal else 'No ictal events were detected in any sweep. The recording appears to be interictal or quiescent.'}
        
        {f'**{labels[0]}** showed **{total_ictal_ch0} ictal event(s)** and **{total_interictal_ch0} interictal spikes**, while **{labels[1]}** showed **{total_ictal_ch1} ictal event(s)** and **{total_interictal_ch1} interictal spikes**. ' if has_ictal else f'**{labels[0]}** recorded **{total_spikes_ch0} spikes** (all interictal), **{labels[1]}** recorded **{total_spikes_ch1} spikes** (all interictal). '}
        
        {f'**{labels[0]}** dominates with ictal activity, while **{labels[1]}** shows predominantly interictal spiking. This pattern is consistent with the expected EC–CA1 relationship, where ictal events originate in EC and propagate to the hippocampus. ' if total_ictal_ch0 > total_ictal_ch1 and 'EC' in labels and 'CA1' in labels else ''}
        """.replace('{labels[0]}', labels[0]).replace('{labels[1]}', labels[1]))
        
        if has_artifact:
            st.warning("**Technical note:** Some sweeps contain amplitude excursions above 1.5 mV, suggesting possible movement artifacts or electrical interference. These sweeps were included in the analysis but flagged for manual review.")
        
        if not df_ictal.empty and has_ictal:
            mean_ictal_freq = df_ictal['mean_freq'].mean()
            max_ictal_freq = df_ictal['mean_freq'].max()
            min_ictal_freq = df_ictal['mean_freq'].min()
            mean_ictal_dur = df_ictal['duration'].mean()
            total_ictal_dur = df_ictal['duration'].sum()
            
            st.markdown(f"""
            **Ictal characteristics:**  
            Mean ictal frequency: **{mean_ictal_freq:.1f} Hz** (range: {min_ictal_freq:.1f}–{max_ictal_freq:.1f} Hz)  
            Mean ictal duration: **{mean_ictal_dur:.1f} s** (total: {total_ictal_dur:.0f} s across all sweeps)
            """)
        
        st.markdown(f"""
        **Spike amplitudes:**  
        Mean amplitude in **{labels[0]}**: **{mean_amp_ch0:.3f} mV**  
        Mean amplitude in **{labels[1]}**: **{mean_amp_ch1:.3f} mV**  
        {'The amplitude distribution (see Distribution plot in Time Analysis) shows the spread of values across sweeps.' if total_spikes_ch0 + total_spikes_ch1 > 0 else 'No spikes were detected, so amplitude statistics are not available.'}
        
        **Total spike counts:**  
        **{labels[0]}**: {total_spikes_ch0} spikes{' — highly active' if total_spikes_ch0 > 1000 else ''}  
        **{labels[1]}**: {total_spikes_ch1} spikes{' — highly active' if total_spikes_ch1 > 1000 else ''}
        """)
        
        st.markdown("**Cross-correlation:**")
        if 'cross_corr_result' in st.session_state:
            cc = st.session_state.cross_corr_result
            if cc['good_sweeps'] >= 2:
                lag_ms = cc['lag_max'] * 1000
                direction = f"{labels[0]} leads {labels[1]}" if lag_ms < 0 else f"{labels[1]} leads {labels[0]}"
                st.markdown(f"""
                Averaged over **{cc['good_sweeps']} sweeps**:  
                Max correlation: **{cc['max_corr']:.3f}** at lag **{lag_ms:.1f} ms**  
                Direction: **{direction}**  
                {'This indicates significant synchronized activity between the channels.' if cc['max_corr'] > 0.15 else 'The correlation is weak, suggesting limited coupling during interictal periods.'}
                """)
            else:
                st.markdown("Insufficient sweeps with stable lag for cross-correlation averaging. Channels may be independent during interictal periods, or the recording may have too few interictal spikes for reliable analysis.")
        else:
            st.markdown("Cross-correlation was not computed. (See cross-correlation tab)")
        
        st.markdown("**Frequency analysis** (see Frequency Analysis tab) shows the spectral power distribution. Background activity typically peaks at low frequencies, while ictal segments show increased power across a broader range.")
        
        early_spikes = df_summary[df_summary['sweep'] <= n_sweeps // 3]['n_spikes'].sum()
        late_spikes = df_summary[df_summary['sweep'] > 2 * n_sweeps // 3]['n_spikes'].sum()
        if early_spikes > 1.5 * late_spikes:
            st.markdown("**Sweep dynamics:** Spike activity decreases from early to late sweeps, which may reflect network rundown or stabilization over the recording.")
        elif late_spikes > 1.5 * early_spikes:
            st.markdown("**Sweep dynamics:** Spike activity increases toward later sweeps, possibly indicating progressive hyperexcitability.")
        
        st.markdown("---")
        
        st.subheader("Export Results")
        st.caption("Download all figures and tables as a ZIP archive.")
        
        with st.spinner("Preparing download..."):
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("tables/sweep_summary.csv", df_summary.to_csv(index=False, sep=';', decimal=','))
                if not df_ictal.empty:
                    zip_file.writestr("tables/ictal_events.csv", df_ictal.to_csv(index=False, sep=';', decimal=','))
                
                from functions import plot_raw_vs_filtered as plot_rf, spectrum_db, lowpass
                
                ictal_chs = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
                sweeps_raw = st.session_state.get('sweeps_raw', None)
                detect_array = np.load(st.session_state.detect_path, mmap_mode='r')
                
                if len(ictal_chs) > 0 and not df_ictal.empty:
                    best_row = df_ictal.loc[df_ictal['duration'].idxmax()]
                    sweep_idx_rf = int(best_row['sweep']) - 1
                    ictal_start = best_row['ictal_start']
                    fig1 = plot_rf(sweeps_raw, detect_array, time, sweep_idx_rf, n_channels, labels,
                                   'Comparison of raw and filtered signals', colors)
                    fig1.savefig(zip_file.open('time_plots/raw_vs_filt.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(fig1)
                    fig2 = plot_rf(sweeps_raw, detect_array, time, sweep_idx_rf, n_channels, labels,
                                   'Comparison of raw and filtered signals (zoomed)', colors,
                                   zoom_start=ictal_start, zoom_duration=10, fs=fs)
                    fig2.savefig(zip_file.open('time_plots/raw_vs_filt_zoom.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                else:
                    sweep_idx_rf = int(df_summary.loc[df_summary['n_interictal'].idxmax()]['sweep']) - 1
                    fig = plot_rf(sweeps_raw, detect_array, time, sweep_idx_rf, n_channels, labels,
                                  'Comparison of raw and filtered signals', colors)
                    fig.savefig(zip_file.open('time_plots/raw_vs_filt.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
                del detect_array
                gc.collect()
                
                sweep_labels_exp = [f"{int((i-1)*sweep_duration)}-{int(i*sweep_duration)}" for i in range(1, n_sweeps + 1)]
                data_exp = df_summary[df_summary['sweep'] >= 2]
                fig, ax = plt.subplots(figsize=(16, 6))
                fig.suptitle('Spike count per sweep', fontsize=25, fontweight='bold')
                for i, ch_name in enumerate(labels):
                    channel_data = data_exp[data_exp['channel'] == ch_name]
                    ax.plot(channel_data['sweep'], channel_data['n_spikes'], marker='o', linestyle='-', linewidth=2, markersize=5, label=ch_name, color=colors[i])
                ax.set_xticks(range(2, n_sweeps + 1))
                ax.set_xticklabels(sweep_labels_exp[1:], rotation=45, fontsize=14)
                ax.set_xlabel('Time interval (min)', labelpad=20)
                ax.set_ylabel('Number of spikes', labelpad=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1.5, n_sweeps + 0.5)
                plt.tight_layout()
                fig.savefig(zip_file.open('time_plots/spike_count.png', 'w'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                gc.collect()
                
                if len(ictal_chs) > 0 and not df_ictal.empty:
                    n_plots = len(ictal_chs)
                    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))
                    fig.suptitle('Ictal duration per sweep', fontsize=25, fontweight='bold')
                    if n_plots == 1:
                        axes = [axes]
                    for ax, channel in zip(axes, ictal_chs):
                        ictal_ch_data = df_ictal[(df_ictal['channel'] == channel) & (df_ictal['sweep'] >= 2)]
                        dur_by_sweep = ictal_ch_data.groupby('sweep')['duration'].sum()
                        ax.bar(dur_by_sweep.index.values, dur_by_sweep.values, color=color_ictal_duration, alpha=1)
                        ax.set_xticks(range(2, n_sweeps + 1))
                        ax.set_xticklabels(sweep_labels_exp[1:], rotation=45, fontsize=14)
                        ax.set_xlabel('Time interval (min)', labelpad=20)
                        ax.set_ylabel('Ictal duration (s)', labelpad=10)
                        ax.set_title(f'{channel}')
                        ax.set_xlim(1.5, n_sweeps + 0.5)
                    plt.tight_layout()
                    fig.savefig(zip_file.open('time_plots/ictal_duration.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    gc.collect()
                
                channels_with_ictal_exp = df_summary[df_summary['n_ictal'] > 0]['channel'].unique()
                channels_to_plot_exp = channels_with_ictal_exp if len(channels_with_ictal_exp) > 0 else []
                if len(channels_to_plot_exp) > 0:
                    n_plots = len(channels_to_plot_exp)
                    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))
                    fig.suptitle('Interictal vs ictal spike count', fontsize=25, fontweight='bold')
                    if n_plots == 1:
                        axes = [axes]
                    for ax, channel in zip(axes, channels_to_plot_exp):
                        summary_ch = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                        ictal_ch = df_ictal[(df_ictal['channel'] == channel) & (df_ictal['sweep'] >= 2)]
                        ictal_peaks_by_sweep = ictal_ch.groupby('sweep')['n_peaks'].sum()
                        sweeps_all = summary_ch['sweep'].values
                        ictal_values = [ictal_peaks_by_sweep.get(sw, 0) for sw in sweeps_all]
                        ax.bar(sweeps_all - 0.2, summary_ch['n_interictal'].values, width=0.4, color=color_interictal_bars, label='Interictal')
                        ax.bar(sweeps_all + 0.2, ictal_values, width=0.4, color=color_ictal_bars, label='Ictal')
                        ax.set_xticks(range(2, n_sweeps + 1))
                        ax.set_xticklabels(sweep_labels_exp[1:], rotation=45, fontsize=14)
                        ax.set_xlabel('Time interval (min)', labelpad=20)
                        ax.set_ylabel('Spike count', labelpad=10)
                        ax.set_title(f'{channel}')
                        ax.legend()
                    plt.tight_layout()
                    fig.savefig(zip_file.open('time_plots/ictal_vs_interictal.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    gc.collect()
                
                channels_exp = df_summary['channel'].unique()
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.set_title('Interictal spikes count', fontsize=25, fontweight='bold')
                for i, channel in enumerate(channels_exp):
                    data_ch = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                    ax.bar(data_ch['sweep'] + (i - (len(channels_exp) - 1) / 2) * 0.25,
                           data_ch['n_interictal'], width=0.25, label=channel, color=colors[i], alpha=1)
                ax.set_xticks(range(2, n_sweeps + 1))
                ax.set_xticklabels(sweep_labels_exp[1:], rotation=45, fontsize=14)
                ax.set_xlabel('Time interval (min)', labelpad=20)
                ax.set_ylabel('Interictal spikes', labelpad=10)
                ax.legend()
                ax.set_xlim(1.5, n_sweeps + 0.5)
                plt.tight_layout()
                fig.savefig(zip_file.open('time_plots/interictal_count.png', 'w'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                gc.collect()
                
                fig, axes = plt.subplots(len(channels_exp), 1, figsize=(16, 6 * len(channels_exp)))
                fig.suptitle('Spike amplitudes', fontsize=25, fontweight='bold')
                if len(channels_exp) == 1:
                    axes = [axes]
                for ax, channel in zip(axes, channels_exp):
                    data_ch = df_summary[(df_summary['channel'] == channel) & (df_summary['sweep'] >= 2)]
                    ictal_ch = df_ictal[df_ictal['channel'] == channel] if not df_ictal.empty else pd.DataFrame()
                    ax.plot(data_ch['sweep'], data_ch['interictal_amplitude'],
                            color=color_interictal_amp, linestyle='-', linewidth=1.2, alpha=0.6)
                    ax.scatter(data_ch['sweep'], data_ch['interictal_amplitude'],
                               label='Interictal', color=color_interictal_amp, s=50, alpha=0.7)
                    if not ictal_ch.empty:
                        ictal_amp_by_sweep = ictal_ch.groupby('sweep')['ictal_amplitude'].mean()
                        ax.scatter(ictal_amp_by_sweep.index, ictal_amp_by_sweep.values,
                                   label='Ictal', color=color_ictal_amp, s=80, marker='s', alpha=0.8)
                    ax.set_xticks(range(2, n_sweeps + 1))
                    ax.set_xticklabels(sweep_labels_exp[1:], rotation=45, fontsize=14)
                    ax.set_xlabel('Time interval (min)', labelpad=20)
                    ax.set_ylabel('Amplitude (mV)', labelpad=10)
                    ax.set_title(f'{channel}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                plt.tight_layout()
                fig.savefig(zip_file.open('time_plots/spike_amplitudes.png', 'w'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                gc.collect()
                
                data_for_box_exp = df_summary[df_summary['mean_amplitude'] > 0]
                palette_exp = {ch: colors[i] for i, ch in enumerate(channels_exp)}
                fig, ax = plt.subplots(figsize=(16, 6))
                sns.boxplot(data=data_for_box_exp, x='channel', y='mean_amplitude', palette=palette_exp, showfliers=False)
                sns.stripplot(data=data_for_box_exp, x='channel', y='mean_amplitude', color='k', alpha=1, size=5)
                ax.set_title('Amplitude distribution', fontweight='bold', fontsize=25)
                ax.set_ylabel('Mean amplitude (mV)')
                ax.set_xlabel('Channel')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(zip_file.open('time_plots/amplitude_boxplot.png', 'w'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                gc.collect()
                
                channels_freq = df_summary['channel'].unique()
                fig, axes = plt.subplots(len(channels_freq), 1, figsize=(16, 6 * len(channels_freq)))
                fig.suptitle('Spectral power', fontsize=25, fontweight='bold')
                if len(channels_freq) == 1:
                    axes = [axes]
                
                detect_array_freq = np.load(st.session_state.detect_path, mmap_mode='r')
                
                for ax, channel in zip(axes, channels_freq):
                    ch_idx = list(channels_freq).index(channel)
                    font_segment = detect_array_freq[0, ch_idx, :int(10 * fs)]
                    xf, font_db = spectrum_db(font_segment, fs)
                    ax.plot(xf, font_db, label='Background', color='#8c9aa6')
                    if not df_ictal.empty:
                        ictal_ch = df_ictal[df_ictal['channel'] == channel]
                        if not ictal_ch.empty:
                            ictal_row = ictal_ch.iloc[0]
                            ictal_sweep_idx = int(ictal_row['sweep']) - 1
                            ictal_start_f = ictal_row['ictal_start']
                            ictal_start_idx_f = int((ictal_start_f + 2) * fs)
                            ictal_segment = detect_array_freq[ictal_sweep_idx, ch_idx, ictal_start_idx_f:ictal_start_idx_f + int(10 * fs)]
                            _, ictal_db = spectrum_db(ictal_segment, fs)
                            ax.plot(xf, ictal_db, label='Ictal', color='#2B6E7A' if ch_idx == 0 else '#D97A5C', linewidth=1.5)
                    ax.set_xlim(0, 25)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power (dB)')
                    ax.set_title(f'{channel}')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower left')
                plt.tight_layout()
                fig.savefig(zip_file.open('freq_plots/spectral_power.png', 'w'), dpi=300, bbox_inches='tight')
                plt.close(fig)
                del detect_array_freq
                gc.collect()
                
                if len(ictal_chs) > 0 and not df_ictal.empty:
                    from scipy.ndimage import gaussian_filter
                    import pywt
                    duration_pre = 5
                    duration_post = 15
                    downsample_factor = 80
                    fs_down = fs / downsample_factor
                    freqs_target = np.linspace(1, 15, 50)
                    fig, axes = plt.subplots(1, len(ictal_chs), figsize=(8 * len(ictal_chs), 6), squeeze=False)
                    fig.suptitle('Wavelet transform', fontsize=25, fontweight='bold')
                    axes = axes[0]
                    
                    detect_array_wavelet = np.load(st.session_state.detect_path, mmap_mode='r')
                    
                    for ax, channel in zip(axes, ictal_chs):
                        ictal_ch_w = df_ictal[df_ictal['channel'] == channel]
                        best_row_w = ictal_ch_w.loc[ictal_ch_w['duration'].idxmax()]
                        sweep_idx_w = int(best_row_w['sweep']) - 1
                        ictal_start_w = best_row_w['ictal_start']
                        ch_idx_w = list(labels).index(channel)
                        start_idx_w = int((ictal_start_w - duration_pre) * fs)
                        end_idx_w = int((ictal_start_w + duration_post) * fs)
                        signal_raw = detect_array_wavelet[sweep_idx_w, ch_idx_w, start_idx_w:end_idx_w]
                        signal = sp.detrend(signal_raw)
                        signal = lowpass(signal, fs, cutoff=40, order=4)
                        signal_down = sp.decimate(signal, downsample_factor, ftype='fir', zero_phase=False)
                        t = np.arange(len(signal_down)) / fs_down - duration_pre
                        central_freq = pywt.central_frequency('morl')
                        scales = central_freq * fs_down / freqs_target
                        coef, freqs = pywt.cwt(signal_down, scales, 'morl', sampling_period=1/fs_down)
                        power = np.abs(coef) ** 2
                        baseline_mask = (t >= -duration_pre) & (t < 0)
                        baseline = np.median(power[:, baseline_mask], axis=1, keepdims=True)
                        power_norm = power / (baseline + 1e-12)
                        power_db = 10 * np.log10(power_norm + 1e-12)
                        vmin, vmax = np.percentile(power_db, [10, 90])
                        im = ax.imshow(power_db, aspect='auto', origin='lower',
                                       extent=[t[0], t[-1], freqs[0], freqs[-1]],
                                       cmap='turbo', vmin=vmin, vmax=vmax)
                        ax.axvspan(-duration_pre, 0, color='white', alpha=0.2, zorder=2)
                        ax.axvline(0, color='k', lw=2, ls='--', alpha=0.9)
                        ax.set_xlim(-duration_pre, duration_post)
                        ax.set_ylim(1, 15)
                        ax.set_xlabel('Time relative to ictal onset (s)')
                        ax.set_ylabel('Frequency (Hz)')
                        ax.set_title(f'{channel} — sweep {sweep_idx_w + 1}')
                    plt.tight_layout()
                    fig.savefig(zip_file.open('freq_plots/wavelet.png', 'w'), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    del detect_array_wavelet
                    gc.collect()
                
                if 'cross_corr_fig' in st.session_state:
                    st.session_state.cross_corr_fig.savefig(zip_file.open('corr/cross_correlation.png', 'w'), dpi=300, bbox_inches='tight')
                    plt.close(st.session_state.cross_corr_fig)
                
                plt.close('all')
                gc.collect()
            
            zip_buffer.seek(0)
        
        st.download_button(
            label="Download all results (ZIP)",
            data=zip_buffer,
            file_name="analysis_results.zip",
            mime="application/zip",
            type="primary"
        )
        
        plt.close('all')
        gc.collect()
        
        st.markdown("---")
        st.markdown("*For the most detailed view, use the tabs above to explore individual plots interactively.*")
else:
    if uploaded_file is None:
        st.markdown("<h3 style='text-align: center; color: white;'>Upload an .abf file in the sidebar to start!</h3>", unsafe_allow_html=True)
