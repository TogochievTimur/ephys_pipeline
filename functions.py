# Copyright (c) 2026 Timur Togochiev

# FUNCTIONS FOR EPILEPTIFORM ACTIVITY

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp
from scipy.fft import fft, fftfreq
import pyabf

def load_abf_sweeps(filepath):
    """
    Load all sweeps from an .abf file.
    Returns: time array, list of sweeps (each with all channels), sampling rate, number of sweeps, number of channels.
    """
    abf = pyabf.ABF(filepath)
    fs = abf.dataRate
    n_sweeps = abf.sweepCount
    n_channels = abf.channelCount

    all_sweeps = []
    for sweep_idx in range(n_sweeps):
        sweep_channels = []
        for ch_idx in range(n_channels):
            abf.setSweep(sweep_idx, channel=ch_idx)
            sweep_channels.append(abf.sweepY)
        all_sweeps.append(sweep_channels)

    first_channel = all_sweeps[0][0]
    total_points = len(first_channel)
    time_per_sweep = np.arange(total_points) / fs

    return time_per_sweep, all_sweeps, fs, n_sweeps, n_channels
    

def get_channel_labels(n_channels):
    """
    Return default labels for channels.
    1 channel → 'Channel 1'
    2 channels → 'EC', 'CA1'
    (Assumed that in file 1 channel is EC, 2 channel is CA1)
    More → 'Channel 1', 'Channel 2', ...
    """
    if n_channels == 1:
        return ["Channel 1"]
    elif n_channels == 2:
        return ["EC", "CA1"]
    else:
        return [f"Channel {i+1}" for i in range(n_channels)]
        
def create_filters(fs, lowcut=500, notch=50, order=6, q=150):
    """
    Create and return a filter that processes a signal
    into two versions: one for spike detection
    and one for amplitude measurement
    """
    b_lp, a_lp = sp.butter(order, lowcut, 'low', fs=fs)
    b_n, a_n = sp.iirnotch(notch, q, fs=fs)
    
    def filter_both(signal):
        notched = sp.filtfilt(b_n, a_n, signal)
        measure = sp.detrend(notched, type='linear')
        detect = sp.filtfilt(b_lp, a_lp, notched)
        detect = sp.filtfilt(b_n, a_n, detect)
        detect = sp.detrend(detect, type='linear')
        return detect, measure
    
    return filter_both
    

def estimate_noise(signal):
    """
    Estimate noise level using MAD (Median Absolute Deviation).
    Robust to spikes — does not overestimate noise in the presence of spikes.
    1.4826 is the constant of transforming from MAD to sigma
    """
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return mad * 1.4826

def detect_spikes(signal, fs, sigma=10.0, prominence_factor=1.0, distance=200, max_amplitude=1.5):
    """
    Detect spikes. Rejects artifact peaks with amplitude above max_amplitude (mV).
    """
    noise_level = estimate_noise(signal)
    height = sigma * noise_level
    peaks, _ = sp.find_peaks(
         np.abs(signal),
         prominence=prominence_factor * height,
         distance=distance
    )
    if len(peaks) > 0:
        peak_amps = np.abs(signal[peaks])
        valid = peak_amps < max_amplitude
        peaks = peaks[valid]
    
    return peaks, height, noise_level


def spectrum_db(signal, fs):
    """
    Function for spectral power
    analysis and demonstrating power
    via dB. Using FFT -
    Fast Fourier Transform
    """
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    power = np.abs(yf[:N//2])**2
    return xf[:N//2], 10 * np.log10(power + 1e-12)

def detect_ictal_events(peak_times, gap=1.0, min_duration=10.0, min_density=3.0, max_density=20.0):
    """
    Detects ictal events based on spike duration and density.
    peak_times — time of each spike (in seconds)
    gap — max interval between spikes to group them into one event (seconds)
    min_duration — minimum event duration to be considered ictal (seconds)
    min_density — minimum spike density within the event (spikes/second)
    max_density — maximum spike density within the event (spikes/second) — to exclude noise
    """
    if len(peak_times) == 0:
        return [], []

    events = []
    current = [0]

    for i in range(1, len(peak_times)):
        if peak_times[i] - peak_times[i-1] < gap:
            current.append(i)
        else:
            duration = peak_times[current[-1]] - peak_times[current[0]]
            density = len(current) / duration if duration > 0 else 0
            
            if duration >= min_duration and min_density <= density <= max_density:
                events.append(current)
            
            current = [i]

    duration = peak_times[current[-1]] - peak_times[current[0]]
    density = len(current) / duration if duration > 0 else 0
    if duration >= min_duration and min_density <= density <= max_density:
        events.append(current)

    all_event_peaks = [p for event in events for p in event]
    interictal = [i for i in range(len(peak_times)) if i not in all_event_peaks]

    return events, interictal

def lowpass(signal, fs, cutoff=40, order=4):
    """
    Low-pass filter
    for wavelet transform
    """
    sos = sp.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sp.sosfiltfilt(sos, signal)

def highpass(signal, fs, lowcut=1.0, order=4):
    """
    High-pass filter to remove slow drift.
    """
    sos = sp.butter(order, lowcut, btype='high', fs=fs, output='sos')
    return sp.sosfiltfilt(sos, signal)

def find_best_segment_60s(signal_1, signal_2, fs, duration=60, step=10, max_amplitude=1.5):
    """
    Find the most active 60-second segment, excluding artifact segments
    with amplitude above max_amplitude.
    """
    window = int(duration * fs)
    step_samples = int(step * fs)
    scores = []
    for i in range(0, len(signal_1) - window, step_samples):
        seg1 = signal_1[i:i+window]
        seg2 = signal_2[i:i+window]
        if np.max(np.abs(seg1)) < max_amplitude and np.max(np.abs(seg2)) < max_amplitude:
            score = np.sum(np.abs(seg1)) + np.sum(np.abs(seg2))
            scores.append((i, score))
    if not scores:
        return 0, window
    best_idx = max(scores, key=lambda x: x[1])[0]
    return best_idx, best_idx + window

def compute_ictal_stats(events, peak_times):
    """
    Compute per-event statistics for ictal events.
    Returns lists: starts, durations, peak_counts, mean_freqs, freq_maxs, freq_mins.
    """
    starts = []
    durations = []
    peak_counts = []
    mean_freqs = []
    freq_maxs = []
    freq_mins = []

    for event in events:
        event_times = peak_times[event]
        start = event_times[0]
        end = event_times[-1]
        duration = end - start
        n_peaks = len(event)
        freq = n_peaks / duration if duration > 0 else 0

        starts.append(round(start, 2))
        durations.append(round(duration, 2))
        peak_counts.append(n_peaks)
        mean_freqs.append(round(freq, 2))

        if duration > 2.0:
            window_dur = 2.0
            w_start = start
            freqs = []
            while w_start < end:
                w_end = w_start + window_dur
                w_peaks = [p for p in event if w_start <= peak_times[p] < w_end]
                if len(w_peaks) > 2:
                    freqs.append(len(w_peaks) / window_dur)
                w_start = w_end
            freq_maxs.append(round(max(freqs) if freqs else freq, 2))
            freq_mins.append(round(min(freqs) if freqs else freq, 2))
        else:
            freq_maxs.append(round(freq, 2))
            freq_mins.append(round(freq, 2))

    return starts, durations, peak_counts, mean_freqs, freq_maxs, freq_mins

def plot_raw_vs_filtered(sweeps, detect_sweeps, 
                         time, sweep_idx, n_channels, labels, title, colors,
                         zoom_start=None, zoom_duration=None, fs=None, dpi=300):
    import matplotlib.colors as mcolors
    
    def shift_color(hex_color, shift):
        rgb = mcolors.hex2color(hex_color)
        if shift > 0:
            return tuple(c + (1 - c) * shift for c in rgb)
        else:
            return tuple(c * (1 + shift) for c in rgb)
    
    colors_raw = [shift_color(colors[0], -0.4), shift_color(colors[1], -0.4)]
    colors_filt = [shift_color(colors[0], 0.1), shift_color(colors[1], 0.1)]
    
    if zoom_start is not None:
        start_idx = int(zoom_start * fs)
        end_idx = int((zoom_start + zoom_duration) * fs)
        t = time[start_idx:end_idx]
        zoom_str = ' (zoomed)'
        target = 5000
    else:
        t = time
        zoom_str = ''
        target = 10000
    
    fig, axes = plt.subplots(n_channels, 2, figsize=(16, 8))
    fig.suptitle(f'{title}{zoom_str}', fontsize=20, fontweight='bold')
    
    for ch_idx in range(n_channels):
        if zoom_start is not None:
            raw = sweeps[sweep_idx][ch_idx][start_idx:end_idx]
            filt = detect_sweeps[sweep_idx][ch_idx][start_idx:end_idx]
        else:
            raw = sweeps[sweep_idx][ch_idx]
            filt = detect_sweeps[sweep_idx][ch_idx]
        
        if len(raw) > target:
            step = len(raw) // target
            raw_plot = raw[::step]
            t_raw = t[::step]
        else:
            raw_plot = raw
            t_raw = t
            
        if len(filt) > target:
            step = len(filt) // target
            filt_plot = filt[::step]
            t_filt = t[::step]
        else:
            filt_plot = filt
            t_filt = t
        
        axes[ch_idx, 0].plot(t_raw, raw_plot, linewidth=0.6, color=colors_raw[ch_idx])
        axes[ch_idx, 0].set_title(f'{labels[ch_idx]} — Raw{zoom_str}')
        axes[ch_idx, 0].set_ylabel('mV')
        axes[ch_idx, 0].grid(True, alpha=0.3)
        
        axes[ch_idx, 1].plot(t_filt, filt_plot, linewidth=0.8, color=colors_filt[ch_idx])
        axes[ch_idx, 1].set_title(f'{labels[ch_idx]} — Filtered{zoom_str}')
        axes[ch_idx, 1].set_ylabel('mV')
        axes[ch_idx, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    plt.tight_layout(pad=1.0)
    fig.set_dpi(dpi)
    return fig
