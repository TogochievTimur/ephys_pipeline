# Copyright (c) 2026 Timur Togochiev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal as sp
from scipy.fft import fft, fftfreq
import pyabf

def load_abf_sweeps(filepath):
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
    if n_channels == 1:
        return ["Channel 1"]
    elif n_channels == 2:
        return ["EC", "CA1"]
    else:
        return [f"Channel {i+1}" for i in range(n_channels)]

def create_filters(fs, lowcut=500, notch=50, order=6, q=150):
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

def estimate_noise_mad(signal):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return mad * 1.4826

def estimate_noise_robust(signal, fs, window_duration=2.0):
    window_samples = int(window_duration * fs)
    if len(signal) < window_samples:
        return estimate_noise_mad(signal)
    
    n_windows = len(signal) // window_samples
    rms_values = []
    for i in range(n_windows):
        segment = signal[i*window_samples : (i+1)*window_samples]
        rms = np.sqrt(np.mean((segment - np.mean(segment))**2))
        rms_values.append(rms)
    
    if not rms_values:
        return estimate_noise_mad(signal)
    
    low_rms_threshold = np.percentile(rms_values, 25)
    quiet_segments = []
    for i, rms in enumerate(rms_values):
        if rms <= low_rms_threshold:
            segment = signal[i*window_samples : (i+1)*window_samples]
            quiet_segments.append(segment)
    
    if not quiet_segments:
        quiet_segments = [signal[:window_samples]]
    
    quiet_signal = np.concatenate(quiet_segments)
    return estimate_noise_mad(quiet_signal)

def detect_spikes_adaptive(signal, fs, noise_level=None, prominence_factor=12.0, 
                           min_distance_ms=75.0, max_amplitude=1.5, 
                           relative_height_threshold=0.8):
    """
    Detect spikes with adaptive thresholding and suppression of secondary peaks.
    
    Parameters:
    - signal: filtered signal
    - fs: sampling rate
    - noise_level: pre-computed noise level (if None, estimated automatically)
    - prominence_factor: how many times above noise a peak must be (default: 12.0 for EC/CA1)
    - min_distance_ms: minimum distance between peaks in ms (default: 75 ms)
    - max_amplitude: maximum allowed amplitude (artifact rejection)
    - relative_height_threshold: secondary peaks must be at least this fraction of 
                                 the main peak height (default: 0.8 = 80%)
    """
    if noise_level is None:
        noise_level = estimate_noise_robust(signal, fs)
    
    min_distance_samples = int(min_distance_ms * fs / 1000.0)
    if min_distance_samples < 1:
        min_distance_samples = 1
    
    # Find peaks in the absolute value (detects both positive and negative spikes)
    peaks, properties = sp.find_peaks(
        np.abs(signal),
        prominence=prominence_factor * noise_level,
        distance=min_distance_samples
    )
    
    if len(peaks) == 0:
        return np.array([], dtype=np.int64), noise_level
    
    # Convert to numpy array of integers
    peaks = np.array(peaks, dtype=np.int64)
    
    # Filter out artifact peaks
    peak_amps = np.abs(signal[peaks])
    valid = peak_amps < max_amplitude
    peaks = peaks[valid]
    peak_amps = peak_amps[valid]
    
    if len(peaks) == 0:
        return np.array([], dtype=np.int64), noise_level
    
    # Secondary peak suppression
    window_samples = int(min_distance_samples * 2)
    if window_samples < 1:
        window_samples = 1
    
    kept_peaks = []
    sorted_indices = np.argsort(peak_amps)[::-1]
    suppressed = np.zeros(len(peaks), dtype=bool)
    
    for i in sorted_indices:
        if suppressed[i]:
            continue
            
        kept_peaks.append(peaks[i])
        
        window_start = peaks[i] - window_samples
        window_end = peaks[i] + window_samples
        
        for j in range(len(peaks)):
            if i == j or suppressed[j]:
                continue
                
            if window_start <= peaks[j] <= window_end:
                if peak_amps[j] < peak_amps[i] * relative_height_threshold:
                    suppressed[j] = True
                elif abs(peaks[j] - peaks[i]) < min_distance_samples * 0.5:
                    suppressed[j] = True
    
    final_peaks = np.array(sorted(kept_peaks), dtype=np.int64)
    
    return final_peaks, noise_level

def spectrum_db(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    power = np.abs(yf[:N//2])**2
    return xf[:N//2], 10 * np.log10(power + 1e-12)

def detect_ictal_events(peak_times, gap=2.0, min_duration=10.0, min_density=2.0, max_density=40.0, min_peak_frequency=3.0):
    """
    Detect ictal events based on spike duration and density.
    
    Parameters:
    - peak_times: time of each spike (in seconds)
    - gap: max interval between spikes to group them into one event (seconds)
    - min_duration: minimum event duration to be considered ictal (seconds)
    - min_density: minimum average spike density within the event (spikes/second)
    - max_density: maximum average spike density within the event (spikes/second)
    - min_peak_frequency: minimum peak frequency within any 2-second window inside the event (spikes/second)
                          Default: 3.0 spikes/s. Events that never reach this frequency are rejected.
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
                # Check peak frequency within the event
                event_times = peak_times[current]
                max_freq = compute_max_frequency(event_times)
                if max_freq >= min_peak_frequency:
                    events.append(current)
            
            current = [i]

    # Check the last event
    duration = peak_times[current[-1]] - peak_times[current[0]]
    density = len(current) / duration if duration > 0 else 0
    if duration >= min_duration and min_density <= density <= max_density:
        event_times = peak_times[current]
        max_freq = compute_max_frequency(event_times)
        if max_freq >= min_peak_frequency:
            events.append(current)

    all_event_peaks = [p for event in events for p in event]
    interictal = [i for i in range(len(peak_times)) if i not in all_event_peaks]

    return events, interictal

def lowpass(signal, fs, cutoff=40, order=4):
    sos = sp.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sp.sosfiltfilt(sos, signal)

def highpass(signal, fs, lowcut=1.0, order=4):
    sos = sp.butter(order, lowcut, btype='high', fs=fs, output='sos')
    return sp.sosfiltfilt(sos, signal)

def find_best_segment_60s(signal_1, signal_2, fs, duration=60, step=10, max_amplitude=1.5):
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
                         zoom_start=None, zoom_duration=None, fs=None, dpi=300, figsize=(12, 6)):
    def shift_color(hex_color, shift):
        rgb = mcolors.hex2color(hex_color)
        if shift > 0:
            return tuple(c + (1 - c) * shift for c in rgb)
        else:
            return tuple(c * (1 + shift) for c in rgb)
    
    colors_raw = [shift_color(colors[0], -0.4), shift_color(colors[1], -0.4)]
    colors_filt = [shift_color(colors[0], 0.1), shift_color(colors[1], 0.1)]
    
    if zoom_start is not None and zoom_duration is not None and fs is not None:
        start_idx = int(zoom_start * fs)
        end_idx = int((zoom_start + zoom_duration) * fs)
        t = time[start_idx:end_idx]
    else:
        t = time
    
    fig, axes = plt.subplots(n_channels, 2, figsize=figsize)
    fig.suptitle(f'{title}', fontsize=18, fontweight='bold')
    
    for ch_idx in range(n_channels):
        if zoom_start is not None and zoom_duration is not None and fs is not None:
            raw = sweeps[sweep_idx][ch_idx][start_idx:end_idx]
            filt = detect_sweeps[sweep_idx][ch_idx][start_idx:end_idx]
        else:
            raw = sweeps[sweep_idx][ch_idx]
            filt = detect_sweeps[sweep_idx][ch_idx]
        
        axes[ch_idx, 0].plot(t, raw, linewidth=0.6, color=colors_raw[ch_idx])
        axes[ch_idx, 0].set_title(f'{labels[ch_idx]} — Raw', fontsize=12)
        axes[ch_idx, 0].set_ylabel('mV', fontsize=10)
        axes[ch_idx, 0].grid(True, alpha=0.3)
        
        axes[ch_idx, 1].plot(t, filt, linewidth=0.8, color=colors_filt[ch_idx])
        axes[ch_idx, 1].set_title(f'{labels[ch_idx]} — Filtered', fontsize=12)
        axes[ch_idx, 1].set_ylabel('mV', fontsize=10)
        axes[ch_idx, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout(pad=0.8)
    fig.set_dpi(dpi)
    return fig

def compute_max_frequency(event_times, window_duration=2.0):
    """
    Compute the maximum spike frequency within any window of given duration.
    """
    
    if len(event_times) < 3:
        return 0.0
    
    max_freq = 0.0
    start_idx = 0
    
    while start_idx < len(event_times):
        window_end = event_times[start_idx] + window_duration
        # Find all spikes within this window
        end_idx = start_idx
        while end_idx < len(event_times) and event_times[end_idx] <= window_end:
            end_idx += 1
        
        n_spikes = end_idx - start_idx
        if n_spikes >= 3:  # Need at least 3 spikes to compute meaningful frequency
            freq = n_spikes / window_duration
            if freq > max_freq:
                max_freq = freq
        
        start_idx += 1
    
    return max_freq
