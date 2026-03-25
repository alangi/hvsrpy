# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

"""Feature extraction helpers for the AutoHVSR extension layer."""

from dataclasses import replace

import numpy as np
from scipy.signal import find_peaks

from ._autohvsr_types import AutoHvsrPeak, AutoHvsrSettings


AUTOHVSR_FEATURE_NAMES = (
    "peak_frequency_log10",
    "peak_amplitude",
    "peak_prominence",
    "mean_curve_amplitude_at_peak",
    "mean_curve_std_at_peak",
    "time_window_feature_0",
    "time_window_feature_1",
    "time_window_feature_2",
    "time_window_feature_3",
    "time_window_feature_4",
    "time_window_feature_5",
    "time_window_feature_6",
    "time_window_feature_7",
    "nearby_frequency_feature_0",
    "nearby_frequency_feature_1",
    "nearby_frequency_feature_2",
    "nearby_frequency_feature_3",
    "nearby_frequency_feature_4",
    "nearby_amplitude_feature_0",
    "nearby_amplitude_feature_1",
    "nearby_amplitude_feature_2",
    "nearby_amplitude_feature_3",
    "nearby_amplitude_feature_4",
)


def safe_mean(amplitude, frequency, f_min, f_max):
    """Return the mean amplitude between frequency bounds or zero."""
    boolean_mask = np.logical_and(frequency > f_min, frequency < f_max)
    if np.sum(boolean_mask) == 0:
        return 0.0
    return float(np.mean(amplitude[boolean_mask]))


def _valid_window_indices(hvsr):
    n_windows = np.asarray(hvsr.amplitude).shape[0]
    if hasattr(hvsr, "valid_window_boolean_mask"):
        mask = np.asarray(hvsr.valid_window_boolean_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != n_windows:
            raise ValueError(
                "valid_window_boolean_mask must be one-dimensional and match the number of HVSR windows."
            )
        return np.where(mask)[0]
    return np.arange(n_windows)


def _validate_autohvsr_input(hvsr):
    """Validate the minimal HvsrTraditional-like interface required here."""
    required = ("frequency", "amplitude", "mean_curve", "std_curve")
    missing = [name for name in required if not hasattr(hvsr, name)]
    if missing:
        raise TypeError(
            f"hvsr is missing required AutoHVSR attributes/methods: {', '.join(missing)}."
        )

    frequency = np.asarray(hvsr.frequency, dtype=float)
    amplitude = np.asarray(hvsr.amplitude, dtype=float)
    if frequency.ndim != 1:
        raise TypeError("hvsr.frequency must be one-dimensional.")
    if amplitude.ndim != 2:
        raise TypeError(
            "AutoHVSR currently requires an HVSR object with two-dimensional amplitude data."
        )
    if amplitude.shape[1] != frequency.size:
        raise TypeError("hvsr.frequency and hvsr.amplitude are not shape-compatible.")


def extract_autohvsr_candidates(hvsr, settings=None, record_index=0):
    """Extract candidate peaks from valid HVSR windows.

    Parameters
    ----------
    hvsr : HvsrTraditional-like
        Already-processed HVSR result exposing ``frequency``,
        window-level ``amplitude``, ``mean_curve()``, and ``std_curve()``.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling peak prominence.
    record_index : int, optional
        Identifier stored on returned peaks for multi-record workflows.

    Returns
    -------
    list of AutoHvsrPeak
        Candidate peaks from windows marked valid by
        ``valid_window_boolean_mask`` when that mask is present. A malformed
        mask raises ``ValueError`` instead of silently falling back to all
        windows.
    """
    _validate_autohvsr_input(hvsr)
    settings = AutoHvsrSettings() if settings is None else settings

    frequency = np.asarray(hvsr.frequency, dtype=float)
    amplitudes = np.asarray(hvsr.amplitude, dtype=float)
    valid_indices = _valid_window_indices(hvsr)
    if len(valid_indices) == 0:
        return []
    mean_curve = np.asarray(hvsr.mean_curve(), dtype=float)
    if len(valid_indices) > 1:
        mean_std_curve = np.asarray(hvsr.std_curve(), dtype=float)
    else:
        mean_std_curve = np.zeros_like(mean_curve, dtype=float)

    peak_data = []
    for window_index in valid_indices:
        amplitude = amplitudes[window_index]
        peak_ids, metadata = find_peaks(
            amplitude,
            prominence=settings.find_peaks_prominence,
        )
        prominences = metadata.get("prominences", np.zeros_like(peak_ids, dtype=float))
        for peak_id, prominence in zip(peak_ids, prominences):
            peak_data.append(
                AutoHvsrPeak(
                    record_index=record_index,
                    window_index=int(window_index),
                    peak_index=int(peak_id),
                    peak_frequency=float(frequency[peak_id]),
                    peak_amplitude=float(amplitude[peak_id]),
                    peak_prominence=float(prominence),
                    mean_curve_amplitude_at_peak=float(mean_curve[peak_id]),
                    mean_curve_std_at_peak=float(mean_std_curve[peak_id]),
                )
            )
    return peak_data


def build_autohvsr_features(hvsr, candidates, settings=None):
    """Build the AutoHVSR feature set for each candidate peak.

    Parameters
    ----------
    hvsr : HvsrTraditional-like
        Source of the window-level HVSR curves.
    candidates : iterable of AutoHvsrPeak
        Candidate peaks returned by :func:`extract_autohvsr_candidates`.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling the feature bins.

    Returns
    -------
    list of AutoHvsrPeak
        New peak objects with ``features`` populated.
    """
    _validate_autohvsr_input(hvsr)
    settings = AutoHvsrSettings() if settings is None else settings
    if len(candidates) == 0:
        return []

    frequency = np.asarray(hvsr.frequency, dtype=float)
    amplitudes = np.asarray(hvsr.amplitude, dtype=float)
    valid_indices = _valid_window_indices(hvsr)
    n_series = max(len(valid_indices), 1)

    all_frequencies = np.asarray([peak.peak_frequency for peak in candidates], dtype=float)
    all_amplitudes = np.asarray([peak.peak_amplitude for peak in candidates], dtype=float)

    time_window_feature_map = {}
    for window_index in valid_indices:
        window_amplitude = amplitudes[window_index]
        values = []
        for f_min, f_max in zip(settings.frequency_bin_edges[:-1], settings.frequency_bin_edges[1:]):
            values.append(safe_mean(window_amplitude, frequency, f_min, f_max))
        time_window_feature_map[int(window_index)] = values

    peaks_with_features = []
    for peak in candidates:
        time_window_values = time_window_feature_map[int(peak.window_index)]
        features = {
            "peak_frequency_log10": float(np.log10(peak.peak_frequency)),
            "peak_amplitude": float(peak.peak_amplitude),
            "peak_prominence": float(peak.peak_prominence),
            "mean_curve_amplitude_at_peak": float(peak.mean_curve_amplitude_at_peak),
            "mean_curve_std_at_peak": float(peak.mean_curve_std_at_peak),
        }
        for index, value in enumerate(time_window_values):
            features[f"time_window_feature_{index}"] = float(value)

        for index, (min_edge, max_edge) in enumerate(
            zip(settings.nearby_frequency_log10_edges[:-1], settings.nearby_frequency_log10_edges[1:])
        ):
            distances = np.abs(np.log10(all_frequencies) - np.log10(peak.peak_frequency))
            nearby = np.sum(np.logical_and(distances > min_edge, distances < max_edge))
            features[f"nearby_frequency_feature_{index}"] = float(nearby / n_series)

        for index, (min_edge, max_edge) in enumerate(
            zip(settings.nearby_amplitude_edges[:-1], settings.nearby_amplitude_edges[1:])
        ):
            distances = np.abs(all_amplitudes - peak.peak_amplitude)
            nearby = np.sum(np.logical_and(distances > min_edge, distances < max_edge))
            features[f"nearby_amplitude_feature_{index}"] = float(nearby / n_series)

        peaks_with_features.append(replace(peak, features=features))

    return peaks_with_features


def candidate_feature_matrix(candidates):
    """Return the stable AutoHVSR feature matrix used by classifiers."""
    if len(candidates) == 0:
        return np.empty((0, len(AUTOHVSR_FEATURE_NAMES)), dtype=float)

    matrix = np.empty((len(candidates), len(AUTOHVSR_FEATURE_NAMES)), dtype=float)
    for row_index, peak in enumerate(candidates):
        for column_index, name in enumerate(AUTOHVSR_FEATURE_NAMES):
            matrix[row_index, column_index] = float(peak.features[name])
    return matrix
