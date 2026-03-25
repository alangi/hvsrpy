# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

"""Optional AutoHVSR post-processing for already-processed HVSR results."""

import numpy as np

from . import statistics
from ._autohvsr_classification import classify_autohvsr_candidates
from ._autohvsr_clustering import cluster_autohvsr_resonances
from ._autohvsr_features import (
    build_autohvsr_features,
    extract_autohvsr_candidates,
)
from ._autohvsr_types import (
    AutoHvsrPeak,
    AutoHvsrResonance,
    AutoHvsrResult,
    AutoHvsrSettings,
)

__all__ = [
    "AutoHvsrSettings",
    "AutoHvsrPeak",
    "AutoHvsrResonance",
    "AutoHvsrResult",
    "extract_autohvsr_candidates",
    "build_autohvsr_features",
    "classify_autohvsr_candidates",
    "cluster_autohvsr_resonances",
    "summarize_autohvsr_resonances",
    "process_autohvsr",
]


def _used_window_indices(hvsr):
    n_windows = np.asarray(hvsr.amplitude).shape[0]
    if hasattr(hvsr, "valid_window_boolean_mask"):
        mask = np.asarray(hvsr.valid_window_boolean_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != n_windows:
            raise ValueError(
                "valid_window_boolean_mask must be one-dimensional and match the number of HVSR windows."
            )
        return tuple(np.where(mask)[0].tolist())
    return tuple(np.arange(n_windows).tolist())


def _safe_weighted_std(distribution, values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(statistics._nanstd_weighted(distribution, values))


def _summarize_one_resonance(peaks, peak_indices, resonance_id, distribution):
    frequencies = np.asarray([peak.peak_frequency for peak in peaks], dtype=float)
    amplitudes = np.asarray([peak.peak_amplitude for peak in peaks], dtype=float)
    return AutoHvsrResonance(
        resonance_id=int(resonance_id),
        peak_indices=tuple(peak_indices),
        peak_count=int(len(peaks)),
        frequency_mean=float(statistics._nanmean_weighted(distribution, frequencies)),
        frequency_std=_safe_weighted_std(distribution, frequencies),
        amplitude_mean=float(statistics._nanmean_weighted(distribution, amplitudes)),
        amplitude_std=_safe_weighted_std(distribution, amplitudes),
        frequency_min=float(np.min(frequencies)),
        frequency_max=float(np.max(frequencies)),
        distribution=distribution,
    )


def summarize_autohvsr_resonances(peaks, settings=None):
    """Summarize clustered AutoHVSR peaks into resonance statistics.

    Parameters
    ----------
    peaks : iterable of AutoHvsrPeak
        Clustered AutoHVSR peaks.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling the resonance summary
        distribution.

    Returns
    -------
    list of AutoHvsrResonance
        Resonance summaries sorted by ``resonance_id``.
    """
    settings = AutoHvsrSettings() if settings is None else settings
    if len(peaks) == 0:
        return []

    resonances = []
    unique_resonance_ids = [
        resonance_id
        for resonance_id in sorted({peak.resonance_id for peak in peaks})
        if resonance_id != -1
    ]
    for resonance_id in unique_resonance_ids:
        member_indices = [index for index, peak in enumerate(peaks) if peak.resonance_id == resonance_id]
        members = [peaks[index] for index in member_indices]
        resonances.append(
            _summarize_one_resonance(
                members,
                peak_indices=member_indices,
                resonance_id=resonance_id,
                distribution=settings.resonance_distribution,
            )
        )
    return resonances


def process_autohvsr(hvsr, settings=None, record_index=0):
    """Run the full AutoHVSR workflow on an already-processed HVSR result.

    Parameters
    ----------
    hvsr : HvsrTraditional-like
        Already-processed HVSR result. The preferred input is
        ``HvsrTraditional``.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling candidate extraction,
        classification, and clustering.
    record_index : int, optional
        Identifier stored on extracted peaks for multi-record workflows.

    Returns
    -------
    AutoHvsrResult
        Structured result containing all candidates, accepted peaks,
        rejected peaks, resonance summaries, and high-level metadata.

    Raises
    ------
    ValueError
        If ``valid_window_boolean_mask`` exists but does not match the number
        of windows in ``hvsr.amplitude``.
    """
    settings = AutoHvsrSettings() if settings is None else settings
    candidates = extract_autohvsr_candidates(hvsr, settings=settings, record_index=record_index)
    candidates = build_autohvsr_features(hvsr, candidates, settings=settings)
    classified_candidates, classifier_mode_used = classify_autohvsr_candidates(
        candidates,
        settings=settings,
    )
    clustered_candidates = cluster_autohvsr_resonances(
        classified_candidates,
        settings=settings,
    )
    resonances = summarize_autohvsr_resonances(clustered_candidates, settings=settings)

    accepted_peaks = tuple(peak for peak in clustered_candidates if peak.is_accepted)
    rejected_peaks = tuple(peak for peak in clustered_candidates if peak.is_accepted is False)
    mean_curve_peak_frequency = None
    mean_curve_peak_amplitude = None
    used_window_indices = _used_window_indices(hvsr)
    if used_window_indices and hasattr(hvsr, "mean_curve_peak"):
        try:
            mean_curve_peak_frequency, mean_curve_peak_amplitude = hvsr.mean_curve_peak(
                distribution=settings.resonance_distribution
            )
            mean_curve_peak_frequency = float(mean_curve_peak_frequency)
            mean_curve_peak_amplitude = float(mean_curve_peak_amplitude)
        except Exception:
            mean_curve_peak_frequency = None
            mean_curve_peak_amplitude = None

    return AutoHvsrResult(
        settings=settings,
        candidates=tuple(clustered_candidates),
        accepted_peaks=accepted_peaks,
        rejected_peaks=rejected_peaks,
        resonances=tuple(resonances),
        mean_curve_peak_frequency=mean_curve_peak_frequency,
        mean_curve_peak_amplitude=mean_curve_peak_amplitude,
        classifier_mode_used=classifier_mode_used,
        used_window_indices=used_window_indices,
    )
