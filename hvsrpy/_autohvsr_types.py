# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.
# Copyright (C) 2019-2026 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Typed containers shared by the AutoHVSR extension modules."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .constants import DISTRIBUTION_MAP


def _validate_strictly_increasing(values, name):
    for lower, upper in zip(values[:-1], values[1:]):
        if not upper > lower:
            raise ValueError(f"{name} must be strictly increasing.")


@dataclass(frozen=True)
class AutoHvsrSettings:
    """Settings controlling AutoHVSR feature extraction and grouping.

    The feature schema is currently fixed and mirrors the bins used by the
    original AutoHVSR workflow in ``hvsrweb.py``. The edge arrays are
    therefore validated against the expected schema lengths.

    ``classifier_model_path`` takes precedence over any bundled model.
    ``classifier_use_bundled_model`` controls whether AutoHVSR should look
    for a packaged default model when no explicit path is supplied.
    """

    find_peaks_prominence: float = 0.25
    frequency_bin_edges: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    nearby_frequency_log10_edges: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
    nearby_amplitude_edges: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 10.0)
    classifier_mode: str = "heuristic"
    classifier_model_path: Optional[str] = None
    classifier_use_bundled_model: bool = True
    classifier_probability_threshold: float = 0.5
    heuristic_min_prominence: float = 0.25
    heuristic_mean_curve_std_factor: float = 1.0
    heuristic_min_nearby_support: float = 0.05
    heuristic_score_threshold: float = 0.45
    cluster_eps: float = 0.2
    cluster_min_samples: int = 10
    split_variance_gain: float = 0.02
    split_min_cluster_size: int = 6
    split_min_subcluster_size: int = 3
    resonance_distribution: str = "lognormal"

    def __post_init__(self):
        if self.classifier_mode not in ("heuristic", "xgboost", "auto"):
            raise ValueError(
                "classifier_mode must be 'heuristic', 'xgboost', or 'auto'."
            )
        if not isinstance(self.classifier_use_bundled_model, bool):
            raise ValueError("classifier_use_bundled_model must be boolean.")
        if DISTRIBUTION_MAP.get(self.resonance_distribution.lower()) is None:
            raise ValueError(
                "resonance_distribution must be one of 'normal', 'lognormal', or 'log-normal'."
            )
        if len(self.frequency_bin_edges) != 9:
            raise ValueError("frequency_bin_edges must contain exactly 9 values.")
        if len(self.nearby_frequency_log10_edges) != 6:
            raise ValueError(
                "nearby_frequency_log10_edges must contain exactly 6 values."
            )
        if len(self.nearby_amplitude_edges) != 6:
            raise ValueError("nearby_amplitude_edges must contain exactly 6 values.")
        _validate_strictly_increasing(self.frequency_bin_edges, "frequency_bin_edges")
        _validate_strictly_increasing(
            self.nearby_frequency_log10_edges,
            "nearby_frequency_log10_edges",
        )
        _validate_strictly_increasing(
            self.nearby_amplitude_edges,
            "nearby_amplitude_edges",
        )
        if any(edge <= 0 for edge in self.frequency_bin_edges):
            raise ValueError("frequency_bin_edges must contain only positive values.")
        if any(edge < 0 for edge in self.nearby_frequency_log10_edges):
            raise ValueError(
                "nearby_frequency_log10_edges must contain only non-negative values."
            )
        if any(edge < 0 for edge in self.nearby_amplitude_edges):
            raise ValueError("nearby_amplitude_edges must contain only non-negative values.")
        if self.find_peaks_prominence < 0:
            raise ValueError("find_peaks_prominence must be non-negative.")
        if self.classifier_probability_threshold < 0 or self.classifier_probability_threshold > 1:
            raise ValueError("classifier_probability_threshold must be between 0 and 1.")
        if self.heuristic_min_prominence < 0:
            raise ValueError("heuristic_min_prominence must be non-negative.")
        if self.heuristic_mean_curve_std_factor < 0:
            raise ValueError("heuristic_mean_curve_std_factor must be non-negative.")
        if self.heuristic_min_nearby_support < 0:
            raise ValueError("heuristic_min_nearby_support must be non-negative.")
        if self.heuristic_score_threshold < 0 or self.heuristic_score_threshold > 1:
            raise ValueError("heuristic_score_threshold must be between 0 and 1.")
        if self.cluster_eps <= 0:
            raise ValueError("cluster_eps must be greater than 0.")
        if self.cluster_min_samples < 1:
            raise ValueError("cluster_min_samples must be at least 1.")
        if self.split_variance_gain < 0:
            raise ValueError("split_variance_gain must be non-negative.")
        if self.split_min_cluster_size < 2:
            raise ValueError("split_min_cluster_size must be at least 2.")
        if self.split_min_subcluster_size < 1:
            raise ValueError("split_min_subcluster_size must be at least 1.")


@dataclass(frozen=True)
class AutoHvsrPeak:
    """One AutoHVSR candidate or accepted peak from a single HVSR window."""

    record_index: int
    window_index: int
    peak_index: int
    peak_frequency: float
    peak_amplitude: float
    peak_prominence: float
    mean_curve_amplitude_at_peak: float
    mean_curve_std_at_peak: float
    features: Dict[str, float] = field(default_factory=dict)
    is_accepted: Optional[bool] = None
    classifier_score: Optional[float] = None
    classifier_mode: Optional[str] = None
    resonance_id: int = -1


@dataclass(frozen=True)
class AutoHvsrResonance:
    """Summary statistics for one AutoHVSR resonance family.

    ``peak_indices`` stores the zero-based positions of member peaks within the
    clustered peak sequence passed into ``summarize_autohvsr_resonances(...)``.
    It does not refer to frequency-bin indices or original window indices.
    """

    resonance_id: int
    peak_indices: Tuple[int, ...]
    peak_count: int
    frequency_mean: float
    frequency_std: float
    amplitude_mean: float
    amplitude_std: float
    frequency_min: float
    frequency_max: float
    distribution: str


@dataclass(frozen=True)
class AutoHvsrResult:
    """Structured result returned by :func:`hvsrpy.autohvsr.process_autohvsr`.

    ``classifier_mode_used`` reports the classifier backend that actually ran.
    It is ``"none"`` when no candidate peaks were available to classify.
    """

    settings: AutoHvsrSettings
    candidates: Tuple[AutoHvsrPeak, ...]
    accepted_peaks: Tuple[AutoHvsrPeak, ...]
    rejected_peaks: Tuple[AutoHvsrPeak, ...]
    resonances: Tuple[AutoHvsrResonance, ...]
    mean_curve_peak_frequency: Optional[float]
    mean_curve_peak_amplitude: Optional[float]
    classifier_mode_used: str
    used_window_indices: Tuple[int, ...]
