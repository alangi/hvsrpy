# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

"""Candidate acceptance helpers for the AutoHVSR extension layer."""

from dataclasses import replace
import warnings

import numpy as np

from ._autohvsr_features import candidate_feature_matrix
from ._autohvsr_models import load_xgboost_classifier
from ._autohvsr_types import AutoHvsrSettings


def _heuristic_score(peak, settings):
    nearby_frequency_support = sum(
        peak.features[f"nearby_frequency_feature_{index}"] for index in range(5)
    )
    nearby_amplitude_support = sum(
        peak.features[f"nearby_amplitude_feature_{index}"] for index in range(5)
    )
    prominence_scale = max(settings.heuristic_min_prominence, 1e-6)
    prominence_score = min(peak.peak_prominence / prominence_scale, 1.0)

    support_scale = max(settings.heuristic_min_nearby_support, 1e-6)
    frequency_support_score = min(nearby_frequency_support / support_scale, 1.0)
    amplitude_support_score = min(nearby_amplitude_support / support_scale, 1.0)

    mc_threshold = peak.mean_curve_amplitude_at_peak - (
        settings.heuristic_mean_curve_std_factor * peak.mean_curve_std_at_peak
    )
    amplitude_margin = peak.peak_amplitude - mc_threshold
    amplitude_scale = max(abs(peak.mean_curve_std_at_peak), 0.25)
    amplitude_score = np.clip(amplitude_margin / amplitude_scale, 0.0, 1.0)

    # The original AutoHVSR path used a trained classifier over peak,
    # mean-curve, and neighborhood-density features. The heuristic
    # fallback mirrors that intent by combining the same feature groups.
    return float(
        (0.35 * prominence_score)
        + (0.30 * amplitude_score)
        + (0.20 * frequency_support_score)
        + (0.15 * amplitude_support_score)
    )


def _classify_with_heuristic(candidates, settings):
    classified = []
    for peak in candidates:
        score = _heuristic_score(peak, settings)
        is_accepted = bool(
            peak.peak_prominence >= settings.heuristic_min_prominence
            and score >= settings.heuristic_score_threshold
        )
        classified.append(
            replace(
                peak,
                is_accepted=is_accepted,
                classifier_score=score,
                classifier_mode="heuristic",
            )
        )
    return classified, "heuristic"


def _classify_with_xgboost(candidates, settings):
    model = load_xgboost_classifier(
        model_path=settings.classifier_model_path,
        use_bundled_model=settings.classifier_use_bundled_model,
    )
    matrix = candidate_feature_matrix(candidates)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(matrix)[:, 1]
        accepted = scores >= settings.classifier_probability_threshold
    else:  # pragma: no cover
        accepted = model.predict(matrix).astype(bool)
        scores = accepted.astype(float)

    classified = []
    for peak, is_accepted, score in zip(candidates, accepted, scores):
        classified.append(
            replace(
                peak,
                is_accepted=bool(is_accepted),
                classifier_score=float(score),
                classifier_mode="xgboost",
            )
        )
    return classified, "xgboost"


def classify_autohvsr_candidates(candidates, settings=None):
    """Classify candidate peaks as accepted or rejected.

    Parameters
    ----------
    candidates : iterable of AutoHvsrPeak
        Candidate peaks with populated ``features``.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling the classifier backend.
        ``classifier_mode='xgboost'`` requires either
        ``classifier_model_path`` or an available bundled model when
        ``classifier_use_bundled_model`` is ``True``. ``classifier_mode='auto'``
        tries the same XGBoost resolution order first and falls back to the
        heuristic classifier with a warning.

    Returns
    -------
    tuple
        ``(classified_candidates, classifier_mode_used)`` where
        ``classified_candidates`` is a list of new ``AutoHvsrPeak``
        objects with ``is_accepted`` and ``classifier_score`` filled in.
        ``classifier_mode_used`` is ``"none"`` when no candidates are
        available, otherwise it identifies the backend that actually ran.
    """
    settings = AutoHvsrSettings() if settings is None else settings
    if len(candidates) == 0:
        return [], "none"

    missing_features = [peak for peak in candidates if len(peak.features) == 0]
    if missing_features:
        raise ValueError(
            "Candidates must have features before classification. "
            "Call build_autohvsr_features(...) first."
        )

    if settings.classifier_mode == "heuristic":
        return _classify_with_heuristic(candidates, settings)
    if settings.classifier_mode == "xgboost":
        return _classify_with_xgboost(candidates, settings)

    try:
        return _classify_with_xgboost(candidates, settings)
    except (ImportError, FileNotFoundError) as exc:
        warnings.warn(
            f"Falling back to heuristic AutoHVSR classification because the "
            f"XGBoost classifier could not be used: {exc}",
            RuntimeWarning,
        )
        return _classify_with_heuristic(candidates, settings)
