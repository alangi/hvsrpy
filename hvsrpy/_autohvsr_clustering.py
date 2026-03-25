# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

"""Resonance clustering helpers for the AutoHVSR extension layer."""

from dataclasses import replace

import numpy as np
from sklearn import cluster

from ._autohvsr_types import AutoHvsrSettings


def _split_resonance_labels(log_frequencies, labels, settings):
    labels = labels.copy()
    unique_labels = sorted(label for label in set(labels) if label != -1)
    delta_label = 0
    for label in unique_labels:
        label += delta_label
        values = log_frequencies[labels == label]
        if len(values) < settings.split_min_cluster_size:
            continue
        if (np.max(values) - np.min(values)) < 1e-3:
            continue

        split_value = np.mean(values)
        left = values[values <= split_value]
        right = values[values > split_value]
        if len(left) < settings.split_min_subcluster_size:
            continue
        if len(right) < settings.split_min_subcluster_size:
            continue

        before_variance = np.var(values)
        after_variance = np.var(left) + np.var(right)
        if (before_variance - after_variance) > settings.split_variance_gain:
            labels[labels > label] += 1
            labels[np.logical_and(labels == label, log_frequencies > split_value)] += 1
            delta_label += 1
    return labels


def _order_resonance_labels(log_frequencies, labels):
    ordered = labels.copy()
    unique_labels = [label for label in sorted(set(labels)) if label != -1]
    if not unique_labels:
        return ordered

    mean_frequencies = []
    for label in unique_labels:
        mean_frequencies.append(np.mean(log_frequencies[labels == label]))

    sorted_old = np.asarray(unique_labels)[np.argsort(mean_frequencies)]
    for new_label, old_label in enumerate(sorted_old):
        ordered[labels == old_label] = new_label
    return ordered


def cluster_autohvsr_resonances(candidates, settings=None):
    """Cluster accepted peaks into resonance families ordered by frequency.

    Parameters
    ----------
    candidates : iterable of AutoHvsrPeak
        Classified peaks with ``is_accepted`` already populated.
    settings : AutoHvsrSettings, optional
        AutoHVSR settings controlling DBSCAN clustering and optional
        split refinement.

    Returns
    -------
    list of AutoHvsrPeak
        New peak objects with ``resonance_id`` assigned. Rejected peaks
        keep ``resonance_id=-1``.
    """
    settings = AutoHvsrSettings() if settings is None else settings
    if len(candidates) == 0:
        return []

    accepted_indices = [index for index, peak in enumerate(candidates) if peak.is_accepted]
    if len(accepted_indices) == 0:
        return [replace(peak, resonance_id=-1) for peak in candidates]

    log_frequencies = np.log(
        np.asarray([candidates[index].peak_frequency for index in accepted_indices], dtype=float)
    )
    dbscan = cluster.DBSCAN(
        eps=settings.cluster_eps,
        min_samples=settings.cluster_min_samples,
        metric="euclidean",
    )
    labels = dbscan.fit_predict(log_frequencies.reshape(-1, 1))
    labels = _split_resonance_labels(log_frequencies, labels, settings)
    labels = _order_resonance_labels(log_frequencies, labels)

    clustered = []
    accepted_counter = 0
    for peak in candidates:
        if not peak.is_accepted:
            clustered.append(replace(peak, resonance_id=-1))
            continue
        clustered.append(replace(peak, resonance_id=int(labels[accepted_counter])))
        accepted_counter += 1
    return clustered
