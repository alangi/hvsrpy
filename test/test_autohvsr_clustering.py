# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

import numpy as np

import hvsrpy
from hvsrpy.autohvsr import (
    AutoHvsrSettings,
    build_autohvsr_features,
    classify_autohvsr_candidates,
    cluster_autohvsr_resonances,
    extract_autohvsr_candidates,
    summarize_autohvsr_resonances,
)
from testing_tools import TestCase, unittest


def _make_hvsr_with_two_resonances(n_windows=24):
    frequency = np.geomspace(0.2, 20.0, 256)
    log_frequency = np.log(frequency)
    windows = []
    for window_index in range(n_windows):
        amplitude = np.ones_like(frequency)
        low_peak = np.exp(-0.5 * ((log_frequency - np.log(0.9 + 0.03 * np.cos(window_index))) / 0.08) ** 2)
        high_peak = np.exp(-0.5 * ((log_frequency - np.log(3.8 + 0.08 * np.sin(window_index))) / 0.07) ** 2)
        amplitude += 2.7 * low_peak
        amplitude += 2.2 * high_peak
        windows.append(amplitude)
    return hvsrpy.HvsrTraditional(frequency, np.asarray(windows))


class TestAutoHvsrClustering(TestCase):

    def setUp(self):
        self.hvsr = _make_hvsr_with_two_resonances()
        self.settings = AutoHvsrSettings(
            classifier_mode="heuristic",
            cluster_min_samples=5,
            heuristic_min_nearby_support=0.02,
        )

    def test_cluster_autohvsr_resonances_orders_by_frequency(self):
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        candidates = build_autohvsr_features(self.hvsr, candidates, settings=self.settings)
        classified, _ = classify_autohvsr_candidates(candidates, settings=self.settings)
        clustered = cluster_autohvsr_resonances(classified, settings=self.settings)
        resonances = summarize_autohvsr_resonances(clustered, settings=self.settings)

        self.assertEqual(len(resonances), 2)
        self.assertLess(resonances[0].frequency_mean, resonances[1].frequency_mean)
        self.assertEqual(resonances[0].resonance_id, 0)
        self.assertEqual(resonances[1].resonance_id, 1)

    def test_cluster_autohvsr_resonances_is_deterministic(self):
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        candidates = build_autohvsr_features(self.hvsr, candidates, settings=self.settings)
        classified, _ = classify_autohvsr_candidates(candidates, settings=self.settings)
        clustered_a = cluster_autohvsr_resonances(classified, settings=self.settings)
        clustered_b = cluster_autohvsr_resonances(classified, settings=self.settings)
        self.assertListEqual(
            [peak.resonance_id for peak in clustered_a],
            [peak.resonance_id for peak in clustered_b],
        )


if __name__ == "__main__":
    unittest.main()
