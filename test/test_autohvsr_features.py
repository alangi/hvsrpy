# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

import numpy as np

import hvsrpy
from hvsrpy.autohvsr import (
    AutoHvsrSettings,
    build_autohvsr_features,
    extract_autohvsr_candidates,
)
from testing_tools import TestCase, unittest


def _make_hvsr_with_two_resonances(n_windows=20):
    frequency = np.geomspace(0.2, 20.0, 256)
    log_frequency = np.log(frequency)
    windows = []
    for window_index in range(n_windows):
        amplitude = np.ones_like(frequency)
        scale = 1.0 + 0.04 * np.sin(window_index)
        low_peak = np.exp(-0.5 * ((log_frequency - np.log(1.0)) / 0.08) ** 2)
        high_peak = np.exp(-0.5 * ((log_frequency - np.log(4.0)) / 0.06) ** 2)
        amplitude += 2.8 * scale * low_peak
        amplitude += 2.1 * scale * high_peak
        amplitude += 0.03 * np.sin(2 * np.pi * np.linspace(0, 1, frequency.size) + window_index)
        windows.append(amplitude)
    return hvsrpy.HvsrTraditional(frequency, np.asarray(windows))


class TestAutoHvsrFeatures(TestCase):

    def setUp(self):
        self.hvsr = _make_hvsr_with_two_resonances()
        self.settings = AutoHvsrSettings(
            classifier_mode="heuristic",
            cluster_min_samples=5,
        )

    def test_extract_autohvsr_candidates(self):
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        self.assertTrue(len(candidates) >= 2 * self.hvsr.n_curves)
        frequencies = np.array([candidate.peak_frequency for candidate in candidates])
        self.assertTrue(np.any(np.abs(frequencies - 1.0) < 0.3))
        self.assertTrue(np.any(np.abs(frequencies - 4.0) < 0.5))

    def test_build_autohvsr_features(self):
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        enriched = build_autohvsr_features(self.hvsr, candidates, settings=self.settings)
        self.assertEqual(len(candidates), len(enriched))
        self.assertTrue("peak_frequency_log10" in enriched[0].features)
        self.assertTrue("nearby_frequency_feature_0" in enriched[0].features)
        self.assertTrue("time_window_feature_0" in enriched[0].features)

    def test_extract_autohvsr_candidates_respects_valid_window_mask(self):
        self.hvsr.valid_window_boolean_mask[:] = False
        self.hvsr.valid_window_boolean_mask[[2, 5]] = True
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        self.assertTrue(len(candidates) > 0)
        self.assertEqual(sorted({candidate.window_index for candidate in candidates}), [2, 5])

    def test_extract_autohvsr_candidates_with_no_valid_windows(self):
        self.hvsr.valid_window_boolean_mask[:] = False
        candidates = extract_autohvsr_candidates(self.hvsr, settings=self.settings)
        self.assertEqual(candidates, [])

    def test_extract_autohvsr_candidates_rejects_malformed_valid_window_mask(self):
        self.hvsr.valid_window_boolean_mask = np.array([True, False], dtype=bool)
        with self.assertRaisesRegex(ValueError, "valid_window_boolean_mask"):
            extract_autohvsr_candidates(self.hvsr, settings=self.settings)

    def test_settings_reject_invalid_feature_edge_lengths(self):
        with self.assertRaisesRegex(ValueError, "frequency_bin_edges"):
            AutoHvsrSettings(frequency_bin_edges=(0.01, 0.1))
        with self.assertRaisesRegex(ValueError, "nearby_frequency_log10_edges"):
            AutoHvsrSettings(nearby_frequency_log10_edges=(0.0, 0.1))
        with self.assertRaisesRegex(ValueError, "nearby_amplitude_edges"):
            AutoHvsrSettings(nearby_amplitude_edges=(0.0, 1.0))

    def test_settings_reject_non_monotonic_feature_edges(self):
        with self.assertRaisesRegex(ValueError, "frequency_bin_edges"):
            AutoHvsrSettings(
                frequency_bin_edges=(0.01, 0.03, 0.1, 0.3, 1.0, 0.9, 10.0, 30.0, 100.0)
            )
        with self.assertRaisesRegex(ValueError, "nearby_frequency_log10_edges"):
            AutoHvsrSettings(
                nearby_frequency_log10_edges=(0.0, 0.025, 0.05, 0.04, 0.2, 0.4)
            )
        with self.assertRaisesRegex(ValueError, "nearby_amplitude_edges"):
            AutoHvsrSettings(
                nearby_amplitude_edges=(0.0, 0.5, 1.0, 0.9, 4.0, 10.0)
            )

    def test_settings_reject_invalid_resonance_distribution(self):
        with self.assertRaisesRegex(ValueError, "resonance_distribution"):
            AutoHvsrSettings(resonance_distribution="gamma")


if __name__ == "__main__":
    unittest.main()
