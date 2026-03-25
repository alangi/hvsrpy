# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

import importlib
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

import hvsrpy
from hvsrpy.autohvsr import (
    AutoHvsrResult,
    AutoHvsrSettings,
    classify_autohvsr_candidates,
    process_autohvsr,
)
from hvsrpy._autohvsr_models import (
    get_bundled_xgboost_model_resource,
    load_xgboost_classifier,
)
from testing_tools import TestCase, unittest


def _make_hvsr_with_two_resonances(n_windows=24):
    frequency = np.geomspace(0.2, 20.0, 256)
    log_frequency = np.log(frequency)
    windows = []
    for window_index in range(n_windows):
        amplitude = np.ones_like(frequency)
        low_peak = np.exp(-0.5 * ((log_frequency - np.log(1.0)) / 0.08) ** 2)
        high_peak = np.exp(-0.5 * ((log_frequency - np.log(4.2)) / 0.06) ** 2)
        amplitude += 2.8 * (1.0 + 0.03 * np.sin(window_index)) * low_peak
        amplitude += 2.0 * (1.0 + 0.05 * np.cos(window_index)) * high_peak
        windows.append(amplitude)
    return hvsrpy.HvsrTraditional(frequency, np.asarray(windows))


def _make_hvsr_with_one_resonance(n_windows=18):
    frequency = np.geomspace(0.2, 20.0, 256)
    log_frequency = np.log(frequency)
    windows = []
    for window_index in range(n_windows):
        amplitude = np.ones_like(frequency)
        peak = np.exp(-0.5 * ((log_frequency - np.log(1.6 + 0.02 * np.sin(window_index))) / 0.08) ** 2)
        amplitude += 3.1 * peak
        windows.append(amplitude)
    return hvsrpy.HvsrTraditional(frequency, np.asarray(windows))


class TestAutoHvsr(TestCase):

    def test_public_imports(self):
        module = importlib.import_module("hvsrpy.autohvsr")
        self.assertTrue(hasattr(module, "AutoHvsrSettings"))
        self.assertTrue(hasattr(module, "AutoHvsrResult"))
        self.assertTrue(hasattr(module, "process_autohvsr"))

    def test_process_autohvsr_end_to_end(self):
        hvsr = _make_hvsr_with_two_resonances()
        settings = AutoHvsrSettings(
            classifier_mode="heuristic",
            cluster_min_samples=5,
            heuristic_min_nearby_support=0.02,
            heuristic_score_threshold=0.3,
        )
        result = process_autohvsr(hvsr, settings=settings)

        self.assertIsInstance(result, AutoHvsrResult)
        self.assertTrue(len(result.candidates) > 0)
        self.assertTrue(len(result.accepted_peaks) > 0)
        self.assertEqual(len(result.resonances), 2)
        self.assertLess(result.resonances[0].frequency_mean, result.resonances[1].frequency_mean)

    def test_process_autohvsr_no_valid_peaks(self):
        frequency = np.geomspace(0.2, 20.0, 256)
        amplitude = np.ones((20, frequency.size))
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        result = process_autohvsr(hvsr, settings=AutoHvsrSettings(classifier_mode="heuristic"))

        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(len(result.accepted_peaks), 0)
        self.assertEqual(len(result.resonances), 0)

    def test_process_autohvsr_single_resonance(self):
        hvsr = _make_hvsr_with_one_resonance()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=4,
                heuristic_score_threshold=0.3,
            ),
        )
        self.assertEqual(len(result.resonances), 1)
        self.assertEqual(result.resonances[0].resonance_id, 0)

    def test_process_autohvsr_with_no_valid_windows(self):
        hvsr = _make_hvsr_with_two_resonances()
        hvsr.valid_window_boolean_mask[:] = False
        result = process_autohvsr(hvsr, settings=AutoHvsrSettings(classifier_mode="heuristic"))
        self.assertEqual(result.used_window_indices, ())
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(len(result.resonances), 0)
        self.assertEqual(result.classifier_mode_used, "none")

    def test_process_autohvsr_raises_on_malformed_valid_window_mask(self):
        hvsr = _make_hvsr_with_two_resonances()
        hvsr.valid_window_boolean_mask = np.array([True, False, True], dtype=bool)
        with self.assertRaisesRegex(ValueError, "valid_window_boolean_mask"):
            process_autohvsr(hvsr, settings=AutoHvsrSettings(classifier_mode="heuristic"))

    def test_zero_candidate_workflow_reports_no_classifier(self):
        frequency = np.geomspace(0.2, 20.0, 256)
        amplitude = np.ones((6, frequency.size))
        hvsr = hvsrpy.HvsrTraditional(frequency, amplitude)
        result = process_autohvsr(hvsr, settings=AutoHvsrSettings(classifier_mode="auto"))
        self.assertEqual(result.classifier_mode_used, "none")
        self.assertEqual(result.accepted_peaks, ())

    def test_xgboost_mode_requires_dependency_and_model(self):
        hvsr = _make_hvsr_with_two_resonances()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=5,
                heuristic_min_nearby_support=0.02,
                heuristic_score_threshold=0.3,
            ),
        )

        settings = AutoHvsrSettings(
            classifier_mode="xgboost",
            classifier_model_path="missing_model.json",
        )
        with self.assertRaises((ImportError, FileNotFoundError)):
            classify_autohvsr_candidates(list(result.candidates), settings=settings)

    def test_xgboost_mode_requires_user_or_bundled_model(self):
        hvsr = _make_hvsr_with_two_resonances()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=5,
                heuristic_min_nearby_support=0.02,
                heuristic_score_threshold=0.3,
            ),
        )
        settings = AutoHvsrSettings(
            classifier_mode="xgboost",
            classifier_model_path=None,
            classifier_use_bundled_model=False,
        )
        with mock.patch(
            "hvsrpy._autohvsr_classification.load_xgboost_classifier",
            side_effect=FileNotFoundError("bundled model lookup is disabled"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "bundled model lookup is disabled"):
                classify_autohvsr_candidates(list(result.candidates), settings=settings)

    def test_xgboost_mode_uses_loaded_model(self):
        hvsr = _make_hvsr_with_two_resonances()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=5,
                heuristic_min_nearby_support=0.02,
                heuristic_score_threshold=0.3,
            ),
        )

        class StubModel:
            def predict_proba(self, matrix):
                probs = np.zeros((matrix.shape[0], 2), dtype=float)
                probs[:, 1] = 0.9
                probs[:, 0] = 0.1
                return probs

        settings = AutoHvsrSettings(
            classifier_mode="xgboost",
            classifier_model_path="stub-model.json",
        )
        with mock.patch(
            "hvsrpy._autohvsr_classification.load_xgboost_classifier",
            return_value=StubModel(),
        ) as loader:
            classified, mode_used = classify_autohvsr_candidates(list(result.candidates), settings=settings)

        loader.assert_called_once_with(
            model_path="stub-model.json",
            use_bundled_model=True,
        )
        self.assertEqual(mode_used, "xgboost")
        self.assertTrue(all(peak.classifier_mode == "xgboost" for peak in classified))
        self.assertTrue(all(peak.is_accepted for peak in classified))

    def test_auto_mode_uses_xgboost_when_model_loader_succeeds(self):
        hvsr = _make_hvsr_with_two_resonances()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=5,
                heuristic_min_nearby_support=0.02,
                heuristic_score_threshold=0.3,
            ),
        )

        class StubModel:
            def predict_proba(self, matrix):
                probs = np.zeros((matrix.shape[0], 2), dtype=float)
                probs[:, 1] = 0.75
                probs[:, 0] = 0.25
                return probs

        with mock.patch(
            "hvsrpy._autohvsr_classification.load_xgboost_classifier",
            return_value=StubModel(),
        ):
            classified, mode_used = classify_autohvsr_candidates(
                list(result.candidates),
                settings=AutoHvsrSettings(classifier_mode="auto"),
            )

        self.assertEqual(mode_used, "xgboost")
        self.assertTrue(all(peak.classifier_mode == "xgboost" for peak in classified))

    def test_auto_mode_falls_back_to_heuristic_when_no_model_is_available(self):
        hvsr = _make_hvsr_with_two_resonances()
        result = process_autohvsr(
            hvsr,
            settings=AutoHvsrSettings(
                classifier_mode="heuristic",
                cluster_min_samples=5,
                heuristic_min_nearby_support=0.02,
                heuristic_score_threshold=0.3,
            ),
        )

        with mock.patch(
            "hvsrpy._autohvsr_classification.load_xgboost_classifier",
            side_effect=FileNotFoundError("no bundled model"),
        ):
            with self.assertWarnsRegex(RuntimeWarning, "Falling back to heuristic AutoHVSR classification"):
                classified, mode_used = classify_autohvsr_candidates(
                    list(result.candidates),
                    settings=AutoHvsrSettings(classifier_mode="auto"),
                )

        self.assertEqual(mode_used, "heuristic")
        self.assertTrue(all(peak.classifier_mode == "heuristic" for peak in classified))

    def test_bundled_model_lookup_returns_none_when_unbundled(self):
        self.assertIsNone(get_bundled_xgboost_model_resource())

    def test_load_xgboost_classifier_uses_user_supplied_model_path(self):
        class StubClassifier:
            def __init__(self):
                self.loaded_path = None

            def load_model(self, path):
                self.loaded_path = path

        class StubXgbModule:
            XGBClassifier = StubClassifier

        model_path = Path("test") / "_autohvsr_custom_model_test.json"
        try:
            model_path.write_text("{}", encoding="utf8")
            with mock.patch.dict("sys.modules", {"xgboost": StubXgbModule()}):
                model = load_xgboost_classifier(
                    model_path=str(model_path),
                    use_bundled_model=False,
                )
        finally:
            if model_path.exists():
                model_path.unlink()

        self.assertEqual(model.loaded_path, str(model_path))

    def test_load_xgboost_classifier_rejects_missing_user_model_path(self):
        with mock.patch.dict("sys.modules", {"xgboost": mock.Mock()}):
            with self.assertRaisesRegex(FileNotFoundError, "XGBoost model file was not found"):
                load_xgboost_classifier(
                    model_path="missing-model.json",
                    use_bundled_model=False,
                )

    def test_load_xgboost_classifier_uses_bundled_model_resource(self):
        class StubClassifier:
            def __init__(self):
                self.loaded_path = None

            def load_model(self, path):
                self.loaded_path = path

        class StubXgbModule:
            XGBClassifier = StubClassifier

        model_dir = Path("test")
        model_path = model_dir / "2_xgboost_peak_classifier.json"
        try:
            model_path.write_text("{}", encoding="utf8")
            with mock.patch.dict("sys.modules", {"xgboost": StubXgbModule()}):
                with mock.patch("hvsrpy._autohvsr_models.resources.files", return_value=model_dir):
                    with mock.patch(
                        "hvsrpy._autohvsr_models.resources.as_file",
                        side_effect=lambda resource: nullcontext(resource),
                    ):
                        model = load_xgboost_classifier(
                            model_path=None,
                            use_bundled_model=True,
                        )
        finally:
            if model_path.exists():
                model_path.unlink()

        self.assertEqual(model.loaded_path, str(model_path))


if __name__ == "__main__":
    unittest.main()
