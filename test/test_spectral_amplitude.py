# This file is part of hvsrpy, a Python package for
# horizontal-to-vertical spectral ratio processing.
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
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

import logging
import importlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hvsrpy
from hvsrpy.spectral_amplitude import (
    compute_fourier_amplitude_spectra,
    smooth_fourier_amplitude_spectra,
    plot_fourier_amplitude_spectra,
)
from testing_tools import unittest, TestCase

logger = logging.getLogger("hvsrpy")
logger.setLevel(level=logging.CRITICAL)


class TestSpectralAmplitude(TestCase):

    @classmethod
    def setUpClass(cls):
        dt = 0.01
        t = np.arange(0, 10, dt)
        f0 = 2.0
        ns = hvsrpy.TimeSeries(np.sin(2*np.pi*f0*t), dt)
        ew = hvsrpy.TimeSeries(0.5*np.sin(2*np.pi*f0*t), dt)
        vt = hvsrpy.TimeSeries(0.25*np.sin(2*np.pi*f0*t), dt)
        record = hvsrpy.SeismicRecording3C(ns, ew, vt)
        cls.records = [record, record]
        cls.settings = hvsrpy.HvsrTraditionalProcessingSettings(
            fft_settings=dict(n=2048)
        )

    def test_module_clean_import(self):
        module = importlib.import_module("hvsrpy.spectral_amplitude")
        self.assertTrue(hasattr(module, "compute_fourier_amplitude_spectra"))
        self.assertTrue(hasattr(module, "smooth_fourier_amplitude_spectra"))
        self.assertTrue(hasattr(module, "plot_fourier_amplitude_spectra"))

    def test_compute_fourier_amplitude_spectra(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        self.assertTrue("frequency" in spectra)
        self.assertEqual(spectra["ns"].shape[0], len(self.records))
        self.assertEqual(spectra["ew"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["vt"].shape, spectra["ns"].shape)
        self.assertEqual(spectra["horizontal"].shape, spectra["ns"].shape)

        # frequency index 0 is DC; peak should be near 2 Hz.
        peak_idx = np.argmax(spectra["ns"][0, 1:]) + 1
        self.assertAlmostEqual(spectra["frequency"][peak_idx], 2.0, places=1)
        self.assertArrayAlmostEqual(
            spectra["horizontal"][0],
            np.sqrt(spectra["ns"][0] * spectra["ew"][0]),
        )

    def test_smooth_fourier_amplitude_spectra_from_settings(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 32),
        )
        smoothed = smooth_fourier_amplitude_spectra(spectra, settings=self.settings)

        self.assertArrayAlmostEqual(
            smoothed["frequency"],
            self.settings.smoothing["center_frequencies_in_hz"],
        )
        self.assertEqual(smoothed["ns"].shape, (len(self.records), 32))
        self.assertEqual(smoothed["ew"].shape, (len(self.records), 32))
        self.assertEqual(smoothed["vt"].shape, (len(self.records), 32))

    def test_smooth_fourier_amplitude_spectra_nyquist_guard(self):
        spectra = compute_fourier_amplitude_spectra(self.records, self.settings)
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.array([1000.0]),
        )
        with self.assertRaises(ValueError):
            smooth_fourier_amplitude_spectra(
                spectra,
                settings=self.settings,
            )

    def test_plot_fourier_amplitude_spectra(self):
        spectra = compute_fourier_amplitude_spectra(
            self.records,
            self.settings,
            include_horizontal=True,
        )
        fig, ax = plot_fourier_amplitude_spectra(spectra, include_horizontal=True)
        self.assertEqual(len(ax.get_lines()), 4)
        plt.close(fig)

    def test_compute_fourier_amplitude_spectra_with_smoothing(self):
        self.settings.smoothing = dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 20, 16),
        )
        spectra = compute_fourier_amplitude_spectra(
            self.records, self.settings, include_horizontal=True, smooth=True)
        self.assertEqual(spectra["ns"].shape, (len(self.records), 16))
        self.assertEqual(spectra["horizontal"].shape, (len(self.records), 16))


if __name__ == "__main__":
    unittest.main()
