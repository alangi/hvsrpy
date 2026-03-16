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
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Helpers for Fourier amplitude spectrum (FAS) workflows.

This module is intentionally isolated from HVSR processing.

Examples
--------
>>> import numpy as np
>>> from hvsrpy.timeseries import TimeSeries
>>> from hvsrpy.seismic_recording_3c import SeismicRecording3C
>>> from hvsrpy.spectral_amplitude import compute_fourier_amplitude_spectra
>>> t = np.arange(0, 10, 0.01)
>>> ns = TimeSeries(np.sin(2*np.pi*2*t), 0.01)
>>> ew = TimeSeries(np.sin(2*np.pi*2*t), 0.01)
>>> vt = TimeSeries(np.sin(2*np.pi*2*t), 0.01)
>>> record = SeismicRecording3C(ns, ew, vt)
>>> from hvsrpy.settings import HvsrTraditionalProcessingSettings
>>> settings = HvsrTraditionalProcessingSettings()
>>> fas = compute_fourier_amplitude_spectra([record], settings)
>>> "frequency" in fas and "ns" in fas and "vt" in fas
True
"""

import numpy as np
from numpy.fft import rfft

from .smoothing import SMOOTHING_OPERATORS

__all__ = [
    "compute_fourier_amplitude_spectra",
    "smooth_fourier_amplitude_spectra",
    "plot_fourier_amplitude_spectra",
]


def _nextpow2(n, minimum_power_of_two=2**15):
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two *= 2


def _check_nyquist_frequency(fnyq, frequencies):
    if np.max(frequencies) > fnyq:
        msg = f"The maximum resampling frequency of {np.max(frequencies):.2f} Hz "
        msg += f"exceeds the records Nyquist frequency of {fnyq:.2f} Hz"
        raise ValueError(msg)


def _resolve_fft_settings(records, fft_settings):
    max_n_samples = 0
    for record in records:
        if record.vt.n_samples > max_n_samples:
            max_n_samples = record.vt.n_samples
    good_n = _nextpow2(max_n_samples)

    if fft_settings is None:
        return dict(n=good_n)

    fft_settings = dict(fft_settings)
    user_n = fft_settings.get("n", max_n_samples)
    if user_n is None:
        fft_settings["n"] = max_n_samples
    else:
        fft_settings["n"] = good_n if good_n > user_n else user_n
    return fft_settings


def _combine_horizontals(fft_ns, fft_ew, method):
    if method == "arithmetic_mean":
        return (fft_ns + fft_ew) / 2
    if method in ("squared_average", "quadratic_mean", "root_mean_square", "effective_amplitude_spectrum"):
        return np.sqrt((fft_ns*fft_ns + fft_ew*fft_ew)/2)
    if method == "geometric_mean":
        return np.sqrt(fft_ns * fft_ew)
    if method in ("total_horizontal_energy", "vector_summation"):
        return np.sqrt(fft_ns*fft_ns + fft_ew*fft_ew)
    if method == "maximum_horizontal_value":
        return np.where(fft_ns > fft_ew, fft_ns, fft_ew)
    msg = f"method_to_combine_horizontals={method} not recognized."
    raise ValueError(msg)


def _method_to_combine_horizontals(settings):
    return getattr(settings, "method_to_combine_horizontals", "geometric_mean")


def compute_fourier_amplitude_spectra(records,
                                      settings,
                                      include_horizontal=False,
                                      smooth=False):
    """Compute FAS from already-windowed accepted 3C traces.

    Parameters
    ----------
    records : iterable of SeismicRecording3C
        3-component traces that are already preprocessed/windowed and
        already represent accepted windows for analysis.
    settings : HvsrTraditionalProcessingSettings-like
        Source of FFT and optional smoothing conventions. Only relevant
        fields are used: ``fft_settings``, ``method_to_combine_horizontals``,
        and ``smoothing``.
    include_horizontal : bool, optional
        If ``True``, include ``horizontal`` in the returned dictionary.
    smooth : bool, optional
        If ``True``, return spectra smoothed using
        :func:`smooth_fourier_amplitude_spectra` and
        ``settings.smoothing``.

    Returns
    -------
    dict
        Dictionary with keys ``frequency``, ``ns``, ``ew``, ``vt`` and
        optional ``horizontal``. Component arrays are shape
        ``(n_records, n_frequencies)``.
    """
    if settings is None:
        raise ValueError("settings is required.")

    records = list(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one SeismicRecording3C.")

    fft_settings = _resolve_fft_settings(records, fft_settings=getattr(settings, "fft_settings", None))
    dt = records[0].ns.dt_in_seconds
    for record in records[1:]:
        if abs(record.ns.dt_in_seconds - dt) > 1E-8:
            msg = "All records must share a common sampling interval to compute a single frequency vector."
            raise ValueError(msg)

    n = fft_settings["n"]
    frequency = np.fft.rfftfreq(n, dt)
    ns = np.empty((len(records), len(frequency)))
    ew = np.empty((len(records), len(frequency)))
    vt = np.empty((len(records), len(frequency)))

    for idx, record in enumerate(records):
        # Compute spectra directly from provided windows. No extra
        # preprocessing/windowing is applied in this module.
        ns[idx] = np.abs(rfft(record.ns.amplitude, **fft_settings))
        ew[idx] = np.abs(rfft(record.ew.amplitude, **fft_settings))
        vt[idx] = np.abs(rfft(record.vt.amplitude, **fft_settings))

    result = dict(frequency=frequency, ns=ns, ew=ew, vt=vt)
    if include_horizontal:
        method_to_combine_horizontals = _method_to_combine_horizontals(settings)
        horizontal = np.empty_like(ns)
        for idx in range(len(records)):
            horizontal[idx] = _combine_horizontals(ns[idx], ew[idx], method_to_combine_horizontals)
        result["horizontal"] = horizontal

    if smooth:
        return smooth_fourier_amplitude_spectra(result, settings)

    return result


def smooth_fourier_amplitude_spectra(spectra,
                                     settings):
    """Smooth Fourier amplitude spectra.

    Parameters
    ----------
    spectra : dict
        Dictionary with entries from
        :func:`compute_fourier_amplitude_spectra`.
    settings : HvsrTraditionalProcessingSettings-like
        Provides smoothing dictionary with keys ``operator``,
        ``bandwidth``, and ``center_frequencies_in_hz``.

    Returns
    -------
    dict
        Smoothed spectra with the same keys as input and updated
        ``frequency`` vector.
    """
    smoothing = getattr(settings, "smoothing", None)
    if smoothing is None:
        msg = "settings.smoothing is required for spectral smoothing."
        raise ValueError(msg)

    operator = smoothing["operator"]
    bandwidth = smoothing["bandwidth"]
    fcs = np.array(smoothing["center_frequencies_in_hz"], dtype=float)

    frequency = np.array(spectra["frequency"], dtype=float)
    # Same Nyquist guard style used by HVSR processing.
    _check_nyquist_frequency(np.max(frequency), fcs)

    result = dict(frequency=fcs)
    for key, value in spectra.items():
        if key == "frequency":
            continue
        value = np.array(value, dtype=float)
        if value.ndim == 1:
            value = np.atleast_2d(value)
        result[key] = SMOOTHING_OPERATORS[operator](frequency, value, fcs, bandwidth)

    return result


def _summary_statistic(values, statistic):
    if statistic == "mean":
        return np.mean(values, axis=0)
    if statistic == "median":
        return np.median(values, axis=0)
    msg = f"statistic={statistic} not recognized. Use 'mean' or 'median'."
    raise ValueError(msg)


def plot_fourier_amplitude_spectra(spectra,
                                   include_horizontal=False,
                                   statistic="median",
                                   ax=None):
    """Plot N, E, Z and optional combined-horizontal amplitude spectra.

    Parameters
    ----------
    spectra : dict
        Spectra dictionary from :func:`compute_fourier_amplitude_spectra`
        or :func:`smooth_fourier_amplitude_spectra`.
    include_horizontal : bool, optional
        If ``True`` and ``horizontal`` exists in ``spectra``, it is
        included in the plot.
    statistic : {"median", "mean"}, optional
        Reduction over windows for each component.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        ``(fig, ax)`` if ``ax`` is ``None`` otherwise ``ax``.
    """
    import matplotlib.pyplot as plt

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=150)

    frequency = np.array(spectra["frequency"], dtype=float)
    to_plot = [("ns", "N"), ("ew", "E"), ("vt", "Z")]
    if include_horizontal and ("horizontal" in spectra):
        to_plot.append(("horizontal", "H"))

    valid_frequency = frequency > 0
    for key, label in to_plot:
        values = np.array(spectra[key], dtype=float)
        if values.ndim == 1:
            values = np.atleast_2d(values)
        reduced = _summary_statistic(values, statistic=statistic)
        ax.plot(frequency[valid_frequency], reduced[valid_frequency], label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fourier Amplitude")
    ax.legend(loc="best")

    if ax_was_none:
        return fig, ax
    return ax
