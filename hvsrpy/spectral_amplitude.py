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

"""Spectral compute helpers for already-accepted 3-component records.

This module computes spectra from the same 3-component records used in
the HVSR workflow, but it intentionally remains separate from the main
HVSR processing pipeline. Inputs are assumed to already be preprocessed,
already split into windows, and already filtered down to the accepted
windows that should be analyzed. No detrending, filtering, windowing,
rejection, or other preprocessing is performed here.

The preferred return type is :class:`SpectralResult`, a small
notebook-friendly dataclass that stores one spectral dataset, including
its frequency vector, component arrays, and lightweight metadata such as
the spectral quantity type and whether smoothing has already been
applied.

Public plotting entry points remain available from this module for
backward compatibility, but plotting lives in
``hvsrpy.spectral_plotting``.
"""

import numpy as np
from numpy.fft import rfft

from ._spectral import (
    SPECTRUM_TYPES,
    SpectralResult,
    as_spectral_result,
    validate_spectrum_type,
)
from .smoothing import SMOOTHING_OPERATORS

__all__ = [
    "SpectralResult",
    "compute_fourier_amplitude_spectra",
    "compute_power_spectral_density",
    "smooth_fourier_amplitude_spectra",
    "smooth_spectra",
    "plot_spectrum_component",
    "plot_spectrum_results",
    "plot_spectrum_summary",
    "plot_spectra",
]


def _nextpow2(n, minimum_power_of_two=2**15):
    """Return the first power of two above ``n``."""
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two *= 2


def _check_nyquist_frequency(fnyq, frequencies):
    """Raise if requested output frequencies exceed the Nyquist limit."""
    if np.max(frequencies) > fnyq:
        msg = f"The maximum resampling frequency of {np.max(frequencies):.2f} Hz "
        msg += f"exceeds the records Nyquist frequency of {fnyq:.2f} Hz"
        raise ValueError(msg)


def _validate_component(component, record_index, component_name):
    """Validate the minimal interface required for one component."""
    missing_attrs = [
        attr for attr in ("amplitude", "dt_in_seconds", "n_samples")
        if not hasattr(component, attr)
    ]
    if missing_attrs:
        missing = ", ".join(missing_attrs)
        msg = (
            f"records[{record_index}].{component_name} is missing required "
            f"attribute(s): {missing}."
        )
        raise ValueError(msg)


def _validate_record_structure(record, record_index):
    """Validate the minimal 3-component record structure used here."""
    missing_components = [
        name for name in ("ns", "ew", "vt") if not hasattr(record, name)
    ]
    if missing_components:
        missing = ", ".join(missing_components)
        msg = f"records[{record_index}] is missing required component(s): {missing}."
        raise ValueError(msg)

    for component_name in ("ns", "ew", "vt"):
        _validate_component(getattr(record, component_name), record_index, component_name)


def _validate_records_and_resolve_dt(records):
    """Validate record structure and enforce a single sampling interval."""
    shared_dt = None
    for record_index, record in enumerate(records):
        _validate_record_structure(record, record_index)

        ns_dt = record.ns.dt_in_seconds
        ew_dt = record.ew.dt_in_seconds
        vt_dt = record.vt.dt_in_seconds
        if abs(ns_dt - ew_dt) > 1E-8 or abs(ns_dt - vt_dt) > 1E-8:
            msg = (
                f"records[{record_index}] has inconsistent component sampling "
                f"intervals: ns={ns_dt}, ew={ew_dt}, vt={vt_dt}. All components "
                "within a record must share the same dt_in_seconds."
            )
            raise ValueError(msg)

        if shared_dt is None:
            shared_dt = ns_dt
            continue

        if abs(ns_dt - shared_dt) > 1E-8:
            msg = (
                f"records[{record_index}].ns.dt_in_seconds={ns_dt} does not match "
                f"the shared sampling interval of {shared_dt}. All records must "
                "share a common dt_in_seconds to compute a single frequency vector."
            )
            raise ValueError(msg)

    return shared_dt


def _resolve_fft_settings(records, fft_settings):
    """Resolve FFT settings while preserving the current policy."""
    max_n_samples = 0
    for record in records:
        for component_name in ("ns", "ew", "vt"):
            n_samples = getattr(record, component_name).n_samples
            if n_samples > max_n_samples:
                max_n_samples = n_samples

    good_n = _nextpow2(max_n_samples)
    if fft_settings is None:
        return {"n": good_n}

    fft_settings = dict(fft_settings)
    user_n = fft_settings.get("n", None)
    if user_n is None:
        fft_settings["n"] = good_n
    else:
        fft_settings["n"] = good_n if good_n > user_n else user_n

    return fft_settings


def _combine_horizontals(spectrum_ns, spectrum_ew, method):
    """Combine horizontal spectra using the configured non-azimuth method."""
    if method == "arithmetic_mean":
        return (spectrum_ns + spectrum_ew) / 2
    if method in ("squared_average", "quadratic_mean", "root_mean_square", "effective_amplitude_spectrum"):
        return np.sqrt((spectrum_ns*spectrum_ns + spectrum_ew*spectrum_ew)/2)
    if method == "geometric_mean":
        return np.sqrt(spectrum_ns * spectrum_ew)
    if method in ("total_horizontal_energy", "vector_summation"):
        return np.sqrt(spectrum_ns*spectrum_ns + spectrum_ew*spectrum_ew)
    if method == "maximum_horizontal_value":
        return np.where(spectrum_ns > spectrum_ew, spectrum_ns, spectrum_ew)
    if method == "single_azimuth":
        msg = (
            "single_azimuth must be computed from a rotated time-domain "
            "horizontal component before spectral transformation."
        )
        raise ValueError(msg)
    msg = f"method_to_combine_horizontals={method} not recognized."
    raise ValueError(msg)


def _method_to_combine_horizontals(settings):
    """Return the configured horizontal-combination method or the default."""
    return getattr(settings, "method_to_combine_horizontals", "geometric_mean")


def _azimuth_in_degrees(settings):
    """Resolve and validate the azimuth used by ``single_azimuth``."""
    if not hasattr(settings, "azimuth_in_degrees"):
        msg = (
            "settings.azimuth_in_degrees is required when "
            "method_to_combine_horizontals='single_azimuth'."
        )
        raise ValueError(msg)

    try:
        azimuth_in_degrees = float(settings.azimuth_in_degrees)
    except (TypeError, ValueError):
        msg = "settings.azimuth_in_degrees must be a finite numeric value."
        raise ValueError(msg)

    if not np.isfinite(azimuth_in_degrees):
        msg = "settings.azimuth_in_degrees must be a finite numeric value."
        raise ValueError(msg)

    return azimuth_in_degrees


def _single_azimuth_horizontal_amplitude(record, settings):
    """Rotate the horizontal components in the time domain."""
    azimuth_in_degrees = _azimuth_in_degrees(settings)
    radians_from_north = np.radians(azimuth_in_degrees)
    return (
        record.ns.amplitude*np.cos(radians_from_north)
        + record.ew.amplitude*np.sin(radians_from_north)
    )


def _compute_component_fas(component, fft_settings):
    """Compute one Fourier amplitude spectrum for a single component."""
    return np.abs(rfft(component.amplitude, **fft_settings))


def _compute_component_psd(component, fft_settings):
    """Compute one one-sided PSD for a single already-windowed component."""
    fft = rfft(component.amplitude, **fft_settings)
    psd = np.real(np.conjugate(fft) * fft)
    psd /= component.n_samples
    psd /= component.fs
    psd *= 2
    return psd


def _compute_single_azimuth_fas(record, settings, fft_settings):
    """Compute single-azimuth FAS using the existing HVSR rotation order."""
    horizontal = _single_azimuth_horizontal_amplitude(record, settings)
    return np.abs(rfft(horizontal, **fft_settings))


def _compute_single_azimuth_psd(record, settings, fft_settings):
    """Compute single-azimuth PSD using the existing HVSR rotation order."""
    horizontal = _single_azimuth_horizontal_amplitude(record, settings)
    fft = rfft(horizontal, **fft_settings)
    psd = np.real(np.conjugate(fft) * fft)
    psd /= record.ns.n_samples
    psd /= record.ns.fs
    psd *= 2
    return psd


def _compute_horizontal_spectra(ns, ew, records, settings, fft_settings, spectrum_type):
    """Compute the optional horizontal spectra according to settings."""
    method_to_combine_horizontals = _method_to_combine_horizontals(settings)
    horizontal = np.empty_like(ns)
    if method_to_combine_horizontals == "single_azimuth":
        for idx, record in enumerate(records):
            if spectrum_type == "fas":
                horizontal[idx] = _compute_single_azimuth_fas(record, settings, fft_settings)
            else:
                horizontal[idx] = _compute_single_azimuth_psd(record, settings, fft_settings)
        return horizontal

    for idx in range(len(records)):
        horizontal[idx] = _combine_horizontals(ns[idx], ew[idx], method_to_combine_horizontals)
    return horizontal


def _compute_spectra(records, settings, spectrum_type, include_horizontal=False, smooth=False):
    """Compute either FAS or PSD for already-accepted 3C records."""
    validate_spectrum_type(spectrum_type)
    if settings is None:
        raise ValueError("settings is required.")

    records = list(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one SeismicRecording3C.")

    dt = _validate_records_and_resolve_dt(records)
    fft_settings = _resolve_fft_settings(
        records,
        fft_settings=getattr(settings, "fft_settings", None),
    )
    n = fft_settings["n"]
    frequency = np.fft.rfftfreq(n, dt)
    ns = np.empty((len(records), len(frequency)))
    ew = np.empty((len(records), len(frequency)))
    vt = np.empty((len(records), len(frequency)))

    component_operator = _compute_component_fas if spectrum_type == "fas" else _compute_component_psd
    for idx, record in enumerate(records):
        ns[idx] = component_operator(record.ns, fft_settings)
        ew[idx] = component_operator(record.ew, fft_settings)
        vt[idx] = component_operator(record.vt, fft_settings)

    result = SpectralResult(
        frequency=frequency,
        ns=ns,
        ew=ew,
        vt=vt,
        horizontal=(
            _compute_horizontal_spectra(ns, ew, records, settings, fft_settings, spectrum_type)
            if include_horizontal else None
        ),
        spectrum_type=spectrum_type,
        is_smoothed=False,
    )

    if smooth:
        return smooth_spectra(result, settings)

    return result


def compute_fourier_amplitude_spectra(records,
                                      settings,
                                      include_horizontal=False,
                                      smooth=False):
    """Compute Fourier amplitude spectra from accepted windows.

    Inputs are assumed to already be preprocessed, already windowed, and
    already accepted for analysis. This function performs no additional
    preprocessing or rejection checks and returns a
    :class:`SpectralResult`.
    """
    return _compute_spectra(
        records,
        settings,
        spectrum_type="fas",
        include_horizontal=include_horizontal,
        smooth=smooth,
    )


def compute_power_spectral_density(records,
                                   settings,
                                   include_horizontal=False,
                                   smooth=False):
    """Compute one-sided PSD curves from accepted windows.

    Inputs are assumed to already be preprocessed, already windowed, and
    already accepted for analysis. This function performs no additional
    preprocessing or rejection checks and returns a
    :class:`SpectralResult`.
    """
    return _compute_spectra(
        records,
        settings,
        spectrum_type="psd",
        include_horizontal=include_horizontal,
        smooth=smooth,
    )


def smooth_spectra(spectra, settings):
    """Smooth a previously computed spectral dataset.

    Parameters
    ----------
    spectra : SpectralResult or dict-like
        Output from one of the compute helpers in this module. Dict-like
        input is still accepted for backward compatibility, but the
        preferred return and input type is
        :class:`SpectralResult`.
    settings : Settings-like
        Provides the smoothing dictionary with keys ``operator``,
        ``bandwidth``, and ``center_frequencies_in_hz``.

    Returns
    -------
    SpectralResult
        Smoothed copy of ``spectra`` with
        ``frequency=settings.smoothing["center_frequencies_in_hz"]`` and
        ``is_smoothed=True``.
    """
    spectra = as_spectral_result(spectra)
    smoothing = getattr(settings, "smoothing", None)
    if smoothing is None:
        msg = "settings.smoothing is required for spectral smoothing."
        raise ValueError(msg)

    operator = smoothing["operator"]
    bandwidth = smoothing["bandwidth"]
    fcs = np.array(smoothing["center_frequencies_in_hz"], dtype=float)

    frequency = np.array(spectra.frequency, dtype=float)
    _check_nyquist_frequency(np.max(frequency), fcs)
    if operator not in SMOOTHING_OPERATORS:
        msg = f"Smoothing operator {operator!r} is not recognized."
        raise ValueError(msg)

    smoothed_components = {}
    for key in ("ns", "ew", "vt", "horizontal"):
        values = getattr(spectra, key)
        if values is None:
            continue
        smoothed_components[key] = SMOOTHING_OPERATORS[operator](
            frequency,
            np.asarray(values, dtype=float),
            fcs,
            bandwidth,
        )

    return SpectralResult(
        frequency=fcs,
        ns=smoothed_components["ns"],
        ew=smoothed_components["ew"],
        vt=smoothed_components["vt"],
        horizontal=smoothed_components.get("horizontal", None),
        spectrum_type=spectra.spectrum_type,
        is_smoothed=True,
    )


def smooth_fourier_amplitude_spectra(spectra, settings):
    """Backward-compatible wrapper for smoothing Fourier spectra."""
    spectra = as_spectral_result(spectra, spectrum_type="fas")
    return smooth_spectra(spectra, settings)


from .spectral_plotting import (  # noqa: E402
    plot_spectra,
    plot_spectrum_component,
    plot_spectrum_results,
    plot_spectrum_summary,
)
