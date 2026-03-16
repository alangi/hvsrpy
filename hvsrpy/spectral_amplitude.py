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

"""Helpers for spectral-analysis workflows on accepted 3-component records.

This module computes spectra from the same 3-component records used in
the HVSR workflow, but it intentionally remains separate from the main
HVSR processing pipeline. Inputs are assumed to already be preprocessed,
already split into windows, and already filtered down to the accepted
windows that should be analyzed. No detrending, filtering, windowing,
rejection, or other preprocessing is performed here.

Two spectral quantities are supported:

``fas``
    Fourier amplitude spectrum. This module preserves the existing
    ``abs(rfft(...))`` convention already used here.
``psd``
    One-sided power spectral density computed from the already-windowed
    inputs. The normalization mirrors the package's PSD processing
    convention where practical by using ``2 * |FFT|^2 / (n_samples *
    fs)`` per record/window, but this module does not apply any extra
    tapering or window-energy correction because the inputs are assumed
    to have already passed through that stage upstream.

The unsmoothed frequency vector is built directly from the FFT bins
defined by the selected FFT length and the shared sampling interval. If
spectra are smoothed, the returned frequency vector becomes
``settings.smoothing["center_frequencies_in_hz"]``. The current
horizontal-combination behavior is preserved for non-azimuth methods.
For ``single_azimuth``, the horizontal trace follows the existing HVSR
methodology: rotate the two horizontal components in the time domain
using the requested azimuth, then transform that rotated component.

The functions in this module use only a small subset of the processing
settings object:

``settings.fft_settings``
    Passed through to :func:`numpy.fft.rfft` after resolving a shared
    ``n`` value.
``settings.method_to_combine_horizontals``
    Used only when ``include_horizontal=True``.
``settings.smoothing``
    Used only when smoothing is requested. Expected keys are
    ``operator``, ``bandwidth``, and ``center_frequencies_in_hz``.

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
    "compute_power_spectral_density",
    "smooth_fourier_amplitude_spectra",
    "smooth_spectra",
    "plot_spectrum_results",
    "plot_spectrum_summary",
    "plot_spectra",
    "plot_fourier_amplitude_spectra",
]


_SPECTRUM_TYPES = {
    "fas": "Fourier Amplitude",
    "psd": "Power Spectral Density",
}


def _nextpow2(n, minimum_power_of_two=2**15):
    """Return the first power of two above ``n``.

    The implementation intentionally mirrors the package's existing FFT
    sizing style, including the minimum power-of-two floor used to keep
    FFT sizes consistent for downstream smoothing workflows.
    """
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
    """Validate the minimal interface required for one component.

    This module intentionally accepts duck-typed record objects, but it
    still checks for the small set of attributes required to compute FAS
    so callers get a clear error instead of a later ``AttributeError``.
    """
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
    """Validate record structure and enforce a single sampling interval.

    The FAS helpers in this module return one shared frequency vector, so
    all components within a record and all records across the input
    iterable must share the same ``dt_in_seconds`` value.
    """
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
    """Resolve FFT settings while preserving the package's current policy.

    A shared FFT length is required because the returned spectra use one
    common frequency vector for all records. The effective ``n`` is based
    on the longest component among all ``ns``, ``ew``, and ``vt`` traces,
    then promoted to the next allowed power-of-two size following the
    existing package convention.
    """

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


def _validate_spectrum_type(spectrum_type):
    """Validate and normalize the requested spectrum type."""
    if spectrum_type not in _SPECTRUM_TYPES:
        msg = (
            f"spectrum_type={spectrum_type!r} not recognized. "
            "Use 'fas' or 'psd'."
        )
        raise ValueError(msg)
    return spectrum_type


def _combine_horizontals(spectrum_ns, spectrum_ew, method):
    """Combine horizontal spectra using the configured non-azimuth method.

    This helper intentionally preserves the existing combination logic
    used by this module. It does not attempt to align the behavior with
    other package pathways beyond what is already implemented here.
    """
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
    """Resolve and validate the azimuth used by ``single_azimuth``.

    The value is interpreted the same way as the existing HVSR
    implementation: degrees from north, clockwise positive. No explicit
    wrapping is required because the trigonometric evaluation is
    periodic, but the value must be finite.
    """
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
    """Rotate the horizontal components in the time domain.

    This mirrors the existing HVSR single-azimuth methodology in
    :mod:`hvsrpy.processing`: ``ns*cos(theta) + ew*sin(theta)`` with
    ``theta`` measured in degrees clockwise from north.
    """
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
    """Compute the optional horizontal spectra according to settings.

    For non-azimuth methods the combination is applied directly to the
    selected spectral quantity. For ``single_azimuth`` the horizontal
    component is formed in the time domain first, matching the HVSR
    implementation, then transformed into the selected spectral quantity.
    """
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
    _validate_spectrum_type(spectrum_type)
    if settings is None:
        raise ValueError("settings is required.")

    records = list(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one SeismicRecording3C.")

    dt = _validate_records_and_resolve_dt(records)
    fft_settings = _resolve_fft_settings(records, fft_settings=getattr(settings, "fft_settings", None))
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

    result = dict(frequency=frequency, ns=ns, ew=ew, vt=vt)
    if include_horizontal:
        result["horizontal"] = _compute_horizontal_spectra(
            ns, ew, records, settings, fft_settings, spectrum_type
        )

    if smooth:
        return smooth_spectra(result, settings)

    return result


def compute_fourier_amplitude_spectra(records,
                                      settings,
                                      include_horizontal=False,
                                      smooth=False):
    """Compute Fourier amplitude spectra from already-accepted windows.

    Parameters
    ----------
    records : iterable of SeismicRecording3C
        3-component traces that are already preprocessed/windowed and
        already represent accepted windows for analysis. This function
        assumes each record exposes ``ns``, ``ew``, and ``vt`` and that
        each component exposes ``amplitude``, ``dt_in_seconds``, and
        ``n_samples``.
    settings : HvsrTraditionalProcessingSettings-like
        Source of FFT and optional smoothing conventions. Only relevant
        fields are used: ``fft_settings``, ``method_to_combine_horizontals``,
        and ``smoothing``. No other processing settings are consumed by
        this module.
    include_horizontal : bool, optional
        If ``True``, include ``horizontal`` in the returned dictionary
        using the current in-module combination behavior.
    smooth : bool, optional
        If ``True``, return spectra smoothed using
        :func:`smooth_fourier_amplitude_spectra` and
        ``settings.smoothing``.

    Returns
    -------
    dict
        Dictionary with keys ``frequency``, ``ns``, ``ew``, ``vt``, and
        optionally ``horizontal``.

        ``frequency`` is shape ``(n_frequencies,)``.
        ``ns``, ``ew``, ``vt``, and optional ``horizontal`` are shape
        ``(n_records, n_frequencies)``.

        When ``smooth=False``, ``frequency`` is the FFT frequency vector
        from :func:`numpy.fft.rfftfreq`. When ``smooth=True``, the result
        is passed through :func:`smooth_fourier_amplitude_spectra`, so
        ``frequency`` becomes
        ``settings.smoothing["center_frequencies_in_hz"]``.

    Notes
    -----
    This function does not apply additional preprocessing, tapering, or
    acceptance checks. It computes FFT magnitudes directly from the
    amplitudes already present on each component.
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
    """Compute one-sided PSD curves from already-accepted windows.

    Parameters
    ----------
    records : iterable of SeismicRecording3C
        Already-preprocessed, already-windowed, already-accepted
        3-component records.
    settings : Settings-like
        Source of FFT, horizontal-combination, azimuth, and optional
        smoothing information. Only the fields needed by this module are
        accessed.
    include_horizontal : bool, optional
        If ``True``, include an optional ``horizontal`` entry.
        ``single_azimuth`` is handled by rotating the horizontal traces
        in the time domain before the PSD is formed.
    smooth : bool, optional
        If ``True``, smooth the PSD curves using ``settings.smoothing``.

    Returns
    -------
    dict
        Dictionary with the same structure as
        :func:`compute_fourier_amplitude_spectra`: ``frequency``, ``ns``,
        ``ew``, ``vt``, and optional ``horizontal``. Component arrays
        are shape ``(n_records, n_frequencies)``.

    Notes
    -----
    The PSD normalization in this module is ``2 * |FFT|^2 / (n_samples *
    fs)`` for each record/window. This follows the package's PSD
    processing convention in broad form, but no additional tapering or
    window-energy normalization is applied here because the inputs are
    assumed to already be windowed upstream.
    """
    return _compute_spectra(
        records,
        settings,
        spectrum_type="psd",
        include_horizontal=include_horizontal,
        smooth=smooth,
    )


def smooth_spectra(spectra, settings):
    """Smooth previously computed spectra of any supported quantity.

    Parameters
    ----------
    spectra : dict
        Dictionary with entries from one of the compute helpers in this
        module. The function expects a ``frequency`` entry plus one or
        more component arrays whose first dimension indexes
        records/windows.
    settings : Settings-like
        Provides smoothing dictionary with keys ``operator``,
        ``bandwidth``, and ``center_frequencies_in_hz``.

    Returns
    -------
    dict
        Smoothed spectra with the same keys as input and updated
        ``frequency`` vector. The output ``frequency`` array is exactly
        ``settings.smoothing["center_frequencies_in_hz"]``.

    Notes
    -----
    This function only changes the sampling of the spectra along the
    frequency axis. It does not alter the component set, amplitude
    convention, or horizontal-combination logic established before
    smoothing.
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
    if operator not in SMOOTHING_OPERATORS:
        msg = f"Smoothing operator {operator!r} is not recognized."
        raise ValueError(msg)

    result = dict(frequency=fcs)
    for key, value in spectra.items():
        if key == "frequency":
            continue
        value = np.array(value, dtype=float)
        if value.ndim == 1:
            value = np.atleast_2d(value)
        result[key] = SMOOTHING_OPERATORS[operator](frequency, value, fcs, bandwidth)

    return result


def smooth_fourier_amplitude_spectra(spectra,
                                     settings):
    """Backward-compatible wrapper for smoothing FAS dictionaries."""
    return smooth_spectra(spectra, settings)


def _summary_statistic(values, statistic):
    """Reduce a 2D array of spectra across records/windows."""
    if statistic == "mean":
        return np.mean(values, axis=0)
    if statistic == "median":
        return np.median(values, axis=0)
    msg = f"statistic={statistic} not recognized. Use 'mean' or 'median'."
    raise ValueError(msg)


def _components_to_plot(spectra, include_horizontal=False):
    """Return the ordered set of component keys and display labels."""
    components = [("ns", "North"), ("ew", "East"), ("vt", "Vertical")]
    if include_horizontal and ("horizontal" in spectra):
        components.append(("horizontal", "Horizontal"))
    return components


def _prepare_component_values(spectra, key):
    """Return one component's spectra as a 2D floating-point array."""
    values = np.array(spectra[key], dtype=float)
    if values.ndim == 1:
        values = np.atleast_2d(values)
    return values


def _configure_spectrum_axis(ax, spectrum_type, ylabel=None, xlabel=False):
    """Apply common axis formatting for spectral plots."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Frequency (Hz)")
    else:
        ax.set_xlabel("")
    ax.set_xmargin(0)


def _valid_plot_mask(frequency, values):
    """Return a mask suitable for positive log-log spectral plotting."""
    return (
        np.isfinite(frequency)
        & (frequency > 0)
        & np.isfinite(values)
        & (values > 0)
    )


def plot_spectrum_results(spectra,
                          spectrum_type,
                          include_horizontal=False,
                          statistic="median",
                          axes=None):
    """Plot detailed spectral results with one subplot per component.

    This plot is intended for quality control and detailed inspection.
    Each subplot shows one component only. All accepted windows are drawn
    faintly, and the selected summary statistic is overlaid as a heavier
    summary curve.

    Parameters
    ----------
    spectra : dict
        Spectra dictionary produced by one of this module's compute
        helpers.
    spectrum_type : {"fas", "psd"}
        Determines the y-axis label.
    include_horizontal : bool, optional
        If ``True`` and ``horizontal`` is present in ``spectra``, an
        additional horizontal subplot is included.
    statistic : {"median", "mean"}, optional
        Summary statistic to overlay on top of the individual windows.
    axes : iterable of matplotlib.axes.Axes, optional
        Existing axes to use. If provided, the number of axes must match
        the number of plotted components.

    Returns
    -------
    tuple
        ``(fig, axes)`` if new axes are created. Otherwise returns the
        provided axes as a NumPy array.
    """
    import matplotlib.pyplot as plt

    spectrum_type = _validate_spectrum_type(spectrum_type)
    components = _components_to_plot(spectra, include_horizontal=include_horizontal)
    frequency = np.array(spectra["frequency"], dtype=float)

    axes_were_provided = axes is not None
    if axes is None:
        fig, axes = plt.subplots(
            len(components),
            1,
            sharex=True,
            figsize=(5.0, 1.9*len(components) + 0.4),
            dpi=150,
        )
    axes = np.atleast_1d(axes)
    if len(axes) != len(components):
        msg = (
            f"Expected {len(components)} axes for the requested components, "
            f"received {len(axes)}."
        )
        raise ValueError(msg)

    ylabel = _SPECTRUM_TYPES[spectrum_type]
    for ax, (key, label) in zip(axes, components):
        values = _prepare_component_values(spectra, key)
        for row in values:
            valid = _valid_plot_mask(frequency, row)
            ax.plot(
                frequency[valid],
                row[valid],
                color="0.65",
                linewidth=0.8,
                alpha=0.35,
            )
        summary = _summary_statistic(values, statistic=statistic)
        valid = _valid_plot_mask(frequency, summary)
        ax.plot(
            frequency[valid],
            summary[valid],
            color="C0",
            linewidth=1.6,
        )
        _configure_spectrum_axis(ax, spectrum_type, ylabel=ylabel, xlabel=False)
        ax.set_title(label)

    _configure_spectrum_axis(axes[-1], spectrum_type, ylabel=ylabel, xlabel=True)

    if axes_were_provided:
        return axes
    return fig, axes


def plot_spectrum_summary(spectra,
                          spectrum_type,
                          include_horizontal=False,
                          statistic="median",
                          ax=None):
    """Plot summary spectral curves for all requested components.

    This plot is intended as a compact spectral overview. Only the
    summary curve for each component is plotted. Individual windows are
    not shown.

    Parameters
    ----------
    spectra : dict
        Spectra dictionary produced by one of this module's compute
        helpers.
    spectrum_type : {"fas", "psd"}
        Determines the y-axis label.
    include_horizontal : bool, optional
        If ``True`` and ``horizontal`` is present in ``spectra``, the
        derived horizontal component is included.
    statistic : {"median", "mean"}, optional
        Summary statistic used for each component.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        ``(fig, ax)`` if ``ax`` is ``None`` otherwise ``ax``.
    """
    import matplotlib.pyplot as plt

    spectrum_type = _validate_spectrum_type(spectrum_type)
    ax_was_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=150)

    frequency = np.array(spectra["frequency"], dtype=float)
    label_map = [("ns", "N"), ("ew", "E"), ("vt", "Z")]
    if include_horizontal and ("horizontal" in spectra):
        label_map.append(("horizontal", "H"))

    for key, label in label_map:
        values = _prepare_component_values(spectra, key)
        reduced = _summary_statistic(values, statistic=statistic)
        valid = _valid_plot_mask(frequency, reduced)
        ax.plot(frequency[valid], reduced[valid], label=label)

    _configure_spectrum_axis(
        ax,
        spectrum_type,
        ylabel=_SPECTRUM_TYPES[spectrum_type],
        xlabel=True,
    )
    ax.legend(loc="best")

    if ax_was_none:
        return fig, ax
    return ax


def plot_spectra(spectra,
                 spectrum_type,
                 include_horizontal=False,
                 statistic="median",
                 ax=None):
    """Plot summary component spectra for either FAS or PSD.

    Parameters
    ----------
    spectra : dict
        Spectra dictionary from one of the compute helpers in this
        module.
    spectrum_type : {"fas", "psd"}
        Selects the y-axis label and validates the intended quantity to
        plot.
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

    Notes
    -----
    This is the compact single-axis summary plot. It summarizes multiple
    windows per component using either the median or mean before
    plotting. Individual windows are not shown. For a per-component QC
    view with faint individual windows, use
    :func:`plot_spectrum_results`.
    """
    return plot_spectrum_summary(
        spectra,
        spectrum_type=spectrum_type,
        include_horizontal=include_horizontal,
        statistic=statistic,
        ax=ax,
    )


def plot_fourier_amplitude_spectra(spectra,
                                   include_horizontal=False,
                                   statistic="median",
                                   ax=None):
    """Backward-compatible wrapper for plotting Fourier amplitude spectra."""
    return plot_spectra(
        spectra,
        spectrum_type="fas",
        include_horizontal=include_horizontal,
        statistic=statistic,
        ax=ax,
    )
