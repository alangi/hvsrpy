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

"""Plotting helpers for spectral results.

The low-level primitive in this module is :func:`plot_spectrum_component`,
which draws one selected component onto a supplied axis. Notebook code
can use that function directly to assemble 2x2 grids or other custom
subplot layouts, while the existing wrapper functions remain available
for backward-compatible convenience.
"""

import numpy as np

from ._spectral import SPECTRUM_TYPES, as_spectral_result

__all__ = [
    "plot_spectrum_component",
    "plot_spectrum_results",
    "plot_spectrum_summary",
    "plot_spectra",
    "plot_fourier_amplitude_spectra",
    "plot_power_spectral_density",
]


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
    if include_horizontal and spectra.horizontal is not None:
        components.append(("horizontal", "Horizontal"))
    return components


def _prepare_component_values(spectra, key):
    """Return one component's spectra as a 2D floating-point array."""
    return np.asarray(getattr(spectra, key), dtype=float)


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


def plot_spectrum_component(spectra,
                            component,
                            spectrum_type=None,
                            statistic="median",
                            ax=None,
                            title=None,
                            show_xlabel=True,
                            show_ylabel=True,
                            individual_color="0.65",
                            summary_color="C0"):
    """Plot one selected spectral component on a supplied axis.

    Parameters
    ----------
    spectra : SpectralResult or dict-like
        Spectral dataset to plot.
    component : {"ns", "ew", "vt", "horizontal"}
        Component key to draw.
    spectrum_type : {"fas", "psd"}, optional
        Spectral quantity to plot. If omitted, it is read from
        ``spectra.spectrum_type`` when available.
    statistic : {"median", "mean"}, optional
        Summary statistic to overlay on top of the individual windows.
    ax : matplotlib.axes.Axes, optional
        Existing axis. If omitted, a new figure and axis are created.
    title : str, optional
        Optional title for the axis. This is useful when composing
        multi-panel notebook layouts externally.
    show_xlabel, show_ylabel : bool, optional
        Control whether frequency and quantity labels are applied.

    Returns
    -------
    tuple or matplotlib.axes.Axes
        Returns ``(fig, ax)`` when ``ax`` is ``None``; otherwise returns
        ``ax``.
    """
    import matplotlib.pyplot as plt

    spectra = as_spectral_result(spectra, spectrum_type=spectrum_type)
    if component not in ("ns", "ew", "vt", "horizontal"):
        raise ValueError(
            f"component={component!r} not recognized. "
            "Use 'ns', 'ew', 'vt', or 'horizontal'."
        )
    if component == "horizontal" and spectra.horizontal is None:
        raise ValueError("horizontal is not present in spectra.")

    ax_was_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=150)

    frequency = np.asarray(spectra.frequency, dtype=float)
    values = _prepare_component_values(spectra, component)
    for row in values:
        valid = _valid_plot_mask(frequency, row)
        ax.plot(
            frequency[valid],
            row[valid],
            color=individual_color,
            linewidth=0.8,
            alpha=0.35,
        )

    summary = _summary_statistic(values, statistic=statistic)
    valid = _valid_plot_mask(frequency, summary)
    ax.plot(
        frequency[valid],
        summary[valid],
        color=summary_color,
        linewidth=1.6,
    )

    _configure_spectrum_axis(
        ax,
        spectra.spectrum_type,
        ylabel=SPECTRUM_TYPES[spectra.spectrum_type] if show_ylabel else None,
        xlabel=show_xlabel,
    )
    if not show_ylabel:
        ax.set_ylabel("")
    if title is not None:
        ax.set_title(title)

    if ax_was_none:
        return fig, ax
    return ax


def plot_spectrum_results(spectra,
                          spectrum_type=None,
                          include_horizontal=False,
                          statistic="median",
                          axes=None):
    """Plot detailed spectral results with one subplot per component.

    Parameters
    ----------
    spectra : SpectralResult or dict-like
        Spectral dataset to plot. Dict-like input is accepted for
        backward compatibility.
    spectrum_type : {"fas", "psd"}, optional
        Spectral quantity to plot. If omitted, it is read from
        ``spectra.spectrum_type`` when available.
    include_horizontal : bool, optional
        If ``True`` and ``horizontal`` is present, an additional
        horizontal subplot is included.
    statistic : {"median", "mean"}, optional
        Summary statistic to overlay on top of the individual windows.
    axes : iterable of matplotlib.axes.Axes, optional
        Existing axes to use.
    """
    import matplotlib.pyplot as plt

    spectra = as_spectral_result(spectra, spectrum_type=spectrum_type)
    components = _components_to_plot(spectra, include_horizontal=include_horizontal)

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

    for index, (ax, (key, label)) in enumerate(zip(axes, components)):
        plot_spectrum_component(
            spectra,
            component=key,
            spectrum_type=spectra.spectrum_type,
            statistic=statistic,
            ax=ax,
            title=label,
            show_xlabel=(index == len(components) - 1),
            show_ylabel=True,
        )

    if axes_were_provided:
        return axes
    return fig, axes


def plot_spectrum_summary(spectra,
                          spectrum_type=None,
                          include_horizontal=False,
                          statistic="median",
                          ax=None):
    """Plot summary spectral curves for all requested components."""
    import matplotlib.pyplot as plt

    spectra = as_spectral_result(spectra, spectrum_type=spectrum_type)
    ax_was_none = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 2.8), dpi=150)

    frequency = np.array(spectra.frequency, dtype=float)
    label_map = [("ns", "N"), ("ew", "E"), ("vt", "Z")]
    if include_horizontal and spectra.horizontal is not None:
        label_map.append(("horizontal", "H"))

    for key, label in label_map:
        values = _prepare_component_values(spectra, key)
        reduced = _summary_statistic(values, statistic=statistic)
        valid = _valid_plot_mask(frequency, reduced)
        ax.plot(frequency[valid], reduced[valid], label=label)

    _configure_spectrum_axis(
        ax,
        spectra.spectrum_type,
        ylabel=SPECTRUM_TYPES[spectra.spectrum_type],
        xlabel=True,
    )
    ax.legend(loc="best")

    if ax_was_none:
        return fig, ax
    return ax


def plot_spectra(spectra,
                 spectrum_type=None,
                 include_horizontal=False,
                 statistic="median",
                 ax=None):
    """Plot summary component spectra for either FAS or PSD."""
    spectra = as_spectral_result(spectra, spectrum_type=spectrum_type)
    return plot_spectrum_summary(
        spectra,
        spectrum_type=spectra.spectrum_type,
        include_horizontal=include_horizontal,
        statistic=statistic,
        ax=ax,
    )


def plot_fourier_amplitude_spectra(spectra,
                                   include_horizontal=False,
                                   statistic="median",
                                   ax=None):
    """Backward-compatible wrapper for plotting Fourier spectra."""
    spectra = as_spectral_result(spectra, spectrum_type="fas")
    return plot_spectra(
        spectra,
        spectrum_type="fas",
        include_horizontal=include_horizontal,
        statistic=statistic,
        ax=ax,
    )


def plot_power_spectral_density(spectra,
                                include_horizontal=False,
                                statistic="median",
                                ax=None):
    """Backward-compatible wrapper for plotting power spectral density."""
    spectra = as_spectral_result(spectra, spectrum_type="psd")
    return plot_spectra(
        spectra,
        spectrum_type="psd",
        include_horizontal=include_horizontal,
        statistic=statistic,
        ax=ax,
    )
