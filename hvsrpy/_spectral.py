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

"""Small shared helpers for spectral compute and plotting modules."""

from dataclasses import dataclass

import numpy as np

SPECTRUM_TYPES = {
    "fas": "Fourier Amplitude",
    "psd": "Power Spectral Density",
}


def validate_spectrum_type(spectrum_type):
    """Validate and normalize the requested spectrum type."""
    if spectrum_type not in SPECTRUM_TYPES:
        msg = (
            f"spectrum_type={spectrum_type!r} not recognized. "
            "Use 'fas' or 'psd'."
        )
        raise ValueError(msg)
    return spectrum_type


@dataclass
class SpectralResult:
    """One computed spectral dataset for accepted 3-component windows.

    Parameters
    ----------
    frequency : array-like
        Shared frequency vector for all component spectra.
    ns, ew, vt : array-like
        North-south, east-west, and vertical component spectra. Arrays
        are normalized to shape ``(n_records, n_frequencies)``.
    horizontal : array-like, optional
        Optional horizontal spectrum produced with the configured HVSR
        horizontal-combination method.
    spectrum_type : {"fas", "psd"}
        Spectral quantity represented by this result.
    is_smoothed : bool, optional
        ``True`` when the frequency vector and component arrays already
        reflect the configured smoothing step.

    Notes
    -----
    The class exposes a small dict-like compatibility surface to ease
    notebook migration. That compatibility layer includes metadata keys
    ``spectrum_type`` and ``is_smoothed`` in addition to the component
    arrays and ``frequency``.
    """

    frequency: np.ndarray
    ns: np.ndarray
    ew: np.ndarray
    vt: np.ndarray
    horizontal: np.ndarray = None
    spectrum_type: str = "fas"
    is_smoothed: bool = False

    def __post_init__(self):
        self.spectrum_type = validate_spectrum_type(self.spectrum_type)
        self.frequency = np.asarray(self.frequency, dtype=float)
        if self.frequency.ndim != 1:
            raise ValueError("frequency must be a one-dimensional array.")

        self.ns = self._normalize_component(self.ns, "ns")
        self.ew = self._normalize_component(self.ew, "ew")
        self.vt = self._normalize_component(self.vt, "vt")
        if self.horizontal is not None:
            self.horizontal = self._normalize_component(self.horizontal, "horizontal")

        for key in ("ns", "ew", "vt", "horizontal"):
            values = getattr(self, key)
            if values is None:
                continue
            if values.shape[1] != self.n_frequencies:
                raise ValueError(
                    f"{key} has {values.shape[1]} frequency samples, expected "
                    f"{self.n_frequencies}."
                )
            if values.shape[0] != self.n_records:
                raise ValueError(
                    f"{key} has {values.shape[0]} records, expected {self.n_records}."
                )

    @staticmethod
    def _normalize_component(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = np.atleast_2d(values)
        if values.ndim != 2:
            raise ValueError(f"{name} must be one- or two-dimensional.")
        return values

    @property
    def has_horizontal(self):
        """Whether a horizontal component is present."""
        return self.horizontal is not None

    @property
    def n_records(self):
        """Number of records or accepted windows represented."""
        return int(self.ns.shape[0])

    @property
    def n_frequencies(self):
        """Number of frequency samples per component."""
        return int(self.frequency.size)

    def keys(self):
        """Return dict-like keys for backward-compatible workflows."""
        keys = ["frequency", "ns", "ew", "vt"]
        if self.horizontal is not None:
            keys.append("horizontal")
        keys.extend(["spectrum_type", "is_smoothed"])
        return tuple(keys)

    def items(self):
        """Yield dict-like items for backward-compatible workflows."""
        for key in self.keys():
            yield key, getattr(self, key)

    def get(self, key, default=None):
        """Return ``key`` if present, matching the dict API."""
        if key in self:
            return getattr(self, key)
        return default

    def as_dict(self):
        """Return a plain dictionary view of the stored data and metadata."""
        return {key: value for key, value in self.items()}

    def __contains__(self, key):
        if key == "horizontal":
            return self.horizontal is not None
        return key in {
            "frequency",
            "ns",
            "ew",
            "vt",
            "spectrum_type",
            "is_smoothed",
        }

    def __getitem__(self, key):
        if key == "horizontal" and self.horizontal is None:
            raise KeyError(key)
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)


def as_spectral_result(spectra, spectrum_type=None, is_smoothed=None):
    """Normalize ``SpectralResult`` or dict-like input into ``SpectralResult``."""
    if spectrum_type is not None:
        spectrum_type = validate_spectrum_type(spectrum_type)

    if isinstance(spectra, SpectralResult):
        result = spectra
        if spectrum_type is not None and result.spectrum_type != spectrum_type:
            raise ValueError(
                f"spectrum_type={spectrum_type!r} does not match "
                f"spectra.spectrum_type={result.spectrum_type!r}."
            )
        if is_smoothed is not None and result.is_smoothed != is_smoothed:
            result = SpectralResult(
                frequency=result.frequency,
                ns=result.ns,
                ew=result.ew,
                vt=result.vt,
                horizontal=result.horizontal,
                spectrum_type=result.spectrum_type,
                is_smoothed=is_smoothed,
            )
        return result

    if spectra is None:
        raise ValueError("spectra is required.")

    resolved_spectrum_type = spectrum_type
    if resolved_spectrum_type is None:
        resolved_spectrum_type = spectra.get("spectrum_type", None)
    if resolved_spectrum_type is None:
        raise ValueError(
            "spectrum_type is required when spectra does not provide one."
        )

    resolved_is_smoothed = is_smoothed
    if resolved_is_smoothed is None:
        resolved_is_smoothed = bool(spectra.get("is_smoothed", False))

    return SpectralResult(
        frequency=spectra["frequency"],
        ns=spectra["ns"],
        ew=spectra["ew"],
        vt=spectra["vt"],
        horizontal=spectra.get("horizontal", None),
        spectrum_type=resolved_spectrum_type,
        is_smoothed=resolved_is_smoothed,
    )
