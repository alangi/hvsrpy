# This file is part of hvsrpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

"""Optional model-loading helpers for the AutoHVSR extension layer."""

from contextlib import nullcontext
from importlib import resources
from pathlib import Path


DEFAULT_XGBOOST_MODEL_FILENAME = "2_xgboost_peak_classifier.json"
DEFAULT_XGBOOST_MODEL_RESOURCE = f"hvsrpy.models/{DEFAULT_XGBOOST_MODEL_FILENAME}"


def get_bundled_xgboost_model_resource():
    """Return the bundled AutoHVSR model resource or ``None`` if absent."""
    try:
        resource = resources.files("hvsrpy.models").joinpath(DEFAULT_XGBOOST_MODEL_FILENAME)
    except ModuleNotFoundError:
        return None

    try:
        if resource.is_file():
            return resource
    except FileNotFoundError:
        return None
    return None


def load_xgboost_classifier(model_path=None, use_bundled_model=True):
    """Load an XGBoost classifier from a user path or bundled resource.

    Parameters
    ----------
    model_path : str or path-like, optional
        User-supplied model path. When provided, it is tried before any
        bundled model lookup.
    use_bundled_model : bool, optional
        If ``True`` and ``model_path`` is not provided, try to load the
        package resource ``hvsrpy.models/2_xgboost_peak_classifier.json``.

    Returns
    -------
    object
        Loaded ``xgboost.XGBClassifier`` instance.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "xgboost is required to use classifier_mode='xgboost'. "
            "Install xgboost or switch to classifier_mode='heuristic'."
        ) from exc

    resource_context = None
    resolved_path = None
    if model_path is not None:
        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"XGBoost model file was not found: {resolved_path}"
            )
        resource_context = nullcontext(resolved_path)
    elif use_bundled_model:
        resource = get_bundled_xgboost_model_resource()
        if resource is None:
            raise FileNotFoundError(
                "No user-supplied AutoHVSR XGBoost model path was provided and "
                f"no bundled model is available at {DEFAULT_XGBOOST_MODEL_RESOURCE}."
            )
        resource_context = resources.as_file(resource)
    else:
        raise FileNotFoundError(
            "No user-supplied AutoHVSR XGBoost model path was provided and "
            "bundled model lookup is disabled."
        )

    with resource_context as usable_path:
        model = xgb.XGBClassifier()
        model.load_model(str(usable_path))
        return model
