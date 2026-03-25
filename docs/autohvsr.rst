AutoHVSR
========

``hvsrpy`` provides an optional AutoHVSR post-processing layer for
window-level resonance grouping after standard HVSR processing. The
expected workflow is:

1. preprocess records,
2. process them into an HVSR result,
3. run ``process_autohvsr(...)`` on that processed result.

Typical usage is:

.. code-block:: python

   import hvsrpy

   records = hvsrpy.preprocess(records, preprocessing_settings)
   hvsr = hvsrpy.process(records, processing_settings)
   result = hvsrpy.process_autohvsr(hvsr)

   print(result.resonances)

To use a custom XGBoost model explicitly:

.. code-block:: python

   settings = hvsrpy.AutoHvsrSettings(
       classifier_mode="xgboost",
       classifier_model_path="path/to/2_xgboost_peak_classifier.json",
   )
   result = hvsrpy.process_autohvsr(hvsr, settings=settings)

The preferred input is ``HvsrTraditional``. AutoHVSR does not require
the web application.

Classifier modes
----------------

``AutoHvsrSettings.classifier_mode`` supports:

* ``"heuristic"``: use the built-in rule-based classifier only.
* ``"xgboost"``: require a usable XGBoost model.
* ``"auto"``: try XGBoost first, then fall back to the heuristic path
  with a warning if XGBoost cannot be used.

The default remains ``"heuristic"`` because no bundled AutoHVSR model
artifact is currently shipped with the package. XGBoost is a supported
dependency, but the original AutoHVSR model artifact from ``hvsrweb`` is
not currently bundled in this repository.

Model loading order
-------------------

When XGBoost is used, AutoHVSR resolves the model in this order:

1. ``classifier_model_path`` if one is supplied.
2. the bundled package resource
   ``hvsrpy.models/2_xgboost_peak_classifier.json`` if
   ``classifier_use_bundled_model=True``.

At present, step 2 does not find a model in this repository, so a
user-supplied model file is required for the XGBoost path. ``"auto"``
therefore falls back to the heuristic classifier with a warning unless a
custom model path is provided or a bundled model is added later.

Validation and errors
---------------------

``AutoHvsrSettings`` validates the fixed AutoHVSR feature schema, the
supported resonance distributions, and the clustering thresholds.
``valid_window_boolean_mask`` is respected when present; if it exists
but does not match the number of HVSR windows, AutoHVSR raises
``ValueError`` instead of silently using all windows.
