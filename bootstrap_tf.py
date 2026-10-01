# bootstrap_tf.py
"""
Import TensorFlow before pandas/pyarrow/streamlit.

On some macOS environments (observed with pandas 3.x, pyarrow 24.x and
TensorFlow 2.21 on arm64), if pandas or pyarrow is imported before
TensorFlow, every subsequent ``model.fit`` and ``model.predict`` call
deadlocks. No thread-limit environment variable fixes it; only import
order does.

Import this module as the very first import in any entry point that uses
both pandas and TensorFlow::

    import bootstrap_tf  # noqa: F401  (must be first)

It is a no-op when TensorFlow is not installed.
"""

try:  # pragma: no cover - trivial import shim
    import tensorflow  # noqa: F401
except ImportError:
    pass
