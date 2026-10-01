# explain.py
"""
SHAP-based explanations for model predictions.

Answers "why did the model predict this?" by attributing a prediction to its
input features. The Random Forest is explained exactly with TreeSHAP.

The LSTM is not explained here: SHAP for recurrent models is approximate and
computationally heavy, and the Random Forest is the model that produces the
headline air-quality category. If needed, a KernelSHAP surrogate can be added
later.
"""

import numpy as np

from preprocessing import FEATURE_COLS, label_for

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - shap is an optional dependency
    SHAP_AVAILABLE = False


FEATURE_DESCRIPTIONS = {
    'temp': 'Temperature',
    'hum': 'Humidity',
    'gas': 'Gas/VOC response',
}

FEATURE_UNITS = {
    'temp': '°C',
    'hum': '%',
    'gas': 'µg/m³ benzene-equiv.',
}


class ModelExplainer:
    """Exact TreeSHAP explainer for a tree-ensemble model."""

    def __init__(self, model, feature_names=None):
        if not SHAP_AVAILABLE:
            raise RuntimeError("shap is not installed; run `pip install shap`.")
        self.model = model
        self.feature_names = list(feature_names or FEATURE_COLS)
        self._explainer = shap.TreeExplainer(model)

    def _contributions(self, features, predicted_class):
        """Return a (n_features,) contribution vector for one sample."""
        raw = self._explainer.shap_values(features)
        values = np.asarray(raw)

        if values.ndim == 2:
            # Binary/single-output: (n_samples, n_features)
            return values[0]

        if values.ndim == 3:
            # Normalise to (n_samples, n_features, n_classes).
            if values.shape[1] != len(self.feature_names):
                values = np.transpose(values, (1, 2, 0))
            return values[0, :, int(predicted_class)]

        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    def explain(self, features, predicted_class, top_k=3):
        """
        Explain one prediction.

        Parameters
        ----------
        features : array-like, shape (1, n_features)
        predicted_class : int
        top_k : int
            How many top factors to return (all factors are still included).

        Returns
        -------
        dict
        """
        features = np.asarray(features, dtype=float)
        contributions = self._contributions(features, predicted_class)

        factors = []
        for name, value, contribution in zip(
            self.feature_names, features[0], contributions
        ):
            factors.append({
                'feature': name,
                'description': FEATURE_DESCRIPTIONS.get(name, name),
                'value': round(float(value), 3),
                'unit': FEATURE_UNITS.get(name, ''),
                'contribution': round(float(contribution), 4),
                'direction': 'increases' if contribution > 0 else 'decreases',
            })

        factors.sort(key=lambda factor: abs(factor['contribution']), reverse=True)

        return {
            'predicted_class': int(predicted_class),
            'predicted_label': label_for(predicted_class),
            'top_factors': factors[:top_k],
            'all_factors': factors,
            'method': 'TreeSHAP',
        }


def explain_prediction(model, features, predicted_class, feature_names=None, top_k=3):
    """Convenience wrapper for a one-off explanation."""
    return ModelExplainer(model, feature_names).explain(
        features, predicted_class, top_k=top_k
    )
