# validation.py
"""
Validation of AeroGuard predictions against reference (certified analyzer) data.

Provides the metrics the review panel asked for:
  * MAE, RMSE, R^2 and Pearson correlation for the predicted AQI value
  * AQI-category agreement: exact-match rate and within-one-category rate
  * A confusion matrix over AQI categories
"""

import numpy as np

from aqi_utils import AQI_CATEGORIES, get_aqi_category

CATEGORY_NAMES = [AQI_CATEGORIES[index]['name'] for index in sorted(AQI_CATEGORIES)]


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R^2 and Pearson r over finite pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'r2': None, 'pearson': None}

    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = None

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        'n': int(y_true.size),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
    }


def category_agreement(y_true_aqi, y_pred_aqi):
    """Exact and within-one AQI category agreement, plus a confusion matrix."""
    y_true_aqi = np.asarray(y_true_aqi, dtype=float)
    y_pred_aqi = np.asarray(y_pred_aqi, dtype=float)
    mask = np.isfinite(y_true_aqi) & np.isfinite(y_pred_aqi)

    true_cat = np.array([get_aqi_category(v) for v in y_true_aqi[mask]])
    pred_cat = np.array([get_aqi_category(v) for v in y_pred_aqi[mask]])

    if true_cat.size == 0:
        return {'n': 0, 'exact_match': None, 'within_one': None, 'confusion': []}

    exact = float(np.mean(true_cat == pred_cat))
    within_one = float(np.mean(np.abs(true_cat - pred_cat) <= 1))

    confusion = np.zeros((6, 6), dtype=int)
    for actual, predicted in zip(true_cat, pred_cat):
        confusion[int(actual), int(predicted)] += 1

    return {
        'n': int(true_cat.size),
        'exact_match': exact,
        'within_one': within_one,
        'confusion': confusion.tolist(),
    }


def validate_against_reference(y_true_aqi, y_pred_aqi, name='model'):
    """Full validation report combining regression and category metrics."""
    return {
        'name': name,
        'regression': regression_metrics(y_true_aqi, y_pred_aqi),
        'categories': category_agreement(y_true_aqi, y_pred_aqi),
    }


def format_report(report):
    """Render a validation report as a printable string."""
    regression = report.get('regression', {})
    categories = report.get('categories', {})

    lines = [f"Validation report: {report.get('name', 'model')}"]
    lines.append(
        f"  n={regression.get('n')} | "
        f"MAE={_fmt(regression.get('mae'))} | "
        f"RMSE={_fmt(regression.get('rmse'))} | "
        f"R2={_fmt(regression.get('r2'))} | "
        f"r={_fmt(regression.get('pearson'))}"
    )
    exact = categories.get('exact_match')
    within = categories.get('within_one')
    lines.append(
        f"  category exact={_pct(exact)} | within-one={_pct(within)}"
    )
    return "\n".join(lines)


def _fmt(value):
    return 'n/a' if value is None else f"{value:.3f}"


def _pct(value):
    return 'n/a' if value is None else f"{value * 100:.1f}%"
