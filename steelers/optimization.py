"""Stable regularized logistic fitting shared by the correction models.

Earlier fitting used a fixed gradient step for a bounded number of iterations
without inspecting the objective. Correlated feature columns can make that
step too large: a synthetic probe of ten identical all-positive rows with six
wins and four losses drove the probability to 0.999999998 (log loss about 8)
where zero weights give 0.693. This module minimizes the same penalized
logistic objective with numerically stable functions and a backtracking step:

    J(w) = mean_i [softplus(z_i) - result_i * z_i] + (penalty / 2) * sum_j w_j^2
    z_i  = offset_i + sum_j w_j * clip(feature_ij)

Features are clipped to [-4, 4], exactly as inference already does; the logit
is never clipped. The objective is non-increasing along every accepted step,
every returned weight is finite, unselected coefficients stay exactly zero,
and the fit reports whether optimization converged rather than assuming it.
"""

import math

CLIP_BOUND = 4.0
MIN_STEP = 1e-12
SUFFICIENT_DECREASE = 1e-4

CONVERGED = "converged"
MAX_ITERATIONS = "max_iterations"
STEP_TOO_SMALL = "step_too_small"
INSUFFICIENT_DATA = "insufficient_data"


def softplus(z):
    """log(1 + exp(z)), stable for every finite input."""
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def sigmoid(z):
    """1 / (1 + exp(-z)), sign-aware so no branch overflows."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    value = math.exp(z)
    return value / (1.0 + value)


def clip(value):
    """Feature clipping shared with inference in matchup/context models."""
    return min(CLIP_BOUND, max(-CLIP_BOUND, value))


def row_logit(row, indices, weights):
    return row["offset"] + sum(weights[i] * clip(row["features"][i]) for i in indices)


def objective(rows, indices, weights, penalty):
    """Penalized mean logistic loss; finite whenever the inputs are finite."""
    if not rows:
        return .5 * penalty * sum(w * w for w in weights)
    total = 0.0
    for row in rows:
        z = row_logit(row, indices, weights)
        total += softplus(z) - row["result"] * z
    return total / len(rows) + .5 * penalty * sum(w * w for w in weights)


def gradient(rows, indices, weights, penalty):
    """Gradient of ``objective`` with respect to every coefficient.

    Unselected coordinates have gradient zero (their loss derivative is
    unused and their penalty term vanishes at zero), so only the selected
    coordinates are accumulated. Finite whenever the inputs are finite.
    """
    result = [0.0] * len(weights)
    if not rows:
        return [penalty * w for w in weights]
    for row in rows:
        z = row_logit(row, indices, weights)
        error = sigmoid(z) - row["result"]
        for index in indices:
            result[index] += error * clip(row["features"][index])
    scale = 1.0 / len(rows)
    return [g * scale + penalty * w for g, w in zip(result, weights)]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(rows, indices, penalty, width):
    """Return (normalized indices, feature width). Raise ValueError on bad input."""
    if not _is_number(penalty) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be a positive finite number")
    inferred = None
    for row in rows:
        if not _is_number(row["offset"]) or not math.isfinite(row["offset"]):
            raise ValueError("Row offsets must be finite numbers")
        result = row["result"]
        if not _is_number(result) or not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError("Outcomes must be finite numbers in [0, 1]")
        features = row["features"]
        if inferred is None:
            inferred = len(features)
        elif len(features) != inferred:
            raise ValueError("Every training row must have the same feature width")
        if not all(_is_number(v) and math.isfinite(v) for v in features):
            raise ValueError("Features must be finite numbers; missing values must not enter the fit")
    if inferred is not None and width is not None and inferred != width:
        raise ValueError("Feature width does not match the training rows")
    total = inferred if rows else (width if width is not None else max((int(i) for i in indices), default=-1) + 1)
    seen = set()
    normalized = []
    for index in indices:
        if isinstance(index, bool) or not _is_number(index) or int(index) != index:
            raise ValueError("Feature indices must be integers")
        index = int(index)
        if index < 0 or index >= total:
            raise ValueError("Feature index out of range")
        if index in seen:
            raise ValueError("Feature indices must be unique")
        seen.add(index)
        normalized.append(index)
    return normalized, total


def fit_logistic_offset(rows, indices, penalty, max_iter=2000, tol=1e-7, width=None):
    """Minimize the penalized logistic objective by backtracking gradient descent.

    Rows carry ``offset``, ``features``, and ``result`` in [0, 1]. Returns
    ``(weights, status)`` where weights has the full feature width (unselected
    coefficients exactly zero) and status reports ``status``, ``converged``,
    ``iterations``, the final ``objective``, and the maximum absolute
    ``gradient_norm``. Failed fits return the best finite weights with
    ``converged=False``; callers must fall back rather than deploy them.
    """
    indices, total = _validate(rows, indices, penalty, width)
    if not rows:
        zero = [0.0] * total
        return zero, {"status": INSUFFICIENT_DATA, "converged": False, "iterations": 0,
                      "objective": objective([], indices, zero, penalty), "gradient_norm": 0.0}
    count = len(rows)
    prepared = []
    for row in rows:
        prepared.append((row["offset"], [clip(v) for v in (row["features"][i] for i in indices)], row["result"]))
    selected = len(indices)
    selected_weights = [0.0] * selected

    def evaluate(weights):
        loss = 0.0
        gradient = [0.0] * selected
        for offset, features, result in prepared:
            z = offset + sum(w * v for w, v in zip(weights, features))
            error = sigmoid(z) - result
            loss += softplus(z) - result * z
            for j in range(selected):
                gradient[j] += error * features[j]
        loss *= 1.0 / count
        for j in range(selected):
            gradient[j] = gradient[j] / count + penalty * weights[j]
        return loss + .5 * penalty * sum(w * w for w in weights), gradient

    objective_value, gradient = evaluate(selected_weights)
    status = MAX_ITERATIONS
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        gradient_norm = max(abs(g) for g in gradient) if selected else 0.0
        if gradient_norm < tol:
            status, converged = CONVERGED, True
            break
        square = sum(g * g for g in gradient)
        trial = 0.5
        accepted = None
        while trial >= MIN_STEP:
            candidate = [w - trial * g for w, g in zip(selected_weights, gradient)]
            candidate_objective = evaluate(candidate)[0]
            if candidate_objective <= objective_value - SUFFICIENT_DECREASE * trial * square:
                accepted = (candidate, candidate_objective)
                break
            trial *= 0.5
        if accepted is None:
            status = STEP_TOO_SMALL
            break
        selected_weights, objective_value = accepted
        _, gradient = evaluate(selected_weights)
    weights = [0.0] * total
    for value, index in zip(selected_weights, indices):
        weights[index] = value
    return weights, {"status": status, "converged": converged, "iterations": iterations,
                     "objective": objective_value,
                     "gradient_norm": max(abs(g) for g in gradient) if selected else 0.0}
