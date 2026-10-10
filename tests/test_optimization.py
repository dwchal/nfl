import math
import unittest

from steelers.optimization import (INSUFFICIENT_DATA, CONVERGED, fit_logistic_offset,
                                   gradient, objective, sigmoid, softplus)


def probe_rows(width=34):
    """The review's synthetic failure: identical all-positive feature vectors."""
    return [{"offset": 0.0, "features": [4.0] * width, "result": float(i < 6)} for i in range(10)]


class StableFunctionsTests(unittest.TestCase):
    def test_softplus_matches_log1pexp_without_overflow(self):
        for value in (-1000.0, -4.0, -0.5, 0.0, 1.3, 12.0, 1000.0):
            self.assertTrue(math.isfinite(softplus(value)))
            if abs(value) < 30:
                self.assertAlmostEqual(softplus(value), math.log1p(math.exp(value)), places=9)

    def test_sign_aware_sigmoid_matches_reference(self):
        for value in (-1000.0, -12.0, -0.2, 0.0, 0.7, 9.0, 1000.0):
            self.assertTrue(math.isfinite(sigmoid(value)))
        for value in (-20.0, -3.5, -0.1, 0.0, 2.2, 20.0):
            self.assertAlmostEqual(sigmoid(value), 1 / (1 + math.exp(-value)), places=12)


class CorrelatedFeatureFailureTests(unittest.TestCase):
    def test_identical_rows_do_not_drive_extreme_probability(self):
        rows = probe_rows()
        weights, status = fit_logistic_offset(rows, tuple(range(34)), penalty=0.1)
        self.assertTrue(all(math.isfinite(w) for w in weights))
        self.assertTrue(status["converged"])
        self.assertEqual(status["status"], CONVERGED)
        # Mixed 6-win/4-loss labels cannot become a near-certain prediction.
        probability = sigmoid(sum(w * 4.0 for w in weights))
        self.assertTrue(0.5 <= probability <= 0.8, probability)

    def test_objective_never_worse_than_zero_initialization(self):
        rows = probe_rows()
        indices = tuple(range(34))
        weights, status = fit_logistic_offset(rows, indices, penalty=0.1)
        zero = [0.0] * 34
        self.assertLessEqual(status["objective"], objective(rows, indices, zero, 0.1) + 1e-9)

    def test_severely_correlated_and_unbalanced_rows_stay_finite(self):
        # One-sided outcomes with a large constant column used to diverge too.
        rows = [{"offset": 3.0, "features": [4.0, 4.0, -4.0], "result": 1.0} for _ in range(40)]
        weights, status = fit_logistic_offset(rows, (0, 1, 2), penalty=0.1)
        self.assertTrue(all(math.isfinite(w) for w in weights))
        self.assertLessEqual(status["objective"], objective(rows, (0, 1, 2), [0.0] * 3, 0.1) + 1e-9)
        self.assertTrue(0.0 <= sigmoid(3.0 + sum(weights[i] * 4.0 * (1 if i < 2 else -1) for i in range(3))) <= 1.0)

    def test_failed_fit_reports_not_converged_and_stays_finite(self):
        rows = [{"offset": 0.0, "features": [4.0] * 34, "result": 1.0} for _ in range(10)]
        weights, status = fit_logistic_offset(rows, tuple(range(34)), penalty=0.1, max_iter=3)
        self.assertFalse(status["converged"])
        self.assertNotEqual(status["status"], CONVERGED)
        self.assertTrue(all(math.isfinite(w) for w in weights))
        self.assertLessEqual(status["objective"], objective(rows, tuple(range(34)), [0.0] * 34, 0.1) + 1e-9)


class GradientTests(unittest.TestCase):
    def test_analytical_gradient_matches_central_finite_differences(self):
        import random
        rng = random.Random(7)
        rows = [{"offset": rng.uniform(-2, 2),
                 "features": [rng.uniform(-4, 4) for _ in range(5)],
                 "result": rng.choice([0.0, 0.5, 1.0])} for _ in range(30)]
        indices = (0, 2, 4)
        for draw in range(5):
            weights = [rng.uniform(-1, 1) for _ in range(5)]
            expected = gradient(rows, indices, weights, 0.1)
            step = 1e-6
            for j in range(5):
                plus, minus = list(weights), list(weights)
                plus[j] += step
                minus[j] -= step
                numeric = (objective(rows, indices, plus, 0.1) - objective(rows, indices, minus, 0.1)) / (2 * step)
                self.assertAlmostEqual(expected[j], numeric, delta=1e-5, msg=f"coordinate {j}")

    def test_nonselected_coordinates_have_zero_derivative(self):
        rows = [{"offset": 0.0, "features": [1.0, 2.0], "result": 1.0}]
        self.assertEqual(gradient(rows, (0,), [0.0, 0.0], 0.1)[1], 0.0)


class ValidationTests(unittest.TestCase):
    def test_empty_training_returns_zero_weights_and_insufficient_data(self):
        weights, status = fit_logistic_offset([], (0, 3), penalty=0.1, width=6)
        self.assertEqual(weights, [0.0] * 6)
        self.assertEqual(status["status"], INSUFFICIENT_DATA)
        self.assertFalse(status["converged"])
        self.assertEqual(status["iterations"], 0)

    def test_constant_columns_fit_and_converge(self):
        rows = [{"offset": 0.0, "features": [1.0, 1.0], "result": float(i % 2)} for i in range(50)]
        weights, status = fit_logistic_offset(rows, (0, 1), penalty=0.05)
        self.assertTrue(status["converged"])
        self.assertTrue(all(math.isfinite(w) for w in weights))

    def test_invalid_inputs_raise(self):
        good = [{"offset": 0.0, "features": [1.0, -1.0], "result": 1.0},
                {"offset": 0.5, "features": [0.0, 2.0], "result": 0.0}]
        cases = (
            [{"offset": float("nan"), "features": [1.0, -1.0], "result": 1.0}],
            [{"offset": 0.0, "features": [1.0, float("inf")], "result": 1.0}],
            [{"offset": 0.0, "features": [1.0, -1.0], "result": 1.5}],
            [{"offset": 0.0, "features": [1.0, -1.0], "result": -0.2}],
            [{"offset": 0.0, "features": [1.0, "NA"], "result": 1.0}],
            good + [{"offset": 0.0, "features": [1.0], "result": 0.0}],  # width mismatch
        )
        for rows in cases:
            with self.assertRaises(ValueError):
                fit_logistic_offset(rows, (0, 1), penalty=0.1)
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, (0, 1), penalty=0.0)
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, (0, 1), penalty=float("nan"))
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, (0, 2), penalty=0.1)      # out of range
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, (0, 0), penalty=0.1)      # duplicate
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, ("x", 1), penalty=0.1)    # non-integer index
        with self.assertRaises(ValueError):
            fit_logistic_offset(good, (0, 1), penalty=0.1, width=5)

    def test_nonconsecutive_indices_keep_other_coefficients_exactly_zero(self):
        rows = [{"offset": 0.0,
                 "features": [1.0 if i % 2 else -1.0, 0.0, -1.5, 2.0 if i % 2 else -2.0, 0.0],
                 "result": float(i % 2)} for i in range(80)]
        weights, status = fit_logistic_offset(rows, (0, 3), penalty=0.05)
        self.assertEqual(len(weights), 5)
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[2], 0.0)
        self.assertEqual(weights[4], 0.0)
        self.assertTrue(status["converged"])
        # The selected coefficients must move the probability away from even odds.
        base = sigmoid(0.0)
        with_signal = sigmoid(weights[0] * 1.0 + weights[3] * 2.0)
        self.assertNotAlmostEqual(with_signal, base, places=6)


if __name__ == "__main__":
    unittest.main()
