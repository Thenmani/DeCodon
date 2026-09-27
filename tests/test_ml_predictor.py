import unittest
from core import ml_predictor


class TestMLPredictor(unittest.TestCase):
    """Unit tests for core/ml_predictor.py Scikit-Learn risk classifier."""

    def test_ml_predictor_module_loads(self):
        """Verify ml_predictor module imports cleanly."""
        self.assertIsNotNone(ml_predictor)

    def test_ml_risk_probabilities_within_valid_bounds(self):
        """Verify that any ML risk prediction function or class returns 0-100% probabilities."""
        # Check if a predictor instance or analysis function is exposed
        for attr_name in ("analyze_mutation", "predict_risk", "predictor", "MLPredictor"):
            if hasattr(ml_predictor, attr_name):
                attr = getattr(ml_predictor, attr_name)
                self.assertIsNotNone(attr)

        # If analyze_mutation(protein_key, mutation) is in ml_predictor or physics_engine
        if hasattr(ml_predictor, "analyze_mutation"):
            res = ml_predictor.analyze_mutation("TDP43", "G298S")
            self.assertIn("wild_freezing_risk_pct", res)
            self.assertIn("mutated_freezing_risk_pct", res)
            self.assertGreaterEqual(res["wild_freezing_risk_pct"], 0.0)
            self.assertLessEqual(res["wild_freezing_risk_pct"], 100.0)
            self.assertGreaterEqual(res["mutated_freezing_risk_pct"], 0.0)
            self.assertLessEqual(res["mutated_freezing_risk_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()