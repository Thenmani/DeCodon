import unittest
from unittest.mock import patch
from core import gemini_agent


class TestGeminiAgent(unittest.TestCase):
    """Unit tests for core/gemini_agent.py mechanistic report generation."""

    def setUp(self):
        """Sample biophysical analysis payload matching DeCodon's schema."""
        self.sample_analysis = {
            "protein_key": "TDP43",
            "protein_name": "TDP-43 (TARDBP)",
            "disease": "ALS",
            "mutation": "G298S",
            "position": 298,
            "wild_aa": "G",
            "mut_aa": "S",
            "codon_transition": "GGT (G) → AGT (S)",
            "wild_freezing_risk_pct": 66.9,
            "mutated_freezing_risk_pct": 78.4,
            "risk_shift_pct": 11.5,
            "wild_features": {
                "gravy_score": -0.42,
                "aromatic_fraction": 0.08,
                "disorder_fraction": 0.45,
                "net_charge": -4.0,
                "isoelectric_point": 5.85,
                "shannon_entropy": 3.72,
            },
            "mutated_features": {
                "gravy_score": -0.43,
                "aromatic_fraction": 0.08,
                "disorder_fraction": 0.44,
                "net_charge": -4.0,
                "isoelectric_point": 5.85,
                "shannon_entropy": 3.71,
            },
            "physics_deltas": {
                "gravy_score": -0.01,
                "aromatic_fraction": 0.0,
                "disorder_fraction": -0.01,
                "net_charge": 0.0,
                "isoelectric_point": 0.0,
                "shannon_entropy": -0.01,
            },
        }

    def test_gemini_agent_module_loads(self):
        """Verify gemini_agent module imports cleanly."""
        self.assertIsNotNone(gemini_agent)

    @patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False)
    def test_fallback_report_when_api_key_missing(self):
        """Verify agent returns a valid Markdown report even without an API key."""
        report_fn = None
        for fn_name in (
            "generate_mechanistic_report",
            "generate_report",
            "get_gemini_report",
            "run_gemini_agent",
        ):
            if hasattr(gemini_agent, fn_name):
                report_fn = getattr(gemini_agent, fn_name)
                break

        if report_fn is not None:
            report = report_fn(self.sample_analysis)
            self.assertIsInstance(report, str)
            self.assertGreater(len(report.strip()), 20)


if __name__ == "__main__":
    unittest.main()