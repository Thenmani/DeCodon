import asyncio
import inspect
import unittest
from unittest.mock import patch
import main


class TestDeCodonAPIPipeline(unittest.TestCase):
    """End-to-end integration tests for the FastAPI application in main.py."""

    def test_fastapi_app_and_routes_registered(self):
        """Verify FastAPI app instance and required routes ('/' and '/api/analyze') exist."""
        self.assertTrue(hasattr(main, "app"), "main.py must define 'app = FastAPI()'")
        route_paths = [ getattr(r, "path", "") for r in main.app.routes ]
        self.assertIn("/", route_paths)
        self.assertIn("/api/analyze", route_paths)

    def _call_analyze_endpoint(self, protein_key: str, mutation: str):
        """Helper to invoke /api/analyze endpoint directly (sync or async)."""
        analyze_route = next(
            r for r in main.app.routes if getattr(r, "path", "") == "/api/analyze"
        )
        endpoint_fn = analyze_route.endpoint

        # Build request payload model if endpoint expects a Pydantic BaseModel
        sig = inspect.signature(endpoint_fn)
        params = list(sig.parameters.values())
        if params and hasattr(params[0].annotation, "model_fields"):
            model_cls = params[0].annotation
            payload = model_cls(protein_key=protein_key, mutation=mutation)
        else:
            payload = {"protein_key": protein_key, "mutation": mutation}

        if inspect.iscoroutinefunction(endpoint_fn):
            res = asyncio.run(endpoint_fn(payload))
        else:
            res = endpoint_fn(payload)

        # If FastAPI JSONResponse is returned, decode its body
        if hasattr(res, "body"):
            import json
            return json.loads(res.body.decode("utf-8"))
        return res

    @patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False)
    def test_end_to_end_benchmark_mutations(self):
        """Test full ML + Physics pipeline for ALS (TDP-43, FUS) and Alzheimer's (Tau)."""
        benchmarks = [
            ("TDP43", "G298S"),
            ("FUS", "G156E"),
            ("TAU", "P301L"),
        ]

        for protein_key, mutation in benchmarks:
            with self.subTest(protein=protein_key, mutation=mutation):
                data = self._call_analyze_endpoint(protein_key, mutation)

                self.assertIsInstance(data, dict)
                self.assertIn("analysis", data)
                self.assertIn("gemini_report", data)

                analysis = data["analysis"]
                self.assertIn("protein_name", analysis)
                self.assertIn("disease", analysis)
                self.assertIn("codon_transition", analysis)
                self.assertIn("wild_freezing_risk_pct", analysis)
                self.assertIn("mutated_freezing_risk_pct", analysis)
                self.assertIn("risk_shift_pct", analysis)
                self.assertIn("wild_features", analysis)
                self.assertIn("mutated_features", analysis)
                self.assertIn("physics_deltas", analysis)

                # Verify 6 biophysical features are returned
                self.assertEqual(len(analysis["wild_features"]), 6)
                self.assertEqual(len(analysis["mutated_features"]), 6)


if __name__ == "__main__":
    unittest.main()