import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Silence the harmless google-genai AFC console warning
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

# Automatic fallback list in case one model has a 503 traffic spike
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


def generate_mechanistic_report(analysis: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "YOUR" in api_key.upper():
        return _local_fallback_report(analysis, "API key not configured")

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are the AI Scientific Copilot for 'DeCodon — Decoding Codons the ML Way'.
    Explain in 3 short, punchy sections using markdown bullets:
    1. **Layman Summary**: How this 1-letter mutation causes the healthy liquid protein droplet inside the neuron to "freeze" into a solid brick.
    2. **Biophysical Mechanism**: Connect the codon change ({analysis['codon_transition']}) and physics shifts to why the droplet loses fluidity.
    3. **Therapeutic Hypothesis**: One concrete molecular strategy to keep the droplet liquid.

    Data:
    - Protein: {analysis['protein_name']} (Disease: {analysis['disease']})
    - Mutation: {analysis['mutation']} (Codon: {analysis['codon_transition']})
    - ML Freezing Risk: {analysis['wild_freezing_risk_pct']}% (Healthy) -> {analysis['mutated_freezing_risk_pct']}% (Mutated) [Shift: +{analysis['risk_shift_pct']}%]
    - Physics Deltas: {analysis['physics_deltas']}
    """

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                ),
            )
            return response.text
        except Exception as e:
            print(f"⚠️ {model_name} busy ({e}), trying next backup model...")

    return _local_fallback_report(analysis, "All Gemini endpoints temporarily at capacity")


def _local_fallback_report(analysis: dict, reason: str) -> str:
    return (
        f"### 🧠 DeCodon Mechanistic Report ({analysis['protein_name']} — {analysis['mutation']})\n"
        f"*(Generated via Local Physics Engine — {reason})*\n\n"
        f"1. **Layman Summary (Liquid-to-Solid Freeze):**\n"
        f"   - In healthy neurons, **{analysis['protein_name']}** forms dynamic liquid droplets. "
        f"The **{analysis['mutation']}** mutation ({analysis['codon_transition']}) spikes the droplet freezing risk from "
        f"**{analysis['wild_freezing_risk_pct']}%** to **{analysis['mutated_freezing_risk_pct']}%**, driving irreversible solid aggregation linked to **{analysis['disease']}**.\n\n"
        f"2. **Biophysical Mechanism:**\n"
        f"   - This single-codon substitution alters local chain flexibility and intermolecular cross-linking (`{analysis['physics_deltas']}`), "
        f"trapping the condensate in a high-viscosity gel/solid state.\n\n"
        f"3. **Therapeutic Hypothesis:**\n"
        f"   - Introduce small-molecule condensate modifiers or steric-shielding antisense oligonucleotides to restore spacer flexibility and prevent pathological cross-linking."
    )