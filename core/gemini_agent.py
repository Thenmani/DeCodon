import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_mechanistic_report(analysis: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "YOUR" in api_key.upper():
        return (
            "⚠️ Gemini API Key not set in .env yet!\n\n"
            f"Local Physics Summary for {analysis['protein_name']} ({analysis['mutation']}):\n"
            f"- Codon Shift: {analysis['codon_transition']}\n"
            f"- Freezing Risk jumped from {analysis['wild_freezing_risk_pct']}% to {analysis['mutated_freezing_risk_pct']}%.\n"
        )

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are the AI Scientific Copilot for 'DeCodon — Decoding Codons the ML Way'.
    Explain in 3 short, clear sections:
    1. Layman Summary (how the liquid droplet freezes into a solid brick inside the neuron)
    2. Biophysical Mechanism (connect the codon change and physics deltas)
    3. Therapeutic Idea (how to keep the droplet liquid)

    Data:
    - Protein: {analysis['protein_name']} (Disease: {analysis['disease']})
    - Mutation: {analysis['mutation']} (Codon: {analysis['codon_transition']})
    - ML Freezing Risk: {analysis['wild_freezing_risk_pct']}% (Healthy) -> {analysis['mutated_freezing_risk_pct']}% (Mutated)
    - Physics Deltas: {analysis['physics_deltas']}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text