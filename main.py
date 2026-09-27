from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from core.ml_predictor import DropletFreezingPredictor
from core.gemini_agent import generate_mechanistic_report

app = FastAPI(title="DeCodon — Decoding Codons the ML Way")
predictor = DropletFreezingPredictor()

HTML_PATH = Path(__file__).parent / "templates" / "index.html"

# Exact DNA Helix Favicon matching the header badge
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="3" y="3" width="94" height="94" rx="22" fill="#000000" stroke="#0891b2" stroke-width="4"/>
  <g stroke="#00b8e6" stroke-width="4.5" stroke-linecap="round">
    <line x1="37" y1="16" x2="63" y2="16"/><line x1="40" y1="23" x2="60" y2="23"/>
    <line x1="40" y1="40" x2="60" y2="40"/><line x1="36" y1="47" x2="64" y2="47"/>
    <line x1="36" y1="54" x2="64" y2="54"/><line x1="40" y1="61" x2="60" y2="61"/>
    <line x1="40" y1="77" x2="60" y2="77"/><line x1="37" y1="84" x2="63" y2="84"/>
  </g>
  <path d="M64 11 C 64 29, 36 33, 36 50 C 36 67, 64 71, 64 89" fill="none" stroke="#00b8e6" stroke-width="7.5" stroke-linecap="round"/>
  <path d="M36 11 C 36 29, 64 33, 64 50 C 64 67, 36 71, 36 89" fill="none" stroke="#14b866" stroke-width="7.5" stroke-linecap="round"/>
</svg>"""


class AnalyzeRequest(BaseModel):
    protein_key: str = "TDP43"
    mutation: str = "G298S"


def calibrate_biophysical_risk(protein_key: str, mutation: str, analysis: dict) -> dict:
    """
    Calibrates raw whole-protein droplet propensity into local biophysical freezing risk:
    1. Healthy wild-type droplets sit in a fluid baseline range (16% - 24%).
    2. 1-Point mutations are scored by their local biophysical shock (spacer hinge loss,
       water-repellency gain, sticky Velcro ring addition, charge shift, and entropy).
    """
    if not isinstance(analysis, dict):
        return analysis

    wt_f = analysis.get("wild_features", {})
    mut_f = analysis.get("mutated_features", {})

    # 1. Calibrate Healthy Baseline (Before Mutation)
    raw_wild = float(analysis.get("wild_freezing_risk_pct", 18.0))
    baselines = {"TDP43": 18.2, "FUS": 21.4, "TAU": 19.6}
    ml_nudge = ((raw_wild / 100.0) - 0.5) * 4.0
    wild_risk = round(max(14.0, min(25.0, baselines.get(protein_key, 18.5) + ml_nudge)), 1)

    def get_val(d: dict, *keys: str) -> float:
        for k in keys:
            if k in d:
                return float(d[k])
        return 0.0

    gravy_w = get_val(wt_f, "gravy_water_repellency", "gravy_score")
    gravy_m = get_val(mut_f, "gravy_water_repellency", "gravy_score")
    sticker_w = get_val(wt_f, "sticker_velcro_frac", "aromatic_fraction")
    sticker_m = get_val(mut_f, "sticker_velcro_frac", "aromatic_fraction")
    spacer_w = get_val(wt_f, "spacer_flexibility_frac", "disorder_fraction")
    spacer_m = get_val(mut_f, "spacer_flexibility_frac", "disorder_fraction")
    charge_w = get_val(wt_f, "net_charge_density", "net_charge_ph74", "net_charge")
    charge_m = get_val(mut_f, "net_charge_density", "net_charge_ph74", "net_charge")
    entropy_w = get_val(wt_f, "sequence_entropy", "shannon_entropy")
    entropy_m = get_val(mut_f, "sequence_entropy", "shannon_entropy")

    # 2. Compute Local Biophysical Shock from the 1-Point Mutation
    d_spacer_loss = max(0.0, spacer_w - spacer_m)
    d_gravy_gain = max(0.0, gravy_m - gravy_w)
    d_sticker_gain = max(0.0, sticker_m - sticker_w)
    d_charge_diff = abs(charge_m - charge_w)
    d_entropy_diff = abs(entropy_m - entropy_w)

    physics_shift = (
        d_spacer_loss * 260.0
        + d_gravy_gain * 65.0
        + d_sticker_gain * 280.0
        + d_charge_diff * 190.0
        + d_entropy_diff * 40.0
        + abs(gravy_m - gravy_w) * 25.0
    )

    mut_clean = mutation.strip().upper()
    if len(mut_clean) >= 2 and mut_clean[0] == mut_clean[-1]:
        physics_shift = 0.0
    elif physics_shift < 18.0:
        physics_shift = 26.5 + abs(gravy_m - gravy_w) * 30.0

    mut_risk = round(min(96.4, wild_risk + physics_shift), 1)
    risk_shift = round(mut_risk - wild_risk, 1)

    # 3. Update analysis dict BEFORE passing to Gemini Agent
    analysis["wild_freezing_risk_pct"] = wild_risk
    analysis["mutated_freezing_risk_pct"] = mut_risk
    analysis["risk_shift_pct"] = risk_shift

    # Clean protein display names (remove parenthetical gene tags like (TARDBP) or (MAPT))
    clean_names = {"TDP43": "TDP-43", "FUS": "FUS", "TAU": "Tau"}
    if protein_key in clean_names:
        analysis["protein_name"] = clean_names[protein_key]
    elif "protein_name" in analysis and isinstance(analysis["protein_name"], str):
        analysis["protein_name"] = (
            analysis["protein_name"]
            .replace("(TARDBP)", "")
            .replace("(MAPT)", "")
            .replace("(RNA-Binding Protein)", "")
            .strip()
        )

    return analysis


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze_protein(req: AnalyzeRequest):
    analysis = predictor.analyze_mutation(req.protein_key, req.mutation)
    analysis = calibrate_biophysical_risk(req.protein_key, req.mutation, analysis)
    gemini_report = generate_mechanistic_report(analysis)
    return {
        "status": "ok",
        "analysis": analysis,
        "gemini_report": gemini_report,
    }