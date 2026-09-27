from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.ml_predictor import DropletFreezingPredictor
from core.gemini_agent import generate_mechanistic_report

app = FastAPI(title="DeCodon — Decoding Codons the ML Way")
predictor = DropletFreezingPredictor()

HTML_PATH = Path(__file__).parent / "templates" / "index.html"


class AnalyzeRequest(BaseModel):
    protein_key: str = "TDP43"
    mutation: str = "G298S"


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze_protein(req: AnalyzeRequest):
    analysis = predictor.analyze_mutation(req.protein_key, req.mutation)
    gemini_report = generate_mechanistic_report(analysis)
    return {
        "status": "ok",
        "analysis": analysis,
        "gemini_report": gemini_report,
    }