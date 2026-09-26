from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.ml_predictor import DropletFreezingPredictor
from core.gemini_agent import generate_mechanistic_report

app = FastAPI(title="DeCodon — Decoding Codons the ML Way")
templates = Jinja2Templates(directory="templates")
predictor = DropletFreezingPredictor()


class AnalyzeRequest(BaseModel):
    protein_key: str = "TDP43"
    mutation: str = "G298S"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze_protein(req: AnalyzeRequest):
    analysis = predictor.analyze_mutation(req.protein_key, req.mutation)
    gemini_report = generate_mechanistic_report(analysis)
    return {
        "status": "ok",
        "analysis": analysis,
        "gemini_report": gemini_report,
    }