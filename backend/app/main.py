"""
main.py
--------
FastAPI backend for PromptShield-AI (Member 2 deliverable).

Run from inside backend/:
    uvicorn app.main:app --reload
"""

from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import database, schemas
from .attack_categories import categorize, explain, safe_rewrite
from .inference import predict as run_prediction

app = FastAPI(title="PromptShield-AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    database.init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=schemas.PredictResponse)
def predict_prompt(request: schemas.PredictRequest, db: Session = Depends(database.get_db)):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        result = run_prediction(request.prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    attack_category = None
    explanation = None
    safe_prompt = request.prompt

    if result["is_injection"]:
        attack_category = categorize(request.prompt)
        explanation = explain(attack_category, result["confidence"])
        safe_prompt = safe_rewrite(request.prompt, attack_category)
    else:
        explanation = f"No injection patterns detected. (Model confidence: {result['confidence']:.2f}%)"

    record = database.PredictionHistory(
        prompt_text=request.prompt,
        label=result["label"],
        is_injection=result["is_injection"],
        confidence=result["confidence"],
        attack_category=attack_category,
        explanation=explanation,
        safe_prompt=safe_prompt,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return schemas.PredictResponse(
        prompt=request.prompt,
        label=result["label"],
        is_injection=result["is_injection"],
        confidence=result["confidence"],
        attack_category=attack_category,
        explanation=explanation,
        safe_prompt=safe_prompt,
    )


@app.get("/history", response_model=List[schemas.HistoryItem])
def get_history(limit: int = 50, db: Session = Depends(database.get_db)):
    records = (
        db.query(database.PredictionHistory)
        .order_by(database.PredictionHistory.id.desc())
        .limit(limit)
        .all()
    )
    return [
        schemas.HistoryItem(
            id=r.id,
            prompt=r.prompt_text,
            label=r.label,
            is_injection=r.is_injection,
            confidence=r.confidence,
            attack_category=r.attack_category,
            explanation=r.explanation,
            safe_prompt=r.safe_prompt,
            created_at=r.created_at,
        )
        for r in records
    ]


@app.delete("/history")
def clear_history(db: Session = Depends(database.get_db)):
    db.query(database.PredictionHistory).delete()
    db.commit()
    return {"message": "History cleared."}