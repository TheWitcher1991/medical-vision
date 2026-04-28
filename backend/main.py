from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import os
from datetime import datetime
import torch

from model import MedClsNet
from inference import predict, CLASS_MAP, DEVICE
from llm import get_llm_analysis, get_recommendations

app = FastAPI(title="Medical Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "medclsnet.pth")

model = None


def load_model():
    global model
    model = MedClsNet(num_classes=4).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        import torch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


load_model()


class DiagnosisResponse(BaseModel):
    diagnosis: str
    diagnosis_ru: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    llm_analysis: Optional[str] = None
    recommendations: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=model is not None, device=DEVICE)


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        image_bytes = await file.read()
        pred_class, confidence, probabilities = predict(image_bytes, model)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        llm_analysis = get_llm_analysis(pred_class, confidence, probabilities)
        recommendations = get_recommendations(pred_class)
        
        return DiagnosisResponse(
            diagnosis=pred_class,
            diagnosis_ru=CLASS_MAP.get(pred_class, pred_class),
            confidence=confidence,
            probabilities=probabilities,
            timestamp=timestamp,
            llm_analysis=llm_analysis,
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    return {"classes": CLASS_MAP}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)