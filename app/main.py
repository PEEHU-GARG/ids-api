import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_BATCH = 1000

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "best_model.joblib")
    state["features"] = joblib.load(MODEL_DIR / "feature_list.joblib")
    state["encoder"] = joblib.load(MODEL_DIR / "label_encoder.joblib")
    yield
    state.clear()


app = FastAPI(
    title="Network Intrusion Detection API",
    description="Classifies network flows (CIC-IDS2017 features) as Benign or an attack type.",
    version="1.0.0",
    lifespan=lifespan,
)


class Flow(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature name -> value. Call GET /features to see the required names.",
    )


class Prediction(BaseModel):
    predicted_class: str
    is_attack: bool
    confidence: float
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    flows: list[Flow] = Field(..., min_length=1, max_length=MAX_BATCH)


def build_frame(flows: list[Flow]) -> pd.DataFrame:
    required = state["features"]
    rows = []
    for i, flow in enumerate(flows):
        missing = [f for f in required if f not in flow.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Flow {i}: missing {len(missing)} feature(s), e.g. {missing[:5]}",
            )
        bad = [f for f in required if not math.isfinite(flow.features[f])]
        if bad:
            raise HTTPException(
                status_code=422,
                detail=f"Flow {i}: NaN/infinite values in {bad[:5]}",
            )
        rows.append([flow.features[f] for f in required])
    return pd.DataFrame(rows, columns=required).astype("float32")


def predict_frame(X: pd.DataFrame) -> list[Prediction]:
    model, encoder = state["model"], state["encoder"]
    proba = model.predict_proba(X)
    class_names = list(encoder.classes_)
    out = []
    for p in proba:
        idx = int(np.argmax(p))
        label = class_names[idx]
        out.append(
            Prediction(
                predicted_class=label,
                is_attack=label != "Benign",
                confidence=round(float(p[idx]), 4),
                probabilities={c: round(float(v), 4) for c, v in zip(class_names, p)},
            )
        )
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "n_features": len(state.get("features", [])),
        "classes": list(state["encoder"].classes_) if "encoder" in state else [],
    }


@app.get("/features")
def features():
    return {"n_features": len(state["features"]), "features": state["features"]}


@app.post("/predict", response_model=Prediction)
def predict(flow: Flow):
    return predict_frame(build_frame([flow]))[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    return predict_frame(build_frame(req.flows))