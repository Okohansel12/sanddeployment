"""
SandSight prediction API.

Serves the two trained models (point + P90 quantile) from the SANDSIGHT-X v2
pipeline. Feature engineering here mirrors the training pipeline exactly, so
predictions match what was validated in the walk-forward evaluation.
"""
import json
from datetime import date
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(title="SandSight Prediction API", version="1.0")

# NOTE: restrict this to your actual frontend origin before going to
# production. "*" is fine for a first deploy / internal testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- load once
point_model = joblib.load(MODELS_DIR / "point_model_v2.pkl")
point_scaler = joblib.load(MODELS_DIR / "point_scaler_v2.pkl")
p90_model = joblib.load(MODELS_DIR / "p90_model_v2.pkl")
p90_scaler = joblib.load(MODELS_DIR / "p90_scaler_v2.pkl")
FEATURES = json.load(open(MODELS_DIR / "final_features.json"))
CFG = json.load(open(MODELS_DIR / "serving_config.json"))


# ---------------------------------------------------------------- schemas
class MonthRecord(BaseModel):
    date: date
    oil_vol_bbls: float = Field(ge=0)
    water_vol_bbls: float = Field(ge=0)
    gas_vol_mscf: float = Field(ge=0)
    cond_vol_bbls: float = Field(ge=0, default=0)
    prod_days: int = Field(ge=1, le=31)
    bean_size: float = Field(ge=0)
    sand_pptb: float = Field(ge=0)

    @field_validator("oil_vol_bbls", "water_vol_bbls", "gas_vol_mscf")
    @classmethod
    def not_all_zero_check(cls, v):
        return v  # cross-field all-zero check happens in the request validator


class TargetMonth(BaseModel):
    date: date
    oil_vol_bbls: float = Field(ge=0)
    water_vol_bbls: float = Field(ge=0)
    gas_vol_mscf: float = Field(ge=0)
    cond_vol_bbls: float = Field(ge=0, default=0)
    prod_days: int = Field(ge=1, le=31)
    bean_size: float = Field(ge=0)


class PredictRequest(BaseModel):
    well_id: str
    field: Optional[str] = None
    history: List[MonthRecord]
    target_month: TargetMonth
    cum_oil_before_history: float = Field(
        default=0.0,
        description="Cumulative oil (bbls) produced by this well BEFORE the "
                    "first history row. Omit only for a well with negligible "
                    "prior production -- otherwise Cum_Oil / log_cum_oil will "
                    "understate the well's maturity.",
    )
    production_start_date: Optional[date] = Field(
        default=None,
        description="First-ever production date for this well. If omitted, "
                    "Production_Age is estimated from the supplied history "
                    "only, which understates age for older wells.",
    )
    has_petrophysical_log: bool = False
    zone_gr: Optional[float] = None
    zone_den: Optional[float] = None
    zone_por: Optional[float] = None
    zone_res_dep: Optional[float] = None

    @field_validator("history")
    @classmethod
    def history_length(cls, v):
        if not (3 <= len(v) <= 12):
            raise ValueError("history must contain between 3 and 12 monthly records")
        return v

    @field_validator("history")
    @classmethod
    def all_zero_rows(cls, v):
        for i, m in enumerate(v):
            if m.oil_vol_bbls == 0 and m.water_vol_bbls == 0 and m.gas_vol_mscf == 0:
                raise ValueError(f"history row {i} has zero oil, water AND gas -- rejected")
        return v

    @field_validator("history")
    @classmethod
    def chronological(cls, v):
        dates = [m.date for m in v]
        if dates != sorted(dates):
            raise ValueError("history rows must be in chronological order")
        return v


class PredictResponse(BaseModel):
    well_id: str
    expected_pptb: float
    ceiling_p90_pptb: float
    risk_level: str
    months_used: int
    imputed_features: List[str]
    gap_warning: bool


# ---------------------------------------------------------------- feature engineering
def build_feature_row(req: PredictRequest) -> tuple[dict, bool]:
    rows = [m.model_dump() for m in req.history]
    rows.append({**req.target_month.model_dump(), "sand_pptb": np.nan})
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    gap_days = d["date"].diff().dt.days
    gap_warning = bool((gap_days.dropna() > 45).any() or (gap_days.dropna() < 20).any())

    d["Oil_Rate_bpd"] = d["oil_vol_bbls"] / d["prod_days"]
    d["Water_Rate_bpd"] = d["water_vol_bbls"] / d["prod_days"]
    d["Gas_Rate_mscfd"] = d["gas_vol_mscf"] / d["prod_days"]
    d["Cond_Rate_bpd"] = d["cond_vol_bbls"] / d["prod_days"]
    d["Liquid_Rate_bpd"] = d["Oil_Rate_bpd"] + d["Water_Rate_bpd"]
    d["Water_Cut"] = d["Water_Rate_bpd"] / d["Liquid_Rate_bpd"].replace(0, np.nan)
    d["GOR"] = (d["Gas_Rate_mscfd"] * 1000) / d["Oil_Rate_bpd"].replace(0, np.nan)
    d["WOR"] = d["Water_Rate_bpd"] / d["Oil_Rate_bpd"].replace(0, np.nan)
    d["GOR"] = d["GOR"].clip(upper=CFG["GOR_cap"])
    d["WOR"] = d["WOR"].clip(upper=CFG["WOR_cap"])

    d["Bean Size"] = d["bean_size"]
    d["Delta_Bean_Size"] = d["Bean Size"].diff().fillna(0)

    start_date = pd.to_datetime(req.production_start_date) if req.production_start_date else d["date"].iloc[0]
    d["Production_Age"] = (d["date"].dt.year - start_date.year) * 12 + (d["date"].dt.month - start_date.month)
    d["Month"] = d["date"].dt.month
    d["Year"] = d["date"].dt.year
    d["Cum_Oil"] = req.cum_oil_before_history + d["oil_vol_bbls"].cumsum()
    d["log_cum_oil"] = np.log1p(d["Cum_Oil"])

    s = d["sand_pptb"]
    for lag in [1, 2, 3, 6, 12]:
        d[f"sand_lag{lag}"] = np.log1p(s.shift(lag))
    d["sand_roll3"] = np.log1p(s.shift(1).rolling(3, min_periods=1).mean())
    d["sand_roll6"] = np.log1p(s.shift(1).rolling(6, min_periods=1).mean())
    d["sand_roll12"] = np.log1p(s.shift(1).rolling(12, min_periods=1).mean())
    d["sand_std6"] = s.shift(1).rolling(6, min_periods=1).std()
    d["sand_max6"] = np.log1p(s.shift(1).rolling(6, min_periods=1).max())
    d["sand_min6"] = np.log1p(s.shift(1).rolling(6, min_periods=1).min())
    d["sand_trend"] = d["sand_lag1"] - d["sand_lag3"]
    d["sand_dev_roll6"] = d["sand_lag1"] - d["sand_roll6"]
    d["sand_ewm"] = np.log1p(s.shift(1).ewm(halflife=3).mean())
    d["sand_cummax"] = np.log1p(s.shift(1).cummax())

    for c in ["Liquid_Rate_bpd", "Water_Cut", "GOR", "Bean Size", "Oil_Rate_bpd"]:
        d[f"{c}_lag1"] = d[c].shift(1)
        d[f"{c}_chg"] = d[c] - d[f"{c}_lag1"]
    d["bean_chg3"] = d["Bean Size"] - d["Bean Size"].shift(3)
    d["wc_chg3"] = d["Water_Cut"] - d["Water_Cut"].shift(3)
    d["liq_chg3"] = d["Liquid_Rate_bpd"] - d["Liquid_Rate_bpd"].shift(3)
    d["liq_roll3"] = d["Liquid_Rate_bpd"].shift(1).rolling(3, min_periods=1).mean()
    d["liq_ratio"] = d["Liquid_Rate_bpd"] / d["liq_roll3"].replace(0, np.nan)
    d["n_obs_sofar"] = np.arange(len(d))  # approximation: true value may be higher if
                                           # this well has history before the supplied window

    target_row = d.iloc[-1].copy()

    imputed = []
    for c in FEATURES:
        if c.startswith("zone_") or c == "has_own_log":
            continue
        if c not in target_row or pd.isna(target_row[c]):
            imputed.append(c)

    # zone / petrophysical features -- only meaningfully populated for a
    # handful of Kokori/Oroni wells in training; default to trained medians
    # otherwise (this reflects the validated finding that these features
    # do not materially change accuracy).
    target_row["zone_GR"] = req.zone_gr if req.zone_gr is not None else CFG["zone_defaults"]["zone_GR"]
    target_row["zone_DEN"] = req.zone_den if req.zone_den is not None else CFG["zone_defaults"]["zone_DEN"]
    target_row["zone_POR"] = req.zone_por if req.zone_por is not None else CFG["zone_defaults"]["zone_POR"]
    target_row["zone_RES_DEP"] = req.zone_res_dep if req.zone_res_dep is not None else CFG["zone_defaults"]["zone_RES_DEP"]
    target_row["has_own_log"] = 1 if req.has_petrophysical_log else 0

    feature_dict = {}
    for c in FEATURES:
        val = target_row.get(c, np.nan)
        feature_dict[c] = val
    return feature_dict, gap_warning, imputed


# ---------------------------------------------------------------- endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        feature_dict, gap_warning, imputed = build_feature_row(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    row = pd.DataFrame([feature_dict])[FEATURES]
    # median-impute anything still missing (mirrors training-time fallback)
    row = row.fillna(row.median(numeric_only=True)).fillna(0)

    X_point = point_scaler.transform(row)
    X_p90 = p90_scaler.transform(row)

    expected = float(np.expm1(point_model.predict(X_point)[0]))
    ceiling = float(np.expm1(p90_model.predict(X_p90)[0]))
    expected = max(expected, 0.0)
    ceiling = max(ceiling, expected)  # ceiling should never read below the point estimate

    if ceiling >= 20:
        risk = "Critical"
    elif ceiling >= 8:
        risk = "Elevated"
    else:
        risk = "Normal"

    return PredictResponse(
        well_id=req.well_id,
        expected_pptb=round(expected, 2),
        ceiling_p90_pptb=round(ceiling, 2),
        risk_level=risk,
        months_used=len(req.history),
        imputed_features=imputed,
        gap_warning=gap_warning,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(FEATURES)}


# ---------------------------------------------------------------- portfolio snapshot
WELLS_SNAPSHOT = json.load(open(MODELS_DIR / "wells_snapshot.json"))


@app.get("/wells")
def list_wells():
    """
    Portfolio snapshot: expected/ceiling/risk for every well in the training
    dataset, computed from each well's most recently RECORDED month (not a
    live forecast). This reflects historical data only (through 2011) --
    swap this for a live feed once one exists.
    """
    return {
        "wells": WELLS_SNAPSHOT,
        "as_of": "most recently recorded month per well (historical dataset, not live)",
        "count": len(WELLS_SNAPSHOT),
    }


# ---------------------------------------------------------------- feature weighting
FEATURE_WEIGHTS = json.load(open(MODELS_DIR / "feature_weights.json"))


@app.get("/feature-weights")
def feature_weights():
    """
    Global feature weighting: which operational inputs show the strongest
    relationship to sanding, as a percentage. Sand-history features
    (lags/rolling stats of the well's own past readings) are excluded by
    design -- they dominate the model and would otherwise crowd out every
    other signal. 'Year' is also excluded; see `note` for why.

    This is a GLOBAL ranking computed once across the full training set
    (permutation importance), not a per-prediction explanation -- it does
    not change based on which well or month is queried.
    """
    return FEATURE_WEIGHTS
