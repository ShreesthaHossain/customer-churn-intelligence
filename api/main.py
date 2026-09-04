"""FastAPI service for churn prediction."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_model_config
from src.inference import build_churn_prediction_response
from src.model import ChurnModelBundle, load_churn_pipeline
from src.policy import NO_INTERNET_SERVICE, NO_PHONE_SERVICE, normalize_service_fields

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
PhoneService = Literal["Yes", "No"]
MultipleLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["No", "DSL", "Fiber optic"]
InternetAddon = Literal["Yes", "No", "No internet service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaperlessBilling = Literal["Yes", "No"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """Raw customer feature payload for churn scoring."""

    model_config = ConfigDict(extra="forbid")

    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: PhoneService
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Contract
    PaperlessBilling: PaperlessBilling
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[float, int, str]
    customerID: str | None = None

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total_charges(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("TotalCharges is required.")
        if isinstance(value, str) and value.strip() == "":
            return ""
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError("TotalCharges must be zero or greater.")
            return float(value)
        if isinstance(value, str):
            try:
                parsed = float(value.strip())
            except ValueError as exc:
                raise ValueError("TotalCharges must be numeric or blank for tenure=0.") from exc
            if parsed < 0:
                raise ValueError("TotalCharges must be zero or greater.")
            return parsed
        raise ValueError("Invalid TotalCharges value.")

    @model_validator(mode="after")
    def validate_service_combinations(self) -> CustomerFeatures:
        if self.tenure != 0 and self.TotalCharges == "":
            raise ValueError("TotalCharges can be blank only when tenure is 0.")

        if self.InternetService == "No":
            for field_name in (
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ):
                if getattr(self, field_name) != NO_INTERNET_SERVICE:
                    raise ValueError(
                        f"{field_name} must be '{NO_INTERNET_SERVICE}' when InternetService is No."
                    )

        if self.PhoneService == "No" and self.MultipleLines != NO_PHONE_SERVICE:
            raise ValueError(
                f"MultipleLines must be '{NO_PHONE_SERVICE}' when PhoneService is No."
            )
        return self


class PredictChurnResponse(BaseModel):
    churn_probability: float
    threshold: float
    prediction: Literal["Churn", "No Churn"]
    risk_level: Literal["Low", "Elevated", "High"]
    recommended_action: str
    model_version: str | None = None
    customerID: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = load_churn_pipeline()
    app.state.model_config = load_model_config()
    yield


app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Score telecom customers for churn risk using the saved calibrated pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline: ChurnModelBundle | None = getattr(request.app.state, "pipeline", None)
    config: dict[str, Any] | None = getattr(request.app.state, "model_config", None)
    model_version = None
    if config:
        from src.policy import model_version_from_config

        model_version = model_version_from_config(config)
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        model_version=model_version,
    )


@app.post("/predict_churn", response_model=PredictChurnResponse)
def predict_churn(customer: CustomerFeatures, request: Request) -> PredictChurnResponse:
    pipeline: ChurnModelBundle = request.app.state.pipeline
    config: dict[str, Any] = request.app.state.model_config

    record = customer.model_dump()
    record, _ = normalize_service_fields(record)

    try:
        result = build_churn_prediction_response(pipeline, record, model_config=config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictChurnResponse(**result)
