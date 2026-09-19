"""Shared FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.policy import NO_INTERNET_SERVICE, NO_PHONE_SERVICE

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
    environment: str
    auth_enabled: bool
    predictions_total: int = 0
    batch_requests_total: int = 0
    last_prediction_at: str | None = None


class BatchPredictRequest(BaseModel):
    primary_key_column: str = "customerID"
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class BatchPredictionItem(BaseModel):
    primary_key: str
    churn_probability: float
    prediction: str
    risk_level: str
    retention_recommended: int
    recommended_action: str
    decision_threshold: float


class BatchPredictResponse(BaseModel):
    model_version: str | None = None
    customers_scored: int
    retention_outreach_flagged: int
    decision_threshold: float
    results: list[BatchPredictionItem]


class MonitoringSummaryResponse(BaseModel):
    service_metrics: dict[str, Any]
    drift_status: Literal["ok", "alert", "no_data"]
    drift_alerts: list[str]
    batch_max_rows: int
