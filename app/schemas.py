"""Pydantic models for every request and response: the API contract.

The store repo is built against these exact field names and shapes (PROJECT_SPEC.md §5).
Do not rename or reshape anything here without updating the store repo too.

Multipart endpoints (/analyze, /try-on) take form fields, which FastAPI declares as
function parameters in the routers; their limits live here as constants so they sit
next to the rest of the contract.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.errors import ErrorCode

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

Category = Literal["upper_body", "lower_body", "dresses"]
FitPreference = Literal["slim", "regular", "loose"]

HEIGHT_MIN_CM = 120
HEIGHT_MAX_CM = 230
WEIGHT_MIN_KG = 20
WEIGHT_MAX_KG = 400

HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$", examples=["#C68E6A"])]


class ErrorResponse(BaseModel):
    error_code: ErrorCode
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error_code": "NO_PERSON_DETECTED",
                    "message": "No person found in the photo. Use a clear full-body photo.",
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


class Measurements(BaseModel):
    """Body measurements in cm. Chest and waist are circumferences, the rest are lengths/widths."""

    shoulder_cm: float = Field(gt=0, examples=[44.1])
    chest_cm: float = Field(gt=0, examples=[96.5])
    waist_cm: float = Field(gt=0, examples=[84.0])
    torso_cm: float = Field(gt=0, examples=[62.3])
    arm_cm: float = Field(gt=0, examples=[60.2])
    leg_cm: float = Field(gt=0, examples=[81.7])


class SkinTone(BaseModel):
    # Tone labels are finalised in Phase 3 (e.g. "fair" | "light" | "medium" | "tan" | "deep"),
    # so this is a plain string for now.
    tone: str = Field(examples=["medium"])
    undertone: Literal["warm", "cool", "neutral"]
    hex: HexColor


class ColorSuggestion(BaseModel):
    name: str = Field(examples=["Olive"])
    hex: HexColor


class AnalyzeResponse(BaseModel):
    measurements: Measurements
    # Nullable on purpose: if the pose is fine but no face is found, we may return
    # measurements with a warning instead of failing with FACE_NOT_FOUND.
    # Final behaviour is decided in R&D / Phase 3. Until then it is always set.
    skin_tone: SkinTone | None
    color_suggestions: list[ColorSuggestion]
    confidence: float = Field(ge=0, le=1, examples=[0.82])
    warnings: list[str]
    debug_image_base64: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "measurements": {
                        "shoulder_cm": 44.1,
                        "chest_cm": 96.5,
                        "waist_cm": 84.0,
                        "torso_cm": 62.3,
                        "arm_cm": 60.2,
                        "leg_cm": 81.7,
                    },
                    "skin_tone": {"tone": "medium", "undertone": "warm", "hex": "#C68E6A"},
                    "color_suggestions": [
                        {"name": "Olive", "hex": "#708238"},
                        {"name": "Mustard", "hex": "#D4A017"},
                    ],
                    "confidence": 0.82,
                    "warnings": ["Arms close to body — shoulder width may be less accurate"],
                    "debug_image_base64": None,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# /recommend-size
# ---------------------------------------------------------------------------


def _check_range(value: list[float]) -> list[float]:
    low, high = value
    if low < 0 or high < 0:
        raise ValueError("range values must be >= 0")
    if low > high:
        raise ValueError("range must be [min, max] with min <= max")
    return value


# One measurement range in cm: [min, max]
CmRange = Annotated[
    list[float],
    Field(min_length=2, max_length=2, examples=[[92, 98]]),
    AfterValidator(_check_range),
]

# The ranges for one size, e.g. {"chest": [92, 98], "waist": [80, 86]}.
# Any field may be missing. Upper-body charts usually use chest / waist / length / shoulder;
# other names are allowed so lower-body charts (e.g. hip, inseam) don't need a contract change.
SizeRanges = dict[str, CmRange]


class RecommendSizeRequest(BaseModel):
    measurements: Measurements
    # Keys are size labels ("S", "M", "L", ...). Order is kept as sent.
    size_chart: dict[str, SizeRanges] = Field(min_length=1)
    category: Category
    fit_preference: FitPreference = "regular"

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "measurements": {
                        "shoulder_cm": 44.1,
                        "chest_cm": 96.5,
                        "waist_cm": 84.0,
                        "torso_cm": 62.3,
                        "arm_cm": 60.2,
                        "leg_cm": 81.7,
                    },
                    "size_chart": {
                        "S": {"chest": [86, 92], "waist": [74, 80], "length": [68, 70], "shoulder": [41, 43]},
                        "M": {"chest": [92, 98], "waist": [80, 86], "length": [70, 72], "shoulder": [43, 45]},
                        "L": {"chest": [98, 104], "waist": [86, 92], "length": [72, 74], "shoulder": [45, 47]},
                    },
                    "category": "upper_body",
                    "fit_preference": "regular",
                }
            ]
        }
    )


class SizeFit(BaseModel):
    score: float = Field(ge=0, le=1, examples=[0.93])
    note: str = Field(examples=["Good fit"])


class RecommendSizeResponse(BaseModel):
    recommended_size: str = Field(examples=["M"])
    per_size: dict[str, SizeFit]

    @model_validator(mode="after")
    def _recommended_size_is_listed(self) -> "RecommendSizeResponse":
        if self.recommended_size not in self.per_size:
            raise ValueError("recommended_size must be one of the keys in per_size")
        return self


# ---------------------------------------------------------------------------
# /try-on
# ---------------------------------------------------------------------------


class TryOnResponse(BaseModel):
    """Exactly one of result_image_url / result_image_base64 is non-null."""

    result_image_url: str | None = Field(examples=["https://replicate.delivery/.../output.jpg"])
    result_image_base64: str | None = None
    latency_ms: int = Field(ge=0, examples=[23400])
    provider: str = Field(examples=["replicate_idm"])

    @model_validator(mode="after")
    def _exactly_one_result(self) -> "TryOnResponse":
        if (self.result_image_url is None) == (self.result_image_base64 is None):
            raise ValueError("exactly one of result_image_url / result_image_base64 must be set")
        return self
