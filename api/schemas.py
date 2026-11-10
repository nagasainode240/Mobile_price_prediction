from typing import Literal, Optional
from pydantic import BaseModel, Field


class MobileInput(BaseModel):
    mobile_id: Optional[str] = Field(default=None, description="Optional custom identifier for the phone")
    
    battery_power: int = Field(
        ..., ge=500, le=3000,
        description="Total energy a battery can store in one time measured in mAh (e.g. 500 - 2500)",
        examples=[1400]
    )
    blue: Literal[0, 1] = Field(
        ...,
        description="Has Bluetooth or not: 1 = Yes, 0 = No",
        examples=[1]
    )
    clock_speed: float = Field(
        ..., ge=0.4, le=3.5,
        description="Speed at which microprocessor executes instructions in GHz (e.g. 0.5 - 3.0)",
        examples=[1.8]
    )
    dual_sim: Literal[0, 1] = Field(
        ...,
        description="Has dual SIM support: 1 = Yes, 0 = No",
        examples=[1]
    )
    fc: int = Field(
        ..., ge=0, le=30,
        description="Front Camera mega pixels (e.g. 0 - 20)",
        examples=[5]
    )
    four_g: Literal[0, 1] = Field(
        ...,
        description="Has 4G network support: 1 = Yes, 0 = No",
        examples=[1]
    )
    int_memory: int = Field(
        ..., ge=2, le=512,
        description="Internal Memory in Gigabytes (e.g. 2 - 64)",
        examples=[32]
    )
    m_dep: float = Field(
        ..., ge=0.05, le=1.5,
        description="Mobile Depth in cm (e.g. 0.1 - 1.0)",
        examples=[0.5]
    )
    mobile_wt: int = Field(
        ..., ge=50, le=300,
        description="Weight of mobile phone in grams (e.g. 80 - 200)",
        examples=[140]
    )
    n_cores: int = Field(
        ..., ge=1, le=16,
        description="Number of cores of processor (1 to 8)",
        examples=[4]
    )
    pc: int = Field(
        ..., ge=0, le=60,
        description="Primary Camera mega pixels (e.g. 0 - 20)",
        examples=[12]
    )
    px_height: int = Field(
        ..., ge=0, le=3000,
        description="Pixel Resolution Height (e.g. 0 - 1960)",
        examples=[720]
    )
    px_width: int = Field(
        ..., ge=400, le=3000,
        description="Pixel Resolution Width (e.g. 500 - 1998)",
        examples=[1280]
    )
    ram: int = Field(
        ..., ge=256, le=16384,
        description="Random Access Memory in Megabytes (e.g. 256 - 3998)",
        examples=[2048]
    )
    sc_h: int = Field(
        ..., ge=4, le=30,
        description="Screen Height of mobile in cm (e.g. 5 - 19)",
        examples=[12]
    )
    sc_w: int = Field(
        ..., ge=0, le=25,
        description="Screen Width of mobile in cm (e.g. 0 - 18)",
        examples=[6]
    )
    talk_time: int = Field(
        ..., ge=1, le=35,
        description="Longest time a battery charge lasts for phone calls in hours (e.g. 2 - 20)",
        examples=[10]
    )
    three_g: Literal[0, 1] = Field(
        ...,
        description="Has 3G network support: 1 = Yes, 0 = No",
        examples=[1]
    )
    touch_screen: Literal[0, 1] = Field(
        ...,
        description="Has touch screen or not: 1 = Yes, 0 = No",
        examples=[1]
    )
    wifi: Literal[0, 1] = Field(
        ...,
        description="Has Wi-Fi connectivity: 1 = Yes, 0 = No",
        examples=[1]
    )


class PredictionResponse(BaseModel):
    mobile_id: Optional[str] = None
    prediction: int = Field(..., description="Target class index: 0, 1, 2, or 3")
    price_category: str = Field(..., description="Price category name")
    price_range_usd: str = Field(..., description="Estimated USD price range")
    price_range_inr: str = Field(..., description="Estimated INR price range")
    confidence: float = Field(..., description="Probability of predicted class")
    probabilities: dict[str, float] = Field(..., description="Probability distribution across all 4 classes")
    latency_ms: Optional[float] = Field(default=None, description="Inference latency in milliseconds")


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    mobiles: list[MobileInput] = Field(..., min_length=1, description="List of mobile specifications")


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
