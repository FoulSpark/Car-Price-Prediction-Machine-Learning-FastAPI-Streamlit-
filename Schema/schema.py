from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum


class FuelType(str, Enum):
    Petrol = "Petrol"
    Diesel = "Diesel"
    CNG = "CNG"


class SellerType(str, Enum):
    Dealer = "Dealer"
    Individual = "Individual"


class TransmissionType(str, Enum):
    Manual = "Manual"
    Automatic = "Automatic"


class CarFeatures(BaseModel):
    Car_Name: str | None = None
    Year: Annotated[int, Field(ge=1000, le=2026)]
    Present_Price: Annotated[float, Field(ge=0)]
    Kms_Driven: Annotated[int, Field(ge=0)]
    Fuel_Type: FuelType
    Seller_Type: SellerType
    Transmission: TransmissionType
    Owner: Annotated[int, Field(ge=0)]


class PredictionPrice(BaseModel):
    prediction_price: float