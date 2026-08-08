from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mtpl.config import get_config, read_levels


class PricingRequest(BaseModel):
    policy_id: int
    exposure: float = Field(gt=0, le=1.5)
    area: str
    veh_power: int = Field(ge=1)
    veh_age: int = Field(ge=0)
    driv_age: int = Field(ge=18, le=100)
    bonus_malus: int = Field(ge=50, le=350)
    veh_brand: str
    veh_gas: str
    density: float = Field(gt=0)
    region: str

    @field_validator("area")
    @classmethod
    def _check_area(cls, v: str) -> str:
        allowed = get_config().features.area_levels
        if v not in allowed:
            raise ValueError(f"unseen area: {v!r} not in {allowed}")
        return v

    @field_validator("veh_gas")
    @classmethod
    def _check_veh_gas(cls, v: str) -> str:
        allowed = get_config().features.veh_gas_levels
        if v not in allowed:
            raise ValueError(f"unseen veh_gas: {v!r} not in {allowed}")
        return v

    @field_validator("region")
    @classmethod
    def _check_region(cls, v: str) -> str:
        allowed = read_levels(get_config().paths.levels_path)["region"]
        if v not in allowed:
            raise ValueError(f"unseen region: {v!r} not in {allowed}")
        return v

    @field_validator("veh_brand")
    @classmethod
    def _check_veh_brand(cls, v: str) -> str:
        allowed = read_levels(get_config().paths.levels_path)["veh_brand"]
        if v not in allowed:
            raise ValueError(f"unseen veh_brand: {v!r} not in {allowed}")
        return v


class ReasonCode(BaseModel):
    feature: str
    value: Any
    factor: float
    direction: Literal["increases", "decreases"]


class PricingResponse(BaseModel):
    premium_pure: float
    premium_technical: float
    frequency: float
    severity: float
    currency: Literal["EUR"] = "EUR"
    model_version: str
    reason_codes: list[ReasonCode]
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class DriftRequest(BaseModel):
    records: list[PricingRequest]
