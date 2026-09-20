"""Request and response shapes for authentication.

`password_hash` appears in none of them. The response models are the boundary
that makes that structural rather than a habit: even if a hash reached this
layer, no field exists to carry it out.
"""

import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.config import get_settings

settings = get_settings()


class SignupRequest(BaseModel):
    # `min_length=1` after Pydantic strips nothing for us — the endpoint
    # trims, then re-checks, because "   " passes a naive length test.
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=settings.password_min_length, max_length=128)


class SigninRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
