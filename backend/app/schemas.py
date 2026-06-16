from pydantic import BaseModel, EmailStr
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    is_active:  bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type:   str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordReset(BaseModel):
    token:        str
    new_password: str


class PredictionOut(BaseModel):
    id:         int
    filename:   str
    result:     str
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}
