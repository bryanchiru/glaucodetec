from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class User(Base):
    __tablename__ = "users"

    id:               Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    username:         Mapped[str]      = mapped_column(String(50),  unique=True, index=True)
    email:            Mapped[str]      = mapped_column(String(120), unique=True, index=True)
    hashed_password:  Mapped[str]      = mapped_column(String(255))
    is_active:        Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:       Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reset_token:      Mapped[str|None] = mapped_column(String(255), nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    user_id:    Mapped[int]      = mapped_column(Integer, index=True)
    filename:   Mapped[str]      = mapped_column(String(255))
    result:     Mapped[str]      = mapped_column(String(10))
    confidence: Mapped[float]    = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
