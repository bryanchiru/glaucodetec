import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Vercel usa PostgreSQL (Supabase). Localmente usa SQLite como fallback.
_raw = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./glaucodetec.db")

# Supabase da URLs con prefijo "postgres://" — SQLAlchemy necesita "postgresql+asyncpg://"
if _raw.startswith("postgres://"):
    DATABASE_URL = _raw.replace("postgres://", "postgresql+asyncpg://", 1)
elif _raw.startswith("postgresql://") and "asyncpg" not in _raw:
    DATABASE_URL = _raw.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = _raw

engine       = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
