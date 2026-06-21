"""
GlaucoDetec API — FastAPI
Autenticación JWT + predicción de glaucoma con EfficientNetB0
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import init_db, get_db
from .models import User, Prediction
from .schemas import UserCreate, UserOut, Token, PasswordResetRequest, PasswordReset, PredictionOut
from .auth import hash_password, verify_password, create_access_token, get_current_user, generate_reset_token
from .predict import predict_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
    except Exception as e:
        print(f"[WARNING] DB init failed: {e}")
    yield


app = FastAPI(
    title="GlaucoDetec API",
    description="Detección de glaucoma con IA — EfficientNetB0",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend como archivos estáticos
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

# ─── Autenticación ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserOut, status_code=201,
          summary="Registro de nuevo usuario")
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    exists = await db.execute(
        select(User).where((User.username == data.username) | (User.email == data.email))
    )
    if exists.scalar_one_or_none():
        raise HTTPException(400, "Usuario o email ya existe")
    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token, summary="Login — retorna JWT")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.username == form.username))
    user   = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciales inválidas")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer"}


@app.post("/auth/logout", summary="Logout (invalida sesión cliente)")
async def logout(current_user: User = Depends(get_current_user)):
    return {"message": f"Sesión cerrada para {current_user.username}"}


@app.post("/auth/password-reset-request", summary="Solicitar recuperación de contraseña")
async def password_reset_request(data: PasswordResetRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == data.email))
    user   = result.scalar_one_or_none()
    if not user:
        return {"message": "Si el email existe recibirás un enlace de recuperación"}
    token            = generate_reset_token()
    user.reset_token = token
    await db.commit()
    # En producción: enviar email con el token
    return {"message": "Token generado", "reset_token": token}


@app.post("/auth/password-reset", summary="Restablecer contraseña con token")
async def password_reset(data: PasswordReset, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.reset_token == data.token))
    user   = result.scalar_one_or_none()
    if not user:
        raise HTTPException(400, "Token inválido o expirado")
    user.hashed_password = hash_password(data.new_password)
    user.reset_token     = None
    await db.commit()
    return {"message": "Contraseña actualizada correctamente"}


@app.get("/auth/me", response_model=UserOut, summary="Datos del usuario actual")
async def me(current_user: User = Depends(get_current_user)):
    return current_user

# ─── Predicción ───────────────────────────────────────────────────────────────

@app.post("/predict", summary="Predecir glaucoma en imagen de fondo de ojo")
async def predict(
    file: UploadFile = File(..., description="Imagen JPG/PNG de fondo de ojo"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se aceptan archivos de imagen")

    image_bytes = await file.read()
    result      = predict_image(image_bytes)

    pred = Prediction(
        user_id=current_user.id,
        filename=file.filename,
        result=result["class"],
        confidence=result["confidence"],
    )
    db.add(pred)
    await db.commit()

    return {**result, "filename": file.filename, "user": current_user.username}


@app.get("/predictions", response_model=list[PredictionOut],
         summary="Historial de predicciones del usuario")
async def get_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()


@app.get("/health", summary="Estado del servicio")
async def health():
    return {"status": "ok", "model": "EfficientNetB0", "version": "1.0.0"}
