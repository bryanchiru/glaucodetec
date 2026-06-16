"""
Vercel serverless entry point — importa la app FastAPI.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.main import app  # noqa: F401 — Vercel necesita 'app' en este módulo
