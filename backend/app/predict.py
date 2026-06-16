"""
Motor de inferencia con ONNX Runtime (sin PyTorch) para deploy en Vercel.
"""
from pathlib import Path
from functools import lru_cache
import io

import numpy as np
import onnxruntime as ort
from PIL import Image

ONNX_PATH = Path(__file__).parent.parent.parent / "ml" / "models" / "glaucodetec.onnx"
CLASSES   = ["NRG", "RG"]
IMG_SIZE  = 224
MEAN      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD       = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def load_session() -> ort.InferenceSession:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Modelo ONNX no encontrado en {ONNX_PATH}. "
            "Ejecuta 'python ml/export_onnx.py' primero."
        )
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(ONNX_PATH), providers=providers)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img   = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr   = np.array(img, dtype=np.float32) / 255.0
    arr   = (arr - MEAN) / STD
    arr   = arr.transpose(2, 0, 1)       # HWC -> CHW
    return arr[np.newaxis, ...]           # batch dim


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_image(image_bytes: bytes) -> dict:
    tensor  = preprocess(image_bytes)
    session = load_session()
    logits  = session.run(None, {"input": tensor})[0][0]
    probs   = softmax(logits)

    idx        = int(probs.argmax())
    class_name = CLASSES[idx]

    return {
        "class":      class_name,
        "label":      "Sin Glaucoma" if class_name == "NRG" else "Referir a Especialista",
        "confidence": round(float(probs[idx]) * 100, 2),
        "prob_NRG":   round(float(probs[0]) * 100, 2),
        "prob_RG":    round(float(probs[1]) * 100, 2),
    }
