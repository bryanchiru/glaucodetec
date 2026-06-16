"""
Convierte el modelo PyTorch a ONNX para deploy en Vercel.
Ejecutar: python ml/export_onnx.py
"""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = Path(__file__).parent / "models" / "glaucodetec_best.pth"
ONNX_PATH  = Path(__file__).parent / "models" / "glaucodetec.onnx"
IMG_SIZE   = 224

def build_model():
    net = models.efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )
    return net

print("Cargando modelo PyTorch...")
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

print("Exportando a ONNX...")
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
print(f"Exportado: {ONNX_PATH}")
print(f"Tamanio: {size_mb:.1f} MB")

# Verificar con onnxruntime
try:
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(ONNX_PATH))
    out  = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"Verificacion OK - output shape: {out.shape}")
except ImportError:
    print("Instala onnxruntime para verificar: pip install onnxruntime")
