import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

from api.predict import predict

app = FastAPI(title="Pet classifier: cat vs dog")

@app.post("/predict")
async def predict_label(file: UploadFile = File(...)) -> dict:
    _, label, confidence = predict(await file.read())
    return {"label": label, "confidence": round(confidence, 3)}

@app.post("/predict/image")
async def predict_annotated(file: UploadFile = File(...)) -> Response:
    image, label, confidence = predict(await file.read())

    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    scale = canvas.shape[1] / 500
    cv2.putText(canvas, f"{label} {confidence:.0%}", (10, int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), max(1, int(2 * scale)))

    _, png = cv2.imencode(".png", canvas)
    return Response(content=png.tobytes(), media_type="image/png")