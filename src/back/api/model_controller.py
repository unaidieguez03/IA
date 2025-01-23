import time
import cv2
from pydantic import BaseModel
from fastapi import FastAPI, File, Response, UploadFile, Request
import numpy as np
from back.model.model import ClassificationModel
import base64
import json

class RequestData(BaseModel):
	id: int
	disease: str

_app = FastAPI(root_path="/python")
model = ClassificationModel("notebooks/checkpoint/best_model.pt")
data = RequestData(
    id = 999,
    disease = ""
)

@_app.post("/send")
async def predict(request: Request) -> Response:
    try:

        body_bytes = await request.body()
        body = json.loads(body_bytes)

        image_base64 = body.get("image")
        file_name = body.get("filename")
        patient_id = body.get("patientId")



        data.id = patient_id
        image_bytes = base64.b64decode(image_base64)

        numpy_array = np.frombuffer(image_bytes, dtype=np.uint8)
        raw_img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

        predicted,_ =  model.classify(raw_img)
        data.disease = predicted

        return {
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "detail": type(proba)}
    
@_app.get("/get")
async def simulation() -> Response:
	return {"patientId": data.id, "disease": data.disease}