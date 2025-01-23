import time
import cv2
from pydantic import BaseModel
from fastapi import FastAPI, File, Response, UploadFile
import numpy as np
from back.model.model import ClassificationModel
import base64

_app = FastAPI(root_path="/python")
model = ClassificationModel("notebooks/checkpoint/best_model.pt")

@_app.post("/send")
async def predict(uploaded_img) -> Response:
    try:
# Parse JSON body
        body = await request.json()

        # Extract fields from JSON
        image_base64 = body.get("image")
        file_name = body.get("filename")
        patient_id = body.get("patientId")
        # RequestData.id = patient_id
        # RequestData.disease = "Osteoporosis o como se diga"


        # RequestData.disease = "lorem" #get_prediction(output_tensor)[0]
        _app.state.diagnosis_data.id = patient_id
        raw_img = cv2.imdecode(image_base64, cv2.IMREAD_COLOR)
        predicted,_ =  model.classify(raw_img)
        _app.state.diagnosis_data.disease = predicted
        _app.state.image = image_base64

        _app.state.diagnosis_ready = True

        # Save the file (optional)
        # output_path = os.path.join("uploads", file_name)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        # with open(output_path, "wb") as f:
        #     f.write(decoded_image)

        # Return success response
        return {
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
@_app.get("/get")
async def simulation() -> Response:
	return {"patientId": _app.state.diagnosis_data.id, "disease": _app.state.diagnosis_data.disease}

class RequestData(BaseModel):
	id: int
	disease: str
