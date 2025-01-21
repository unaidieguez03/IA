from fastapi import FastAPI, Body, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import pydicom
from io import BytesIO
import base64
import os
from models import Data

app = FastAPI(root_path="/python")

@app.post("/send")
async def handle_diagnosis(request: Request):
    try:
        # Parse JSON body
        body = await request.json()

        # Extract fields from JSON
        image_base64 = body.get("image")
        file_name = body.get("filename")
        patient_id = body.get("patientId")
        Data.id = body.get("patientId")

        if not image_base64 or not file_name or not patient_id:
            return {"status": "error", "detail": "Missing required fields"}

        # Decode the Base64 image
        decoded_image = base64.b64decode(image_base64)

        # Save the file (optional)
        output_path = os.path.join("uploads", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        with open(output_path, "wb") as f:
            f.write(decoded_image)

        # Return success response
        return {
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/get")
async def get_disease():
	return {"patientId": Data.id, "disease": "Fastidiado"}
