import uvicorn

def start_server():
    uvicorn.run("back.api.model_controller:_app", app_dir="src/", host="0.0.0.0", port=8000, log_level="info", reload=True)
