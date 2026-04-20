from fastapi import FastAPI, UploadFile, File
import os
import shutil
from detector import detect as run_detect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, img_base64 = run_detect(file_path)

    return {
        "filename": file.filename,
        "total": len(detections),
        "detections": detections,
        "image": img_base64,
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()