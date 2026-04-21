from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
import shutil
from detector import detect as run_detect, detect_video

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
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

@app.post("/detect/video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    input_path  = f"uploads/{file.filename}"
    output_path = f"outputs/annotated_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stats, converted_path = detect_video(input_path, output_path)
    video_filename = os.path.basename(converted_path)

    return {
        "filename": file.filename,
        "output_video": f"/video/{video_filename}",
        "stats": stats
    }

@app.get("/video/{filename}")
async def get_video(filename: str):
    path = f"outputs/{filename}"
    return FileResponse(path, media_type="video/mp4")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()