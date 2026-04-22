from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
import shutil
from detector import detect as run_detect, detect_video
from database import SessionLocal, Detection, init_db

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def startup():
    init_db()

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, img_base64 = run_detect(file_path)

    db = SessionLocal()
    record = Detection(
        filename=file.filename,
        type="image",
        total=len(detections),
        detections=detections
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {
        "id": record.id,
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

    db = SessionLocal()
    record = Detection(
        filename=file.filename,
        type="video",
        total=stats["unique_vehicles"],
        detections=stats
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {
        "id": record.id,
        "filename": file.filename,
        "output_video": f"/video/{video_filename}",
        "stats": stats
    }

@app.get("/history")
async def get_history():
    db = SessionLocal()
    records = db.query(Detection).order_by(Detection.created_at.desc()).limit(50).all()
    db.close()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "type": r.type,
            "total": r.total,
            "created_at": r.created_at
        }
        for r in records
    ]

@app.get("/history/{detection_id}")
async def get_detection(detection_id: int):
    db = SessionLocal()
    record = db.query(Detection).filter(Detection.id == detection_id).first()
    db.close()
    if not record:
        return {"error": "Not found"}
    return record

@app.get("/video/{filename}")
async def get_video(filename: str):
    path = f"outputs/{filename}"
    return FileResponse(path, media_type="video/mp4")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()