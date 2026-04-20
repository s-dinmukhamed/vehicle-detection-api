# vehicle-detection-api

REST API for detecting vehicles in images using YOLOv8.

<img width="1264" height="862" alt="Снимок экрана 2026-04-20 в 09 24 48" src="https://github.com/user-attachments/assets/3966466f-b3b8-4685-bded-43035c243544" />

## Features 
1. Detects cars, trucks, buses and motorcycles
2. Returns bounding boxes and confidence rate
3. Responses with annotated image
4. Simple web interface

## Stack 
1. Python
2. YOLOv8
3. FastAPI
4. OpenCV

## Instalation 
''' bash \
git clone https://github.com/s-dinmukhamed/vehicle-detection-api.git \
cd vehicle-detection-api \
python -m venv venv \
Mac: source venv/bin/activate Windows: venv\Scripts\activate \ 
pip install -r requirements.txt \
'''

## Run 
'''bash \
uvicorn main:app --reload \

Open http://localhost:8000
```

### POST /detect

Upload an image and get vehicle detections.

Request:  multipart/form-data with `file` field

Response: 
json
{
  "filename": "test.jpg",
  "total": 3,
  "detections": [
    {
      "class": "car",
      "confidence": 0.95,
      "bbox": [191, 857, 5477, 2862]
    }
  ],
  "image": ""
}
```

## Project Structure

```
vehicle-detection-api/
├── main.py          # FastAPI app
├── detector.py      # YOLOv8 inference
├── static/
│   └── index.html   # Web interface
├── uploads/         # Temporary uploaded files
└── requirements.txt
```
