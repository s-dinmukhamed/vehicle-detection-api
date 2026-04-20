from ultralytics import YOLO
import cv2
import base64

model = YOLO("yolov8n.pt")

def detect(image_path: str):
    results = model(image_path)
    detections = []

    img = cv2.imread(image_path)

    for r in results:
        for box in r.boxes:

            cls = model.names[int(box.cls)]
            conf = round(float(box.conf), 2)
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class":cls,
                "confidence":conf,
                "bbox": [x1,y1,x2,y2]
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls} {conf}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", img)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

    return detections, img_base64