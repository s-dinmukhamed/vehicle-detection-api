from ultralytics import YOLO
import cv2
import base64
import subprocess

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat"}

def detect(image_path: str):
    results = model(image_path)
    detections = []
    img = cv2.imread(image_path)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            if cls not in VEHICLE_CLASSES:
                continue
            conf = round(float(box.conf), 2)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls} {conf}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return detections, img_base64


def detect_video(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    unique_ids = set()
    class_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, verbose=False, persist=True)

        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id in zip(r.boxes, r.boxes.id):
                cls = model.names[int(box.cls)]
                if cls not in VEHICLE_CLASSES:
                    continue

                tid = int(track_id)
                unique_ids.add(tid)

                if tid not in class_counts:
                    class_counts[tid] = cls

                conf = round(float(box.conf), 2)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls} {conf} id:{tid}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    converted_path = output_path.replace(".mp4", "_web.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", output_path,
        "-vcodec", "libx264", "-acodec", "aac",
        converted_path
    ], capture_output=True)

    stats = {}
    for tid, cls in class_counts.items():
        stats[cls] = stats.get(cls, 0) + 1

    return {"unique_vehicles": len(unique_ids), "by_class": stats}, converted_path