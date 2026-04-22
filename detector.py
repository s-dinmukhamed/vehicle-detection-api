from ultralytics import YOLO
import cv2
import base64
import subprocess
import os

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat"}

COLORS = [
    (255, 99, 99),   (99, 255, 99),   (99, 99, 255),
    (255, 199, 99),  (99, 255, 255),  (255, 99, 255),
    (180, 255, 99),  (99, 180, 255),  (255, 99, 180),
    (200, 200, 99),
]

def get_color(track_id: int):
    return COLORS[track_id % len(COLORS)]

def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def detect(image_path: str):
    results = model(image_path)
    detections = []
    img = cv2.imread(image_path)

    for idx, r in enumerate(results):
        for i, box in enumerate(r.boxes):
            cls = model.names[int(box.cls)]
            if cls not in VEHICLE_CLASSES:
                continue
            conf = round(float(box.conf), 2)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            track_id = i + 1
            color = get_color(track_id)

            detections.append({
                "id": track_id,
                "class": cls,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "color": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            })

            draw_box(img, x1, y1, x2, y2, f"#{track_id} {cls} {conf}", color)

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
    id_colors = {}

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

                color = get_color(tid)
                id_colors[tid] = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                conf = round(float(box.conf), 2)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                draw_box(frame, x1, y1, x2, y2, f"#{tid} {cls} {conf}", color)

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

    objects = [
        {"id": tid, "class": cls, "color": id_colors.get(tid, "#ffffff")}
        for tid, cls in class_counts.items()
    ]

    return {
        "unique_vehicles": len(unique_ids),
        "by_class": stats,
        "objects": objects
    }, converted_path