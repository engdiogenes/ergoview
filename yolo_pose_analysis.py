import os
from ultralytics import YOLO
import cv2
import numpy as np

def run_pose_estimation(video_path, progress_callback=None):
    # Verifica se o modelo existe localmente, senão baixa automaticamente
    model_path = "yolov8n-pose.pt"
    if not os.path.exists(model_path):
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_path)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        for person in results[0].keypoints.xy:
            keypoints = person.cpu().numpy().tolist()
            pose_data.append({"keypoints": keypoints})

        frame_count += 1
        if progress_callback:
            progress_callback(frame_count / total_frames)

    cap.release()
    out.release()

    return pose_data, output_path
