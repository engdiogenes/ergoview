from ultralytics import YOLO
import cv2
import numpy as np
import os

def run_pose_estimation(video_path, progress_callback=None, frame_skip=2, resize_to=(640, 360), save_annotated_video=False):
    model_path = "yolo11n-pose.pt"

    if not os.path.exists(model_path):
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_path)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output_video.mp4"
    out = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resize_to)

    pose_data = []
    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, resize_to)
        results = model(frame, verbose=False)

        if save_annotated_video:
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        if results[0].keypoints is not None and results[0].keypoints.xy is not None:
            for person in results[0].keypoints.xy:
                keypoints = person.cpu().numpy().tolist()
                pose_data.append({"keypoints": keypoints})
        else:
            pose_data.append({"keypoints": []})

        frame_count += 1
        processed_frames += 1

        if progress_callback:
            try:
                progress_callback(frame_count / total_frames)
            except Exception:
                pass

    cap.release()
    if save_annotated_video and out:
        out.release()

    return pose_data, output_path if save_annotated_video else None
