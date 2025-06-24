import cv2
import numpy as np
from ultralytics import YOLO

def run_pose_estimation(video_path):
    model = YOLO("yolov8n-pose.pt")  # modelo leve de pose
    cap = cv2.VideoCapture(video_path)
    pose_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        keypoints = results[0].keypoints

        if keypoints is not None and keypoints.xy is not None:
            joints = {}
            for idx, (x, y) in enumerate(keypoints.xy[0]):
                joints[f'joint_{idx}'] = {
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(keypoints.conf[0][idx])
                }
            pose_data.append(joints)

    cap.release()
    return pose_data
