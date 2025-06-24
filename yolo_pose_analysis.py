from ultralytics import YOLO
import cv2
import numpy as np

def run_pose_estimation(video_path):
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_data = []

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

    cap.release()
    out.release()

    return pose_data, output_path
