from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
import torch
import cv2
import numpy as np
import os

def run_pose_estimation(video_path, progress_callback=None, frame_skip=2, save_annotated_video=False):
    model_path = "yolo11n-pose.pt"

    if not os.path.exists(model_path):
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_path)

    # ✅ Permitir deserialização segura do modelo
    torch.serialization.add_safe_globals([PoseModel])
    model = YOLO(model_path)

    ...
