from ultralytics import YOLO
import cv2
import numpy as np
import os

def run_pose_estimation(video_path, progress_callback=None):
    model_path = "yolo11n-pose.pt"
    
    # Baixa o modelo se não estiver presente
    if not os.path.exists(model_path):
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_path)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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

        # Verifica se há keypoints detectados
        if results[0].keypoints is not None and results[0].keypoints.xy is not None:
            for person in results[0].keypoints.xy:
                keypoints = person.cpu().numpy().tolist()
                pose_data.append({"keypoints": keypoints})
        else:
            pose_data.append({"keypoints": []})  # Frame sem detecção

        frame_count += 1

        # Atualiza a barra de progresso, se fornecida
        if progress_callback:
            try:
                progress_callback(frame_count / total_frames)
            except Exception as e:
                print(f"Erro ao atualizar progresso: {e}")

    cap.release()
    out.release()

    return pose_data, output_path
