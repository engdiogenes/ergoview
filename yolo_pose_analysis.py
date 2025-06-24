import cv2
from ultralytics import YOLO

def run_pose_estimation(video_path):
    """
    Executa a detecção de pose com YOLOv8 em um vídeo.
    Retorna uma lista de dicionários com coordenadas das articulações por frame.
    """
    model = YOLO("yolov8n-pose.pt")  # Certifique-se de que o modelo está disponível
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, verbose=False)

        joints = {}
        if results and results[0].keypoints and results[0].keypoints.xy is not None:
            keypoints = results[0].keypoints
            if len(keypoints.xy) > 0:
                for idx, (x, y) in enumerate(keypoints.xy[0]):
                    joints[f'joint_{idx}'] = {
                        'x': float(x),
                        'y': float(y),
                        'z': 0.0,
                        'visibility': 1.0
                    }

        frame_data.append(joints)

    cap.release()
    return frame_data
