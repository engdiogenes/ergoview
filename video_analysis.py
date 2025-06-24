import cv2
import mediapipe as mp

def analyze_video(video_path):
    """
    Processa um vídeo para detectar poses humanas usando MediaPipe.
    Retorna uma lista de dicionários com coordenadas das articulações por frame.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converte a imagem para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        joints = {}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                joints[f'joint_{idx}'] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }

        frame_data.append(joints)

    cap.release()
    return frame_data
