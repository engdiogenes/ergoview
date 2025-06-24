import streamlit as st
import os
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis

st.title("Análise Ergonômica com Webcam")

# Botão para iniciar a gravação
st.write("Grave um vídeo com sua câmera para análise ergonômica.")
start_recording = st.button("📷 Iniciar Gravação")

# Captura de vídeo pela câmera
video_file = None
if start_recording:
    video_file = st.camera_input("Gravando...")

if video_file is not None:
    # Salva o vídeo capturado
    with open("webcam_video.mp4", "wb") as f:
        f.write(video_file.getvalue())

    st.video("webcam_video.mp4")

    st.write("Executando detecção de pose com YOLOv8...")
    pose_data = run_pose_estimation("webcam_video.mp4")

    st.write("Gerando diagnóstico ergonômico...")
    diagnosis = generate_diagnosis(pose_data)

    st.subheader("Diagnóstico:")
    for item in diagnosis:
        st.write("-", item)
