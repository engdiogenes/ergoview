import streamlit as st
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis

st.title("Análise Ergonômica com Webcam")

# Captura de vídeo pela câmera
video_file = st.camera_input("Grave um vídeo com sua câmera")

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
