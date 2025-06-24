import streamlit as st
import os
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis
from angle_graphs import generate_angle_graphs  # nova função para gráficos

st.title("Análise Ergonômica com Vídeo")

st.write("📹 Grave um vídeo com seu celular ou computador e envie abaixo para análise ergonômica.")

video_file = st.file_uploader("Faça upload do vídeo (formato .mp4)", type=["mp4"])

if video_file is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.video("uploaded_video.mp4")

    st.write("🔍 Executando detecção de pose com YOLOv8...")
    pose_data, processed_video_path = run_pose_estimation("uploaded_video.mp4")

    st.write("📊 Gerando gráficos dos ângulos...")
    elbow_graph, knee_graph = generate_angle_graphs(pose_data)

    st.image(elbow_graph, caption="Ângulo do Cotovelo ao Longo do Tempo")
    st.image(knee_graph, caption="Ângulo do Joelho ao Longo do Tempo")

    st.write("🩺 Gerando diagnóstico ergonômico...")
    diagnosis = generate_diagnosis(pose_data)

    st.subheader("Diagnóstico:")
    for item in diagnosis:
        st.write("-", item)

    st.subheader("🎥 Vídeo com Esqueleto Detectado:")
    st.video(processed_video_path)
