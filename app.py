import streamlit as st
import os
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis
from angle_graphs import generate_angle_graphs

st.set_page_config(page_title="Análise Ergonômica com YOLOv8", layout="centered")
st.title("📊 Análise Ergonômica com Vídeo")

st.write("Grave um vídeo com seu celular ou computador e envie abaixo para análise ergonômica.")

# Upload do vídeo
video_file = st.file_uploader("📁 Envie um vídeo no formato .mp4", type=["mp4"])

if video_file is not None:
    # Salva o vídeo enviado
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.subheader("🎬 Vídeo Original")
    st.video("uploaded_video.mp4")

    # Barra de progresso
    progress_bar = st.progress(0.0)
    st.info("🔍 Processando vídeo... Isso pode levar alguns segundos.")

    # Processamento com detecção de pose
    pose_data, processed_video_path = run_pose_estimation(
        "uploaded_video.mp4",
        progress_callback=lambda p: progress_bar.progress(min(p, 1.0))
    )

    st.success("✅ Detecção de pose concluída!")

    # Geração de gráficos
    st.subheader("📈 Gráficos dos Ângulos")
    elbow_graph, knee_graph = generate_angle_graphs(pose_data)
    st.image(elbow_graph, caption="Ângulo do Cotovelo ao Longo do Tempo")
    st.image(knee_graph, caption="Ângulo do Joelho ao Longo do Tempo")

    # Diagnóstico
    st.subheader("🩺 Diagnóstico Ergonômico")
    diagnosis = generate_diagnosis(pose_data)
    for item in diagnosis:
        st.write("•", item)

    # Vídeo com esqueleto
    st.subheader("🎥 Vídeo com Esqueleto Detectado")
    st.video(processed_video_path)
