import streamlit as st
import os
import numpy as np
import time
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis
from angle_graphs import generate_angle_graphs

# Configuração da página
st.set_page_config(page_title="Análise Ergonômica com YOLOv11", layout="centered")

st.title("📊 Análise Ergonômica com Vídeo")
st.write("Grave um vídeo com seu celular ou computador e envie abaixo para análise ergonômica.")

# Função para calcular ângulo entre três pontos
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Upload do vídeo
video_file = st.file_uploader("📁 Envie um vídeo no formato .mp4", type=["mp4"])

if video_file is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.subheader("🎬 Vídeo Original")
    st.video("uploaded_video.mp4")

    progress_bar = st.progress(0.0)
    st.info("🔍 Processando vídeo... Isso pode levar alguns segundos.")
    start_time = time.time()

    try:
        pose_data, processed_video_path = run_pose_estimation(
            "uploaded_video.mp4",
            progress_callback=lambda p: progress_bar.progress(min(p, 1.0))
        )
        elapsed_time = time.time() - start_time
        st.success(f"✅ Detecção de pose concluída em {elapsed_time:.2f} segundos.")
    except Exception as e:
        st.error(f"Erro ao processar o vídeo: {e}")
        st.stop()

    st.subheader("📈 Gráficos dos Ângulos")
    elbow_graph, knee_graph = generate_angle_graphs(pose_data)

    if elbow_graph:
        st.image(elbow_graph, caption="Ângulo do Cotovelo ao Longo do Tempo")
    else:
        st.warning("⚠️ Nenhum dado válido para o cotovelo.")

    if knee_graph:
        st.image(knee_graph, caption="Ângulo do Joelho ao Longo do Tempo")
    else:
        st.warning("⚠️ Nenhum dado válido para o joelho.")

    st.subheader("🩺 Diagnóstico Ergonômico")
    diagnosis = generate_diagnosis(pose_data)
    if diagnosis:
        for item in diagnosis:
            st.write("•", item)
    else:
        st.info("Nenhum alerta ergonômico detectado.")

    # Cálculo do ângulo do tronco
    tronco_angles = []
    for frame in pose_data:
        keypoints = frame["keypoints"]
        try:
            neck = np.mean([keypoints[5], keypoints[6]], axis=0)
            hip = np.mean([keypoints[11], keypoints[12]], axis=0)
            knee = keypoints[13]
            angle = calculate_angle(neck, hip, knee)
            tronco_angles.append(angle)
        except Exception:
            tronco_angles.append(np.nan)

    posturas_inclinadas = sum(1 for ang in tronco_angles if not np.isnan(ang) and ang < 135)

    st.metric(
        label="Posturas Inclinadas (Tronco < 135°)",
        value=f"{posturas_inclinadas} vezes"
    )

    st.subheader("📥 Baixar Vídeo com Esqueleto Detectado")
    if os.path.exists(processed_video_path):
        with open(processed_video_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo com esqueleto",
                data=f,
                file_name="video_esqueleto.mp4",
                mime="video/mp4"
            )
    else:
        st.error("❌ O vídeo com esqueleto não foi gerado.")
