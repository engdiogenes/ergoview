import streamlit as st
import os
import numpy as np
import time
import pandas as pd
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis
from angle_graphs import generate_angle_graphs

# Configuração da página
st.set_page_config(page_title="Análise Ergonômica com YOLOv11", layout="centered")

# Inicialização do estado
if "pose_data" not in st.session_state:
    st.session_state.pose_data = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

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

if video_file is not None and st.session_state.pose_data is None:
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
           _callback=lambda p: progress_bar.progress(min(p, 1.0))
        )
        elapsed_time = time.time() - start_time
        st.success(f"✅ Detecção de pose concluída em {elapsed_time:.2f} segundos.")
        st.session_state.pose_data = pose_data
        st.session_state.processed_video_path = processed_video_path
    except Exception as e:
        st.error(f"Erro ao processar o vídeo: {e}")
        st.stop()

# Se já houver dados processados
if st.session_state.pose_data:
    pose_data = st.session_state.pose_data

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

    # Análise NR-17 por frame
    desvios = []
    fps = 30  # ajuste conforme o vídeo
    for i, frame in enumerate(pose_data):
        keypoints = frame["keypoints"]
        tempo_segundos = i / fps
        try:
            neck = np.mean([keypoints[5], keypoints[6]], axis=0)
            hip = np.mean([keypoints[11], keypoints[12]], axis=0)
            knee = keypoints[13]
            shoulder = np.mean([keypoints[11], keypoints[12]], axis=0)
            elbow = keypoints[13]

            tronco_angle = calculate_angle(neck, hip, knee)
            if tronco_angle < 135:
                desvios.append({
                    "Frame": i,
                    "Tempo (s)": round(tempo_segundos, 2),
                    "Desvio": "Inclinação excessiva do tronco",
                    "Ângulo": round(tronco_angle, 2)
                })

            # Braço elevado
            shoulder_left = keypoints[5]
            elbow_left = keypoints[7]
            wrist_left = keypoints[9]
            arm_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
            if arm_angle > 90:
                desvios.append({
                    "Frame": i,
                    "Tempo (s)": round(tempo_segundos, 2),
                    "Desvio": "Braço elevado acima do ombro",
                    "Ângulo": round(arm_angle, 2)
                })

            # Flexão profunda do joelho
            hip_left = keypoints[11]
            knee_left = keypoints[13]
            ankle_left = keypoints[15]
            knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
            if knee_angle < 90:
                desvios.append({
                    "Frame": i,
                    "Tempo (s)": round(tempo_segundos, 2),
                    "Desvio": "Flexão profunda do joelho",
                    "Ângulo": round(knee_angle, 2)
                })

        except Exception:
            continue

    # Exibir métricas
    st.subheader("📊 Desvios Posturais Detectados (NR-17)")
    if desvios:
        df_desvios = pd.DataFrame(desvios)
        st.dataframe(df_desvios)
        csv = df_desvios.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar CSV dos Desvios", data=csv, file_name="desvios_nr17.csv", mime="text/csv")
    else:
        st.info("Nenhum desvio postural detectado conforme NR-17.")

    st.subheader("📥 Baixar Vídeo com Esqueleto Detectado")
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        with open(st.session_state.processed_video_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo com esqueleto",
                data=f,
                file_name="video_esqueleto.mp4",
                mime="video/mp4"
            )
    else:
        st.error("❌ O vídeo com esqueleto não foi gerado.")
