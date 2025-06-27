import streamlit as st
import os
import numpy as np
import time
import pandas as pd
import threading
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis
from angle_graphs import generate_angle_graphs
import cv2
from analise_ergonomica import analisar_metricas_ergonomicas


st.set_page_config(page_title="Análise Ergonômica com YOLOv11", layout="centered")

if "pose_data" not in st.session_state:
    st.session_state.pose_data = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

st.title("📊 Análise Ergonômica com Vídeo")
st.write("Grave um vídeo com seu celular ou computador e envie abaixo para análise ergonômica.")


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=30):
    desvios_por_tipo = {
        "Inclinação excessiva do tronco": [],
        "Braço elevado acima do ombro": [],
        "Flexão profunda do joelho": []
    }

    for i, frame in enumerate(pose_data):
        keypoints = frame["keypoints"]
        try:
            neck = np.mean([keypoints[5], keypoints[6]], axis=0)
            hip = np.mean([keypoints[11], keypoints[12]], axis=0)
            knee = keypoints[13]
            tronco_angle = calculate_angle(neck, hip, knee)
            if tronco_angle < 135:
                desvios_por_tipo["Inclinação excessiva do tronco"].append((i, tronco_angle))

            shoulder_left = keypoints[5]
            elbow_left = keypoints[7]
            wrist_left = keypoints[9]
            arm_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
            if arm_angle > 90:
                desvios_por_tipo["Braço elevado acima do ombro"].append((i, arm_angle))

            hip_left = keypoints[11]
            knee_left = keypoints[13]
            ankle_left = keypoints[15]
            knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
            if knee_angle < 90:
                desvios_por_tipo["Flexão profunda do joelho"].append((i, knee_angle))
        except Exception:
            continue

    desvios_filtrados = []
    for tipo, ocorrencias in desvios_por_tipo.items():
        if not ocorrencias:
            continue
        sequencia = []
        for idx in range(len(ocorrencias)):
            if not sequencia:
                sequencia.append(ocorrencias[idx])
            elif ocorrencias[idx][0] == sequencia[-1][0] + 1:
                sequencia.append(ocorrencias[idx])
            else:
                if len(sequencia) >= persistencia_minima:
                    for frame_idx, angulo in sequencia:
                        tempo_segundos = frame_idx / fps
                        desvios_filtrados.append({
                            "Frame": frame_idx,
                            "Tempo (s)": round(tempo_segundos, 2),
                            "Desvio": tipo,
                            "Ângulo": round(angulo, 2)
                        })
                sequencia = [ocorrencias[idx]]
        if len(sequencia) >= persistencia_minima:
            for frame_idx, angulo in sequencia:
                tempo_segundos = frame_idx / fps
                desvios_filtrados.append({
                    "Frame": frame_idx,
                    "Tempo (s)": round(tempo_segundos, 2),
                    "Desvio": tipo,
                    "Ângulo": round(angulo, 2)
                })

    return pd.DataFrame(desvios_filtrados)


video_file = st.file_uploader("📁 Envie um vídeo no formato .mp4", type=["mp4"])

if video_file is not None and st.session_state.pose_data is None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.subheader("🎬 Vídeo Original")
    st.video("uploaded_video.mp4")

    timer_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    st.info("🔍 Processando vídeo... Isso pode levar alguns segundos.")
    start_time = time.time()
    running = True

    def update_timer():
        while running:
            elapsed = time.time() - start_time
            timer_placeholder.markdown(f"⏱️ Tempo decorrido: **{elapsed:.1f} segundos**")
            time.sleep(0.1)

    # ✅ Função de progresso definida corretamente
    def atualizar_progresso(p):
        progress_bar.progress(min(p, 1.0))

    timer_thread = threading.Thread(target=update_timer)
    timer_thread.start()

    try:
        # ✅ Chamada otimizada com parâmetros adicionais
        pose_data, processed_video_path = run_pose_estimation(
            "uploaded_video.mp4",
            progress_callback=atualizar_progresso,
            frame_skip=2,
            save_annotated_video=True
        )
        running = False
        timer_thread.join()
        elapsed_time = time.time() - start_time
        st.success(f"✅ Detecção de pose concluída em {elapsed_time:.2f} segundos.")
        st.session_state.pose_data = pose_data
        st.session_state.processed_video_path = processed_video_path
    except Exception as e:
        running = False
        timer_thread.join()
        st.error(f"Erro ao processar o vídeo: {e}")
        st.stop()



if st.session_state.pose_data:
    pose_data = st.session_state.pose_data
    metricas = analisar_metricas_ergonomicas(pose_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Posturas Inadequadas", metricas["Posturas Inadequadas"], "NR-17")
    col2.metric("Movimentos Repetitivos", metricas["Movimentos Repetitivos"], "NR-17")
    col3.metric("Posturas Forçadas", metricas["Posturas Forçadas (>90s)"], "NR-17")

    col4, col5, col6 = st.columns(3)
    col4.metric("Pausas/Ritmo de Trabalho", metricas["Pausas/Ritmo de Trabalho"], "NR-17")
    col5.metric("Mobiliário/Layout", metricas["Mobiliário/Layout"], "NR-17")
    col6.metric("Ângulos Articulares Extremos", metricas["Ângulos Articulares Extremos"], "ISO 11226")

    col7, col8, col9 = st.columns(3)
    col7.metric("Posturas Estáticas", metricas["Posturas Estáticas (>4s)"], "ISO 11226")
    col8.metric("Risco Postural", metricas["Risco Postural"], "ISO 11226")
    col9.metric("Postura Sentada", metricas["Postura Sentada"], "ISO 9241")

    st.markdown("### 📌 Descrições detalhadas das violações")

    if metricas["Posturas Inadequadas"] > 0:
        st.warning(
            f"🔸 Foram detectadas {metricas['Posturas Inadequadas']} ocorrências de posturas inadequadas, como inclinação excessiva do tronco, elevação dos braços acima do ombro ou flexão profunda dos joelhos. Essas posturas devem ser evitadas conforme a NR-17.")

    if metricas["Movimentos Repetitivos"] > 0:
        st.warning(
            "🔸 Foram identificados movimentos repetitivos com os membros superiores, o que pode causar fadiga muscular e lesões por esforço repetitivo (LER/DORT), conforme a NR-17.")

    if metricas["Posturas Forçadas (>90s)"] > 0:
        st.warning(
            "🔸 Foram detectadas posturas forçadas mantidas por mais de 90 segundos, como flexão profunda dos joelhos. Isso representa risco ergonômico elevado segundo a NR-17.")

    if metricas["Pausas/Ritmo de Trabalho"] == 0:
        st.warning(
            "🔸 Não foram detectadas pausas significativas durante a atividade. A NR-17 recomenda pausas para recuperação física e mental.")

    if metricas["Ângulos Articulares Extremos"] > 0:
        st.warning(
            f"🔸 Foram identificadas {metricas['Ângulos Articulares Extremos']} ocorrências de ângulos articulares extremos (ex: tronco < 90°, braço > 150°, joelho < 60°), o que representa risco postural segundo a ISO 11226.")

    if metricas["Posturas Estáticas (>4s)"] > 0:
        st.warning(
            f"🔸 Foram detectadas {metricas['Posturas Estáticas (>4s)']} posturas estáticas mantidas por mais de 4 segundos, o que pode causar fadiga muscular e deve ser evitado conforme a ISO 11226.")

    if metricas["Risco Postural"] in ["Moderado", "Alto"]:
        st.warning(
            f"🔸 A classificação geral de risco postural foi **{metricas['Risco Postural']}**, indicando necessidade de intervenção ergonômica segundo a ISO 11226.")

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

    st.subheader("📊 Desvios Posturais Detectados (NR-17)")
    df_desvios = detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=90)

    def gerar_video_com_alertas(df_desvios2, input_video_path="uploaded_video.mp4",
                                output_video_path="video_com_alertas.mp4"):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("❌ Erro ao abrir o vídeo original para gerar alertas.")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        alert_frames = {}
        alert_duration = 2 * fps  # 2 segundos

        for _, row in df_desvios2.iterrows():
            start = int(row["Frame"])
            for f2 in range(start, start + alert_duration):
                if f2 not in alert_frames:
                    alert_frames[f2] = []
                alert_frames[f2].append(row["Desvio"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in alert_frames:
                for i, desvio in enumerate(alert_frames[frame_idx]):
                    text = f"⚠️ {desvio}"
                    y = 50 + i * 40
                    cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        return output_video_path


    if not df_desvios.empty:
        contagem = df_desvios["Desvio"].value_counts().to_dict()

        col1, col2, col3 = st.columns(3)
        col1.metric("Inclinação do Tronco", contagem.get("Inclinação excessiva do tronco", 0))
        col2.metric("Braço Elevado", contagem.get("Braço elevado acima do ombro", 0))
        col3.metric("Flexão do Joelho", contagem.get("Flexão profunda do joelho", 0))

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
        st.error("❌ Dados de pose não carregados. Verifique se o vídeo foi processado corretamente.")
        st.stop()
