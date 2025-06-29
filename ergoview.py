import streamlit as st
import os
import numpy as np
import time
import pandas as pd
import threading
from ergonomics import generate_diagnosis
from analise_ergonomica import analisar_metricas_ergonomicas
from yolo_pose_analysis import run_pose_estimation
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ErgoView - Análise Ergonômica", layout="wide")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=90):
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

def gerar_diagnostico_avancado(metricas, df_desvios):
    diagnostico = []

    # Diagnóstico baseado nas métricas
    if metricas["Posturas Inadequadas"] > 0:
        diagnostico.append(f"⚠️ Foram detectadas {metricas['Posturas Inadequadas']} posturas inadequadas, indicando risco ergonômico conforme a NR-17.")
    if metricas["Movimentos Repetitivos"] > 0:
        diagnostico.append("⚠️ Movimentos repetitivos foram identificados, o que pode levar a LER/DORT.")
    if metricas["Posturas Forçadas (>90s)"] > 0:
        diagnostico.append("⚠️ Posturas forçadas foram mantidas por mais de 90 segundos, o que representa risco elevado.")
    if metricas["Pausas/Ritmo de Trabalho"] == 0:
        diagnostico.append("⚠️ Ausência de pausas detectada. A NR-17 recomenda pausas regulares para recuperação.")
    if metricas["Ângulos Articulares Extremos"] > 0:
        diagnostico.append(f"⚠️ Foram detectados {metricas['Ângulos Articulares Extremos']} ângulos articulares extremos, o que pode causar sobrecarga muscular.")
    if metricas["Posturas Estáticas (>4s)"] > 0:
        diagnostico.append(f"⚠️ {metricas['Posturas Estáticas (>4s)']} posturas estáticas foram mantidas por mais de 4 segundos.")
    if metricas["Risco Postural"] in ["Moderado", "Alto"]:
        diagnostico.append(f"⚠️ O risco postural geral foi classificado como **{metricas['Risco Postural']}**, indicando necessidade de intervenção.")

    # Diagnóstico baseado nos desvios detectados
    if not df_desvios.empty:
        tipos = df_desvios["Desvio"].value_counts()
        for tipo, qtd in tipos.items():
            diagnostico.append(f"🔍 Foram detectadas {qtd} ocorrências de \"{tipo}\" com persistência mínima de 3 segundos.")

    if not diagnostico:
        diagnostico.append("✅ Nenhum risco ergonômico relevante foi identificado. A postura está dentro dos limites recomendados.")

    return diagnostico


# Interface com abas
st.title("📊 ErgoView - Análise Ergonômica com Visão Computacional")
st.markdown("Bem-vindo ao **ErgoView**, uma ferramenta para auxiliar ergonomistas na análise de operações industriais com base em vídeo. Software desenvolvido por Eng Diógenes Oliveira")

tab1, tab2, tab3, tab4 = st.tabs(["📥 Upload e Processamento", "📊 Métricas e Alertas", "📈 Gráficos e Diagnóstico", "📎 Relatórios e Downloads"])

with tab1:
    st.header("📥 Upload de Vídeo")
    video_file = st.file_uploader("Envie um vídeo no formato .mp4", type=["mp4"])
    if video_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())
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

        def atualizar_progresso(p):
            progress_bar.progress(min(p, 1.0))

        timer_thread = threading.Thread(target=update_timer)
        timer_thread.start()

        try:
            pose_data, processed_video_path = run_pose_estimation(
                "uploaded_video.mp4",
                progress_callback=atualizar_progresso,
                frame_skip=1,
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

with tab2:
    st.header("📊 Métricas Ergonômicas")
    if "pose_data" in st.session_state:
        pose_data = st.session_state.pose_data
        df_desvios = detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=90)
        metricas = analisar_metricas_ergonomicas(df_desvios)

        col1, col2, col3 = st.columns(3)
        col1.metric("Posturas Inadequadas", metricas["Posturas Inadequadas"], "NR-17")
        col2.metric("Movimentos Repetitivos", metricas["Movimentos Repetitivos"], "NR-17")
        col3.metric("Posturas Forçadas", metricas["Posturas Forçadas (>90s)"], "NR-17")

        col4, col5, col6 = st.columns(3)
        col4.metric("Pausas/Ritmo", metricas["Pausas/Ritmo de Trabalho"], "NR-17")
        col5.metric("Mobiliário/Layout", metricas["Mobiliário/Layout"], "NR-17")
        col6.metric("Ângulos Extremos", metricas["Ângulos Articulares Extremos"], "ISO 11226")

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

with tab3:
    st.header("📈 Gráficos e Diagnóstico")
    if "pose_data" in st.session_state:
        pose_data = st.session_state.pose_data
        # Gerar df_angulos a partir de pose_data
        dados = []
        fps = 30
        for i, frame in enumerate(pose_data):
            keypoints = frame["keypoints"]
            try:
                ang_cotovelo = calculate_angle(keypoints[5], keypoints[7], keypoints[9])  # ombro, cotovelo, punho
                ang_joelho = calculate_angle(keypoints[11], keypoints[13], keypoints[15])  # quadril, joelho, tornozelo
                tempo = i / fps
                dados.append({
                    "Tempo (s)": round(tempo, 2),
                    "Ângulo Cotovelo": round(ang_cotovelo, 2),
                    "Ângulo Joelho": round(ang_joelho, 2)
                })
            except Exception:
                continue

        df_angulos = pd.DataFrame(dados)
        df_desvios = detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=90)

        # Gráfico 1: Evolução dos Ângulos ao Longo do Tempo
        fig1 = go.Figure()

        # Traços dos ângulos
        fig1.add_trace(go.Scatter(x=df_angulos["Tempo (s)"], y=df_angulos["Ângulo Cotovelo"],
                                  mode='lines', name='Ângulo Cotovelo'))
        fig1.add_trace(go.Scatter(x=df_angulos["Tempo (s)"], y=df_angulos["Ângulo Joelho"],
                                  mode='lines', name='Ângulo Joelho'))

        # Linhas horizontais de referência para cotovelo
        fig1.add_hline(y=90, line=dict(color="red", dash="dash"),
                       annotation_text="Limite Inferior Cotovelo (90°)", annotation_position="top left")
        fig1.add_hline(y=150, line=dict(color="red", dash="dash"),
                       annotation_text="Limite Superior Cotovelo (150°)", annotation_position="top left")

        # Linhas horizontais de referência para joelho
        fig1.add_hline(y=60, line=dict(color="blue", dash="dot"),
                       annotation_text="Limite Inferior Joelho (60°)", annotation_position="bottom left")
        fig1.add_hline(y=90, line=dict(color="blue", dash="dot"),
                       annotation_text="Limite Superior Joelho (90°)", annotation_position="bottom left")

        # Layout
        fig1.update_layout(title="Evolução dos Ângulos com Limites Ergonômicos",
                           xaxis_title="Tempo (s)", yaxis_title="Ângulo (graus)")

        st.plotly_chart(fig1)

        # Gráfico 2: Histograma dos Ângulos
        fig2 = go.Figure()

        # Histogramas
        fig2.add_trace(go.Histogram(x=df_angulos["Ângulo Cotovelo"], name="Cotovelo", opacity=0.6))
        fig2.add_trace(go.Histogram(x=df_angulos["Ângulo Joelho"], name="Joelho", opacity=0.6))

        # Linhas verticais de referência
        fig2.add_vline(x=90, line=dict(color="red", dash="dash"), annotation_text="Limite Inferior Cotovelo",
                       annotation_position="top right")
        fig2.add_vline(x=150, line=dict(color="red", dash="dash"), annotation_text="Limite Superior Cotovelo",
                       annotation_position="top right")
        fig2.add_vline(x=60, line=dict(color="blue", dash="dot"), annotation_text="Limite Inferior Joelho",
                       annotation_position="top right")
        fig2.add_vline(x=90, line=dict(color="blue", dash="dot"), annotation_text="Limite Superior Joelho",
                       annotation_position="top right")

        # Layout
        fig2.update_layout(
            barmode='overlay',
            title="Distribuição dos Ângulos Articulares com Limites Ergonômicos",
            xaxis_title="Ângulo (graus)",
            yaxis_title="Frequência"
        )

        st.plotly_chart(fig2)

        # Gráfico 3: Contagem de Desvios por Tipo
        st.dataframe(df_desvios.head())
        contagem = df_desvios["Desvio"].value_counts().reset_index()
        contagem.columns = ["Desvio", "Contagem"]
        fig3 = px.bar(contagem, x="Desvio", y="Contagem", title="Contagem de Desvios por Tipo")
        st.plotly_chart(fig3)

        # Gráfico 4: Dispersão de Ângulo vs Tempo
        fig4 = px.scatter(df_desvios, x="Tempo (s)", y="Ângulo", color="Desvio",
                          title="Desvios Detectados: Ângulo vs Tempo")
        st.plotly_chart(fig4)

        st.subheader("🧠 Diagnóstico Ergonômico")
        diagnosis = gerar_diagnostico_avancado(metricas, df_desvios)
        if diagnosis:
            for item in diagnosis:
                st.write("•", item)
        else:
            st.info("Nenhum alerta ergonômico detectado.")


with tab4:
    st.header("📎 Relatórios e Downloads")
    if "pose_data" in st.session_state:
        pose_data = st.session_state.pose_data
        df_desvios = detectar_desvios_com_persistencia(pose_data, fps=30, persistencia_minima=90)

        if not df_desvios.empty:
            st.dataframe(df_desvios)
            csv = df_desvios.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar CSV dos Desvios", data=csv, file_name="desvios_nr17.csv", mime="text/csv")
        else:
            st.info("Nenhum desvio postural detectado conforme NR-17.")

        if "processed_video_path" in st.session_state and os.path.exists(st.session_state.processed_video_path):
            with open(st.session_state.processed_video_path, "rb") as f:
                st.download_button(
                    label="📥 Baixar vídeo com esqueleto",
                    data=f,
                    file_name="video_esqueleto.mp4",
                    mime="video/mp4"
                )
