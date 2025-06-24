import streamlit as st
from video_analysis import analyze_video
from ergonomics import generate_diagnosis

st.title("Análise Ergonômica de Vídeo")

uploaded_file = st.file_uploader("Faça upload de um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")

    st.write("Processando vídeo...")
    pose_data = analyze_video("temp_video.mp4")

    st.write("Gerando diagnóstico ergonômico...")
    diagnosis = generate_diagnosis(pose_data)

    st.subheader("Diagnóstico:")
    for item in diagnosis:
        st.write("-", item)
