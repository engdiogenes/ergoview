import streamlit as st
import tempfile
import os
from yolo_pose_analysis import run_pose_estimation
from ergonomics import generate_diagnosis

st.title("Análise Ergonômica com YOLOv8 Pose")

uploaded_file = st.file_uploader("Faça upload de um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    st.write("Executando detecção de pose com YOLOv8...")
    pose_data = run_pose_estimation(video_path)

    st.write("Gerando diagnóstico ergonômico...")
    diagnosis = generate_diagnosis(pose_data)

    st.subheader("Diagnóstico:")
    for item in diagnosis:
        st.write("-", item)

    os.unlink(video_path)
