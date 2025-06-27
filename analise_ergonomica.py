import numpy as np

def calcular_angulo(a, b, c):
    """Calcula o ângulo entre três pontos (em graus)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def analisar_metricas_ergonomicas(pose_data, fps=30):
    """
    Analisa métricas ergonômicas com base em dados de pose extraídos de vídeo.
    Retorna um dicionário com contagens e classificações por norma.
    """
    posturas_inadequadas = 0
    movimentos_repetitivos = 0
    posturas_forcadas = 0
    pausas_ritmo = 0
    angulos_articulares_extremos = 0
    posturas_estaticas = 0
    risco_postural = "Baixo"
    postura_sentada = "Não Avaliado"
    mobiliario_layout = "Não Avaliado"

    tronco_inclinado_frames = []
    braco_elevado_frames = []
    joelho_flexionado_frames = []
    inatividade_frames = 0

    for i, frame in enumerate(pose_data):
        keypoints = frame.get("keypoints", [])
        if len(keypoints) < 16:
            continue

        try:
            # Tronco: pescoço (média ombros) - quadril (média quadris) - joelho esquerdo
            neck = np.mean([keypoints[5], keypoints[6]], axis=0)
            hip = np.mean([keypoints[11], keypoints[12]], axis=0)
            knee = keypoints[13]
            tronco_angle = calcular_angulo(neck, hip, knee)
            if tronco_angle < 135:
                tronco_inclinado_frames.append(i)

            # Braço esquerdo: ombro - cotovelo - punho
            shoulder = keypoints[5]
            elbow = keypoints[7]
            wrist = keypoints[9]
            braco_angle = calcular_angulo(shoulder, elbow, wrist)
            if braco_angle > 90:
                braco_elevado_frames.append(i)

            # Joelho esquerdo: quadril - joelho - tornozelo
            ankle = keypoints[15]
            joelho_angle = calcular_angulo(keypoints[11], keypoints[13], ankle)
            if joelho_angle < 90:
                joelho_flexionado_frames.append(i)

            # Ângulos articulares extremos
            if tronco_angle < 90 or braco_angle > 150 or joelho_angle < 60:
                angulos_articulares_extremos += 1

        except Exception:
            continue

    # Posturas inadequadas = soma de ocorrências
    posturas_inadequadas = len(set(tronco_inclinado_frames + braco_elevado_frames + joelho_flexionado_frames))

    # Posturas forçadas = joelho flexionado por mais de 90s
    if joelho_flexionado_frames:
        duracao = len(joelho_flexionado_frames) / fps
        if duracao > 90:
            posturas_forcadas = 1

    # Posturas estáticas = tronco inclinado por mais de 4s
    if tronco_inclinado_frames:
        duracao = len(tronco_inclinado_frames) / fps
        if duracao > 4:
            posturas_estaticas = len(tronco_inclinado_frames)

    # Risco postural baseado em número de desvios
    total_desvios = posturas_inadequadas + angulos_articulares_extremos
    if total_desvios > 100:
        risco_postural = "Alto"
    elif total_desvios > 30:
        risco_postural = "Moderado"

    # Pausas e ritmo de trabalho: detectar inatividade (sem keypoints) por mais de 10s
    for i, frame in enumerate(pose_data):
        if not frame.get("keypoints"):
            inatividade_frames += 1
    if inatividade_frames / fps > 10:
        pausas_ritmo = 1

    # Movimentos repetitivos: braço elevado em mais de 50% dos frames
    if len(braco_elevado_frames) > 0.5 * len(pose_data):
        movimentos_repetitivos = 1

    return {
        "Posturas Inadequadas": posturas_inadequadas,
        "Movimentos Repetitivos": movimentos_repetitivos,
        "Posturas Forçadas (>90s)": posturas_forcadas,
        "Pausas/Ritmo de Trabalho": pausas_ritmo,
        "Mobiliário/Layout": mobiliario_layout,
        "Ângulos Articulares Extremos": angulos_articulares_extremos,
        "Posturas Estáticas (>4s)": posturas_estaticas,
        "Risco Postural": risco_postural,
        "Postura Sentada": postura_sentada
    }
