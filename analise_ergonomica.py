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

    tronco_inclinado = False
    braco_elevado = False
    joelho_flexionado = False

    tronco_inclinado_eventos = 0
    braco_elevado_eventos = 0
    joelho_flexionado_eventos = 0

    tronco_inclinado_duracao = 0
    joelho_flexionado_duracao = 0
    inatividade_frames = 0

    for i, frame in enumerate(pose_data):
        keypoints = frame.get("keypoints", [])
        if len(keypoints) < 16:
            inatividade_frames += 1
            continue

        try:
            neck = np.mean([keypoints[5], keypoints[6]], axis=0)
            hip = np.mean([keypoints[11], keypoints[12]], axis=0)
            knee = keypoints[13]
            tronco_angle = calcular_angulo(neck, hip, knee)
            tronco_desvio = tronco_angle < 135

            shoulder = keypoints[5]
            elbow = keypoints[7]
            wrist = keypoints[9]
            braco_angle = calcular_angulo(shoulder, elbow, wrist)
            braco_desvio = braco_angle > 90

            ankle = keypoints[15]
            joelho_angle = calcular_angulo(keypoints[11], keypoints[13], ankle)
            joelho_desvio = joelho_angle < 90

            if tronco_desvio and not tronco_inclinado:
                tronco_inclinado_eventos += 1
            tronco_inclinado = tronco_desvio
            if tronco_desvio:
                tronco_inclinado_duracao += 1

            if braco_desvio and not braco_elevado:
                braco_elevado_eventos += 1
            braco_elevado = braco_desvio

            if joelho_desvio and not joelho_flexionado:
                joelho_flexionado_eventos += 1
            joelho_flexionado = joelho_desvio
            if joelho_desvio:
                joelho_flexionado_duracao += 1

            if tronco_angle < 90 or braco_angle > 150 or joelho_angle < 60:
                angulos_articulares_extremos += 1

        except Exception:
            continue

    posturas_inadequadas = tronco_inclinado_eventos + braco_elevado_eventos + joelho_flexionado_eventos

    if joelho_flexionado_duracao / fps > 90:
        posturas_forcadas = 1

    if tronco_inclinado_duracao / fps > 4:
        posturas_estaticas = tronco_inclinado_eventos

    total_desvios = posturas_inadequadas + angulos_articulares_extremos
    if total_desvios > 100:
        risco_postural = "Alto"
    elif total_desvios > 30:
        risco_postural = "Moderado"

    if inatividade_frames / fps > 10:
        pausas_ritmo = 1

    if braco_elevado_eventos > 10:
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
