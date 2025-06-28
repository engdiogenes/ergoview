import numpy as np


def calcular_angulo(a, b, c):
    """Calcula o ângulo entre três pontos (em graus)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)



def analisar_metricas_ergonomicas(df_desvios):
    metricas = {
        "Posturas Inadequadas": 0,
        "Movimentos Repetitivos": 0,
        "Posturas Forçadas (>90s)": 0,
        "Pausas/Ritmo de Trabalho": 0,
        "Mobiliário/Layout": 1,
        "Ângulos Articulares Extremos": 0,
        "Posturas Estáticas (>4s)": 0,
        "Risco Postural": "Baixo",
        "Postura Sentada": 0
    }

    if df_desvios.empty:
        return metricas

    # Contagem total de desvios persistentes
    # Contar eventos únicos por tipo de desvio
    df_desvios = df_desvios.sort_values(by=["Desvio", "Frame"])
    df_desvios["Frame_diff"] = df_desvios["Frame"].diff().fillna(1)
    df_desvios["Desvio_diff"] = df_desvios["Desvio"] != df_desvios["Desvio"].shift()
    df_desvios["Novo_evento"] = (df_desvios["Frame_diff"] != 1) | (df_desvios["Desvio_diff"])
    df_desvios["Evento"] = df_desvios["Novo_evento"].cumsum()
    metricas["Posturas Inadequadas"] = df_desvios.groupby(["Desvio", "Evento"]).ngroups

    # Filtrar apenas os desvios com ângulos extremos
    extremos = df_desvios[
        ((df_desvios["Desvio"] == "Braço elevado acima do ombro") & (df_desvios["Ângulo"] > 150)) |
        ((df_desvios["Desvio"] == "Flexão profunda do joelho") & (df_desvios["Ângulo"] < 60)) |
        ((df_desvios["Desvio"] == "Inclinação excessiva do tronco") & (df_desvios["Ângulo"] < 90))
        ].copy()
    # Contar eventos únicos de flexão profunda do joelho
    joelho = df_desvios[df_desvios["Desvio"] == "Flexão profunda do joelho"].copy()
    joelho = joelho.sort_values(by="Frame")
    joelho["Frame_diff"] = joelho["Frame"].diff().fillna(1)
    joelho["Novo_evento"] = joelho["Frame_diff"] != 1
    joelho["Evento"] = joelho["Novo_evento"].cumsum()
    num_eventos_joelho = joelho["Evento"].nunique()

    # Se quiser adicionar ao dicionário de métricas:
    metricas["Flexão profunda do joelho"] = num_eventos_joelho

    # Identificar eventos únicos
    extremos = extremos.sort_values(by=["Desvio", "Frame"])
    extremos["Frame_diff"] = extremos["Frame"].diff().fillna(1)
    extremos["Desvio_diff"] = extremos["Desvio"] != extremos["Desvio"].shift()
    extremos["Novo_evento"] = (extremos["Frame_diff"] != 1) | (extremos["Desvio_diff"])
    extremos["Evento"] = extremos["Novo_evento"].cumsum()

    # Contar eventos únicos
    metricas["Ângulos Articulares Extremos"] = extremos.groupby(["Desvio", "Evento"]).ngroups

    # Posturas forçadas: número de tipos de desvio com persistência mínima já garantida
    metricas["Posturas Forçadas (>90s)"] = len(df_desvios["Desvio"].unique())

    # Posturas estáticas: desvios com pouca variação de ângulo
    desvios_estaticos = df_desvios.groupby("Desvio")["Ângulo"].std()
    metricas["Posturas Estáticas (>4s)"] = (desvios_estaticos < 5).sum()

    # Risco postural baseado na quantidade de desvios
    total = metricas["Posturas Inadequadas"]
    if total >= 10:
        metricas["Risco Postural"] = "Alto"
    elif total >= 5:
        metricas["Risco Postural"] = "Moderado"

    return metricas
