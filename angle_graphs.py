import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (em radianos convertido para graus)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_angle_graphs(pose_data):
    elbow_angles = []
    knee_angles = []

    for frame in pose_data:
        keypoints = frame["keypoints"]

        # Índices dos keypoints no formato COCO:
        # 5: ombro esquerdo, 7: cotovelo esquerdo, 9: punho esquerdo
        # 11: quadril esquerdo, 13: joelho esquerdo, 15: tornozelo esquerdo
        try:
            elbow_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            knee_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        except Exception:
            elbow_angle = np.nan
            knee_angle = np.nan

        elbow_angles.append(elbow_angle)
        knee_angles.append(knee_angle)

    # Gráfico do cotovelo
    plt.figure(figsize=(10, 4))
    plt.plot(elbow_angles, label="Ângulo do Cotovelo (esquerdo)", color='blue')
    plt.xlabel("Frame")
    plt.ylabel("Ângulo (graus)")
    plt.title("Variação do Ângulo do Cotovelo ao Longo do Tempo")
    plt.legend()
    plt.tight_layout()
    elbow_path = "cotovelo.png"
    plt.savefig(elbow_path)
    plt.close()

    # Gráfico do joelho
    plt.figure(figsize=(10, 4))
    plt.plot(knee_angles, label="Ângulo do Joelho (esquerdo)", color='green')
    plt.xlabel("Frame")
    plt.ylabel("Ângulo (graus)")
    plt.title("Variação do Ângulo do Joelho ao Longo do Tempo")
    plt.legend()
    plt.tight_layout()
    knee_path = "joelho.png"
    plt.savefig(knee_path)
    plt.close()

    return elbow_path, knee_path
