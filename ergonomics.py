import math

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos (a, b, c) usando o produto escalar.
    """
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    dot_product = sum(ba[i] * bc[i] for i in range(3))
    magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(3)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(3)))

    if magnitude_ba * magnitude_bc == 0:
        return 0

    angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def generate_diagnosis(pose_data):
    """
    Gera um diagnóstico ergonômico simples com base nos ângulos do cotovelo e joelho.
    """
    diagnostics = []

    for frame_idx, joints in enumerate(pose_data):
        try:
            # Coordenadas dos ombros, cotovelos e punhos
            shoulder_r = [joints['joint_12']['x'], joints['joint_12']['y'], joints['joint_12']['z']]
            elbow_r = [joints['joint_14']['x'], joints['joint_14']['y'], joints['joint_14']['z']]
            wrist_r = [joints['joint_16']['x'], joints['joint_16']['y'], joints['joint_16']['z']]

            # Coordenadas dos quadris, joelhos e tornozelos
            hip_r = [joints['joint_24']['x'], joints['joint_24']['y'], joints['joint_24']['z']]
            knee_r = [joints['joint_26']['x'], joints['joint_26']['y'], joints['joint_26']['z']]
            ankle_r = [joints['joint_28']['x'], joints['joint_28']['y'], joints['joint_28']['z']]

            # Cálculo dos ângulos
            elbow_angle = calculate_angle(shoulder_r, elbow_r, wrist