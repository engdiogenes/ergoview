import math

def calculate_angle(a, b, c):
    ba = [a[i] - b[i] for i in range(2)]
    bc = [c[i] - b[i] for i in range(2)]

    dot_product = sum(ba[i] * bc[i] for i in range(2))
    magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(2)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(2)))

    if magnitude_ba * magnitude_bc == 0:
        return 0

    angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle_rad)

def generate_diagnosis(pose_data):
    diagnostics = []

    for frame_idx, joints in enumerate(pose_data):
        try:
            # Índices do modelo YOLOv8 Pose:
            # 5: shoulder_r, 7: elbow_r, 9: wrist_r
            # 11: hip_r, 13: knee_r, 15: ankle_r

            shoulder = [joints['joint_5']['x'], joints['joint_5']['y']]
            elbow = [joints['joint_7']['x'], joints['joint_7']['y']]
            wrist = [joints['joint_9']['x'], joints['joint_9']['y']]

            hip = [joints['joint_11']['x'], joints['joint_11']['y']]
            knee = [joints['joint_13']['x'], joints['joint_13']['y']]
            ankle = [joints['joint_15']['x'], joints['joint_15']['y']]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)

            if elbow_angle < 45 or elbow_angle > 160:
                diagnostics.append(f"Frame {frame_idx}: Ângulo do cotovelo fora da faixa ergonômica ({elbow_angle:.1f}°).")
            if knee_angle < 60 or knee_angle > 160:
                diagnostics.append(f"Frame {frame_idx}: Ângulo do joelho fora da faixa ergonômica ({knee_angle:.1f}°).")

        except KeyError:
            continue

    if not diagnostics:
        diagnostics.append("Postura dentro dos limites ergonômicos em todos os quadros analisados.")

    return diagnostics
