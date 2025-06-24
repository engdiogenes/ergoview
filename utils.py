def normalize_coordinates(x, y, width, height):
    """
    Normaliza as coordenadas x e y com base na largura e altura do frame.
    """
    return x / width, y / height

def format_diagnosis(diagnostics):
    """
    Formata a lista de diagnósticos em uma string legível.
    """
    return "\n".join(diagnostics)
