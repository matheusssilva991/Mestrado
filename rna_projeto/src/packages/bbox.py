import numpy as np
from PIL import Image

def bboxes_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    # Verifica se NÃO há sobreposição e inverte
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


def extract_and_save_crop(image_path, bbox, output_path):
    """Extrai o crop da imagem usando bbox e salva"""
    x, y, w, h = bbox
    img = Image.open(image_path)
    crop = img.crop((x, y, x + w, y + h))
    crop.save(output_path)
    return output_path
