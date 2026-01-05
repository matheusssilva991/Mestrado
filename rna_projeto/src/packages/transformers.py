import cv2
from PIL import Image
import numpy as np


class CLAHETransform:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) em imagens PIL.
    Pode ser aplicado em imagens grayscale ou RGB.
    Args:
        clip_limit (float): Limite de contraste para CLAHE.
        tile_grid_size (tuple): Tamanho da grade de blocos para CLAHE.

    Returns:
        Imagem PIL com CLAHE aplicado.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        # Converter PIL para array NumPy
        img = np.array(img_pil)

        # Verificar se é grayscale ou RGB
        if img.ndim == 2:  # grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_eq = clahe.apply(img)
        else:
            # converter para LAB e aplicar CLAHE no canal L
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l_eq = clahe.apply(l_channel)
            lab_eq = cv2.merge((l_eq, a_channel, b_channel))
            img_rgb = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            img_eq = img_rgb

        # voltar para PIL
        return Image.fromarray(img_eq)


class ToGrayscalePIL:
    """Converte uma imagem PIL para escala de cinza."""

    def __call__(self, img_pil):
        return img_pil.convert('L')


class QuantizeLevelsTransform:
    """Quantiza uma imagem PIL para um número específico de níveis de cinza."""

    def __init__(self, levels=64):
        self.levels = levels
        self.scale = (levels - 1) / 255.0

    def __call__(self, img_pil):
        arr = np.array(img_pil)
        # entrada esperada uint8 [0..255]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        arr_q = (arr.astype(np.float32) * self.scale).astype(np.uint8)
        return Image.fromarray(arr_q, mode='L')


class WhiteTopHatTransform:
    """Aplica a transformação White Top-Hat em uma imagem PIL em escala de cinza."""

    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size

    def __call__(self, img_pil):
        arr = np.array(img_pil.convert('L'))

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        wth = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, se)
        return Image.fromarray(wth, mode='L')
