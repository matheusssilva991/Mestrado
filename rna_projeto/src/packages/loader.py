import os # noqa
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms # noqa


class CropsDataset(Dataset):
    def __init__(self, df, transform=None, glcm_cfg=None, return_mode='image'):
        """
        transform: pipeline de pré-processamento (PIL -> Tensor), ex.: Resize/Normalize
        glcm_cfg: dict com configs para GLCM. Ex.:
            {
                'enabled': True,
                'distances': [1, 2, 3],
                'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
                'levels': 256,           # quantização (0..levels-1)
                'features': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            }
        return_mode: 'image' | 'features' | 'both'
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.glcm_cfg = glcm_cfg or {'enabled': False}
        self.return_mode = return_mode

        # Verificações básicas
        assert 'crop_path' in self.df.columns, "Falta coluna crop_path"
        assert 'label' in self.df.columns, "Falta coluna label"

        # Configurações padrão GLCM
        if self.glcm_cfg.get('enabled', False):
            # defaults
            self.glcm_cfg.setdefault('distances', [1])
            self.glcm_cfg.setdefault('angles', [0])
            self.glcm_cfg.setdefault('levels', 256)
            self.glcm_cfg.setdefault('features', ['contrast', 'homogeneity'])

    def __len__(self):
        return len(self.df)

    # Calcular features GLCM
    def _compute_glcm_features(self, img_pil):
        # Converter para escala de cinza e quantizar
        img_gray = img_pil.convert('L')
        arr = np.array(img_gray)
        """ print("Tipo:", arr.dtype)
        print("Valor mínimo:", arr.min())
        print("Valor máximo:", arr.max()) """

        # Opcional: normalizar/quantizar para níveis desejados
        levels = self.glcm_cfg['levels']
        if arr.max() > (levels - 1):
            # Requantização simples para levels
            arr = (arr.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)

        # Matriz GLCM
        glcm = graycomatrix(
            arr,
            distances=self.glcm_cfg['distances'],
            angles=self.glcm_cfg['angles'],
            levels=levels,
            symmetric=True,
            normed=True
        )

        # Extrair propriedades e agregar por média sobre distâncias/ângulos
        feats = []
        for f in self.glcm_cfg['features']:
            vals = graycoprops(glcm, f)  # shape: (len(dist), len(angles))
            feats.append(vals.mean())
        feats = np.array(feats, dtype=np.float32)
        return feats

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['crop_path']
        label = int(row['label'])

        img = Image.open(path).convert('RGB')

        # Pré-processamentos (ex.: Resize, Augment, Normalize)
        img_out = self.transform(img) if self.transform else img

        # Features GLCM (se habilitado)
        if self.glcm_cfg.get('enabled', False):
            glcm_feats = self._compute_glcm_features(img)
        else:
            glcm_feats = None

        if self.return_mode == 'image':
            return img_out, label
        elif self.return_mode == 'features':
            # Retorna vetor de features como tensor
            import torch
            return torch.from_numpy(glcm_feats), label
        elif self.return_mode == 'both':
            import torch
            return (img_out, torch.from_numpy(glcm_feats)), label
        else:
            raise ValueError("return_mode deve ser 'image', 'features' ou 'both'")
