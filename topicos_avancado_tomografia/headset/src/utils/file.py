import numpy as np
import SimpleITK as sitk
import os


def read_raw_img_from_detector(file_name, dim_x, dim_y, data_type, offset=0):
    """
    Read a single raw image from detector.

    Parameters
    ----------
    file_name : str
        Name of the file.
    dim_x : int
        Image dimension in pixels (X axis).
    dim_y : int
        Image dimension in pixels (Y axis).
    data_type : dtype
        Data type in the file, in XRMC float64.
    offset : int, optional
        Header block of the image file in bytes, in XRMC (60 bytes).

    Returns
    -------
    numpy.ndarray
        Image matrix with shape (dim_x, dim_y).
    """
    # Lê arquivo binário raw do detector
    img = np.fromfile(file_name, dtype=data_type, count=dim_x * dim_y, offset=offset)

    # Redimensiona array 1D para matriz 2D
    img = img.reshape((dim_x, dim_y))
    return img


def read_raw_img_folder(
    folder_name,
    file_name_prefix,
    init_img,
    final_img,
    step,
    dim_x,
    dim_y,
    data_type,
    offset=0,
):
    """
    Read a directory (or subset) of images from a directory containing simulation images.

    Parameters
    ----------
    folder_name : str
        Name (path) of the directory.
    file_name_prefix : str
        Prefix of the image file name.
    init_img : int
        Index (degree) of the initial image, usually 0.
    final_img : int
        Index (degree) of the final image, usually 359.
    step : int
        Step in degrees between images, usually 1.
    dim_x : int
        Image dimension in pixels (X axis).
    dim_y : int
        Image dimension in pixels (Y axis).
    data_type : dtype
        Data type in the file, in XRMC float64.
    offset : int, optional
        Header block of the image file in bytes, in XRMC (60 bytes).

    Returns
    -------
    numpy.ndarray
        Stack of images with shape (num_images, dim_x, dim_y).
    """
    # Lê primeira imagem da sequência
    num_images = len(range(init_img, final_img, step))
    imgs = np.empty((num_images, dim_x, dim_y), dtype=data_type)

    # Lê todas as imagens e preenche diretamente no array pré-alocado
    for idx, i in enumerate(range(init_img, final_img, step)):
        file_name = f"{folder_name}/{file_name_prefix}{i:d}.dat"
        imgs[idx] = read_raw_img_from_detector(
            file_name, dim_x, dim_y, data_type, offset
        )

    return imgs


def save_as_mhd_raw(np_array, base_name="image", voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Save a numpy array as MHD/RAW format.

    Parameters
    ----------
    np_array : numpy.ndarray
        Numpy array to be saved.
    base_name : str, optional
        Base name for the output file.
    voxel_spacing : tuple of float, optional
        Voxel spacing in order (Z, Y, X).
    """
    # Converte para imagem SimpleITK em formato uint16
    sitk_image = sitk.GetImageFromArray(np_array.astype(np.uint16))

    # Define espaçamento entre voxels (ordem Z, Y, X)
    sitk_image.SetSpacing(voxel_spacing)

    # Salva em formato .mhd/.raw
    sitk.WriteImage(sitk_image, f"{base_name}.mhd")


def safe_load_npz(path: str):
    """Carrega um volume de um arquivo .npz com verificações.
    - Verifica existência do arquivo.
    - Tenta usar a chave 'volume' se existir; caso contrário, pega o primeiro array.
    - Imprime shape, dtype, min/max para confirmação.
    Retorna: np.ndarray ou None se falhar.
    """
    if not os.path.exists(path):
        print(f"[Aviso] Arquivo não encontrado: {path}")
        return None
    try:
        data = np.load(path)
    except Exception as e:
        print(f"[Erro] Falha ao carregar {path}: {e}")
        return None
    vol = None
    if 'volume' in data.files:
        vol = data['volume']
    else:
        # Seleciona o primeiro array encontrado
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                vol = arr
                break
    if vol is None:
        print(f"[Aviso] Nenhum array encontrado em {path}. Chaves: {data.files}")
        return None
    try:
        vmin = float(vol.min())
        vmax = float(vol.max())
    except Exception:
        vmin, vmax = (None, None)
    print(
        f"[OK] {os.path.basename(path)} carregado: "
        f"shape={vol.shape}, dtype={vol.dtype}, min={vmin}, max={vmax}"
    )
    return vol