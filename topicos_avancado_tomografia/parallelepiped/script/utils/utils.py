from sys import prefix
from tkinter import font
import numpy as np
import os

def read_raw_xrmc_data(file_path: str, shape: tuple, dtype: str= 'float64', offset: int = 0) -> np.ndarray:
    """
    Reads raw XRMc data from a binary file and reshapes it.

    Parameters:
    - file_path: str, path to the binary file.
    - shape: tuple, desired shape of the output array.
    - dtype: data type of the binary data (default is float64).
    - offset: number of bytes to skip at the beginning of the file.

    Returns:
    - numpy array with the specified shape.
    """
    try:
        data = np.fromfile(file_path, dtype=dtype, offset=offset)
        return data.reshape(shape)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise e


def print_image_summary(image: np.ndarray, name: str = "Image") -> None:
    """
    Prints basic statistics of the given image array.

    Parameters:
    - image: numpy array, the image data.
    - name: str, name of the image for identification in the output.
    """
    print(f"{name} Statistics:")
    print(f" - Shape: {image.shape}")
    print(f" - Min: {np.min(image):.3e}")
    print(f" - Max: {np.max(image):.3e}")
    print(f" - Mean: {np.mean(image):.3e}")
    print(f" - Std Dev: {np.std(image):.3e}")
    print(f" - Type: {image.dtype}")


def read_raw_xrmc_folder(folder_path: str, **kwargs) -> np.ndarray:
    """
    Reads raw XRMc data files from a specified folder and stacks them into a 3D numpy array,
    only for the range of indices provided and with the specified step.

    Parameters:
    - folder_path: str, path to the folder containing binary files.
    - kwargs: additional keyword arguments for file reading (e.g., prefix, slice_projection_range, shape, dtype, offset, step).

    Returns:
    - 3D numpy array with stacked slices.
    """
    prefix = kwargs.get('prefix', 'img_')
    shape = kwargs.get('shape', (512, 512))
    dtype = kwargs.get('dtype', 'float64')
    step = kwargs.get('step', 1)
    offset = kwargs.get('offset', 0)
    file_extension = kwargs.get('file_extension', '.dat')
    start_index, end_index = kwargs.get('slice_projection_range', (0, None))

    # Filtra e ordena os arquivos pelo índice numérico
    file_list = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(file_extension)]
    file_list.sort(key=lambda x: int(x[len(prefix):-len(file_extension)]))

    # Aplica o range de índices e o step
    if end_index is not None:
        indices = range(start_index, end_index, step)
        file_list = [f"{prefix}{i:04d}{file_extension}" for i in indices if f"{prefix}{i:04d}{file_extension}" in file_list]

    slices = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        slice_data = read_raw_xrmc_data(file_path, shape, dtype, offset)
        slices.append(slice_data)

    return np.stack(slices, axis=2)  # Empilha ao longo do eixo z


def load_sino_data(file_path: str) -> np.ndarray:
    """
    Carrega dados de sinograma de um arquivo binário.

    Parâmetros:
    - file_path: caminho para o arquivo binário.

    Retorna:
    - array numpy com os dados do sinograma.
    """
    try:
        sino_data = np.load(file_path)
        return sino_data
    except Exception as e:
        print(f"Erro ao carregar dados do sinograma do arquivo {file_path}: {e}")
        raise e
