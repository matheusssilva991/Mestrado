import os
import numpy as np
import pandas as pd


def dataset_to_dataframe(dataset, feature_names, return_xy=False):
    """
    Converte um PyTorch Dataset em um DataFrame do Pandas.
    """
    # Pré-aloca arrays numpy
    n = len(dataset)
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        feats, label = dataset[i]
        X[i] = feats.numpy()
        y[i] = label
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    if return_xy:
        return df, X, y

    return df


def ensure_dir_exists(dir_path):
    """
    Garante que o diretório especificado existe; cria-o se não existir.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def check_if_numpy_data_exists(save_dir):
    """
    Verifica se todos os arquivos numpy necessários existem no diretório especificado.
    Retorna True se todos os arquivos existirem, caso contrário False.
    """
    required_files = [
        'X_train.npy', 'y_train.npy',
        'X_val.npy', 'y_val.npy',
        'X_test.npy', 'y_test.npy'
    ]
    return all(os.path.isfile(os.path.join(save_dir, f)) for f in required_files)


def save_all_numpy_data(save_dir, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Salva todos os arrays numpy do dataset em arquivos .npy.
    """
    ensure_dir_exists(save_dir)
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)


def load_all_numpy_data(save_dir):
    """
    Carrega todos os arrays numpy do dataset a partir de arquivos .npy.
    Retorna: X_train, y_train, X_val, y_val, X_test, y_test
    """

    if not check_if_numpy_data_exists(save_dir):
        raise FileNotFoundError(f"Numpy data files not found in directory: {save_dir}")

    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test
