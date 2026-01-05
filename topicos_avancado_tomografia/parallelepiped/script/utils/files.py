import os
import time
import re


def is_file_in_use(path):
    if not os.path.exists(path):
        return False
    try:
        # Try to open for exclusive access (Windows may raise)
        f = open(path, "a+")
        f.close()
        return False
    except IOError:
        return True


def wait_for_file_stable(path, timeout=30, poll=0.5):
    """Espera até que o arquivo exista e seu tamanho esteja estável."""
    start = time.time()
    last_size = -1
    while time.time() - start < timeout:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size == last_size and size > 0:
                return True
            last_size = size
        time.sleep(poll)
    return False


def replace_rotate_angle(file_path, new_angle):
    """
    Substitui o primeiro valor numérico após 'RotateAll 0 0 0 0 0 1 ' por new_angle.
    Lança RuntimeError se o padrão não for encontrado ou FileNotFoundError se o arquivo não existir.
    """
    # Regex local ao módulo para localizar a linha RotateAll
    rotate_re = re.compile(r"(RotateAll\s+0\s+0\s+0\s+0\s+0\s+1\s+)([-+]?\d+(\.\d+)?)")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # Use callable replacement para evitar interpretações de backreferences (ex: \11)
    new_txt, n = rotate_re.subn(lambda m: f"{m.group(1)}{new_angle:.1f}", txt, count=1)
    if n == 0:
        raise RuntimeError(
            "Pattern 'RotateAll 0 0 0 0 0 1 <angle>' not found in " + file_path
        )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_txt)
    return True
