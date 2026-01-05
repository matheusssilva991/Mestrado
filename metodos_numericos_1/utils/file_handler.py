import numpy as np

def read_file(file: str) -> list[str]:
    lines: list[str] = []

    with open(file, "r") as file_descriptor:
        for line in file_descriptor:
            lines.append(line)

    return lines

def write_result(file: str, result: np.float64|np.int32) -> None:
    with open(file, "w") as file_descriptor:
        file_descriptor.write(result) # type: ignore
