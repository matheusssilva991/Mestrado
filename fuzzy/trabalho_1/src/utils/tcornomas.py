from numpy import sqrt


def tconorma_max(x: float, y: float) -> float:
    """T-conorma do máximo"""
    return max(x, y)


def tconorma_produto(x: float, y: float) -> float:
    """T-conorma dual da t-norma do produto"""
    return x + y - x * y


def tconorma_lukasiewicz(x: float, y: float) -> float:
    """T-conorma de Lukasiewicz"""
    return min(1, x + y)


def tconorma_hamacher(x: float, y: float) -> float:
    """T-conorma de Hamacher"""
    if x == 1 and y == 1:
        return 1
    num = x + y - 2 * x * y
    denom = 1 - x * y
    return num / denom


def tconorma_sugeno_weber(x: float, y: float) -> float:
    """T-conorma de Sugeno-Weber"""
    return min(1, x + y - (2 * x * y) / 3)


def tconorma_dombi(x: float, y: float) -> float:
    """T-conorma de Dombi"""
    if x == 1 or y == 1:
        return 1
    if x == 0 and y == 0:
        return 0

    x = float(x)
    y = float(y)
    eps = 1e-12
    fator_denom1 = (x / max(1 - x, eps)) ** 2
    fator_denom2 = (y / max(1 - y, eps)) ** 2
    denom = 1 + sqrt(fator_denom1 + fator_denom2)

    return 1 - 1 / denom
