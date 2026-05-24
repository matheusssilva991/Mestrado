from numpy import sqrt

def tnorma_min(x: float, y: float) -> float:
    """T-norma do mínimo"""
    return min(x, y)


def tnorma_produto(x: float, y: float) -> float:
    """T-norma do produto"""
    return x * y


def tnorma_lukasiewicz(x: float, y: float) -> float:
    """T-norma de Lukasiewicz"""
    return max(0, x + y - 1)


def tnorma_hamacher(x: float, y: float) -> float:
    """T-norma de Hamacher"""
    if x == 0 and y == 0:
        return 0
    num = x * y
    denom = x + y - num
    return num / denom


def tnorma_sugeno_weber(x: float, y: float) -> float:
    """T-norma de Sugeno-Weber"""
    num = x + y - 1 + 2 * x * y
    denom = 3
    return max(0, num / denom)


def tnorma_dombi(x: float, y: float) -> float:
    """T-norma de Dombi"""
    fator_denom1 = ((1 - x) / x) ** 2
    fator_denom2 = ((1 - y) / y) ** 2
    denom = 1 + sqrt(fator_denom1 + fator_denom2)

    return 1 / denom
