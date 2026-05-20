def young_membership(age, tail_end=30):
    """Pertinencia para 'jovem' com trechos lineares."""
    # Ate 18 anos, a pertinencia e maxima.
    if age <= 18:
        return 1.0

    # De 18 a 21, cai linearmente de 1.0 para 0.7.
    if age <= 21:
        return 1.0 - 0.1 * (age - 18.0)

    # De 21 a 25, cai linearmente de 0.7 para 0.3.
    if age <= 25:
        return 0.7 - 0.1 * (age - 21.0)

    if tail_end <= 25:
        return 0.0

    # Depois de 25, a pertinencia continua caindo linearmente ate 0.
    if age < tail_end:
        return 0.3 * (tail_end - age) / (tail_end - 25.0)
    return 0.0


def adult_membership(age, tail_end=70):
    """Pertinencia para 'adulto' por faixas lineares."""
    # Antes de 18, nao ha pertinencia para adulto.
    if age <= 18:
        return 0.0

    # De 18 a 21, sobe de 0.0 para 0.3.
    if age <= 21:
        return 0.1 * (age - 18.0)

    # De 21 a 25, sobe de 0.3 para 0.7.
    if age <= 25:
        return 0.3 + 0.1 * (age - 21.0)

    # De 25 a 30, sobe suavemente de 0.7 para 1.0.
    if age <= 30:
        return 0.7 + 0.06 * (age - 25.0)

    # De 30 a 55, a pertinencia fica no maximo.
    if age <= 55:
        return 1.0

    # De 55 a 60, desce linearmente de 1.0 para 0.7.
    if age <= 60:
        return 1.0 - 0.06 * (age - 55.0)

    # De 60 a 65, desce linearmente de 0.7 para 0.3.
    if age <= 65:
        return 0.7 - 0.08 * (age - 60.0)

    if tail_end <= 65:
        return 0.0

    # Depois de 65, a cauda cai de 0.3 ate 0.0 em tail_end.
    if age < tail_end:
        return 0.3 * (tail_end - age) / (tail_end - 65.0)

    return 0.0
