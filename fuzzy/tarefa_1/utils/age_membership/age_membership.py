def young_membership(age, tail_end=30):
    """Pertinencia para 'jovem' com trechos lineares."""
    # Ate 18 anos, 1
    if age <= 18:
        return 1.0

    # De 18 a 21, [1, 0.7]
    if age <= 21:
        return 1.0 - 0.1 * (age - 18.0)

    # De 21 a 25, [0.7, 0.3]
    if age <= 25:
        return 0.7 - 0.1 * (age - 21.0)

    if tail_end <= 25:
        return 0.0

    # Depois de 25, [].
    if age < tail_end:
        return 0.3 * (tail_end - age) / (tail_end - 25.0)

    return 0.0


def adult_membership(age, tail_end=70):
    """Pertinencia para 'adulto' por faixas lineares."""
    # Antes de 18, 0
    if age <= 18:
        return 0.0

    # De 18 a 21, [0, 0.3]
    if age <= 21:
        return 0.1 * (age - 18.0)

    # De 21 a 25, [0.3, 0.7].
    if age <= 25:
        return 0.3 + 0.1 * (age - 21.0)

    # De 25 a 30, [0.7, 1].
    if age <= 30:
        return 0.7 + 0.06 * (age - 25.0)

    # De 30 a 55, 1.
    if age <= 55:
        return 1.0

    # De 55 a 60, [1, 0.7].
    if age <= 60:
        return 1.0 - 0.06 * (age - 55.0)

    # De 60 a 65, [0.7, 0.3].
    if age <= 65:
        return 0.7 - 0.08 * (age - 60.0)

    if tail_end <= 65:
        return 0.0

    # Depois de 65, [0.3, 0].
    if age < tail_end:
        return 0.3 * (tail_end - age) / (tail_end - 65.0)

    return 0.0
