# T-normas
def min_tnorma(a, b):
    return min(a, b)

def prod_tnorma(a, b):
    return a * b


# T-conormas
def max_tconorma(a, b):
    return max(a, b)

def lukasiewicz_tconorma(a, b):
    return min(1, a + b)


# Negacao
def sugeno_negation(a, lamb=0):
    return (1 - a) / (1 + lamb * a)


def zadeh_negation(a):
    return 1 - a
