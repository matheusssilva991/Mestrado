from .tcornomas import (
    tconorma_dombi,
    tconorma_hamacher,
    tconorma_lukasiewicz,
    tconorma_max,
    tconorma_produto,
    tconorma_sugeno_weber,
)
from .tnormas import (
    tnorma_dombi,
    tnorma_hamacher,
    tnorma_lukasiewicz,
    tnorma_min,
    tnorma_produto,
    tnorma_sugeno_weber,
)


TNORMAS_DISPONIVEIS = {
    "min": tnorma_min,
    "prod": tnorma_produto,
    "lukasiewicz": tnorma_lukasiewicz,
    "hamacher": tnorma_hamacher,
    "sugeno-weber": tnorma_sugeno_weber,
    "dombi": tnorma_dombi,
}


TCONORMAS_DISPONIVEIS = {
    "max": tconorma_max,
    "prod": tconorma_produto,
    "lukasiewicz": tconorma_lukasiewicz,
    "hamacher": tconorma_hamacher,
    "sugeno-weber": tconorma_sugeno_weber,
    "dombi": tconorma_dombi,
}
