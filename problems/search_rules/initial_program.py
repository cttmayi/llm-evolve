# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np
from utils.init import init

init()


def search_algorithm(input_data: dict) -> bool:
    """A simple search algorithm that checks if the input data meets a certain condition"""
    x = input_data.get("x", 0)
    y = input_data.get("y", 0)
    # Example condition: check if x and y are within a certain range
    cond1 = (-2 <= x <= 2 and -2 <= y <= 2)

    cond2 = (x == 0 and y == 0)

    return cond1 and cond2



# EVOLVE-BLOCK-END


