import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    result = []
    for elem in x:
        if elem>0:
            result.append(elem)
        else:
            elu = alpha * (math.exp(elem) - 1)
            result.append(elu)
    return result