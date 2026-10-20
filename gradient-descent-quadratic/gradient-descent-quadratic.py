def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    result = x0
    for _ in range(steps):
        result = result - lr*gradient(a, b, result)
    return result
    
def gradient(a: float, b: float, x: float) -> float:
    return 2*a*x + b