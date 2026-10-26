def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """

    EMA =[values[0]]
    for i, elem in enumerate(values[1:]):
        EMA.append(alpha*elem + (1-alpha)*EMA[i])
    return EMA