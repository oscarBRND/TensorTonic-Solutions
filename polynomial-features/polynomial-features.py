def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """

    result = []
    for value in values:
        result.append([value**i for i in range(degree+1)])
    return result