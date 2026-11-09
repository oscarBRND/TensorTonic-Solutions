def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    # Write code here
    counts = {}
    for elem in values:
        counts[elem] = counts.get(elem, 0) + 1
    return [ counts[v] / len(values) for v in values]