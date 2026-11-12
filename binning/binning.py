import math

def binning(values: list, num_bins: int) -> list:
    """
    Returns the equal-width bin index of every value.
    """
    mini = values[0]
    max = values[-1]
    w = (max - mini)/num_bins
    if mini==max:
        return [0 for _ in values]
    else:
        return [ min(math.floor((x - values[0])/w), num_bins-1) for x in values]
