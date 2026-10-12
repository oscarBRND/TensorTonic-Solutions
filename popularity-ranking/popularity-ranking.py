def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    WRs = []
    for elem in items:
        v = elem[1]
        R = elem[0]
        WR = (v/(v + min_votes))*R + (min_votes/(v + min_votes))*global_mean
        WRs.append(WR)
    return WRs
    