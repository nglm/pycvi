import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple
from .exceptions import ScoreError

def better_score(
    score1: float,
    score2: float,
    maximize: bool,
    or_equal: bool = False
) -> bool:
    """
    Determines whether `score1` is indeed better than `score2`.

    If both scores are None, return a ScoreError.

    It is assumed that if one (and only one) score is `None` it means
    that it hasn't been reached yet, which means that it is probably
    the best.
    """
    if score1 is None and score2 is None:
        msg = "Better score not determined, both scores are None."
        raise ScoreError(msg)
    elif score1 is None:
        return True
    elif score2 is None:
        return False
    elif score1 == score2:
        return or_equal
    elif score1 > score2:
        return maximize
    elif score1 < score2:
        return not maximize
    else:
        msg = "Better score could not be determined: {} | {}".format(
            score1, score2
        )
        raise ScoreError(msg)

def argbest(
    scores: List[float],
    maximize: bool,
    ignore_None: bool = False,
) -> int:
    """
    Returns index of best score
    """
    # In some cases we want "None" to be the best score
    if not ignore_None:
        try:
            res = scores.index(None)
            return res
        except ValueError:
            # If we wanted None to be the best but there is no None
            # just continue with the regular case
            pass
    scores_with_nans = [s if s is not None else np.nan for s in scores]
    if maximize:
        return int(np.nanargmax(scores_with_nans))
    else:
        return int(np.nanargmin(scores_with_nans))

def best_score(
    scores: List[float],
    maximize: bool,
    ignore_None: bool = False,
) -> float:
    """
    Returns best score
    """
    return scores[argbest(scores, maximize, ignore_None)]

def argworst(
    scores: List[float],
    maximize: bool,
) -> int:
    """
    Returns index of worst score
    """
    scores_with_nans = [s if s is not None else np.nan for s in scores]
    if maximize:
        return int(np.nanargmin(scores_with_nans))
    else:
        return int(np.nanargmax(scores_with_nans))

def worst_score(
    scores: List[float],
    maximize: bool,
) -> float:
    """
    Returns worst score
    """
    return scores[argworst(scores, maximize)]