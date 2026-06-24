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
    Determine whether `score1` is better than `score2`.

    If both scores are None, a `ScoreError` is raised.

    Parameters
    ----------
    score1 : float
        First score to compare.
    score2 : float
        Second score to compare.
    maximize : bool
        Whether higher scores are better.
    or_equal : bool, optional
        Whether to return True when scores are equal.

    Returns
    -------
    bool
        True if `score1` is better than `score2`, else False.

    Raises
    ------
    ScoreError
        If both scores are None or comparison cannot be determined.
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
    Return index of best score.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    maximize : bool
        Whether higher scores are better.
    ignore_None : bool, optional
        If False, `None` is treated as best score.

    Returns
    -------
    int
        Index of best score in `scores`.
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
    Return best score.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    maximize : bool
        Whether higher scores are better.
    ignore_None : bool, optional
        If False, `None` is treated as best score.

    Returns
    -------
    float
        Best score in `scores`.
    """
    return scores[argbest(scores, maximize, ignore_None)]

def argworst(
    scores: List[float],
    maximize: bool,
) -> int:
    """
    Return index of worst score.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    maximize : bool
        Whether higher scores are better.

    Returns
    -------
    int
        Index of worst score in `scores`.
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
    Return worst score.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    maximize : bool
        Whether higher scores are better.

    Returns
    -------
    float
        Worst score in `scores`.
    """
    return scores[argworst(scores, maximize)]