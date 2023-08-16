
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from .cvi import gap_statistic, silhouette, score_function, CH, hartigan
from .compute_scores import (
    best_score, better_score, worst_score, argbest, argworst
)


class Score():

    def __init__(
        self,
        score_function: callable = None,
        maximise: bool = True,
        improve: bool = True,
        score_type: str = "monotonous",
        k_condition: callable = None,
    ) -> None:
        self.function = score_function
        self.maximise = maximise
        self.improve = improve
        self.score_type = score_type
        self.k_condition = k_condition

    def __call__(
        self,
        X: np.ndarray,
        clusters_data: List[Tuple[List[int], Dict]],
        dist_kwargs: dict = {},
        *args: Any, **kwds: Any,
    ) -> Any:
        return self.function(X, clusters_data, dist_kwargs, )

    def is_relevant(self, score, k, score_prev, k_prev) -> bool:
        # A score is always relevant when it is absolute
        if self.score_type == "absolute":
            return True
        else:
            # When relative:
            if self.improve:
                # better "and" greater or worse and lesser
                return (
                    self.better_score(score, score_prev) == k > k_prev
                )
            else:
                # worse "and" greater or better and lesser
                return (
                    self.better_score(score_prev, score) == k_prev > k
                )

    def better_score(
        self,
        score1: float,
        score2: float,
        or_equal: bool = False
    ) -> bool:
        return better_score(score1, score2, self.maximise, or_equal)

    def argbest(
        self,
        score1: float,
        score2: float,
    ) -> int:
        return argbest(score1, score2, self.maximise)

    def best_score(
        self,
        score1: float,
        score2: float,
    ) -> float:

        return best_score(score1, score2, self.maximize)

    def argworst(
        self,
        score1: float,
        score2: float,
    ) -> int:
        return argworst(score1, score2, self.maximize)

    def worst_score(
        self,
        score1: float,
        score2: float,
    ) -> float:
        """
        Returns worst score
        """
        return worst_score(score1, score2, self.maximize)

class Hartigan(Score):

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:

        super().__init__(
            score_function=hartigan,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=0 and k<N)
        )

class CalinskiHarabasz(Score):

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:

        # Note that the case k=1 for the absolute version will always
        # give CH=0
        k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        super().__init__(
            score_function=CH,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition = k_condition,
        )
