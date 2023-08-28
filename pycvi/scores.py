
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from .cvi import gap_statistic, silhouette, score_function, CH, hartigan
from .compute_scores import (
    best_score, better_score, worst_score, argbest, argworst, compute_score,
    reduce
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
        # If score_type == "absolute", 'improve' is irrelevant
        self.improve = improve
        self.score_type = score_type
        self.k_condition = k_condition

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {},
    ) -> dict:
        return score_kwargs

    def __call__(
        self,
        X: np.ndarray,
        clusters: List[List[int]],
        score_kwargs: dict = {},
    ) -> float:
        n_clusters = len(clusters)
        if self.k_condition(n_clusters):
            return self.function(X, clusters, **score_kwargs)
        else:
            msg = (
                f"{self.__class__} called with an incompatible number of "
                + f"clusters: {n_clusters}"
            )
            raise ValueError(msg)

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
        scores: List[float],
        ignore_None: bool = False,
    ) -> int:
        return argbest(scores, self.maximise, ignore_None)

    def best_score(
        self,
        scores: List[float],
        ignore_None: bool = False,
    ) -> float:

        return best_score(scores, self.maximise, ignore_None)

    def argworst(
        self,
        scores: List[float],
    ) -> int:
        return argworst(scores, self.maximise)

    def worst_score(
        self,
        scores: List[float],
    ) -> float:
        """
        Returns worst score
        """
        return worst_score(scores, self.maximise)

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
            k_condition= lambda k: (k>=0)
        )

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:
        s_kw = {}
        s_kw["k"] = n_clusters
        if n_clusters < len(X_clus):
            s_kw["clusters_next"] = clusterings_t.get(n_clusters+1, None)
        s_kw.update(score_kwargs)
        return s_kw

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

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:

        s_kw = {}
        s_kw["k"] = n_clusters
        if n_clusters == 0:
            s_kw["X1"] = X_clus[clusterings_t.get(1, None)]
        s_kw.update(score_kwargs)
        return s_kw

class GapStatistic(Score):

    def __init__(
        self,
        score_type: str = "monotonous",
        B: int = 10
    ) -> None:

        k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        super().__init__(
            score_function= gap_statistic,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition=k_condition
        )

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:
        s_kw = {"B" : 10}
        s_kw["k"] = n_clusters
        s_kw.update(score_kwargs)
        return s_kw

class Silhouette(Score):

    def __init__(self) -> None:
        super().__init__(
            score_function=silhouette,
            maximise=True,
            improve=None,
            score_type="absolute",
            k_condition= lambda k: (k>=2)
        )

class ScoreFunction(Score):

    def __init__(self) -> None:
        super().__init__(
            score_function=score_function,
            maximise=True,
            improve=None,
            score_type="absolute",
            k_condition= lambda k: (k>=1)
        )

class Inertia(Score):

    def __init__(
        self,
        reduction: Union[str, callable] = "sum",
    ) -> None:
        """
        reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce`
        """

        k_condition = lambda k: k>=0

        def score_function(X, clusters):
            return reduce(compute_score(
                "list_inertia", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            score_function=score_function,
            maximise=False,
            improve=True,
            score_type="monotonous",
            k_condition = k_condition,
        )

class Diameter(Score):

    def __init__(
        self,
        reduction: Union[str, callable] = "sum",
    ) -> None:
        """
        reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce`
        """

        k_condition = lambda k: k>=0

        def score_function(X, clusters):
            return reduce(compute_score(
                "list_diameter", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            score_function=score_function,
            maximise=False,
            improve=True,
            score_type="monotonous",
            k_condition = k_condition,
        )

SCORES = [
    Hartigan,  # Remove because of the annoying clustering2 argument
    CalinskiHarabasz,
    GapStatistic,
    Silhouette,
    ScoreFunction,
    Inertia,
    Diameter,
]