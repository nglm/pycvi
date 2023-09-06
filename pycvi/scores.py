
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from .cvi import gap_statistic, silhouette, score_function, CH, hartigan
from .compute_scores import (
    best_score, better_score, worst_score, argbest, argworst, compute_score,
    reduce
)
from .exceptions import InvalidKError

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

    def __call__(
        self,
        X: np.ndarray,
        clusters: List[List[int]],
        score_kwargs: dict = {},
    ) -> float:
        if "k" in score_kwargs:
            n_clusters = score_kwargs["k"]
        else:
            n_clusters = len(clusters)
        if self.k_condition(n_clusters):
            return self.function(X, clusters, **score_kwargs)
        else:
            msg = (
                f"{self.__class__} called with an incompatible number of "
                + f"clusters: {n_clusters}"
            )
            raise InvalidKError(msg)

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {},
    ) -> Union[dict, None]:
        return score_kwargs

    def criterion(self, scores: Dict[int, float]):
        if self.score_type == "monotonous":
            selected_k = None
            max_diff = 0

            list_k = list(scores.keys())
            last_relevant_score = scores[list_k[0]]
            last_relevant_k = list_k[0]

            # If it is improving compare last to current
            # Otherwise compare current to next
            if self.improve:
                list_k_compare = list_k[1:]
            else:
                list_k_compare = list_k[:-1]
            # Find the biggest increase / drop
            for i, k in enumerate(list_k_compare):
                # Special case if the score couldn't be computed
                # either because this score doesn't allow this k
                # or because the clustering model didn't converge
                if scores[k] is None:
                    continue
                elif (
                    i==0
                    or self.is_relevant(
                        scores[k], k,
                        last_relevant_score, last_relevant_k)
                ):
                    diff = abs(scores[k] - last_relevant_score)
                    if max_diff < diff:
                        selected_k = k
                        max_diff = diff

                    last_relevant_k = k
                    last_relevant_score = scores[k]
        else:
            #
            scores_valid = {k: s for k,s in scores.items() if s is not None}
            # Special case if None if all scores were None (because of
            # the condition on k or because the model didn't converge)
            if scores_valid == {}:
                selected_k = None
            else:
                if self.maximise:
                    selected_k = max(scores_valid, key=scores_valid.get)
                else:
                    selected_k = min(scores_valid, key=scores_valid.get)
        return selected_k

    def is_relevant(self, score, k, score_prev, k_prev) -> bool:
        # A score is always relevant when it is absolute
        if self.score_type == "absolute":
            return True
        else:
            # When relative:
            if self.improve:
                # better "and" greater or worse and lesser
                return (
                    self.better_score(score, score_prev) == (k > k_prev)
                )
            else:
                # worse "and" greater or better and lesser
                return (
                    self.better_score(score_prev, score) == (k_prev > k)
                )

    def select(
        self,
        scores_t_k: List[Dict[int, float]]
    ) -> List[int]:
        """
        Select the best $k$ for each $t$ according to the scores given.

        :param scores_t_k: _description_
        :type scores_t_k: List[Dict[int, float]]
        :return: _description_
        :rtype: List[int]
        """
        return [self.criterion(s_t) for s_t in scores_t_k]

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
        if n_clusters == 0:
            s_kw["X1"] = X_clus
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
            s_kw["X1"] = X_clus
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
        reduction: Union[str, callable] = "max",
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