
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict, Tuple

from .cvi import (
    gap_statistic, silhouette, score_function, CH, hartigan, MB, SD_index,
    SDbw_index, dunn, xie_beni, xie_beni_star, davies_bouldin
)
from .compute_scores import (
    best_score, better_score, worst_score, argbest, argworst, compute_score,
    reduce
)
from .exceptions import InvalidKError

class Score():

    score_types: List[str] = ["monotonous", "absolute"]
    reductions: List[str] = None

    def __init__(
        self,
        score_function: callable = None,
        maximise: bool = True,
        improve: bool = True,
        score_type: str = "monotonous",
        criterion_function: callable = None,
        k_condition: callable = None,
        ignore0: bool = False,
    ) -> None:
        self.function = score_function
        self.criterion_function = criterion_function
        self.maximise = maximise
        # If score_type == "absolute", 'improve' is irrelevant
        self.improve = improve
        if score_type in self.score_types:
            self.score_type = score_type
        else:
            raise ValueError(
                "score_type {} invalid for {}. Please choose among {}".format(score_type, str(self), self.score_types)
            )
        self.k_condition = k_condition
        self.ignore0 = ignore0
        self.N = None
        self.d = None

    def __call__(
        self,
        X: np.ndarray,
        clusters: List[List[int]],
        score_kwargs: dict = {},
    ) -> float:
        """
        Compute the score of the clustering.

        :param X: Values of all members
        :type X: np.ndarray, shape: (N, d*w_t) or (N, w_t, d)
        :param clusters: List of (members, info) tuples
        :type clusters: List[List[int]]
        :param score_kwargs: kwargs specific for the score, defaults to
            {}
        :type score_kwargs: dict, optional
        :raises InvalidKError: _description_
        :return: _description_
        :rtype: float
        """
        dims = X.shape
        self.N = dims[0]
        self.d = dims[-1]
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

    def criterion(
        self,
        scores: Dict[int, float],
        score_type: str = None,
    ) -> int:
        """
        Treats the regular cases .
        Cases included monotonous/absolute/pseudo-monotonous and
        maximise=True/False with improve=True/False.

        Does not take into account rules that are specific to a score.
        """
        # Because some custom criterion_function relies on the general
        # case "monotonous"/"absolute"
        if score_type is None:
            score_type = self.score_type
        # For monotonous scores
        if score_type == "monotonous":
            scores_valid = {k: s for k,s in scores.items() if s is not None}
            selected_k = None
            max_diff = 0

            list_k = list(scores_valid.keys())
            last_relevant_score = scores_valid[list_k[0]]
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
                if scores_valid[k] is None:
                    continue
                # Special case for example for Hartigan where
                # we don't use k=0 as a reference score if k=0 is more
                # relevant than k=1
                elif (i==0 and self.ignore0 and not self.is_relevant(
                        scores_valid[k], k,
                        last_relevant_score, last_relevant_k)
                    ):
                    selected_k = 1
                    last_relevant_k = 1
                    last_relevant_score = scores_valid[k]
                elif (
                    (i==0 and not self.ignore0)
                    or self.is_relevant(
                        scores_valid[k], k,
                        last_relevant_score, last_relevant_k
                    )
                ):
                    diff = abs(scores_valid[k] - last_relevant_score)
                    if max_diff < diff:
                        selected_k = k
                        max_diff = diff

                    last_relevant_k = k
                    last_relevant_score = scores_valid[k]
        # For absolute scores
        elif score_type == "absolute":
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
        # For scores with a special selection process
        else:
            selected_k = self.criterion_function(scores)
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

    score_types: List[str] = ["monotonous", "original"]

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:
        """
        According to de Amorim and Hennig [2015] "In the original paper,
        the lowest $k$ to yield Hartigan $<= 10$ was proposed as the
        optimal choice. However, if no $k$ meets this criteria, choose
        the $k$ whose difference in Hartigan for $k$ and $k+1$ is the
        smallest". According to Tibshirani et al. [2001] it is "the
        estimated number of clusters is the smallest k such that
        Hartigan $<= 10$ and $k=1$ could then be possible.

        Possible score_type: monotonous, or original
        """

        f_k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        super().__init__(
            score_function=hartigan,
            maximise=False,
            improve=True,
            score_type=score_type,
            criterion_function=self.f_criterion,
            k_condition= f_k_condition,
            ignore0 = True,
        )

    def f_criterion(
        self,
        scores: Dict[int, float],
    ) -> int:
        valid_k = {
            k: s for k,s in scores.items()
            if ((s is not None) and (s<=10))
        }
        if valid_k:
            # Take the lowest $k$ to yield Hartigan $<= 10$
            selected_k = min(valid_k)
        else:
            # Otherwise, choose the $k$ whose difference in Hartigan for
            # $k$ and $k+1$ is the smallest, or when
            # hartigan(k)<hartigan(k+1)
            ks = sorted([k for k in scores.keys() if scores[k] is not None])
            arg_selected_k = np.argmin(
                [
                    max(scores[ks[i]]-scores[ks[i+1]], 0)
                    for i in range(len(ks)-1)
                ]
            )
            selected_k = ks[arg_selected_k]
        return selected_k

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

    def __str__(self) -> str:
        return 'Hartigan_{}'.format(self.score_type)

class CalinskiHarabasz(Score):

    score_types: List[str] = ["monotonous", "original", "absolute"]

    def __init__(
        self,
        score_type: str = "monotonous"
    ) -> None:
        """
        Originally this index is absolute and has to be maximised to
        find the best $k$. A monotonous approach can also be taken, so
        that the case k=1 can be selected, with CH(1) = 0 and CH(0)
        extended (see `pycvi.cvi.CH`)
        """

        # Note that the case k=1 for the absolute version will always
        # give CH=0
        f_k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        # original and absolute are the same
        if score_type == "original":
            score_type = "absolute"

        super().__init__(
            score_function=CH,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition = f_k_condition,
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
        if "zero_type" not in s_kw or s_kw["zero_type"] == None:
            if len(X_clus) > 100:
                s_kw["zero_type"] = "variance"
            else:
                s_kw["zero_type"] = "bounds"
        s_kw.update(score_kwargs)
        return s_kw

    def __str__(self) -> str:
        score_type = self.score_type
        # original and absolute are the same
        if score_type == "absolute":
            score_type = "original"
        return 'CalinskiHarabasz_{}'.format(score_type)

class GapStatistic(Score):

    score_types: List[str] = ["monotonous", "absolute", "original"]

    def __init__(
        self,
        score_type: str = "monotonous",
    ) -> None:

        f_k_condition = lambda k: (
            k>=0 if score_type == "monotonous" else k>=1
        )

        super().__init__(
            score_function= gap_statistic,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition=f_k_condition,
            criterion_function=self.f_criterion,
        )

        self.s = {}

    def f_criterion(
        self,
        scores: Dict[int, float],
    ) -> int:
        """
        Select the smallest k such that "Gap(k) >= Gap(k+1) - s(k+1)"
        """
        # Keep only k corresponding to scores that are not None
        ks = sorted([k for k in scores.keys() if scores[k] is not None])
        # Get all k such that "Gap(k) >= Gap(k+1) - s(k+1)"
        selected_k_tmp = [
            ks[k] for k in range(len(ks)-1)
            # Gap(k) >= Gap(k+1) - s(k+1)
            if scores[ks[k]] >= scores[ks[k+1]] - self.s[ks[k+1]]
        ]
        # Find the smallest k such that "Gap(k) >= Gap(k+1) - s(k+1)"
        if selected_k_tmp:
            selected_k = int(np.amin(selected_k_tmp))
        else:
            selected_k = None
        return selected_k

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:
        s_kw = {"B" : 10}
        s_kw["k"] = n_clusters
        if self.score_type == "original":
            s_kw["return_s"] = True
        s_kw.update(score_kwargs)
        if "zero_type" not in s_kw or s_kw["zero_type"] == None:
            if len(X_clus) > 100:
                s_kw["zero_type"] = "variance"
            else:
                s_kw["zero_type"] = "bounds"
        return s_kw

    def __call__(
        self,
        X: np.ndarray,
        clusters: List[List[int]],
        score_kwargs: dict = {}
    ) -> float:
        if self.score_type == "original":
            gap, s =  super().__call__(X, clusters, score_kwargs)
            k = score_kwargs["k"]
            self.s[k] = s
            return gap
        else:
            return super().__call__(X, clusters, score_kwargs)

    def __str__(self) -> str:
        return 'GapStatistic_{}'.format(self.score_type)

class Silhouette(Score):

    score_types: List[str] = ["absolute"]

    def __init__(
        self,
        score_type: str = "absolute",
    ) -> None:
        super().__init__(
            score_function=silhouette,
            maximise=True,
            improve=None,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'Silhouette'

class ScoreFunction(Score):

    score_types: List[str] = ["absolute", "original"]

    def __init__(self, score_type: str = "original") -> None:
        """
        Originally this index has to be maximised to find the best $k$,
        but the original paper Saitta et al. [2007] adds special cases:

        - if the score always increases, then the number $k=1$ is chosen
        - if a maximum is found, outside the extreme $k$ values, then
          the argument of this maximum is chosen.
        - it is empirically decided that if $(SF2 − SF1) \times d <=
          0.2$ then, $k = 1$ is also chosen (d being the dimensionality
          of the data)

        The absolute case can also be chosen (i.e. special case are
        ignored)
        """
        super().__init__(
            score_function=score_function,
            maximise=True,
            improve=None,
            score_type=score_type,
            criterion_function=self.f_criterion,
            k_condition= lambda k: (k>=1)
        )

    def f_criterion(
        self,
        scores: Dict[int, float],
    ) -> int:
        # General case first
        best_k = self.criterion(scores, score_type="absolute")
        # If score always increases, then choose k=1
        if best_k == max(scores):
            best_k = 1
        if (
            (1 in scores) and (2 in scores)
            and ((scores[2] - scores[1]) * self.d <= 0.2)
        ):
            best_k=1
        return best_k

    def __str__(self) -> str:
        return 'ScoreFunction'

class MaulikBandyopadhyay(Score):

    score_types: List[str] = ["absolute", "monotonous"]

    def __init__(self, score_type: str = "original") -> None:
        """
        Originally this index has to be maximised to find the best $k$.
        The monotonous case can also be chosen.

        The case k=1 always returns 0.
        """
        super().__init__(
            score_function=MB,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=1)
        )

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:
        s_kw = {"p" : 2}
        s_kw["k"] = n_clusters
        s_kw.update(score_kwargs)
        return s_kw

    def __str__(self) -> str:
        return f'MaulikBandyopadhyay_{self.score_type}'

class SD(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.

        Note that if two clusters have equal centroids, then `SD = inf`
        which means that this clustering is irrelevant, which works as
        intended (even though two clusters could be well separated and
        still have equal centroids, as in the case of two concentric
        circles).
        """
        super().__init__(
            score_function=SD_index,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def get_score_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        score_kwargs: dict = {}
    ) -> None:
        s_kw = {"alpha" : None}
        s_kw.update(score_kwargs)
        return s_kw

    def __str__(self) -> str:
        return 'SD'

class SDbw(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.

        Note that if two clusters have all datapoints further away to
        their respective centroids than what is called in the original
        paper "the average standard deviation of clusters", then `SDbw =
        inf`, which means that this clustering is irrelevant, which
        works as intended.
        """
        super().__init__(
            score_function=SDbw_index,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'SDbw'

class Dunn(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            score_function=dunn,
            maximise=True,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'Dunn'

class XB(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            score_function=xie_beni,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'XB'

class XBStar(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            score_function=xie_beni_star,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'XB_star'

class DB(Score):

    score_types: List[str] = ["absolute"]

    def __init__(self, score_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            score_function=davies_bouldin,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'DB'

class Inertia(Score):

    score_types: List[str] = ["monotonous"]
    reductions: List[str] = [
        "sum", "mean", "max", "median", "min", "", None
    ]

    def __init__(
        self,
        reduction: Union[str, callable] = "sum",
        score_type: str = "monotonous",
    ) -> None:
        """
        reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce`
        """

        f_k_condition = lambda k: k>=0

        def score_function(X, clusters):
            return reduce(compute_score(
                "list_inertia", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            score_function=score_function,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition = f_k_condition,
        )

        self.reduction = reduction

    def __str__(self) -> str:
        if hasattr(self, "reduction"):
            return 'Inertia_{}'.format(self.reduction)
        else:
            return 'Inertia'

class Diameter(Score):

    score_types: List[str] = ["monotonous"]
    reductions: List[str] = [
        "sum", "mean", "max", "median", "min", "", None
    ]

    def __init__(
        self,
        reduction: Union[str, callable] = "max",
        score_type: str = "monotonous",
    ) -> None:
        """
        reduction available: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce`
        """

        f_k_condition = lambda k: k>=0

        def score_function(X, clusters):
            return reduce(compute_score(
                "list_diameter", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            score_function=score_function,
            maximise=False,
            improve=True,
            score_type=score_type,
            k_condition = f_k_condition,
        )

        self.reduction = reduction

    def __str__(self) -> str:
        return 'Diameter_{}'.format(self.reduction)

SCORES = [
    Hartigan,
    CalinskiHarabasz,
    GapStatistic,
    Silhouette,
    ScoreFunction,
    MaulikBandyopadhyay,
    SD,
    SDbw,
    Dunn,
    XB,
    XBStar,
    DB,
    Inertia,
    Diameter,
]