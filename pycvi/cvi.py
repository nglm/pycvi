"""
Python implementation of state-of the-art internal CVIs.

Internal CVIs are used to select the best clustering among a set of
pre-computed clustering when no information about the true clusters nor
the number of clusters is available.

.. [Hartigan] D. J. Strauss and J. A. Hartigan, “Clustering algorithms,”
   Biometrics, vol. 31, p. 793, sep 1975.
.. [CH] T. Calinski and J. Harabasz, “A dendrite method for cluster
   analysis,” Communications in Statistics - Theory and Methods, vol. 3,
   no. 1, pp. 1–27, 1974.
.. [Gap] R. Tibshirani, G. Walther, and T. Hastie, “Estimating the
   number of clusters in a data set via the gap statistic,” Journal
   of the Royal Statistical Society Series B: Statistical
   Methodology, vol. 63, pp. 411–423, July 2001.
.. [Silhouette] P. J. Rousseeuw, “Silhouettes: a graphical aid to the
   interpretation and validation of cluster analysis,” Journal of
   computational and applied mathematics, vol. 20, pp. 53–65, 1987.
.. [Dunn] J. C. Dunn, “Well-separated clusters and optimal fuzzy
   partitions,” Journal of Cybernetics, vol. 4, pp. 95–104, Jan. 1974.
.. [DB] D. L. Davies and D. W. Bouldin, “A cluster separation measure,”
   IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.
   PAMI-1, pp. 224–227, Apr. 1979.
.. [SD] M. Halkidi, M. Vazirgiannis, and Y. Batistakis, “Quality scheme
   assessment in the clustering process,” in Principles of Data Mining
   and Knowledge Discovery, pp. 265–276, Springer Berlin Heidelberg,
   2000
.. [SDbw] M. Halkidi and M. Vazirgiannis, “Clustering validity
   assessment: finding the optimal partitioning of a data set,” in
   Proceedings 2001 IEEE International Conference on Data Mining, pp.
   187–194, IEEE Comput. Soc, 2001.
.. [XB] X. Xie and G. Beni, “A validity measure for fuzzy clustering,”
   IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.
   13, no. 8, pp. 841–847, 1991.
.. [XB*] M. Kim and R. Ramakrishna, “New indices for cluster validity
   assessment,” Pattern Recognition Letters, vol. 26, pp. 2353–2363, Nov.
   2005.
.. [SF] S. Saitta, B. Raphael, and I. F. C. Smith, “A bounded index for
   cluster validity,” in Machine Learning and Data Mining in Pattern
   Recognition, pp. 174–187, Springer Berlin Heidelberg, 2007.
.. [MB] U. Maulik and S. Bandyopadhyay, “Performance evaluation of some
   clustering algorithms and validity indices,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, pp. 1650–1654, Dec. 2002.
"""

import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple

from .cvi_func import (
    gap_statistic, silhouette, score_function, CH, hartigan, MB, SD_index,
    SDbw_index, dunn, xie_beni, xie_beni_star, davies_bouldin
)
from .dist import reduce
from .compute_scores import _compute_score
from ._compare_scores import (
    best_score, better_score, worst_score, argbest, argworst
)
from .exceptions import InvalidKError, SelectionError
from ._utils import _check_list_of_dict

class CVI():
    """
    Base class for all Cluster Validity Indices in PyCVI.

    To create a custom CVI class, inherit from this class with well
    defined parameters.

    Parameters
    ----------
    cvi_function : callable, optional
        Function used to assess each clustering, by default None
    maximise : bool, optional
        Determines whether higher values mean better clustering
        quality according to this CVI , by default True
    improve : bool, optional
        Determines whether the quality of the clustering is expected
        to improve with increasing values of :math:`k` (concerns
        only monotone CVIs), by default True
    cvi_type : str, optional
        Determines whether the CVI is to be interpreted as being
        "absolute", "monotonous" or "original" (note that not all
        CVIs can have these 3 interpretations), by default
        "monotonous"
    criterion_function : callable, optional
        Determines how the best clustering should be selected
        according to their corresponding CVI values, by default None
    k_condition : callable, optional
        :math:`k` values that are compatible with this CVI, by
        default None
    ignore0 : bool, optional
        Determines how to treat the special case :math:`k=0` (when
        available) when selecting the best clustering. This is for
        example used in the Hartigan index where we don't use
        :math:`k=0` as a reference score if :math:`k=0` is more
        relevant than :math:`k=1`, by default False

    Raises
    ------
    ValueError
        Raised if the `cvi_type` given is not among the available
        options for this CVI.
    """

    cvi_types: List[str] = ["monotonous", "absolute"]
    reductions: List[str] = None

    def __init__(
        self,
        cvi_function: callable = None,
        maximise: bool = True,
        improve: bool = True,
        cvi_type: str = "monotonous",
        criterion_function: callable = None,
        k_condition: callable = None,
        ignore0: bool = False,
    ) -> None:
        self.function = cvi_function
        self.criterion_function = criterion_function
        self.maximise = maximise
        # If cvi_type == "absolute", 'improve' is irrelevant
        self.improve = improve
        if cvi_type in self.cvi_types:
            self.cvi_type = cvi_type
        else:
            raise ValueError(
                "cvi_type {} invalid for {}. Please choose among {}".format(cvi_type, str(self), self.cvi_types)
            )
        self.k_condition = k_condition
        self.ignore0 = ignore0
        self.N = None
        self.d = None

    def __call__(
        self,
        X: np.ndarray,
        clustering: List[List[int]],
        cvi_kwargs: dict = {},
    ) -> float:
        """
        Computes the CVI value of the clustering.

        Parameters
        ----------
        X : np.ndarray, shape: `(N, d*w_t)` or `(N, w_t, d)`
            Dataset/
        clustering : List[List[int]]
            List of clusters.
        cvi_kwargs : dict, optional
            kwargs specific for the CVI, by default {}

        Returns
        -------
        float
            The CVI value for this clustering.

        Raises
        ------
        InvalidKError
            If the CVI has been called with an incompatible number of
            clusters. Note that most CVIs don't accept the case
            :math:`k=0`
        """
        dims = X.shape
        self.N = dims[0]
        self.d = dims[-1]
        if "k" in cvi_kwargs:
            n_clusters = cvi_kwargs["k"]
        else:
            n_clusters = len(clustering)
        if self.k_condition(n_clusters):
            return self.function(X, clustering, **cvi_kwargs)
        else:
            msg = (
                f"{self.__class__} called with an incompatible number of "
                + f"clusters: {n_clusters}"
            )
            raise InvalidKError(msg)

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {},
    ) -> Union[dict, None]:
        """
        Get the kwargs parameters specific to the CVI.

        Base method to override when defining a CVI if the CVI function
        requires additional parameters than the standard `X` and
        `clusters` representing respectively the data values (already
        processed) and the partition representing the clustering.

        Parameters
        ----------
        X_clus : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`,
        optional
            Dataset to cluster (already processed), by default None
        clusterings_t : Dict[int, List], optional
            All the clusterings computed for the provided :math:`k`
            range. Having an overview of the clusterings can be needed
            in some CVI such as the Hartigan index. By default None.
        n_clusters : int, optional
            Current number of clusters considered, by default None
        cvi_kwargs : dict, optional
            Pre-defined kwargs, typically the metric to use when
            computing the CVI values, by default {}

        Returns
        -------
        Union[dict, None]
            The dictionary of kwargs necessary to compute the CVI.
        """
        return cvi_kwargs

    def criterion(
        self,
        scores: Dict[int, float],
        cvi_type: str = None,
    ) -> Union[int, None]:
        """
        The default selection method for regular cases.

        Regular cases included monotonous/absolute/pseudo-monotonous and
        maximise=True/False with improve=True/False.

        Does not take into account rules that are specific to a CVI,
        such as the Gap statistic, Hartigan, etc.

        Returns None if no clustering could be selected (probably
        because the CVI values were None or NaN).

        Parameters
        ----------
        scores : Dict[int, float]
            The CVI values obtained for the provided :math:`k` range.
        cvi_type : str, optional
            The type of CVI to use in the selection scheme. Note that
            for most cases it is redundant with the attribute
            `CVI.cvi_type` but it may facilitate the selection for
            CVIs that use base cases with small adjustments, by
            default None.

        Returns
        -------
        Union[int, None]
            The :math:`k` value corresponding to the selected
            clustering. Returns `None` if no clustering could be
            selected.
        """
        # Because some custom criterion_function relies on the general
        # case "monotonous"/"absolute"
        if cvi_type is None:
            cvi_type = self.cvi_type
        # For monotonous CVIs
        if cvi_type == "monotonous":
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
                # either because this CVI doesn't allow this k
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
        # For absolute CVIs
        elif cvi_type == "absolute":
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
        # For CVIs with a special selection process
        else:
            selected_k = self.criterion_function(scores)
        return selected_k

    def is_relevant(
        self,
        score: float,
        k: int,
        score_prev: float,
        k_prev: int,
    ) -> bool:
        """
        Determines if a score is relevant based on the CVI properties.

        This is particularly useful for pseudo-monotonous CVIs, to know
        whether we should ignore a specific CVI value.

        Parameters
        ----------
        score : float
            The current CVI value.
        k : int
            The current :math:`k` value.
        score_prev : float
            The previous CVI value.
        k_prev : int
            The previous :math:`k` value (which must be smaller).

        Returns
        -------
        bool
            True is the current value is relevant, which means that it
            is following the expected scheme of CVI values given the
            properties of the CVI (its `cvi_type`, `maximise`,
            `improve` properties).
        """
        # A score is always relevant when the CVI is absolute
        if self.cvi_type == "absolute":
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
        scores_t_k: List[Dict[int, float]],
        return_list: bool = False,
    ) -> Union[List[int], int]:
        """
        Select the best clusterings according to the CVI values given.

        Select the best :math:`k` for each :math:`t` according to the
        CVI values given. If the data is not time series or if the time
        series are clustered considering all time steps at once, then
        the returned list has only one element.

        If no k could be selected given the scores_t_k values, then k is
        set to None.

        Parameters
        ----------
        scores_t_k : List[Dict[int, float]]
            The CVI values for the provided :math:`k` range and for the
            potential number :math:`t` of iterations to consider in
            time.
        return_list: bool, optional
            Determines whether the output should be forced to be a
            List[Dict], even when no sliding window is used by default
            False.

        Returns
        -------
        Union[List[int], int]
            The list of :math:`k` values corresponding to the best
            clustering for each potential number :math:`t` of iterations
            to consider in time. Some elements can be `None` if no
            clustering could be selected at a given iteration :math:`t`.

        Raises
        ------
        ValueError
            If ```scores_t_k``` is empty or not of the right type (a
            list of dictionaries in the case of time series data
            clustered by sliding windows or a dictionary).
        SelectionError
            If no clustering could be selected with the given CVI values
            (probably because the CVI values are None or NaN)
        """
        try:
            scores_t_k, should_return_list = _check_list_of_dict(scores_t_k)
        except ValueError as e:
            msg = f"scores_t_k in CVI.select: {e}"
            raise ValueError(msg)
        # Time series case with sliding window
        return_list = return_list or should_return_list
        if return_list:
            k_selected = [self.criterion(s_t) for s_t in scores_t_k]
            if None in k_selected:
                msg = (
                    "No clustering could be selected by {self} "
                    + f"with the CVI values given: {scores_t_k}"
                )
                raise SelectionError(msg)
        # Other cases (non time-series or time-series without sliding window)
        else:
            k_selected = self.criterion(scores_t_k[0])
            if k_selected == None:
                msg = (
                    "No clustering could be selected by {self} "
                    + f"with the CVI values given: {scores_t_k}"
                )
                raise SelectionError(msg)
        return k_selected

    def better_score(
        self,
        score1: float,
        score2: float,
        or_equal: bool = False
    ) -> bool:
        """
        Checks if a CVI value is better than another.

        Takes into account the properties of the CVI (its `cvi_type`,
        `maximise`, `improve` properties)

        Parameters
        ----------
        score1 : float
            The first CVI value
        score2 : float
            The second CVI value
        or_equal : bool, optional
            Determines whether 2 equal scores should yield `True` or
            not, by default False

        Returns
        -------
        bool
            True if `score1` is better than `score2`.
        """
        return better_score(score1, score2, self.maximise, or_equal)

    def argbest(
        self,
        scores: List[float],
        ignore_None: bool = False,
    ) -> int:
        """
        Returns the index of the best score.

        Parameters
        ----------
        scores : List[float]
            A list of CVI values
        ignore_None : bool, optional
            If `True`, `None` values will be ignored, otherwise, a
            `None` value will be considered as the best score, by
            default False

        Returns
        -------
        int
            The index of the best score
        """
        return argbest(scores, self.maximise, ignore_None)

    def best_score(
        self,
        scores: List[float],
        ignore_None: bool = False,
    ) -> float:
        """
        Returns the best score.

        Parameters
        ----------
        scores : List[float]
            A list of CVI values
        ignore_None : bool, optional
            If `True`, `None` values will be ignored, otherwise, a
            `None` value will be considered as the best score, by
            default False

        Returns
        -------
        float
            The best score
        """
        return best_score(scores, self.maximise, ignore_None)

    def argworst(
        self,
        scores: List[float],
    ) -> int:
        """
        Returns the index of the worst score.

        Parameters
        ----------
        scores : List[float]
            A list of CVI values

        Returns
        -------
        int
            The index of the worst score
        """
        return argworst(scores, self.maximise)

    def worst_score(
        self,
        scores: List[float],
    ) -> float:
        """
        Returns the worst score.

        Parameters
        ----------
        scores : List[float]
            A list of CVI values

        Returns
        -------
        float
            The worst score
        """
        return worst_score(scores, self.maximise)

class Hartigan(CVI):
    """
    The Hartigan index. [Hartigan]_

    Originally, this index is absolute and the selection criteria is as
    follows:

    According to de Amorim and Hennig [2015] "In the original paper, the
    lowest :math:`k` to yield Hartigan :math:`<= 10` was proposed as the
    optimal choice. However, if no :math:`k` meets this criteria, choose
    the :math:`k` whose difference in Hartigan for :math:`k` and
    :math:`k+1` is the smallest". According to Tibshirani et al. [2001]
    it is "the estimated number of clusters is the smallest :math:`k`
    such that Hartigan :math:`<= 10` and :math:`k=1` could then be
    possible.

    A monotonous approach can also be taken.

    Possible `cvi_type` values: "monotonous" or "original".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "monotonous".
    """

    cvi_types: List[str] = ["monotonous", "original"]

    def __init__(
        self,
        cvi_type: str = "monotonous"
    ) -> None:
        f_k_condition = lambda k: (
            k>=0 if cvi_type == "monotonous" else k>=1
        )

        super().__init__(
            cvi_function=hartigan,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            criterion_function=self._f_criterion,
            k_condition= f_k_condition,
            ignore0 = True,
        )

    def _f_criterion(
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
            if len(ks) > 1:
                arg_selected_k = np.argmin(
                    [
                        max(scores[ks[i]]-scores[ks[i+1]], 0)
                        for i in range(len(ks)-1)
                    ]
                )
                selected_k = ks[arg_selected_k]
            else:
                selected_k = None
        return selected_k

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        """
        Get the kwargs parameters specific to Hartigan.

        Hartigan has 3 additional parameters:

        - `k` (int): the current number of clusters.
        - `clusters_next` (np.ndarray, shape: `(N, d*w_t)` or `(N, w_t,
          d))`: the clustering for the next :math:`k` value
          considered.
        - `X1` (np.ndarray, shape: (N, d*w_t) or (N, w_t, d)): the
          dataset to cluster (already processed). This is needed for
          the case :math:`k=0`, and in that case `X_clus` is sampled
          from a uniform distribution with similar parameters as the
          original distribution.

        Parameters
        ----------
        X_clus : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`,
        optional
            Dataset to cluster (already processed), by default None
        clusterings_t : Dict[int, List], optional
            All the clusterings computed for the provided :math:`k`
            range. Having an overview of the clusterings can be needed
            in some CVI such as the Hartigan index. By default None.
        n_clusters : int, optional
            Current number of clusters considered, by default None
        cvi_kwargs : dict, optional
            Pre-defined kwargs, typically the metric to use when
            computing the CVI values, by default {}

        Returns
        -------
        Union[dict, None]
            The dictionary of kwargs necessary to compute the CVI.
        """
        cvi_kw = {}
        cvi_kw["k"] = n_clusters
        if n_clusters < len(X_clus):
            cvi_kw["clusters_next"] = clusterings_t.get(n_clusters+1, None)
        if n_clusters == 0:
            cvi_kw["X1"] = X_clus
        cvi_kw.update(cvi_kwargs)
        return cvi_kw

    def __str__(self) -> str:
        return 'Hartigan_{}'.format(self.cvi_type)

class CalinskiHarabasz(CVI):
    """
    The Calinski-Harabasz index. [CH]_

    Originally, this index is absolute and has to be maximised to find
    the best :math:`k`. A monotonous approach can also be taken, so that
    the case k=1 can be selected, with CH(1) = 0 and CH(0) extended (see
    `pycvi.cvi.CH`)

    Possible `cvi_type` values: "monotonous", "absolute" or "original".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "original".
    """

    cvi_types: List[str] = ["monotonous", "original", "absolute"]

    def __init__(
        self,
        cvi_type: str = "original"
    ) -> None:
        # Note that the case k=1 for the absolute version will always
        # give CH=0
        f_k_condition = lambda k: (
            k>=0 if cvi_type == "monotonous" else k>=1
        )

        # original and absolute are the same
        if cvi_type == "original":
            cvi_type = "absolute"

        super().__init__(
            cvi_function=CH,
            maximise=True,
            improve=True,
            cvi_type=cvi_type,
            k_condition = f_k_condition,
        )

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        """
        Get the kwargs parameters specific to Calinski-Harabasz.

        Calinski-Harabasz has 3 additional parameters:

        - `k` (int): the current number of clusters.
        - `X1` (np.ndarray, shape: `(N, d*w_t)` or `(N, w_t, d)`): the
          dataset to cluster (already processed). This is
          needed for the case :math:`k=0`, and in that case `X_clus`
          is sampled from a uniform distribution with similar
          parameters as the original distribution
        - `zero_type` (str): determines how to parametrize the uniform
          distribution to sample from in the case :math:`k=0`. Possible
          options:

          - `"variance"`: the uniform distribution is defined such that
            it has the same variance and mean as the original data.
          - `"bounds"`: the uniform distribution is defined such that
            it has the same bounds as the original data.

        Parameters
        ----------
        X_clus : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`,
        optional
            Dataset to cluster (already processed), by default None
        clusterings_t : Dict[int, List], optional
            All the clusterings computed for the provided :math:`k`
            range. Having an overview of the clusterings can be needed
            in some CVI such as the Hartigan index. By default None.
        n_clusters : int, optional
            Current number of clusters considered, by default None
        cvi_kwargs : dict, optional
            Pre-defined kwargs, typically the metric to use when
            computing the CVI values, by default {}

        Returns
        -------
        Union[dict, None]
            The dictionary of kwargs necessary to compute the CVI.
        """
        cvi_kw = {}
        cvi_kw["k"] = n_clusters
        if n_clusters == 0:
            cvi_kw["X1"] = X_clus
        if "zero_type" not in cvi_kw or cvi_kw["zero_type"] == None:
            if len(X_clus) > 100:
                cvi_kw["zero_type"] = "variance"
            else:
                cvi_kw["zero_type"] = "bounds"
        cvi_kw.update(cvi_kwargs)
        return cvi_kw

    def __str__(self) -> str:
        cvi_type = self.cvi_type
        # original and absolute are the same
        if cvi_type == "absolute":
            cvi_type = "original"
        return 'CalinskiHarabasz_{}'.format(cvi_type)

class GapStatistic(CVI):
    """
    The Gap statistic. [Gap]_

    Originally, this index is absolute and the selection criteria is as
    follow:

    Take the smallest :math:`k` such that :math:`Gap(k) \\geq Gap(k+1) -
    s(k+1)`.

    A monotonous approach can also be taken.

    Possible `cvi_type` values: "monotonous", or "original".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "original".
    """

    cvi_types: List[str] = ["monotonous", "original"]

    def __init__(
        self,
        cvi_type: str = "original",
    ) -> None:

        f_k_condition = lambda k: (
            k>=0 if cvi_type == "monotonous" else k>=1
        )

        super().__init__(
            cvi_function= gap_statistic,
            maximise=True,
            improve=True,
            cvi_type=cvi_type,
            k_condition=f_k_condition,
            criterion_function=self._f_criterion,
        )

        self.s = {}

    def _f_criterion(
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

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        """
        Get the kwargs parameters specific to Gap statistic.

        Gap statistic has 4 additional parameters:

        - `k` (int): the current number of clusters.
        - `B` (int): the number of uniform samples drawn.
        - `zero_type` (str): determines how to parametrize the uniform
          distribution to sample from in the case :math:`k=0`. Possible
          options:

          - `"variance"`: the uniform distribution is defined such that
            it has the same variance and mean as the original data.
          - `"bounds"`: the uniform distribution is defined such that
            it has the same bounds as the original data.

        - `return_s` (bool): determines whether the s value should also
          be returned

        Parameters
        ----------
        X_clus : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`,
        optional
            Dataset to cluster (already processed), by default None
        clusterings_t : Dict[int, List], optional
            All the clusterings computed for the provided :math:`k`
            range. Having an overview of the clusterings can be needed
            in some CVI such as the Hartigan index. By default None.
        n_clusters : int, optional
            Current number of clusters considered, by default None
        cvi_kwargs dict, optional
            Pre-defined kwargs, typically the metric to use when
            computing the CVI values, by default {}

        Returns
        -------
        Union[dict, None]
            The dictionary of kwargs necessary to compute the CVI.
        """
        cvi_kw = {"B" : 10}
        cvi_kw["k"] = n_clusters
        if self.cvi_type == "original":
            cvi_kw["return_s"] = True
        cvi_kw.update(cvi_kwargs)
        if "zero_type" not in cvi_kw or cvi_kw["zero_type"] == None:
            if len(X_clus) > 100:
                cvi_kw["zero_type"] = "variance"
            else:
                cvi_kw["zero_type"] = "bounds"
        return cvi_kw

    def __call__(
        self,
        X: np.ndarray,
        clustering: List[List[int]],
        cvi_kwargs: dict = {}
    ) -> float:
        """
        Computes the CVI value of the clustering.

        If `cvi_type=original`, then an attribute `s[k]` is computed,
        corresponding to the std around the CVI value (with `k` begin
        the number of clusters).

        Parameters
        ----------
        X : np.ndarray, shape: `(N, d*w_t)` or `(N, w_t, d)`
            Dataset/
        clustering : List[List[int]]
            List of clusters.
        cvi_kwargs : dict, optional
            kwargs specific for the CVI, by default {}

        Returns
        -------
        float
            The CVI value for this clustering.

        Raises
        ------
        InvalidKError
            If the CVI has been called with an incompatible number of
            clusters. Note that most CVIs don't accept the case
            :math:`k=0`
        """
        if self.cvi_type == "original":
            gap, s =  super().__call__(X, clustering, cvi_kwargs)
            k = cvi_kwargs["k"]
            self.s[k] = s
            return gap
        else:
            return super().__call__(X, clustering, cvi_kwargs)

    def __str__(self) -> str:
        return 'GapStatistic_{}'.format(self.cvi_type)

class Silhouette(CVI):
    """
    The Silhouette score. [Silhouette]_

    This index is absolute, bounded in :math:`[-1, 1]` range and has to
    be maximised.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(
        self,
        cvi_type: str = "absolute",
    ) -> None:
        super().__init__(
            cvi_function=silhouette,
            maximise=True,
            improve=None,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'Silhouette'

class ScoreFunction(CVI):
    """
    The Score function. [SF]_

    This index has to be maximised to find the best clustering, but the
    original paper Saitta et al. [2007] adds special cases:

    - if the score always increases, then the number :math:`k = 1` is
      chosen.
    - if a maximum is found, outside the extreme :math:`k` values, then
      the argument of this maximum is chosen.
    - it is empirically decided that if :math:`(SF_2 − SF_1) \\times d
      \\leq 0.2` then, :math:`k = 1` is also chosen (:math:`d` being the
      dimensionality of the data).

    The absolute case can also be chosen (i.e. special case are
    ignored).

    Possible `cvi_type` values: "absolute", or "original".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "original".
    """

    cvi_types: List[str] = ["absolute", "original"]

    def __init__(self, cvi_type: str = "original") -> None:
        super().__init__(
            cvi_function=score_function,
            maximise=True,
            improve=None,
            cvi_type=cvi_type,
            criterion_function=self._f_criterion,
            k_condition= lambda k: (k>=1)
        )

    def _f_criterion(
        self,
        scores: Dict[int, float],
    ) -> int:
        # General case first
        best_k = self.criterion(scores, cvi_type="absolute")
        # If score always increases, then choose k=1
        if best_k == max(scores):
            best_k = 1
        if (
            (1 in scores) and (2 in scores)
            and ((scores[2] - scores[1]) * self.d <= 0.2)
        ):
            best_k=1
        return best_k

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        """
        Get the kwargs parameters specific to Score Function.

        Score Function has no additional parameters, but :math:`k` is used to distinguish between the case :math:`k=0` and :math:`k=1`, to make sure that the case :math:`k=0` is never computed.

        Parameters
        ----------
        X_clus : np.ndarray, shape `(N, d*w_t)` or `(N, w_t, d)`,
        optional
            Dataset to cluster (already processed), by default None
        clusterings_t : Dict[int, List], optional
            All the clusterings computed for the provided :math:`k`
            range. Having an overview of the clusterings can be needed
            in some CVI such as the Hartigan index. By default None.
        n_clusters : int, optional
            Current number of clusters considered, by default None
        cvi_kwargs : dict, optional
            Pre-defined kwargs, typically the metric to use when
            computing the CVI values, by default {}

        Returns
        -------
        Union[dict, None]
            The dictionary of kwargs necessary to compute the CVI.
        """
        cvi_kw = {}
        cvi_kw["k"] = n_clusters
        cvi_kw.update(cvi_kwargs)
        return cvi_kw

    def __str__(self) -> str:
        return 'ScoreFunction'

class MaulikBandyopadhyay(CVI):
    """
    The Maulik-Bandyopadhyay index. [MB]_

    Originally, this index is absolute and has to be maximised to find
    the best :math:`k`.

    A monotonous approach can also be taken.

    Possible `cvi_type` values: "monotonous", or "absolute".

    Note that the case :math:`k=1` always returns 0.

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute", "monotonous"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        super().__init__(
            cvi_function=MB,
            maximise=True,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=1)
        )

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        cvi_kw = {"p" : 2}
        cvi_kw["k"] = n_clusters
        cvi_kw.update(cvi_kwargs)
        return cvi_kw

    def __str__(self) -> str:
        return f'MaulikBandyopadhyay_{self.cvi_type}'

class SD(CVI):
    """
    The SD index. [SD]_

    This index is absolute and has to be minimised to find the best
    :math:`k`.

    Note that if two clusters have equal centroids, then `SD = inf`
    which means that this clustering is irrelevant, which works as
    intended (even though two clusters could be well separated and still
    have equal centroids, as in the case of two concentric circles).

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        super().__init__(
            cvi_function=SD_index,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def get_cvi_kwargs(
        self,
        X_clus: np.ndarray = None,
        clusterings_t: Dict[int, List] = None,
        n_clusters: int = None,
        cvi_kwargs: dict = {}
    ) -> None:
        cvi_kw = {"alpha" : None}
        cvi_kw.update(cvi_kwargs)
        return cvi_kw

    def __str__(self) -> str:
        return 'SD'

class SDbw(CVI):
    """
    The SDbw index. [SDbw]_

    This index is absolute and has to be minimised to find the best
    :math:`k`.

    Note that if two clusters have all datapoints further away to
    their respective centroids than what is called in the original
    paper "the average standard deviation of clusters", then `SDbw =
    inf`, which means that this clustering is irrelevant, which
    works as intended.

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        super().__init__(
            cvi_function=SDbw_index,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'SDbw'

class Dunn(CVI):
    """
    The Dunn index. [Dunn]_

    This index is absolute and has to be maximised to find the best
    :math:`k`.

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        super().__init__(
            cvi_function=dunn,
            maximise=True,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'Dunn'

class XB(CVI):
    """
    The Xie-Beni index. [XB]_

    This index is absolute and has to be minimised to find the best
    :math:`k`.

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            cvi_function=xie_beni,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'XB'

class XBStar(CVI):
    """
    The Xie-Beni* index. [XB*]_

    This index is absolute and has to be minimised to find the best
    :math:`k`.

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        super().__init__(
            cvi_function=xie_beni_star,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'XB_star'

class DB(CVI):
    """
    The Davies-Bouldin index. [DB]_

    This index is absolute and has to be minimised to find the best
    :math:`k`.

    The case :math:`k=1` is not possible.

    Possible `cvi_type` values: "absolute".

    Parameters
    ----------
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "absolute".
    """

    cvi_types: List[str] = ["absolute"]

    def __init__(self, cvi_type: str = "absolute") -> None:
        """
        The case k=1 is not possible.
        """
        super().__init__(
            cvi_function=davies_bouldin,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition= lambda k: (k>=2)
        )

    def __str__(self) -> str:
        return 'DB'

class Inertia(CVI):
    """
    The inertia of a clustering.

    This index is monotonous and and smaller values are considered
    better.

    Possible `cvi_type` values: "monotonous".

    Parameters
    ----------
    reduction : str, optional
        Determines how to combine the inertia values of each cluster to
        compute the inertia of the whole clustering, by default "sum".
        Available options: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce` for more information.
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "monotonous".
    """

    cvi_types: List[str] = ["monotonous"]
    reductions: List[str] = [
        "sum", "mean", "max", "median", "min", "", None
    ]

    def __init__(
        self,
        reduction: Union[str, callable] = "sum",
        cvi_type: str = "monotonous",
    ) -> None:

        f_k_condition = lambda k: k>=0

        def cvi_function(X, clusters):
            return reduce(_compute_score(
                "list_inertia", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            cvi_function=cvi_function,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition = f_k_condition,
        )

        self.reduction = reduction

    def __str__(self) -> str:
        if hasattr(self, "reduction"):
            return 'Inertia_{}'.format(self.reduction)
        else:
            return 'Inertia'

class Diameter(CVI):
    """
    The Diameter of a clustering.

    This index is monotonous and and smaller values are considered
    better.

    Possible `cvi_type` values: "monotonous".

    Parameters
    ----------
    reduction : str, optional
        Determines how to combine the diameter values of each cluster to
        compute the diameter of the whole clustering, by default "sum".
        Available options: `"sum"`, `"mean"`, `"max"`, `"median"`,
        `"min"`, `""`, `None` or a callable. See
        `pycvi.compute_scores.reduce` for more information.
    cvi_type : str, optional
        Determines how the index should be interpreted, when selecting
        the best clustering, by default "monotonous".
    """

    cvi_types: List[str] = ["monotonous"]
    reductions: List[str] = [
        "sum", "mean", "max", "median", "min", "", None
    ]

    def __init__(
        self,
        reduction: Union[str, callable] = "max",
        cvi_type: str = "monotonous",
    ) -> None:

        f_k_condition = lambda k: k>=0

        def cvi_function(X, clusters):
            return reduce(_compute_score(
                "list_diameter", X, clusters, dist_kwargs={}, score_kwargs={}
                ), reduction)

        super().__init__(
            cvi_function=cvi_function,
            maximise=False,
            improve=True,
            cvi_type=cvi_type,
            k_condition = f_k_condition,
        )

        self.reduction = reduction

    def __str__(self) -> str:
        return 'Diameter_{}'.format(self.reduction)

CVIs = [
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
"""
List of available CVI indices in PyCVI
"""