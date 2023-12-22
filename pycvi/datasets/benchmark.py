"""
Few samples from benchmarking datasets.

Datasets aggregated by Thomas Barton, on his GitHub repository [Bart]_:

- "target"
- "zelnik1"
- "long1"

Datasets from the UCR Time Series Classification Archive [UCR]_:

- "Trace"
- "SmallKitchenAppliances"

.. [UCR] H. A. Dau, E. Keogh, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S.
  Gharghabi, C. A. Ratanamahatana, Yanping, B. Hu, N. Begum, A. Bagnall,
  A. Mueen, G. Batista, and Hexagon-ML, “The ucr time series
  classification archive,” October 2018.
  https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

.. [Bart]  T. Barton, “Clustering benchmarks.”
  ”https://github.com/deric/clusteringbenchmark”, 2015. [Online;
  accessed 06-December-2023].
"""

import numpy as np
import csv
from importlib import resources
from typing import Tuple

def load_data(
    fname: str = "target",
    data_source: str ="barton",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset and labels.

    Parameters
    ----------
    fname : str
        Filename of the dataset, by default "target".
    path : str, optional
        Path to the file, by default "./Barton/".
    bool : bool, optional
        Verbosity.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The dataset and labels
    """
    # Load data from arff to dataframe
    if data_source.lower() == "barton":
        subfolder = "Barton"
    else:
        subfolder = "UCR"
    f_data = fname + "_data.csv"
    with resources.open_text(f"pycvi.datasets.{subfolder}", f_data) as f:
        reader = csv.reader(f)
        data = np.array(list(reader), dtype=float)
        # reshape to (N, T, 1) if UCR
        if subfolder == "UCR":
            data = np.expand_dims(data, -1)
    f_labels = fname + "_labels.csv"
    with resources.open_text(f"pycvi.datasets.{subfolder}", f_labels) as f:
        reader = csv.reader(f)
        labels = np.array(list(reader)).squeeze()
        classes = np.unique(labels)
        map_classes = {c:i for i,c in enumerate(classes)}
        labels = np.array([map_classes[label] for label in labels], dtype=int)
    if verbose:
        print((
            f"Source: {subfolder} | Dataset: {fname} | Shape: {data.shape}"
            + f" | Labels: {np.unique(labels)}"
        ))

    return data, labels