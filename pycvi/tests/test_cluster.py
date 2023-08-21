import numpy as np
from numpy.testing import assert_array_equal

from ..cluster import sliding_window

def test_sliding_window():
    # From examples drawn by hand
    windows = list(range(1,7))
    T = 7
    p_r =     [0, 1, 1, 2, 2, 3]
    p_l =     [0, 0, 1, 1, 2, 2]
    lengths = [
        [1, 1, 1, 1, 1, 1, 1], # 1
        [2, 2, 2, 2, 2, 2, 1], # 2
        [2, 3, 3, 3, 3, 3, 2], # 3
        [3, 4, 4, 4, 4, 3, 2], # 4
        [3, 4, 5, 5, 5, 4, 3], # 5
        [4, 5, 6, 6, 5, 4, 3], # 6
    ]
    midpoints = [
        [0, 0, 0, 0, 0, 0, 0], # 1
        [0, 0, 0, 0, 0, 0, 0], # 2
        [0, 1, 1, 1, 1, 1, 1], # 3
        [0, 1, 1, 1, 1, 1, 1], # 4
        [0, 1, 2, 2, 2, 2, 2], # 5
        [0, 1, 2, 2, 2, 2, 2], # 6
    ]
    origins = [
        [[0], [1], [2], [3], [4], [5], [6]], # 1
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6]], # 2
        [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6]], # 3
        [[0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6], [5, 6]], # 4
        [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]], # 5
        [[0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]], # 6
    ]
    for i, w in enumerate(windows):
        window = sliding_window(T, w)

        output = [window["padding_left"], window["padding_right"]]
        output_exp = [p_l[i], p_r[i]]

        output_np = [
            window["length"], window["midpoint_w"],
        ]
        output_np += window["origin"]
        output_np_exp = [lengths[i], midpoints[i]]
        output_np_exp += origins[i]

        for out, out_exp in zip(output, output_exp):
            assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
        for out, out_exp in zip(output_np, output_np_exp):
            assert_array_equal(out, out_exp)

