import pytest

from .._compare_scores import (
    better_score, argbest, best_score, argworst, worst_score,
)

def test_comparisons():
    maximize = False
    score1 = [
        -3,    -3,  -3,    0,     2,    -2,    5,    8,    -3,     3,
        None,  None]
    score2 = [
        -4,    -1,   0,    0,    -2,     2,    8,    5,     None,  None,
        2,    -2]
    better_max = [
        False, True, True, False, False, True, True, False, False, False,
        True, True]
    better_min = [
        True, False, False, False, True, False, False, True, False, False,
        True, True]
    for i, (s1, s2) in enumerate(zip(score1, score2)):
        msg = "Better score was wrong, i: {} | s1: {} | s2 {}".format(i, s1, s2)
        out = better_score(s1, s2, maximize)
        assert out == better_max[i], msg
        out = better_score(s1, s2, not maximize)
        assert out == better_min[i], msg

        if better_max[i]:
            out_exp_arg = 0
            out_exp_score = s1
        else:
            out_exp_arg = 1
            out_exp_score = s2
        out = argbest([s1, s2], maximize)
        msg = "argbest was wrong,      i: {} | s1: {} | s2 {}".format(i, s1, s2)
        assert (out == out_exp_arg or s1 == s2), msg


        out = best_score([s1, s2], maximize)
        msg = "best_score was wrong,      i: {} | s1: {} | s2 {}".format(i, s1, s2)
        assert (out == out_exp_score), msg