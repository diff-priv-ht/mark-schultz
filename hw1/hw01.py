import functools
import scipy.stats as sp
import numpy as np
import tqdm

MAX_NUM = 500

@functools.lru_cache(MAX_NUM + 1)
def sum_binom_pmf(a, p):
    """
    Returns a function that computes Pr[Z = k] for Z:

    Z ~ Binom(a, p) + Binom(500 - a, 1 - p) 

    by using the convolution formula.
    """
    X1 = lambda k: sp.binom.pmf(k, a, p)
    X2 = lambda k: sp.binom.pmf(k, MAX_NUM - a, 1 - p)
    pmf_as_list = np.convolve([X1(k) for k in range(a+1)], [X2(k) for k in range(MAX_NUM - a + 1)])
    return lambda k: pmf_as_list[k]

"""
Differential Privacy Definition:

M is (e, 0)-differentially private if:

1. For all S <= Range(M)
2. For all databases that are neighboring

it holds that:

Pr[M(x) \in S] <= exp(e) Pr[M(y) \in S]
"""

"""
Strategy for solving:
We want:

\min_{a\in [0,500]}\min_{s \in [0,500]} \frac{\Pr[M(x) = s]}{\Pr[M(y) = s]}

this will give us exp(e), which we can take the logarithm of to find e.

We can also interchange the databases and expect this to hold, so we'll need to look at all of the reciprocals as well.
"""

def find_differential_privacy_given_p(p):
    ls_of_expEs = []
    for a in tqdm.trange(MAX_NUM):  # Y_pmf goes to a + 1, so this range only being [0, MAX_NUM) isn't an issue.
        X_pmf = sum_binom_pmf(a, p)
        Y_pmf = sum_binom_pmf(a + 1, p)
        for s in tqdm.trange(MAX_NUM + 1):
            ls_of_expEs.append(X_pmf(s) / Y_pmf(s))
            ls_of_expEs.append(Y_pmf(s) / X_pmf(s))
    return np.log(max(ls_of_expEs))