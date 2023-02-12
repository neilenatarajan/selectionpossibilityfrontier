import numpy as np
import pandas as pd

def make_combined_scores(*ss, scale=False):
    '''Returns Combined Scoring Function

    Parameters
    ----------
    ss : list of (np.array -> float, float)
        List containing tuples of target functions and maximums

    scale : optional, boolean
        If true, indicates that functions should be scaled on [0, 1] before combining

    Returns
    -------
    divfunc : np.array -> float
        Function for calculating combined diversity

    dfmax : float
        Maximum value of the diversity function
    '''

    # Scale target functions if desired
    if scale:
        return (lambda p: sum([sf(p)/smax for (sf, smax) in ss]), sum([smax/smax for (_, smax) in ss]))

    else:
        return (lambda p: sum([sf(p) for (sf, _) in ss]), sum([smax for (_, smax) in ss]))

def make_mean_scores(k, q, alpha=1):
    '''Returns The Mean Score
    I.e. Mean value of all scores of selected candidates

    Parameters
    ----------
    k : int in (0, len(df)]
        The number of participants to select
    
    q : float
        The max score value.

    alpha : optional, float in (0, 1]
        Scaling exponent for the objective function.

    Returns
    -------
    scorefunc : np.array -> float
        Function for calculating mean scores

    sfmax : float
        Maximum value of the score function
    '''

    return (lambda s: __mean_score(s, alpha), __mean_score(np.repeat(q, k), alpha))

def __mean_score(s, alpha=1):
    '''Compute the mean scores of a cohort.'''
    return np.sum(s**(alpha), axis=-1)
