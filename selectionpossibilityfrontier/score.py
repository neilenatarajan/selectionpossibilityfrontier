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

def make_mean_scoring(q, n, alpha=0.5):
    '''Returns The Mean Score
    I.e. Mean value of all scores of selected candidates

    Parameters
    ----------
    q : np.array
        1D Array of dim n. Contains target numbers by attribute.

    n : int in (0, len(q)]
        The number of attributes to consider
        
    alpha : optional, float in (0, 1]
        Scaling exponent for the objective function.

    Returns
    -------
    divfunc : np.array -> float
        Function for calculating presence-based diversity

    dfmax : float
        Maximum value of the diversity function
    '''

    assert n > 0 and n <= len(q)

    return (lambda p: __presence_targets(p, q, n, alpha), __presence_targets(q, q, n, alpha))

def __proportional_targets(p, q, w, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    return ((np.minimum(q, p))**(alpha)).dot(w)