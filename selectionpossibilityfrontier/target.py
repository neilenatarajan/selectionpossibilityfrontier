import numpy as np
import pandas as pd

def make_combined_targets(*ts, scale=False):
    '''Returns Combined Diversity Function

    Parameters
    ----------
    ts : list of (np.array -> float, float)
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
        return (lambda p: sum([tf(p)/tmax for (tf, tmax) in ts]), sum([tmax/tmax for (_, tmax) in ts]))

    else:
        return (lambda p: sum([tf(p) for (tf, _) in ts]), sum([tmax for (_, tmax) in ts]))

def make_proportional_targets(k, q, w=None, alpha=0.5):
    '''Returns Proportional Diversity Function

    Parameters
    ----------
    k : int in (0, len(df)]
        The number of participants to select
    
    q : np.array
        1D Array of dim n. Contains target proportions by attribute.

    w : optional, dict {column: float}
        Weighting over df columns
        By default, a uniform weighting is used
        
    alpha : optional, float in (0, 1]
        Scaling exponent for the objective function.

    Returns
    -------
    divfunc : np.array -> float
        Function for calculating proportional diversity

    dfmax : float
        Maximum value of the diversity function
    '''
    if w is None:
        w = np.ones(X.shape[1])

    assert not np.any(w < 0)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)

    # Convert fractions to sums
    q = np.round(k * q)

    return (lambda p: __proportional_targets(p, q, w, alpha), __proportional_targets(q, q, w, alpha))

def make_presence_targets(q, n, alpha=0.5):
    '''Returns Presence Diversity Function

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

def __presence_targets(p, q, n, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    # First ensure that nothing higher than q is counted
    p = np.sort(np.minimum(q, p))
    p[..., :-n] = 0 
    
    return (np.sum(p, axis=-1)**(alpha))
