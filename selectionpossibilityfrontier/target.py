import numpy as np
import pandas as pd
import warnings

def apply_target(tf, C, scale=False, tfmax=None):
    '''Applies a target function to a given cohort

    Parameters
    ----------
    tf : np.array -> float
        Diversity target function

    C : np.array
        2D Array of dims n by m. Rows are participants, columns are (binarised) attributes.

    scale : optional, boolean
        If true, indicates that function should be scaled on [0, 1] before returning

    tfmax : optional, float
        Maximum value of tf

    Returns
    -------
    diversity : float
        Diversity score of C according to tf
    '''

    assert not(scale and (tfmax is None))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p = np.nansum(C, axis=0)

    p[np.isnan(p)] = 0.0

    if scale:
        return tf(p)/tfmax
    else:
        return tf(p)

def make_combined_targets(ts, tmaxes, scale=False, weights=None):
    '''Returns Combined Diversity Function

    Parameters
    ----------
    ts : list of (np.array, float) -> float
        List containing target functions 

    tmaxes : list of float
        List containing maximums

    scale : optional, boolean
        If true, indicates that functions should be scaled on [0, 1] before combining

    weights : optional, np.array
        If scale, uses these weights when combining targets

    Returns
    -------
    divfunc : np.array -> float
        Function for calculating combined diversity

    dfmax : float
        Maximum value of the diversity function
    '''

    # Scale target functions if desired
    if scale:
        return (lambda p: sum([tf(p)*w/tmax for (tf, tmax, w) in zip(ts, tmaxes, weights)]), sum([w for w in weights]))

    else:
        return (lambda p: sum([tf(p) for tf in ts]), sum([tmax for tmax in tmaxes]))

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

    return (lambda p, s: __proportional_targets(p, q, w, alpha), __proportional_targets(q, q, w, alpha))

def make_mean_targets(k, q, w=None, alpha=0.5):
    '''Returns Proportional Diversity Function

    Parameters
    ----------
    k : int in (0, len(df)]
        The number of participants to select
    
    m : float
        The target score mean by attribute.

    w : optional, float
        Weighting
        
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

    # Convert means to sums
    m = np.round(k * m)

    return (lambda p, s: __mean_targets(s, m, w, alpha), __mean_targets(m, m, w, alpha))

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

    return (lambda p, s: __presence_targets(p, q, n, alpha), __presence_targets(q, q, n, alpha))

def __mean_targets(s, m, w, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    return (np.minimum(s, m)**(alpha))*w

def __proportional_targets(p, q, w, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    return ((np.minimum(q, p))**(alpha)).dot(w)

def __presence_targets(p, q, n, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    # First ensure that nothing higher than q is counted
    p = np.sort(np.minimum(q, p))
    p[..., :-n] = 0 
    
    # Then sum the remaining values
    return (np.sum(p, axis=-1)**(alpha))
