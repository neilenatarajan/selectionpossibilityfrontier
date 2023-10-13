import numpy as np
import pandas as pd

import warnings

from .utils import check_random_state

def return_frontier(X, s, k, divfunc, dfmax, res=20, ext=True, pre_selects=None, seed=None, scale=True, verbose=False):
    '''Return Cohorts on a Frontier

    Parameters
    ----------
    X : np.array
        2D Array of dims n by m. Rows are participants, and columns are (binarised) attributes.
        
    s : np.array
        1D Array of dim m. Contains performance scores by participant.

    k : int in (0, len(df)]
        The number of participants to select
    
    q : np.array
        1D Array of dim n. Contains target proportions by attribute.

    w : optional, dict {column: float}
        Weighting over df columns
        By default, a uniform weighting is used
        
    res : optional, int
        Resolution of Frontier curve
        
    ext : optional, Bool
        Determines whether to include the extremes of the curve

    pre_selects : None or iterable
        Optionally, you may pre-specify a set of rows to be forced into the
        solution.
        Values must be valid indices for df.
        
    seed : optional, int or numpy.random.RandomState
        An optional seed or random number state.
    
    scale : optional, bool
        If True, returns diversity and performance scores as fraction of theoretical maximum.

    verbose : optional, bool
        If True, print progress to stdout.
    
    Returns
    -------
    idx : pd.Index, length=(k,)
        Indices of the selected rows

    score : float
        The score of the solution found.  Larger is better.
    '''


    ds = []
    qs = []
    cs = []

    if verbose:
            print('Beginning Frontier Plotting')

    for i in range(res):
        if ext:  
            sratio = i/(res-1)
        else:
            sratio = (i+1)/(res+1)

        div, perf, c = optimise_cohort(
            X, 
            k, 
            check_random_state(seed),
            df = divfunc,
            dfmax = dfmax,
            quantile=0,
            s=s,
            sratio=sratio,
            pre_selects=pre_selects,
            scale=scale
        )
        
        ds.append(div)
        qs.append(perf)
        cs.append(c)

        if verbose:
            print('Completed step {} of {} | sratio: {:.2f} | diversity score: {:.2f} | performance score: {:.2f}'.format(i+1, res, sratio, div, perf))
        
        
    return (ds, qs, cs)

# The following is borrowed from entrofy
def optimise_cohort(X, k, rng, w=None, df=None, dfmax=None, pre_selects=None, quantile=0.0, s=None, sratio=0, scale=True):
    '''Finds an optimal cohort with given s and q

    Parameters
    ----------
    X : np.array
        2D Array of dims n by m. Rows are participants, and columns are (binarised) attributes.
        
    k : int in (0, len(df)]
        The number of participants to select

    rng : np.random.RandomState
        The random state.

    w : optional, dict {column: float}
        Weighting over df columns
        By default, a uniform weighting is used

    df : function (vectorised)
        Vectorised function on 1d np arrays. Returns diversity values.

    dfmax : optional, float
        Maximum value of df.

    pre_selects : None or iterable
        Optionally, you may pre-specify a set of rows to be forced into the
        solution.
        Values must be valid indices for df.

    quantile : optional, float in [0,1]
        Define the quantile to be used in tie-breaking between top choices at every step.

    s : np.array
        1D Array of dim m. Contains scores by participant.

    sratio : float in [0, 1]
        Weighting ratio of score : diversity  

    Returns
    -------

    div : float
        The diversity of the solution found.  Larger is better.

    perf : float
        The mean performance score of the solution found.  Larger is better.

    idx : pd.Index, length=(k,)
        Indices of the selected rows

    '''

    n_participants, n_attributes = X.shape
    X = np.array(X, dtype=float)

    assert 0 < k <= n_participants
    assert (s is not None) or (sratio == 0)
    assert sratio >= 0 and sratio <= 1

    if k == n_participants:
        return np.arange(n_participants)

    if s is not None:
        qmax = dfmax
        s[np.isnan(s)] = s.min()
        s = (s + s.min()) / (s.min() + s.max())
        
    
    # Initialization
    y = np.zeros(n_participants, dtype=bool)

    if pre_selects is not None:
        y[pre_selects] = True

    # Where do we have missing data?
    Xn = np.isnan(X)

    while True:
        i = y.sum()
        if i >= k:
            break

        # Initialize the distribution vector
        # We suppress empty-slice warnings here:
        #   even if y is non-empty, some column of X[y] may be all nans
        #   in this case, the index set (y and not-nan) becomes empty.
        # It's easier to just ignore this warning here and recover below
        # than to prevent it by slicing out each column independently.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p = np.nansum(X[y], axis=0)

        p[np.isnan(p)] = 0.0

        # Compute the candidate distributions
        p_new = p + X

        # Wherever X is nan, propagate the old p since we have no new information
        p_new[Xn] = (Xn * p)[Xn]

        # Compute marginal gain for each candidate
        # Simple if not calculating performance
        
        if sratio == 0:
            delta = df(p_new) - df(p)
           
        # If calculating performance, scale both scores and sum
        else:
            delta_div_scaled = (df(p_new) - df(p)) / qmax
            delta_perf_scaled = s / k
            delta = delta_div_scaled*(1-sratio) + delta_perf_scaled*sratio
            
        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score.  
        # If quantile is enabled, break near-ties randomly.
        delta_real = delta[np.isfinite(delta)]
        target_score = np.percentile(delta_real, 100 * (1.0-quantile))

        new_idx = rng.choice(np.flatnonzero(delta >= target_score))
        y[new_idx] = True
    
    if scale:
        dval = (df(np.nansum(X[y], axis=0)) / qmax)
    else:
        dval = df(np.nansum(X[y], axis=0))

    return (dval, (sum(s[y]) / k), np.flatnonzero(y))
