import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numbers
import warnings



# The following is borrowed from scikit-learn v0.17
def check_random_state(seed):
    '''Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


# The following is borrowed from entrofy
def __optimise_cohort(X, k, rng, w=None, df=None, dfmax=None, pre_selects=None, quantile=0.0, s=None, sratio=0):
    '''Finds an optimal cohort with given s and q

    Parameters
    ----------
    X : np.array
        2D Array of dims n by m. Rows are participants, and columns are (binarised) attributes.
        
    k : int in (0, len(df)]
        The number of participants to select

    rng : np.random.RandomState
        the random state

    w : optional, dict {column: float}
        Weighting over df columns
        By default, a uniform weighting is used

    df : function (vectorised)
        Vectorised function on 1d np arrays. Returns diversity function.

    pre_selects : None or iterable
        Optionally, you may pre-specify a set of rows to be forced into the
        solution.
        Values must be valid indices for df.

    quantile : float, values in [0,1]
        Define the quantile to be used in tie-breaking between top choices at
        every step; choose e.g. 0.01 for the top 1% quantile
        By default, 0.0

    alpha : float in (0, 1]
        Scaling exponent for the objective function.

    s : np.array
        1D Array of dim m. Contains scores by participant.

    sratio : float in [0, 1]
        Weighting ratio of score : diversity  

    Returns
    -------

    div : float
        The diversity of the solution found.  Larger is better.

    qual : float
        The mean score of the solution found.  Larger is better.

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
        # Simple if not calculating quality
        
        if sratio == 0:
            delta = df(p_new) - df(p)
           
        # If calculating quality, scale both scores and sum
        else:
            delta_div_scaled = (df(p_new) - df(p)) / qmax
            delta_qual_scaled = s / k
            delta = delta_div_scaled*(1-sratio) + delta_qual_scaled*sratio

        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score.  
        # If quantile is enabled, break near-ties randomly.
        delta_real = delta[np.isfinite(delta)]
        target_score = np.percentile(delta_real, 100 * (1.0-quantile))

        new_idx = rng.choice(np.flatnonzero(delta >= target_score))
        y[new_idx] = True
    
    return ((df(np.nansum(X[y], axis=0)) / qmax), (sum(s[y]) / k), np.flatnonzero(y))

def __proportional_targets(p, q, w, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    return ((np.minimum(q, p))**(alpha)).dot(w)

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
        
    alpha : float in (0, 1]
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

def __presence_targets(p, q, n, alpha=0.5):
    '''Compute the diversity value of a solution.'''
    # Should this be the straightforward sum of all of the nations, up to the limit
    # Or should this be the sum of the greatest n nations, up to the limit?
    return (np.minumum(np.sum(np.minimum(q, p)), n*q)**(alpha))

def return_frontier(X, s, k, divfunc, dfmax, res=20, ext=True, seed=None):
    '''Return Cohorts on a Frontier

    Parameters
    ----------
    X : np.array
        2D Array of dims n by m. Rows are participants, and columns are (binarised) attributes.
        
    s : np.array
        1D Array of dim m. Contains scores by participant.

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
        
    seed : [optional] int or numpy.random.RandomState
        An optional seed or random number state.

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

    for i in range(res):
        
        if ext:  
            sratio = i/(res-1)
        else:
            sratio = (i+1)/(res+1)

        div, qual, c = __optimise_cohort(
            X, 
            k, 
            check_random_state(seed),
            df = divfunc,
            dfmax = dfmax,
            quantile=0,
            s=s,
            sratio=sratio
        )
        
        ds.append(div)
        qs.append(qual)
        cs.append(c)
        
        
    return (ds, qs, cs)


def plot_frontier(ds, qs, dlabel='Diversity', qlabel='Observed Quality', title='Selection Possibility Frontier', lims=False):
    plt.scatter(qs, ds)
    if lims:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.xlabel(qlabel)
    plt.ylabel(dlabel)
    plt.title(title)
    plt.show()
