import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
def __optimise_cohort(X, k, rng, w=None, q=None, pre_selects=None, quantile=0.0, alpha=0.5, s=None, sratio=0):
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

    q : np.array
        1D Array of dim n. Contains target proportions by attribute.

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

    if w is None:
        w = np.ones(n_attributes)

    if q is None:
        q = 0.5 * np.ones(n_attributes)

    assert 0 < k <= n_participants
    assert not np.any(w < 0)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
    assert len(w) == n_attributes
    assert len(q) == n_attributes
    assert (s is not None) or (sratio == 0)
    assert sratio >= 0 and sratio <= 1
    
    if k == n_participants:
        return np.arange(n_participants)

    # Convert fractions to sums
    q = np.round(k * q)

    if s is not None:
        qmax = __objective_diversity(q, w, q, alpha=alpha)
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
            delta = __objective_diversity(p_new, w, q, alpha=alpha) - __objective_diversity(p, w, q, alpha=alpha)
           
        # If calculating quality, scale both scores and sum
        else:
            delta_div_scaled = (__objective_diversity(p_new, w, q, alpha=alpha) - __objective_diversity(p, w, q, alpha=alpha)) / qmax
            delta_qual_scaled = s / k
            delta = delta_div_scaled*(1-sratio) + delta_qual_scaled*sratio

        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score.  Break near-ties randomly.
        delta_real = delta[np.isfinite(delta)]
        target_score = np.percentile(delta_real, 100 * (1.0-quantile))

        new_idx = rng.choice(np.flatnonzero(delta >= target_score))
        y[new_idx] = True
    
    return ((__objective_diversity(np.nansum(X[y], axis=0), w, q, alpha=alpha) / qmax), (sum(s[y]) / k), np.flatnonzero(y))



def __return_frontier(X, s, k, q, w=None, res=20, ext=True, seed=None):
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

    if k is None:
        k = categories.shape[0] // 2

    for i in range(res):
        
        if ext:  
            sratio = i/(res-1)
        else:
            sratio = i+1/(res+1)

        div, qual, c = __optimise_cohort(
            X, 
            k, 
            check_random_state(seed),
            w=w,
            q=q,
            quantile=0,
            s=s,
            sratio=sratio
        )
        
        ds.append(div)
        qs.append(qual)
        cs.append(c)
        
        
    return (ds, qs, cs)


def __plot_frontier(ds, qs, dlabel='Diversity', qlabel='Quality', lims=False):
    plt.scatter(qs, ds)
    if lims:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.xlabel(qlabel)
    plt.ylabel(dlabel)
    plt.title('SPF for Possible C1 Finalist Cohorts')
    plt.show()
