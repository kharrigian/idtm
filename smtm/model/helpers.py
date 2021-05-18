

###################
### Imports
###################

## External Library
import numpy as np
from scipy.special import logsumexp

###################
### Functions
###################

def sample_categorical(p):
    """

    """
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

def sample_multinomial(p):
    """
    Multinomial sample that gets around floating point error
    with small numbers

    Args:
        p (1d-array): Probability distribution (sums to 1)

    Returns:
        i (int): Sampled index based on probability distribution
    """
    pcum = p.cumsum()
    u = np.random.uniform(0,1)
    for i, v in enumerate(pcum):
        if u < v:
            return i
    return len(p) - 1

def logistic_transform(x,
                       is_log=False,
                       axis=0,
                       **kwargs):
    """

    """
    ## Check Type
    if isinstance(x, list):
        x = np.array(x)
    ## Tranformation
    if is_log:
        Z = logsumexp(x, axis=axis, **kwargs)
        transform = x - Z
    else:
        b = x.max(axis=axis, **kwargs)
        xexp = np.exp(x - b)
        Z = np.sum(xexp, axis=axis, **kwargs)
        transform = np.divide(xexp,
                              Z,
                              where=Z>0,
                              out=np.zeros_like(xexp))
    return transform
        
    