import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from .Richardson_Lucy_Deconvolution import R_L_deconvolve_step


def credibility_interval(samples, weights=None, level=0.95):
    assert level<1, "Level >= 1!"
    weights = np.ones(len(samples)) if weights is None else weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = np.array(samples)[order]
    weights = np.array(weights)[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invCDF = interp1d(CDF, S)
    # Find smallest interval
    distance = lambda Y, level=level: invCDF(Y+level)-invCDF(Y)
    res = minimize_scalar(distance, bounds=(0, 1-level), method="Bounded")
    return invCDF(res.x).item(), invCDF(res.x+level).item()


def get_CI(param, S):
    lowers, uppers = [], []
    for t in S.rawTimestamps:
        samples, weights = S.getPD(t=t, name=param, density=False)
        lower, upper = credibility_interval(samples, weights)
        lowers.append(lower)
        uppers.append(upper)
    return lowers, uppers


def get_CIDict(params, S):
    CIDict = dict()
    for param in params:
        lowers, uppers = get_CI(param, S)
        CIDict[param] = {"lowers": lowers, "uppers": uppers}
        
    return CIDict


def reconstruct(C, Ds, delay, max_iter):
    H = np.zeros((len(C), len(C)))
    for i, D in enumerate(Ds):
        delay_ = delay[:D+1]
        delay_ = delay_ / np.sum(delay_)
        if i+D+1 <= H.shape[0]:
            H[i, i:i+D+1] = delay_
        else:
            H[i, i:H.shape[0]] = delay_[:H.shape[0]-i-D-1]
    I = R_L_deconvolve_step(C, H, max_iter)
    return [int(i) for i in I]


def l1_norm(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    return np.sum(np.abs(array1 - array2))

