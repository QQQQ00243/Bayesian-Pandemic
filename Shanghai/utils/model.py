import numpy as np
import bayesloop as bl

from .utils import get_CIDict
from scipy.stats import poisson
from inspect import getfullargspec
from bayesloop.exceptions import ConfigurationError
from bayesloop.transitionModels import SerialTransitionModel, GaussianRandomWalk, CombinedTransitionModel, BreakPoint


def likelihood(data, R, D):
    now, prev = data[0], data[1:]
    beta = [0.01355014, 0.03523035, 0.08130081, 0.14905149, 0.1897019, 0.21680217, 0.13550136, 0.08130081, 0.05420054, 0.02710027, 0.01355014, 0.00271003]
    mean = 0
    for i in range(int(D)):
        mean += prev[-(i+1)] * beta[i]
    mean *= R
    return poisson.pmf(k=now, mu=mean)

def likelihood_h1n1(data, R, D):
    now, prev = data[0], data[1:]
    si = [2.24, 2.85, 9.31, 49.83, 64.49, 49.07, 23.76, 7.07, 1.51]
    scale = (1.37 / 10.92 * 0.05 + 0.3) / 64.49
    beta = scale * np.array(si)
    mean = 0
    for i in range(int(D)):
        mean += prev[-(i+1)] * beta[i]
    mean *= R
    return poisson.pmf(k=now, mu=mean)


class customOM(bl.observationModels.ObservationModel):
    def __init__(self, function, *args, **kwargs):
        # check if first argument is valid
        if not hasattr(function, '__call__'):
            raise ConfigurationError('Expected a function as the first argument of NumPy observation model')

        self.function = function
        self.name = function.__name__
        self.segmentLength = 1  # all required data for one time step is bundled
        self.multiplyLikelihoods = False  # no more than one observation per time step

        # get specified parameter names/values
        self.parameterNames = args[::2]
        self.parameterValues = args[1::2]

        # check if number of specified parameters matches number of arguments of function (-1 for data)
        nArgs = len(getfullargspec(self.function).args)
        if not len(self.parameterNames) == nArgs-1:
            raise ConfigurationError('Supplied function has {} parameters, observation model has {}'
                                     .format(nArgs-1, len(self.parameterNames)))

        # check if first argument of supplied function is called 'data'
        if not getfullargspec(self.function).args[0] == 'data':
            raise ConfigurationError('First argument of supplied function must be called "data"')

        # check for unknown keyword-arguments
        for key in kwargs.keys():
            if key not in ['prior']:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(key))

        # get allowed keyword-arguments
        self.prior = kwargs.get('prior', None)

    def pdf(self, grid, dataSegment):
        gridR, gridD = grid
        gridR = gridR[:, 0]
        gridD = gridD[0, :]
        PDF = np.empty((len(gridR), len(gridD)))
        for i, R in enumerate(gridR):
            for j, D in enumerate(gridD):
                PDF[i, j] = self.function(dataSegment[0], R, D)
        return PDF


class myStudy(bl.Study):
    def __init__(
        self,
        data_seg,
        hpDict,
        silent,
    ):
        bl.Study.__init__(self, silent=silent)
        self.data_seg = data_seg
        self.hpDict = hpDict
        self.silent = silent

    def estimate_params(self):
        self.loadData(self.data_seg, silent=self.silent)
        params = ["R", "D"]
        L = customOM(likelihood, "R", bl.oint(0, 10, 200), "D", bl.cint(0, 11, 12))
        T = SerialTransitionModel(
            CombinedTransitionModel(
                GaussianRandomWalk('s1R', self.hpDict["s1R"], target='R'),
                GaussianRandomWalk('s1D', self.hpDict["s1D"], target='D'),
            ),
            BreakPoint("t1", self.hpDict["t1"]),
            CombinedTransitionModel(
                GaussianRandomWalk('s2R', self.hpDict["s2R"], target='R'),
                GaussianRandomWalk('s2D', self.hpDict["s2D"], target='D'),
            ),
            BreakPoint("t2", self.hpDict["t2"]),
            CombinedTransitionModel(
                GaussianRandomWalk('s3R', self.hpDict["s3R"], target='R'),
                GaussianRandomWalk('s3D', self.hpDict["s3D"], target='D'),
            ),
        )
        self.set(L, T, silent=self.silent)
        self.fit(silent=self.silent)
        
        # res = {"R":{"lowers": [...], "uppers": [...], "means": [...]},
        #        "D":{"lowers": [...], "uppers": [...], "means": [...]}}
        res = get_CIDict(params=params, S=self)
        res["R"]["means"] = self.getParameterMeanValues("R").tolist()
        res["D"]["means"] = [int(D) for D in self.getParameterMeanValues("D").tolist()]
        return res

