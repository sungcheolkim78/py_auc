"""

test new algorithm

"""

import numpy as np
import py_auc

class Score_generator(object):
    """ two class score generator """

    def __init__(self):
        """ """

        self._kind0 = ''
        self._kind1 = ''
        self._mu0 = 0
        self._mu1 = 0
        self._std0 = 0
        self._std1 = 0
        self._n0 = 100
        self._n1 = 100

        self._s0 = []
        self._s1 = []

    def generate(self, kind, mu, std, n):
        """ set parameters of class """

        if kind.lower() not in ['uniform', 'gaussian', 'triangle']:
            kind = 'uniform'

        if kind.lower() == 'uniform':
            temp = np.random.uniform(low=mu-std, high=self._mu0+self._std0, size=n)
        elif kind.lower() == 'gaussian':
            temp = np.random.normal(loc=mu, scale=std, size=n)
        elif kind.lower() == 'triangle':
            temp = np.random.triangular(mu-std, mu, mu+std, size=n)

        return temp, kind, mu, std, n

    def set0(self, kind, mu, std, n):
        """
        kind : ['uniform', 'gaussian', 'triangle']
        mu : mean
        std : standard deviation
        n : number of samples
        """

        self._s0, self._kind0, self._mu0, self._std0, self._n0 = self.generate(kind, mu, std, n)

    def set1(self, kind, mu, std, n):
        self._s1, self._kind1, self._mu1, self._std1, self._n1 = self.generate(kind, mu, std, n)

    def get(self):
        return self._s0, self._s1
