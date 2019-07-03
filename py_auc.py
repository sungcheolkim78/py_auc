#!/usr/local/bin/python3
"""
py_auc - python library for calculating the area under the curve (ROC, PR) of binary classifiers

author: Sungcheol Kim @ IBM
email: kimsung@us.ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#import seaborn as sns


class AUC(object):
    """ object for area under the curve (AUC) calculation """

    def __init__(self, data=None, debug=False):
        """
        initialize with scores and classes
        input: data - panda DataFrame with "Score" and "Class" columns
        """

        self._scores = []
        self._classes = []
        self._data = data

        self._debug = debug

        self._n0 = 0       # class 0
        self._n1 = 0       # class 1
        self._n = 0

        self._spm = None

        if (data is not None) and ("Score" in data.columns) and ("Class" in data.columns):
            self.get_classes_scores(data["Class"], data["Score"])

    def get_classes_scores(self, c, s):
        """ add scores and classes """

        if len(c) != len(s):
            print('... dimension is not matches: score - %d    class - %d'.format(len(s), len(c)))
            return

        self._scores = s
        self._classes = c

        self._n = len(self._scores)
        self._n1 = np.count_nonzero(self._classes)
        self._n0 = self._n - self._n1

        self._prepare()

    def get_scores(self, s0, s1):
        """ add class 0 score and class 1 score separately """

        self._n0 = len(s0)
        self._n1 = len(s1)
        self._n = self._n0 + self._n1

        self._scores = np.zeros(self._n)
        self._scores[:self._n0] = s0
        self._scores[self._n0:] = s1
        self._classes = np.ones(self._n)
        self._classes[:self._n0] = 0

        self._prepare()

    def n(self):
        return self._n

    def n0(self):
        return self._n0

    def n1(self):
        return self._n1

    def _prepare(self):
        """ calculate rank """

        if self._data is not None:
            return

        self._data = pd.DataFrame()

        self._data['Score'] = self._scores
        self._data['Class'] = self._classes

    def cal_auc_rank(self, measure_time=False):
        """ calculate area under ROC using rank algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = np.arange(self._n)/self._n
        mask = self._spm['Class'] == 0

        auc = 0.5 + (self._spm[mask].Rank.mean() - self._spm[~mask].Rank.mean())

        if measure_time: print("--- %s seconds ---" % (time.time() - start_time))

        return(auc)

    def cal_auc_trapz(self, measure_time=False):
        """ calculate area under ROC using trapz function """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        x = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.trapz(y, x=x)

        if measure_time: print("--- %s seconds ---" % (time.time() - start_time))

        return(auc)

    def cal_auc_sklearn(self, measure_time=False):
        """ calculate area under ROC using scikit-learning """

        from sklearn.metrics import roc_auc_score

        if measure_time: start_time = time.time()

        auc = roc_auc_score(self._classes, self._scores)

        if measure_time: print("--- %s seconds ---" % (time.time() - start_time))

        return(auc)

    def cal_auprc_rank(self, measure_time=False):
        """ calculate area under precision-recall curve using rank algorithm """

        if measure_time: start_time = time.time()

        rho = self._n1/self._n

        self._spm = self._data.sort_values(by='Score', ascending=False)
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = 0.5*rho*(1.0 + np.sum(p*p)/(self._n*rho*rho))

        if measure_time: print("--- %s seconds ---" % (time.time() - start_time))

        return(auprc)

    def cal_auprc_trapz(self, measure_time=False):
        """ calculate area under precision-recall curve using trapz algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = np.trapz(p, x=y)

        if measure_time: print("--- %s seconds ---" % (time.time() - start_time))

        return(auprc)

    def plot_rank(self, width=800):
        """ plot rank vs class """

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

        auc = self.cal_auc_rank()

        ax0 = axs[0]
        ax0.plot(self._spm['Rank'], self._spm['Class'])
        ax0.set_xlabel('Normalized Rank')
        ax0.set_ylabel('Class')

        cmatrix = self._spm['Class'].values[::-1]

        ax1 = axs[1]
        ax1.imshow(cmatrix.reshape(width, -1).T)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Rank')
        ax1.set_title('Class heatmap by Rank')

    def plot_ROC(self):
        """ calculate ROC curve (receiver operating characteristic curve) """

        self._spm = self._data.sort_values(by='Score', ascending=False)
        x = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision
        #print(p)

        auc = np.trapz(y, x=x)
        print('AUC (area under the ROC curve): {0:8.3f}'.format(auc))

        auc = np.trapz(p, x=y)
        print('AUPRC (area under the PRC curve): {0:8.3f}'.format(auc))

        # ROC plot
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

        ax0 = axs[0]
        ax0.plot(x, y)
        ax0.set_xlabel('FPR (false positive rate)')
        ax0.set_ylabel('TPR (true positive rate)')
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.set_title('Receiver Operating Characteristic Curve)')

        # PR plot
        ax1 = axs[1]
        ax1.plot(y, p)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Precision-Recall Curve')
        ax1.set_xlabel('Recall (TPR)')
        ax1.set_ylabel('Precision')

        # score distribution

        bins = np.linspace(min(self._scores), max(self._scores), 100)

        ax2 = axs[2]
        ax2.hist(self._data.loc[self._data['Class'] == 0, 'Score'], bins, alpha=0.5, label='Category 0')
        ax2.hist(self._data.loc[self._data['Class'] == 1, 'Score'], bins, alpha=0.5, label='Category 1')
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Scores')
        ax2.set_ylabel('#')

        plt.show()

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
            temp = np.random.uniform(low=mu-std, high=mu+std, size=n)
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
        """
        kind : ['uniform', 'gaussian', 'triangle']
        mu : mean
        std : standard deviation
        n : number of samples
        """
        self._s1, self._kind1, self._mu1, self._std1, self._n1 = self.generate(kind, mu, std, n)

    def get(self):
        """ get scores """

        return [self._s0, self._s1]

    def get_asDataFrame(self):
        """ get scores as DataFrame """

        n = self._n0 + self._n1
        scores = np.zeros(n)
        scores[:self._n0] = self._s0
        scores[self._n0:] = self._s1
        classes = np.ones(n)
        classes[:self._n0] = 0

        return pd.DataFrame({'Score': scores, 'Class': classes})

