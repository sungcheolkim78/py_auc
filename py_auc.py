#!/usr/local/bin/python3
"""
py_auc - python library for calculating the area under the curve (ROC, PR) of binary classifiers

author: Sungcheol Kim @ IBM
email: kimsung@us.ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns


class AUC(object):
    """ object for area under the curve (AUC) calculation """

    def __init__(self, scores=None, classes=None, debug=False):
        """ initialize with scores and classes """

        self._scores = scores
        self._classes = classes
        self._data = None

        self._debug = debug

        self._n0 = 0       # class 0
        self._n1 = 0       # class 1
        self._n = 0

        self._data = None
        self._spm = None

        if (scores is not None) and (classes is not None):
            self.get_classes_scores(classes, scores)

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

        self._data = pd.DataFrame()

        self._data['Score'] = self._scores
        self._data['Class'] = self._classes

    def cal_auc_rank(self):
        """ calculate area under ROC using rank algorithm """

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = range(self._n)

        auc = 0.5 + (self._spm.loc[self._spm['Class'] == 0, 'Rank'].mean() - self._spm.loc[self._spm['Class'] == 1, 'Rank'].mean())/self._n

        return(auc)

    def cal_auprc_rank(self):
        """ calculate area under precision-recall curve using rank algorithm """

        rho = self._n1/self._n

        self._spm = self._data.sort_values(by='Score', ascending=False)
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = 0.5*rho*(1.0 + np.sum(p*p)/(self._n*rho*rho))

        return(auprc)

    def cal_auc_trapz(self):

        self._spm = self._data.sort_values(by='Score', ascending=False)
        x = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.trapz(y, x=x)

        return(auc)

    def cal_auprc_trapz(self):

        self._spm = self._data.sort_values(by='Score', ascending=False)
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = np.trapz(p, x=y)

        return(auprc)

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
