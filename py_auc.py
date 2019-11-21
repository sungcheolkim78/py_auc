#!/usr/local/bin/python3
"""
py_auc - python library for calculating the area under the curve (ROC, PR) of binary classifiers

author: Sungcheol Kim @ IBM
email: kimsung@us.ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from scipy.optimize import curve_fit


class AUC(object):
    """ object for area under the curve (AUC) calculation
    example:
        import py_auc

        sg = py_auc.Score_generator()
        a = py_auc.AUC(sg.get_asDataFrame())
        a.cal_auc_rank()
        a.plot_ROC()
    """

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

    def _prepare(self):
        """ calculate rank """

        self._data = pd.DataFrame()

        self._data['Score'] = self._scores
        self._data['Class'] = self._classes

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = range(self._n+1)[1:]
        self._spm['FPR'] = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        self._spm['TPR'] = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        self._spm['Prec'] = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

    def cal_auc_rank(self, measure_time=False):
        """ calculate area under ROC using rank algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['Rank'] = np.arange(self._n+1)[1:]/self._n
        mask = self._spm['Class'] == 0

        auc = 0.5 + (self._spm[mask].Rank.mean() - self._spm[~mask].Rank.mean())

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_bac(self, measure_time=False):
        """ calculate area under ROC using rank algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        self._spm['FPR'] = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        self._spm['TPR'] = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.sum(self._spm['TPR'] + 1 - self._spm['FPR'])/self._n - 0.5

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_trapz(self, measure_time=False):
        """ calculate area under ROC using trapz function """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        x = (self._spm['Class'] == 0).cumsum().values/self._n0         # FPR
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall

        auc = np.trapz(y, x=x)

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auc_sklearn(self, measure_time=False):
        """ calculate area under ROC using scikit-learning """

        if measure_time: start_time = time.time()

        auc = roc_auc_score(self._classes, self._scores)

        if measure_time:
            return (auc, (time.time() - start_time))
        else:
            return (auc)

    def cal_auprc_rank(self, measure_time=False):
        """ calculate area under precision-recall curve using rank algorithm """

        if measure_time: start_time = time.time()

        rho = self._n1/self._n

        self._spm = self._data.sort_values(by='Score', ascending=False)
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = 0.5*rho*(1.0 + np.sum(p*p)/(self._n*rho*rho))

        if measure_time:
            return (auprc, (time.time() - start_time))
        else:
            return(auprc)

    def cal_auprc_trapz(self, measure_time=False):
        """ calculate area under precision-recall curve using trapz algorithm """

        if measure_time: start_time = time.time()

        self._spm = self._data.sort_values(by='Score', ascending=False)
        y = (self._spm['Class'] == 1).cumsum().values/self._n1         # TPR or recall
        p = (self._spm['Class'] == 1).cumsum().values/(np.arange(self._n) + 1)    # precision

        auprc = np.trapz(p, x=y)

        if measure_time:
            return (auprc, (time.time() - start_time))
        else:
            return(auprc)

    def cal_auprc_sklearn(self, measure_time=False):
        """ calculate area under PRC using scikit-learning """

        if measure_time: start_time = time.time()

        auprc = average_precision_score(self._classes, self._scores)

        if measure_time:
            computetime = time.time() - start_time
            return (auprc, computetime)
        else:
            return auprc

    def plot_rank(self, sampling=10, filename=''):
        """ plot rank vs class """

        self._prepare()

        cmatrix = self._spm['Class'].values.reshape(sampling, -1)
        prob = cmatrix.mean(axis=1)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        ax0 = axs[0]
        ax0.plot(prob, '.')
        ax0.set_xlabel('Sampled Rank r\'')
        ax0.set_ylabel('P(1|r\')')

        ax1 = axs[1]
        ax1.plot(self._spm['Rank'], self._spm['TPR'], label='TPR')
        ax1.plot(self._spm['Rank'], self._spm['FPR'], '--', label='FPR')
        ax1.set_xlabel('Rank r')
        ax1.set_ylabel('TPR(r), FPR(r)')
        ax1.legend()

        ax2 = axs[2]
        ax2.plot(self._spm['Rank'], self._spm['Prec'], label='prec')
        bac = (self._spm['TPR'].values + 1.0 - self._spm['FPR'].values)/2.0
        ax2.plot(self._spm['Rank'], bac, '--', label='bac')
        ax2.set_xlabel('Rank r')
        ax2.set_ylabel('Precision(r), bac(r)')
        ax2.legend()

        if filename == '': filename = 'rank_plot.pdf'
        plt.savefig(filename, dpi=150)

    def plot_ROC(self, bins=50, filename=''):
        """ calculate ROC curve (receiver operating characteristic curve) """

        self._prepare()

        auc = np.trapz(self._spm['TPR'], x=self._spm['FPR'])
        print('AUC (area under the ROC curve): {0:8.3f}'.format(auc))

        auc = np.trapz(self._spm['Prec'], x=self._spm['TPR'])
        print('AUPRC (area under the PRC curve): {0:8.3f}'.format(auc))

        # ROC plot
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

        ax0 = axs[0]
        ax0.plot(self._spm['FPR'], self._spm['TPR'], label='ROC')
        ax0.set_xlabel('FPR')
        ax0.set_ylabel('TPR')
        #ax0.set_xlim(0, 1.1)
        #ax0.set_ylim(0, 1.1)

        # PR plot
        ax1 = axs[1]
        ax1.plot(self._spm['TPR'], self._spm['Prec'], label='PRC')
        ax1.set_xlabel('Recall (TPR)')
        ax1.set_ylabel('Precision')
        #ax1.set_xlim(0, 1.1)
        #ax1.set_ylim(0, 1.1)
        #ax1.set_title('Precision-Recall Curve')

        # score distribution
        ax2 = axs[2]
        sns.distplot(self._spm.loc[self._spm['Class']==0, 'Score'], bins=bins, kde=False, rug=True, label='Class 0')
        sns.distplot(self._spm.loc[self._spm['Class']==1, 'Score'], bins=bins, kde=False, rug=True, label='Class 1')
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Scores')
        ax2.set_ylabel('#')

        if filename == '':
            filename = 'auc_summary.pdf'
        plt.savefig(filename, dpi=150)
        plt.show()


class Score_generator(object):
    """ two class score generator
    example:
        import py_auc
        sg = py_auc.Score_generator()
        sg.set0('gaussian', 0, 1, 1000)
        sg.set1('gaussian', 3, 1, 1000)

        OR
        sg.set(n=10000, rho=0.5, kind0='gaussian', mu0=0, std0=2, kind1='gaussian', mu1=1, std1=2)

        res = sg.get_classProbability(sampleSize=100, sampleN=100, measure_time=False)
        lambda = sg.get_lambda(cprob=res)
        sg.plot_prob(cprob=res)
        sg.plot_rank(cprob=res)
    """

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
        self._n = self._n0 + self._n1
        self._rho = self._n1/self._n

        self._s0 = []
        self._s1 = []
        self._prob = []
        self._sampleN = 0
        self._sampling = []
        self._fit_vals = []

        self._debug = False

    def _generate(self, kind, mu, std, n, k):
        """ set parameters of class """

        if kind.lower() not in ['uniform', 'gaussian', 'triangle', 'gfamily']:
            kind = 'uniform'

        if kind.lower() == 'uniform':
            temp = np.random.uniform(low=mu-std, high=mu+std, size=n)
            shapex = np.array([mu-2*std, mu-std, mu-std, mu+std, mu+std, mu+2*std])
            shapey = np.array([0, 0, 1/(2*std), 1/(2*std), 0, 0])
        elif kind.lower() == 'gaussian':
            temp = np.random.normal(loc=mu, scale=std, size=n)
            shapex = np.linspace(mu-2.5*std, mu+2.5*std, 50)
            shapey = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (shapex - mu)**2 / (2 * std**2))
        elif kind.lower() == 'triangle':
            #if self._debug: print('... generating gaussian distribution with n={}, mu={}, sigma={}'.format(n, mu, std))
            temp = np.random.triangular(mu-std, mu, mu+std, size=n)
            shapex = np.array([mu-2*std, mu-std, mu, mu+std, mu+2*std])
            shapey = np.array([0, 0, 1/std, 0, 0])
        elif kind.lower() == 'gfamily':
            #if self._debug: print('... generating gaussian family with k = {}, n={}, mu={}, sigma={}'.format(k, n, mu, std))
            temp = gaussfamily(k=k, n=n, mu=mu, std=std, show=False)
            if k == 1:
                std_k = std * np.sqrt(3*k)
                shapex = np.array([mu-2*std_k, mu-std_k, mu-std_k, mu+std_k, mu+std_k, mu+2*std_k])
                shapey = np.array([0, 0, 1/(2*std_k), 1/(2*std_k), 0, 0])
            elif k == 2:
                std_k = std * np.sqrt(3*k)
                shapex = np.array([mu-2*std_k, mu-std_k, mu, mu+std_k, mu+2*std_k])
                shapey = np.array([0, 0, 1/std_k, 0, 0])
            else:
                shapex = np.linspace(mu-2.5*std, mu+2.5*std, 50)
                shapey = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (shapex - mu)**2 / (2 * std**2))

        return temp, kind, mu, std, n, shapex, shapey

    def set0(self, kind, mu, std, n, k=1):
        """
        kind : ['uniform', 'gaussian', 'triangle', 'gfamily']
        mu : mean
        std : standard deviation
        n : number of samples
        """

        self._s0, self._kind0, self._mu0, self._std0, self._n0, self._shapex0, self._shapey0 = self._generate(kind, mu, std, n, k)
        self._n = self._n0 + self._n1
        self._rho = float(self._n1/self._n)
        self._k0 = k

    def set1(self, kind, mu, std, n, k=1):
        """
        kind : ['uniform', 'gaussian', 'triangle', 'gfamily']
        mu : mean
        std : standard deviation
        n : number of samples
        """
        self._s1, self._kind1, self._mu1, self._std1, self._n1, self._shapex1, self._shapey1 = self._generate(kind, mu, std, n, k)
        self._n = self._n0 + self._n1
        self._rho = float(self._n1/self._n)
        self._k1 = k

    def set(self, n=10000, rho=0.5, kind0='gaussian', mu0=0, std0=2, k0=1, kind1='gaussian', mu1=1, std1=2, k1=1):
        """ generate score distribution """

        n1 = int(n*rho)
        n0 = n - n1
        if self._debug: print('... generating {} positive class'.format(n1))
        self.set1(kind1, mu1, std1, n1, k=k1)
        if self._debug: print('... generating {} negative class'.format(n0))
        self.set0(kind0, mu0, std0, n0, k=k0)

    def get(self):
        """ get scores """

        return [self._s0, self._s1]

    def get_asDataFrame(self):
        """ get scores as DataFrame """

        scores = np.zeros(self._n)
        scores[:self._n0] = self._s0
        scores[self._n0:] = self._s1
        classes = np.ones(self._n)
        classes[:self._n0] = 0

        return pd.DataFrame({'Score': scores, 'Class': classes})

    def get_randomSample(self, n):
        """ get sample of scores """

        temp = self.get_asDataFrame()
        return temp.sample(n)

    def get_classProbability(self, sampleSize=100, sampleN=100, measure_time=False):
        """ calculate probability of class at given rank r """

        if measure_time: start_time = time.time()

        temp = self.get_asDataFrame()
        temp0 = temp[temp['Class'] == 0]
        temp1 = temp[temp['Class'] == 1]
        n1 = int(sampleSize*self._rho)
        n0 = sampleSize - n1

        res = pd.DataFrame()
        res['Rank'] = range(sampleSize+1)[1:]

        for i in range(sampleN):
            a = pd.concat([temp0.sample(n0), temp1.sample(n1)]).sort_values(by='Score', ascending=False)
            res['Class_{}'.format(i)] = a['Class'].values

        self._prob = res.values[:, 1:sampleN+1].mean(axis=1)
        res['P(1|r)'] = self._prob
        res['P(0|r)'] = 1 - self._prob

        res['TPR'] = np.cumsum(res['P(1|r)'])/n1
        res['FPR'] = np.cumsum(res['P(0|r)'])/n0
        res['Prec'] = np.cumsum(res['P(1|r)'])/res['Rank']
        res['bac'] = 0.5*(res['TPR'] + 1.0 - res['FPR'])

        self._sampling = res
        self._sampleN = sampleN
        self._sampleSize = sampleSize
        self._sampleN0 = n0
        self._sampleN1 = n1
        self._auc = np.sum(res['P(0|r)']*res['Rank']/n0 - res['P(1|r)']*res['Rank']/n1)/sampleSize + 0.5
        self._aucbac = 2*np.sum(res['bac'])/sampleSize - 0.5
        prec = res['Prec'].values
        self._auprc = 0.5*self._rho + 0.5*np.sum(prec[1:]*prec[:-1])/n1    # new formula
        if self._debug:
            print('... sampling: N {}, M {}, auc {}'.format(sampleSize, sampleN, self._auc))

        if measure_time:
            return (res, (time.time()-start_time))
        else:
            return res

    def get_cprob(self, sampleN=100, measure_time=False):
        """ calculate probability of class at given rank r """

        if measure_time: start_time = time.time()

        n1 = int(sampleN*self._rho)
        n0 = sampleN - n1
        self._sampleSize = sampleN

        res = pd.DataFrame()
        res['Rank'] = range(sampleN+1)[1:]

        for i in range(sampleN):
            self.set0(self._kind0, self._mu0, self._std0, n0, k=self._k0)
            self.set1(self._kind1, self._mu1, self._std1, n1, k=self._k1)
            temp = self.get_asDataFrame()
            temp0 = temp[temp['Class'] == 0]
            temp1 = temp[temp['Class'] == 1]

            a = pd.concat([temp0.sample(n0), temp1.sample(n1)]).sort_values(by='Score', ascending=False)
            res['Class_{}'.format(i)] = a['Class'].values

        self._prob = res.values[:, 1:sampleN+1].mean(axis=1)
        res['P(1|r)'] = self._prob
        res['P(0|r)'] = 1 - self._prob

        res['TPR'] = np.cumsum(res['P(1|r)'])/n1
        res['FPR'] = np.cumsum(res['P(0|r)'])/n0
        res['Prec'] = np.cumsum(res['P(1|r)'])/res['Rank']
        res['bac'] = 0.5*(res['TPR'] + 1.0 - res['FPR'])

        self._sampling = res
        self._sampleN = sampleN
        self._sampleSize = sampleN
        self._sampleN0 = n0
        self._sampleN1 = n1
        self._auc = np.sum(res['P(0|r)']*res['Rank']/n0 - res['P(1|r)']*res['Rank']/n1)/sampleN + 0.5
        self._aucbac = 2*np.sum(res['bac'])/sampleN - 0.5
        prec = res['Prec'].values
        self._auprc = 0.5*self._rho + 0.5*np.sum(prec[1:]*prec[:-1])/n1    # new formula
        if self._debug:
            print('... sampling: N {}, M {}, auc {}'.format(sampleN, sampleN, self._auc))

        if measure_time:
            mtime = time.time() - start_time
            if self._debug:
                print('compute time: {} sec'.format(mtime))
            return [res, mtime]
        else:
            return res

    def get_lambda(self, cprob=None, init_vals=None, sampleN=100):
        """ fit with Fermi-dirac distribution """

        if cprob is None:
            self.get_cprob(sampleN=sampleN)
        else:
            self._sampling = cprob

        if init_vals is None:
            init_vals = [0.1, self._sampleN*self._rho*0.1]

        if self._debug: print('... fitting: initial l2, l1 = {}'.format(init_vals))
        x = self._sampling['Rank'].values
        self._fit_vals, covar = curve_fit(fd, x, self._prob, p0=init_vals)
        if self._debug: print('... fitting: final l2, l1 = {}'.format(self._fit_vals))

        return (-self._fit_vals[1], self._fit_vals[0])

    def plot_hist(self, filename='', show=True):
        """ plot histogram """

        plt.close('all')
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(121)

        bins = 50

        sns.distplot(self._s0, bins=bins, norm_hist=True, kde=False, rug=True, label='Class 0 (#={})'.format(self._n0))
        sns.distplot(self._s1, bins=bins, norm_hist=True, kde=False, rug=True, label='Class 1 (#={})'.format(self._n1))
        plt.plot(self._shapex0, self._shapey0)
        plt.plot(self._shapex1, self._shapey1)
        
        plt.annotate("mu={}\ns={}".format(self._mu0, self._std0), xy=(self._mu0, 0), xytext=(0.25, 0.25),
                textcoords='axes fraction', horizontalalignment='left',
                arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate("mu={}\ns={}".format(self._mu1, self._std0), xy=(self._mu1, 0), xytext=(0.75, 0.25),
                textcoords='axes fraction', horizontalalignment='right',
                arrowprops=dict(facecolor='black', shrink=0.05))
        plt.xlabel('Score')
        plt.ylabel('#')
        plt.legend()

        ax2 = fig.add_subplot(122)
        r = self._sampling['Rank'].values
        c = self._sampling['P(1|r)'].values
        label = 'Size={}, #={}'.format(len(self._prob), self._sampleN)

        ax2.plot(r, c, '.', label=label, alpha=0.5)
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('P(1|r)')
        ax2.set_ylim((-0.05, 1.05))
        if len(self._fit_vals) == 2:
            msg = 'Fit: {:.3f}, {:.3f}'.format(self._fit_vals[0], self._fit_vals[1])
            ax2.plot(r, fd(r, self._fit_vals[0], self._fit_vals[1]), label=msg)
        ax2.legend()

        if filename == '':
            filename = 'score_hist.pdf'

        plt.savefig(filename, dpi=150)
        if show: plt.show()

    def plot_prob(self, filename='', ss=100, sn=100, axs=None, show=True, cprob=None, label=None, figsize=None):
        """ plot class probability """

        if cprob is not None:
            a = cprob
        else:
            a = self.get_classProbability(sampleSize=ss, sampleN=sn)

        if axs is None:
            plt.close('all')
            if figsize is None:
                fig = plt.figure(figsize=(16, 5))
            else:
                fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            axs = (ax1, ax2, ax3)
        else:
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]

        if label is None:
            label = 'Size={}, #={}'.format(len(self._prob), self._sampleN)

        r = range(len(self._prob))
        ax1.plot(r, self._prob, '.', label=label, alpha=0.5)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('P(1|r)')
        ax1.set_ylim((-0.05, 1.05))
        ax1.legend()

        ax2.plot(a['FPR'], a['TPR'], '.', label=label, alpha=0.5)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title('ROC')
        ax2.legend()

        ax3.plot(a['TPR'], a['Prec'], '.', label=label, alpha=0.5)
        ax3.set_xlabel('TPR')
        ax3.set_ylabel('Prec')
        ax3.set_title('PRC')
        ax3.legend()

        if filename == '':
            filename = 'classProbability.pdf'
        plt.savefig(filename, dpi=150)

        if show:
            plt.show()
            return
        else:
            return axs

    def plot_rank(self, filename='', ss=100, sn=100, axs=None, show=True, cprob=None, label=None, figsize=None):
        """ plot 3 panels of rank related functions """

        if cprob is not None:
            a = cprob
        else:
            a = self.get_classProbability(sampleSize=ss, sampleN=sn)

        if axs is None:
            plt.close('all')
            if figsize is None:
                fig = plt.figure(figsize=(14, 5))
            else:
                fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            axs = (ax1, ax2, ax3)
        else:
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]

        if label is None:
            label = 'Size={}, #={}'.format(len(self._prob), self._sampleN)

        r = a['Rank'].values
        ax1.plot(r, self._prob, '.', label=label, alpha=0.5)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('P(1|r)')
        ax1.set_ylim((-0.05, 1.05))
        if len(self._fit_vals) == 2:
            msg = 'Fit: {:.3f}, {:.3f}'.format(self._fit_vals[0], self._fit_vals[1])
            ax1.plot(r, fd(r, self._fit_vals[0], self._fit_vals[1]), label=msg)
        ax1.legend()

        ax2.plot(a['Rank'], a['bac'], '.', label=label, alpha=0.5)
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('bac')
        #ax2.set_title('ROC')
        ax2.legend()

        ax3.plot(a['Rank'], a['Prec']*a['Prec'], '.', label=label, alpha=0.5)
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Prec^2')
        #ax3.set_title('PRC')
        ax3.legend()

        if filename == '':
            filename = 'rank_class_Probability.pdf'
        plt.savefig(filename, dpi=150)

        if show:
            plt.show()
            return
        else:
            return axs

    def plot_fermi(self, sn=100, label=None, show=True, cprob=None, axs=None, figsize=None):

        if cprob is not None:
            a = cprob
        else:
            a = self.get_cprob(sampleN=sn)

        if axs is None:
            plt.close('all')
            if figsize is None:
                fig = plt.figure(figsize=(10, 6))
            else:
                fig = plt.figure(figsize=figsize)
            axs = fig.gca()

        if label is None:
            label = 'Size={}, #={}'.format(len(self._prob), self._sampleN)

        r = a['Rank'].values
        axs.plot(r, self._prob, '.', label=label, alpha=0.5)
        axs.set_xlabel('Rank')
        axs.set_ylabel('P(1|r)')
        axs.set_ylim((-0.05, 1.05))
        if len(self._fit_vals) == 2:
            msg = 'Fit: {:.3f}, {:.3f}'.format(self._fit_vals[0], self._fit_vals[1])
            axs.plot(r, fd(r, self._fit_vals[0], self._fit_vals[1]), label=msg)
        axs.legend()

        if show:
            plt.show()
            return
        else:
            return axs


def fd(x, l1, l2):
    """ fermi-dirac distribution """
    return 1./(1.+np.exp(l1*x - l2))


def gaussfamily(k=1, n=100, mu=0, std=1, show=True):
    """ generate random numbers from gaussian family distribution """

    x = np.sum(np.random.rand(k, n), axis=0)/k*2*std*np.sqrt(3*k) + mu - std*np.sqrt(3*k)

    if show:
        c_std = np.std(x)
        c_mu = np.mean(x)
        print('... mean: {}, std: {}'.format(c_mu, c_std))

        plt.hist(x, bins=50)
        plt.show()

    return x

# vim:foldmethod=indent:foldlevel=0
