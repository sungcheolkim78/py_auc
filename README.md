# py_auc: probability based binary classifier metric calculation library

library for calculating the area under the curve (ROC, PR) of binary classifiers

## Installation

Download or clone this repository

```{bash}
> git clone git@github.com:sungcheolkim78/py_auc.git
```
  
Install libary locally

```{bash}
> pip3 install -e .
```

## Usage

```{python}
import py_auc

sg0 = py_auc.Score_generator()
sg0.set(rho=0.75, kind0='gaussian', mu0=0, std0=2, kind1='gaussian', mu1=4, std1=2)

res = sg0.get_classProbability(sampleSize=200, sampleN=500)
sg0.plot_rank(cprob=res)
```

Score_generator class has key methods;