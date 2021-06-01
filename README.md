# Infinite Dynamic Topic Model (iDTM)

**Johns Hopkins University**
**Nonparametric Bayesian Stats**
**Spring 2021**
**Professor Yanxun Xu**

## Overview

Implementation and evaluation of the Infinite Dynamic Topic Model (iDTM). iDTM serves as an extension to Latent Dirichlet Allocation (LDA), Hierarchical Dirichlet Processes (HDP), and Dynamic Topic Models (DTM). In particular, it does not require a priori knowledge of the latent dimensionality, nor does it restrict the model to learning static topic components over time. For an in-depth review of iDTM, please checkout the `documentation/` folder.

## Installation

The `smtm` package can be installed using `pip install -e .`. All code was developed for Python 3.7. We recommend using a conda environment to manage the installation.

## Execution

To run synthetic data experiments, you can run `python scripts/model/synthetic.py <MODEL_TYPE>`, where `<MODEL_TYPE>` is one of "lda", "hdp", "dtm", or "idtm".

To see what observational data experiments look like, check out `scripts/model/observed.py`. They require access to preprocessed text data. See examples of building a dataset in `scripts/model/build_dataset.py`.
