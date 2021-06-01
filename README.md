# Nonparametric Bayesian Stats Final Project (Spring 2021 - Johns Hopkins University)

Implementation and evaluation of the Infinite Dynamic Topic Model (iDTM). iDTM serves as an extension to Latent Dirichlet Allocation (LDA), Hierarchical Dirichlet Processes (HDP), and Dynamic Topic Models (DTM). In particular, it does not require a priori knowledge of the latent dimensionality, nor does it restrict the model to learning static topic components over time. 

In its current state, the implementation of iDTM is still quite poor -- too slow and memory inefficient to run on any reasonably sized real world text dataset. We also continue to struggle to get the model to converge, even on synthetic data. For this reason, we recommend avoiding iDTM until this message has been removed.

## Installation

The `smtm` package can be installed using `pip install -e .`. Please use Python 3.7.X.

## Execution

To run synthetic data experiments, you can run `python scripts/model/synthetic.py <MODEL_TYPE>`, where `<MODEL_TYPE>` is one of "lda", "hdp", "dtm", or "idtm".

To see what observational data experiments look like, check out `scripts/model/observed.py`. They require access to preprocessed text data. See examples of building a dataset in `scripts/model/build_dataset.py`.

## Documentation

For a background refresher and in-depth review of iDTM, please checkout the `documentation/` folder.