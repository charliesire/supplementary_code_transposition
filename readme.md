The repository contains the Supplementary Material related to the article "Bayesian Calibration for prediction in a multi-output transposition context". 

The directory "codes" presents all the codes and data: 

- investigate_alphamap.py is the implementation of the algorithm to investigate $\boldsymbol{\alpha}_{\text{MAP}}.$
- bayes_lambda.py is dedicated to the MCMC sampling of $\boldsymbol{\lambda}$, for the methods No error, Uniform_error and Hierarchical MAP. 
- bayes_alpha.py is dedicated to the MCMC sample of $\boldsymbol{A}$, for the method Full-bayesian.
- full_bayes.py uses the outputs of bayes_lambda.py and bayes_alpha.py to compute te results of the Full-bayesian approach.
- embedded_discrepancy.py is dedidcated to the method Embedded discrepancy.
- utils_calib.py provides different functions useful for the implementation: Monte Carlo sampling of $\boldsymbol{\lambda}$, computation of likelihoods, normalization of $\boldsymbol{\lambda}$, etc.
- utils_plot_errors.py provides differents functions useful for plotting the results.
- run_all_strategies.ipynb uses the previous .py files to perform the different methods. 
- gp_simus.py is dedicated to the surrogate model.
- plot_summary.ipynb uses the functions of utils_plot_errors.ipynb to plot the different resutls.
- The subdirectories starting with "design_" are the results of the different methods, each one is associated with a different design $\mathbb{X}$.
- The subdirectories "measurement_points" and "surrogate_models" provides the designs and the surrogate models built for each design, respectively.
- The subdirectory "design_1" presents the results of the core of the manuscript. 


The directory "proof" provides a proof of the consistency of the estimators introduced in the manuscript.
