# time-dependent-estimation

This repository contains experiments on the estimation of time-dependent sensitivities /
factor loadings in factor models. Random factor models with time-dependent sensitivities
are created and the performance of various estimation models are compared. The models
implemented include
* Constant OLS linear regression
* Rolling OLS linear regression
* Exponentially weighted OLS linear regression
* Sensitivities modeled as random walk, estimated with a Kalman filter (henceforth "Kalman")
* Sensitivities modeled as a local trend model, estimated with a Kalman filter (henceforth "Kalman local trend").

Check the slides "Slides-TimeDependentEstimation.pdf" for more information.

Inspired by the paper:
Bentz: Quantitative Equity Investment Management with Time-Varying Factor Sensitivities, in Dunis, Laws, Naim: Applied quantitative methods for trading and investment.

