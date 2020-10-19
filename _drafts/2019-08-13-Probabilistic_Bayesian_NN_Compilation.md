---
layout: post
title: Probabilistic / Bayesian (Deep) Neural Networks
usemathjax: true
---

This post is going to be a compilation of the probabilistic and/or Bayesian treatment of neural networks.

A conventional neural network, deep or not, find the maximum likelihood of the parameters by backpropagating the training error and updating the parameters by gradient descent. However, this is a point estimate, which does not take into account the uncertainty of these estimations, and this is one of the main reasons of overfitting in neural networks.

One thing that we can use to avoid overfitting is regularization, like the L1 penalty, which is equal to placing priors on the parameters in the "probability lingo". In this case, we do the maximum a priori estimation, but again, this gives us a point estimates even if our "prior beliefs" about the parameters are considered. Still not Bayesian enough.

To go full Bayesian, we need to model, or measure, the uncertainty. This is performed by estimating the posterior distributions of the parameters. In this case, we encounter scalability issues easily since we need to sum or integrate over the possible values of parameters, which goes insane easily.

In Bayesian inference, there are two main approaches for these computation issues, which are sampling based, stochastic approximation methods, and variational approximation methods which are deterministic optimization techniques.

Applying these approximate inference methods to neural networks is one of the popular research topics in the machine (or deep) learning world these days.

In this post, I aim to compile the papers, books, software, and blog posts I find useful for probabilistic and Bayesian methods for neural networks. As my other posts, this is a "note to the self", but I hope you find something to take away, too.

Papers
------
- "[Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks]()" by José Miguel Hernández-Lobato and Ryan P. Adams.
