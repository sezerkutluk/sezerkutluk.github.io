---
layout: post
title: Gradient Based Explanations
usemathjax: true
---

Gradient Based Explanations of Machine Learning Models
--
One of the main group of methods in explainable AI is gradient based explanations. In very basic terms, the procedure is calculating the gradient of the output with respect to the input data. It is very similar to what we do when we are estimating the parameters of a model, but we calculate the gradients w.r.t. the input, not the parameters (Simonyan et al., 2014).

For the differentiable models, it can be very useful. For instance, in image classification by a huge neural network, it is hard to investigate all the parameters and make sense of it. However, by using a gradient based method, we can highlight the regions in the image where the classifier is focusing on.

Let us have a softmax classifier:

$$ p(y=i | \mathbf{x}, \boldsymbol{\omega} _ {1:K})=\frac{e ^ {\boldsymbol{\omega}_ {i} \mathbf{x}+b_i}}{ \sum ^{K}_ {j=1} e^{\boldsymbol{\omega}_ j\mathbf{x} + b_j}} $$

We can write the gradient w.r.t. the input as follows:

$$ \frac{\partial}{\partial \mathbf{x}} \frac{e ^ {\boldsymbol{\omega}_ i \mathbf{x}+b_i}}{\sum _ {j=1}^K e^{\boldsymbol{\omega}_ j\mathbf{x} + b_j}} = \frac{e^{\boldsymbol{\omega}_ i\mathbf{x}+b_i}\boldsymbol{\omega}_ i \sum_ {j=1}^K e^{\boldsymbol{\omega}_ j\mathbf{x}+b_j}-\left(\sum _ {j=1}^K e^{\boldsymbol{\omega}_ j\mathbf{x}+b_j}\boldsymbol{\omega}_ j \right) e^{\boldsymbol{\omega}_ i \mathbf{x}+b_i}}{\left( \sum_ {j=1}^K e^{\boldsymbol{\omega}_ j\mathbf{x}+b_j}\right) ^2} $$

References
--
K. Simonyan, A. Vedaldi, and A. Zisserman, *"Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"*, Workshop at International Conference on Learning Representations, 2014.
