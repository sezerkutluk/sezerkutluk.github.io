---
layout: post
title: "Implicit Acceleration via Overparameterization"
usemathjax: true
---

In machine learning optimization and linear network literature, many works have shown that overparameterization can change optimization dynamics. In this short post, we will look into a simple example by Arora et al. (2018), and see how a small trick helps accelerate gradient descent by implicitly combining adaptive learning rate and past gradient trajectories as in momentum.

Let's look into linear regression with $$\ell_p$$ loss:

$$
\begin{aligned}
L(\mathbf{w})=\mathbb{E}_{(\mathbf{x},y) \sim S} \left[\frac{1}{p}(\mathbf{x}^\top \mathbf{w}-y)^p\right]
\end{aligned}
$$

Here $$\mathbf{x} \in \mathbb{R}^d$$ is the input vector, $$y \in \mathbb{R}$$ is the output which is a scalar, $$S$$ is the training set, $$\mathbf{w} \in \mathbb{R}^d$$ is the parameter vector. 

Something very cool happens when we write $$\mathbf{w}_1w_2$$ instead of $$\mathbf{w}$$, where $$\mathbf{w}_1 \in \mathbb{R}^d$$ and $$w_2 \in \mathbb{R}$$. So the optimization objective becomes:

$$
\begin{aligned}
L(\mathbf{w}_1, w_2)=\mathbb{E}_{(\mathbf{x},y) \sim S} \left[\frac{1}{p}(\mathbf{x}^\top \mathbf{w}_1 w_2-y)^p\right]
\end{aligned}
$$

If we write the gradients of the objective functions of $$L(\mathbf{w})$$ and $$L(\mathbf{w}_1, w_2)$$ with respect to $$\mathbf{w}$$ and $$\mathbf{w}_1$$ and $$w_2$$: 

$$
\begin{aligned}
\nabla_\mathbf{w}&=\mathbb{E}_{(\mathbf{x},y)\sim S} \left[(\mathbf{x}^\top\mathbf{w}-y)^{p-1}\mathbf{x} \right]\\

\nabla_{\mathbf{w}_1}&=\mathbb{E}_{(\mathbf{x},y)\sim S} \left[(\mathbf{x}^\top\mathbf{w}_1 w_2-y)^{p-1}w_2\mathbf{x} \right]=w_2 \nabla _\mathbf{w}\\

\nabla_{w_2}&=\mathbb{E}_{(\mathbf{x},y)\sim S} \left[(\mathbf{x}^\top\mathbf{w}_1 w_2-y)^{p-1}\mathbf{w}_1^\top\mathbf{x} \right]
\end{aligned}
$$

With $$\eta$$ as the learning rate, we can write the gradient descent update rule as follows:

$$
\begin{aligned}
\mathbf{w}_1^{(t+1)} &\leftarrow \mathbf{w}_1^{(t)}-\eta\nabla_{\mathbf{w}_1^{(t)}}\\

w_2^{(t+1)} &\leftarrow w_2^{(t)}-\eta\nabla_{w_2^{(t)}}
\end{aligned}
$$

Since $$\mathbf{w}=\mathbf{w}_1w_2$$,
$$
\begin{aligned}
\mathbf{w}^{(t+1)}&=\mathbf{w}_1^{(t+1)} w_2^{(t+1)}\\
&\leftarrow (\mathbf{w}_1^{(t)}-\eta\nabla_{\mathbf{w}_1^{(t)}})(w_2^{(t)}-\eta\nabla_{w_2^{(t)}})\\
&=\mathbf{w}_1^{(t)}w_2^{(t)}-\eta w_2^{(t)} \nabla_{\mathbf{w}_1^{(t)}}-\eta \nabla_{w_2^{(t)}}\mathbf{w}_1^{(t)}+\eta ^2\nabla_{\mathbf{w}_1^{(t)}}\nabla_{w_2^{(t)}}
\end{aligned}
$$

Because $$\eta$$ is chosen as a small number, we can drop the quadratic term, $$\eta ^2\nabla_{\mathbf{w}_1^{(t)}}\nabla_{w_2^{(t)}}$$. With $$\nabla_{\mathbf{w}_1} = w_2 \nabla_{\mathbf{w}}$$ and $$\mathbf{w}_1=\mathbf{w}/w_2$$, this becomes:

$$
\begin{aligned}
\mathbf{w}^{(t+1)} &\leftarrow \mathbf{w}^{(t)}-\eta w_2^{(t)}w_2^{(t)}\nabla_{\mathbf{w}^{(t)}}-\eta \nabla_{w_2^{(t)}}\mathbf{w}^{(t)}/w_2^{(t)}\\
&=\mathbf{w}^{(t)}-\eta (w_2^{(t)})^2\nabla_{\mathbf{w}^{(t)}}-\eta (w_2^{(t)})^{-1}\nabla_{w_2^{(t)}}\mathbf{w}^{(t)}
\end{aligned}
$$

We can define $$p^{(t)}=\eta (w_2^{(t)})^2 \in \mathbb{R}$$ and $$\gamma^{(t)} = \eta (w_2^{(t)})^{-1}\nabla_{w_2^{(t)}} \in \mathbb{R}$$:

$$
\begin{aligned}
\mathbf{w}^{(t+1)} &\leftarrow \mathbf{w}^{(t)}-p^{(t)}\nabla_{\mathbf{w}^{(t)}}-\gamma^{(t)}\mathbf{w}^{(t)}\\
&=(1-\gamma^{(t)})\mathbf{w}^{(t)}-p^{(t)}\nabla_{\mathbf{w}^{(t)}}
\end{aligned}
$$

If we initialize the parameters $$\mathbf{w}_1$$ and $$w_2$$ near zero, $$\mathbf{w}$$ is near zero as well. Therefore, the parameters along iterations are weighted combinations of past gradients.

Let's write down a few steps:

$$
\begin{aligned}
\mathbf{w}^{(1)} & \approx 0\\

\mathbf{w}^{(2)} & \approx (1-\gamma^{(1)})\mathbf{w}^{(1)}-p^{(1)}\nabla_{\mathbf{w}^{(1)}}\\
&= -p^{(1)}\nabla_{\mathbf{w}^{(1)}}\\

\mathbf{w}^{(3)} & \approx (1-\gamma^{(2)})\mathbf{w}^{(2)}-p^{(2)}\nabla_{\mathbf{w}^{(2)}}\\
&=(1-\gamma^{(2)})(-p^{(1)}\nabla_{\mathbf{w}^{(1)}})-p^{(2)}\nabla_{\mathbf{w}^{(2)}}\\

\mathbf{w}^{(4)} & \approx (1-\gamma^{(3)})\mathbf{w}^{(3)}-p^{(3)}\nabla_{\mathbf{w}^{(3)}}\\
&=(1-\gamma^{(3)})\left[ (1-\gamma^{(2)})(-p^{(1)}\nabla_{\mathbf{w}^{(1)}})-p^{(2)}\nabla_{\mathbf{w}^{(2)}} \right] -p^{(3)}\nabla_{\mathbf{w}^{(3)}}\\
&= -(1-\gamma^{(3)})(1-\gamma^{(2)})p^{(1)}\nabla_{\mathbf{w}^{(1)}}-(1-\gamma^{(3)})p^{(2)}\nabla_{\mathbf{w}^{(2)}} -p^{(3)}\nabla_{\mathbf{w}^{(3)}}
\end{aligned}
$$

From these, we can write the following:
$$
\begin{aligned}
\mathbf{w}^{(t)} & \approx \sum_{\tau=1}^{t-1} \left( -p^{(\tau)}\nabla_{\mathbf{w}^{(\tau)}} \prod _{j={\tau+1}}^{t-1}(1-\gamma^{(j)}) \right)\\
&=\sum_{\tau=1}^{t-1} \left( -p^{(\tau)} \prod _{j={\tau+1}}^{t-1}(1-\gamma^{(j)}) \right) \nabla_{\mathbf{w}^{(\tau)}}
\end{aligned}
$$

If we rewrite the update rule with this:
$$
\begin{aligned}
\mathbf{w}^{(t+1)} &\leftarrow \mathbf{w}^{(t)}-p^{(t)}\nabla_{\mathbf{w}^{(t)}}-\gamma^{(t)}\mathbf{w}^{(t)}\\
&=\mathbf{w}^{(t)}-p^{(t)}\nabla_{\mathbf{w}^{(t)}}-\gamma^{(t)} \sum_{\tau=1}^{t-1} \left( -p^{(\tau)} \prod _{j={\tau+1}}^{t-1}(1-\gamma^{(j)}) \right) \nabla_{\mathbf{w}^{(\tau)}}\\
&=\mathbf{w}^{(t)}-p^{(t)}\nabla_{\mathbf{w}^{(t)}}- \sum_{\tau=1}^{t-1} \gamma^{(t)} \left( -p^{(\tau)} \prod _{j={\tau+1}}^{t-1}(1-\gamma^{(j)}) \right) \nabla_{\mathbf{w}^{(\tau)}}
\end{aligned}
$$

And we can simplify this by defining $$\mu^{(t,\tau)}=-\gamma^{(t)}p^{(\tau)} \prod _{j={\tau+1}}^{t-1}(1-\gamma^{(j)})$$:

$$
\begin{aligned}
\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)}-p^{(t)} \nabla _{\mathbf{w}^{(t)}}-\sum_{\tau=1}^{t-1} \mu^{(t,\tau)}\nabla_{\mathbf{w}^{(\tau)}}
\end{aligned}
$$

So, since $$p^{(t)}$$ depends on $$t$$, the effective learning rate scales adaptively with $$w_2$$. Because $$w_2$$ is initialized near zero, updates push it away from zero, growing its magnitude $$\vert w_2 \vert$$ which results in larger step sizes as optimization progresses. Meanwhile, the momentum term accumulates past gradients, dynamically adjusting its influence based on the value of $$w_2$$ instead of relying on a fixed hyperparameter.

Although what we do is plain gradient descent, this scalar overparameterization and near-zero initialization demonstrate implicit acceleration through the combination of an adaptive effective learning rate and an adaptive momentum.

Cool, huh?

References
--
Sanjeev Arora, Nadav Cohen, Elad Hazan, "On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization", ICML 2018.