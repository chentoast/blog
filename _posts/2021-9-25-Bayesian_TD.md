---
layout: post
author: Tony Chen
title: Gaussian Processes and Bayesian TD Learning
---

It's been a long break from this blog, so I thought I would make another post as a distraction from applying to jobs and grad school.
Today, we'll be talking about a common Bayesian nonparametric model called the *gaussian process*, and its application to reinforcement learning.

1. [Gaussian Proceses](#gauss)
2. [The Posterior of a Gaussian Process](#posterior)
3. [Gaussian Processes and TD Learning](#TD)
4. [Conclusion](#conclusion)

## Gaussian Proceses <a name="gauss"></a>

For simplicity, we'll start in one dimension. A *gaussian process* is a stochastic process with index set \\(\mathbb{R}\\), parameterized by two functions:
a mean function $$m: \mathbb{R} \to \mathbb{R}$$, and a positive semi-definite symmetric covariance function or kernel \\(k: \mathbb{R} \times \mathbb{R} \to \mathbb{R}^+\\).
It's common to use \\(x\\) to denote the index variable.
A gaussian process \\(G(x)\\) is uniquely characterized by the following property: for any finite collection of points $$x_1, x_2, \ldots x_n$$, the
random vector $$G(x_1), \ldots G(x_n)$$ has a multivariate normal distribution with mean $$m(x_1), \ldots m(x_n)$$ and covariance matrix $$\Sigma_{ij} = k(x_i, x_j)$$ (it should be clear why we require $$k$$ to be positive semi-definite and symmetric). Note that multivariate normal distributions also have this similar property of multivariate normal marginals, and so the gaussian process can be interpreted as an infinite dimensional generalization of the multivariate normal distribution.

Another interpretation of the gaussian process, is as a distribution over random functions. Sample paths of the gaussian process can be seen as functions $$\mathbb{R} \to \mathbb{R}; t \mapsto X(t)$$, and the continuity/smoothness properties of these sample paths depends on the choice of kernel.
Interpreting the gaussian process as a distribution over functions allows it to serve as a prior for Bayesian inference, where we feed it data, and compute a posterior distribution over functions.
This interpretation is the more applicable one for machine learning, so we'll think of the gaussian process as a distribution over functions from now on.
Note: the gaussian process is only a distribution in some sense of the word. Function spaces are way too big to even have useful sigma-algebras, let alone measures, and so its pretty hopeless to find some probability measure on even a nice space of functions, such as an Lp space. However, the gaussian process is a distribution in the sense that we are able to sample from it - and thus, compute the posterior. Posterior computations in nonparametric Bayesian statistics don't utilize Bayes rule, and rather rely on the conjugacy between the marginals of stochastic processes, as we'll see soon.

## The Posterior of a Gaussian Process <a name="posterior"></a>

Suppose we have data $$(x_1, y_1), \ldots (x_n, y_n)$$ that we think arises from a random function. We'd like to fit a gaussian process to this data for the purposes of prediction: that is, given a new point $$x^*$$, we'd like to make a guess at the corresponding $$y^*$$.

To do this, we'll assume that the data points are realizations from a random function drawn from a gaussian process with mean, covariance $$m, k$$. Because of the marginal property of a GP, we thus have that $$(y_1, y_2, \ldots y_n)$$ has a multivariate normal distribution with mean $$m(x_1), \ldots m(x_n)$$ and covariance $$\Sigma_{ij} = k(x_i, x_j).$$
Thus, we can compute the posterior $$y^* \vert x^*, x_1, \ldots x_n, y_1, \ldots y_n.$$
This is a pretty straightforward, albeit tedious calculation that exploits the conjugacy of the multivariate normal distribution.

After a bunch of matrix algebra, we arrive at the following result. Let $$K(x^*, X)$$ denote the $$1 \times n$$ vector with components $$K(x^*, X)_i = k(x^*, x_i)$$ ($$K(X, x^*)$$ will denote the corresponding $$1 \times n$$ vector).
Define the $$n \times n$$ matrix $$K(X, X)$$ similarly.

The posterior distribution is again a normal distribution:

$$y^* \vert x^*, x, y = N\left(K(x^*, X)K(X, X)^{-1}y, k(x^*, x^*) - K(x^*, X)K(X, X)^{-1}K(X, x^*)\right).$$

Note that this result is easily generalized to multiple evaluation points $$x_1^*, \ldots$$ - the posterior is just a multivariate normal instead of a normal.
In this way, we are not explicitly computing the entire posterior function (since that would be impossible), but rather the posterior function values
at a set of finite test points.

Also note that this model assumed that there was no measurement noise: the y's were exact samples from the underlying function f.
It's probably a more realistic assumption that the data is instead a nosiy signal centered at f: $$y = f(x) + \epsilon$$, where
epsilon is a multivariate normal with zero mean and some covariance matrix $$\Sigma$$. For everything that follows, we'll assume a
diagonal covariance for the error: $$\Sigma = \sigma^2 I$$, although this assumption is not strictly necessary.

In this case, the posterior becomes

$$y^* \vert x^*, x, y = N\left(K(x^*, X)(K(X, X) + \sigma^2 I)^{-1}y, k(x^*, x^*) - K(x^*, X)(K(X, X) + \sigma^2 I)^{-1}K(X, x^*)\right).$$

As a final topic, I'll talk a little bit about hyperparameter optimization.
Common kernel function choices include the *squared-exponential kernel*

$$k(x, x') = \exp - \dfrac{(x - x')^2}{2\tau},$$

and the *periodic kernel*

$$k(x, x') = \sigma \exp -\dfrac{2}{l} \sin \left( \frac{\vert x - x' \vert }{p} \right).$$

Note that these kernel functions depend on hyperparameters such as $$\sigma, \tau$$, and kernel behavior can often be quite sensitive to these choices.
Ideally, we would like to find a data driven way to set these hyperparameters, instead of fixing them to a set value.

The solution, is to optimize these hyperparameters - I'll call them $$\theta$$ - with respect to the log marginal likelihood of the gaussian process.
The marginal likelihood of the dataset $$p(y | X, \theta)$$ is obtained by marginalizing over the underlying function f:

$$p(y | X, \theta) = \int p(y | f(X)) p(f(X) | X, \theta) dy.$$

Since we assumed that $$y | X \sim N(f(X), \sigma^2 I),$$ and $$f(X) | X \sim N(m(X), k(X, X))$$ by definition of a GP, we have that this integral
is an integral over the product of two normal pdfs and is thus tractable.
After a bunch of tedious math which I'll skip, we arrive at our result:

$$\log p(y | X, \theta) = -\frac{1}{2} \left[ y^T (K_\theta(X, X) + \sigma^2 I)^{-1} y + \log \left\vert K_\theta(X, X) + \sigma^2 I\right\vert \right],$$

up to a constant.
We'll thus maximize our hyperparameters with respect to this quantity.

## Gaussian Processes and TD Learning <a name="TD"></a>

I'll wrap up this post with an application of GPs to reinforcement learning. First, here's a very quick recap of TD Learning.

Given some MDP with states $$S$$, actions $$A$$, discount factor $$\gamma$$, and reward dynamics $$R : S \times A \to \mathbb{R}$$, TD Learning proposes to bootstrap
the value function $$V: S \to \mathbb{R}$$ by applying the following "delta rule":

$$V^{t+1}(s_{t}) = V^{t}(s_{t}) + (r_t + \gamma V^{t}(s_{t+1}) - V^{t}(s_{t})).$$

The idea is that we try to adjust our estimate of $$V(s_t)$$ towards the more accurate signal $$r_t + V(s_{t+1})$$.

The above learning rule is a purely algorithmic solution to the problem. But, we can also adopt a Bayesian solution by endowing $$V$$ with
a probability distribution.
The key insight is that for the true value function and an optimal policy $$\pi$$, $$V(s) - \gamma V(s') = \mathbb{E}[R(s, \pi(s))],$$
where the expectation is with respect to the transition dynamics of the MDP.
Thus, we have an (albeit approximate) generative process for our rewards, conditioned on the value function.

Suppose we have a length $$T$$ episode, and have observed trajectories $$(s_t, a_t, r_t)$$.
We'll assume the observed rewards are noisy realizations of the true/optimal rewards, so that

$$r_t = V(s_t) - \gamma V(s_{t+1}) + \epsilon_t,$$

where $$\epsilon \sim N(0, \sigma^2 I).$$ The diagonal covariance assumption implies that the rewards are stationary, or time-invariant.
Now, we need to put a prior over the value function. As you may have guessed, we'll use a GP:

$$V \sim MVN(0, K(S, S)).$$

For simplicity, we take the mean function to be zero.
The choice of kernel probably heavily depends on what environment we are dealing with; for example,
in a spatial gridworld task, we may choose to use a kernel that reflects euclidian distances, etc.

Define the matrix $$(T-1) \times T$$ matrix $$H$$ by

$$H = \begin{bmatrix}
1 & -\gamma & 0 & \ldots & \\
0 & 1 & -\gamma & \ldots & \\
0 & \ldots & 1 & -\gamma & \\
\end{bmatrix}.$$

It should be pretty clear that $$r = HV + \epsilon,$$ where $$V = [V(s_1), \ldots V(s_t)]^T.$$
As a consequence, we have that

$$r \sim N(HV, H K(S, S) H^T + \sigma^2 I).$$

Again, using the wonderful powers of matrix algebra and multivariate normal distributions, we can easily find the posterior of $$V$$
in an analogous fashion as the derivation for the posterior of the GP:

$$V(s_t) | r \sim N(\mu(s_t), \tau(s_t))$$

where

$$\mu(s_t) = K(s_t, S)H^T (HK(S, S) H^T + \sigma^2I)^{-1}r,$$

$$\tau(s_t) = k(s_t, s_t) - K(s_t, S)^TH^T (HK(S, S) H^T + \sigma^2I)^{-1} HK(S, s_t).$$

The similarities between this posterior and the GP posterior should be noticable.

And, in similar fashion, we can perform marginal likelihood optimization by marginalizing out the value function $$V$$:

$$\log p(r \vert \theta) = \int p(r \vert V, \theta) p(V \vert \theta) dV.$$

## Conclusion <a name="conclusion"></a>

Gaussian processes are incredibly powerful models, but their most interesting property is that they happen to be useful for
a bunch of different things. From regression, to classification, to latent variable models and dimensionality reduction, and
even for hyperparameter optimization, GPs can do it all. They're also quite heavily studied in ML theory - their connections
to neural networks and reproducing kernel hilbert spaces (RKHS) make them useful theoretical tools.

In this post, I've only just barely scraped the surface of a very large literature on GPs and their applications. One of the main
problems behind GPs is their limited scalability - computing the posterior requires a large matrix inversion, which scales
cubically with the number of datapoints. A significant body of research is dedicated to leveraging sparsity or low rank approximations
to speed up computation.

GPs are just one family of models in a whole zoo of Bayesian nonparametric oddities - I'm hoping that in the future, we can explore some
more of these models together.