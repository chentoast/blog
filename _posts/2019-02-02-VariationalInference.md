---
layout: post
author: Tony Chen
title: Variational Inference
---

The biggest challenge of Bayesian Statistics lies in inference, where one combines the prior and likelihood to get a posterior.  Most of the time, the posterior will not have an analytic form and so we have to rely on sampling methods such as MCMC.  However, MCMC can be very computationally expensive, and is often a hassle to implement and re-implement every time you change your model.  As such, in the last couple of years, Variational Inference has become a viable and scalable alternative Bayesian models.

## Introduction and the Problem

Recall Bayes rule, which gives us the procedure for drawing inferences from data:

$$p(\theta \vert d) = \frac{p(d \vert \theta)p(\theta)}{p(d)} $$

The term in the denominator is often referred to as the marginal likelihood, or the evidence, and is what causes inference to be intractable.  This is because to calculate \\(p(d) \\), we have to integrate over all of our parameters:  \\(p(d) = \int_{\theta \in \Theta} p(d \vert \theta)p(\theta) \\).  Now if \\(\theta \\) is one dimensional this could work, but as the dimensionality of our parameter space increases, the computational power needed also increases exponentially, making this an unfeasible numerical computation.

Variational Inference, or VI, belongs to a class of inference algorithms referred to as approximate inference, which is designed to get around this problem in a different way from MCMC.  The idea is that instead of drawing samples from the exact posterior, such as MCMC, you trade off a little bit of bias in exchange for computational tractability.  Most of these algorithms work by transforming inference from a sampling problem, to an optimization problem, which is typically an easier class of problems to deal with.  VI is just one of the approximate sampling algorithms, but it has been the one that has seen the most success and has been the most widely adopted.

## Statement of the objective

Like previously stated, VI transforms inference from a sampling problem into a optimization problem.  It does this by approximating the posterior \\(p(\theta \vert x) \\) with some class of parametric distributions Q.  Then, we want to find the \\(q(\theta) \in Q \\) such that our variational distribution q is "close" to the posterior.  Typically, we use the KL Divergence as a measure for the closeness of two distributions.  Recall that for two distributions p, q, the KL divergence is defined as:

$$KL(p || q) = \int p(x) log(\frac{p(x)}{q(x)})dx $$

Our objective then becomes

$$\text{Find } q^{*}(\theta) = \underset{q \in Q}{argmin} KL(q(\theta) || p(\theta | x)) $$

And then once we have found \\(q^{*}(\theta) \\), we we use that as our approximate posterior.  However, we run into crucial problem when we actually try and solve this.  Lets expand out the definition of the KL divergence:

$$KL(q(\theta) || p(\theta | x)) = \int q(\theta) log(\frac{q(\theta)}{p(\theta)}) $$

$$ = \mathbb{E}_{q}(log(\frac{q(\theta)}{p(\theta)}) $$

$$ = \mathbb{E}_{q}(log(q(\theta))) - \mathbb{E}_{q}(log(p(\theta| x))) $$

$$ = \mathbb{E}_{q}(log(q(\theta))) - \mathbb{E}_{q}(log(p(\theta, x))) - \mathbb{E}_{q}(log(p(x))) $$

We can see that the evil term, \\(p(x) \\) pops up again in this term.  Thus we have that the KL divergence is intractable and useless to us.  

What can we do from here?  The way forward is to note that since \\(p(x) \\) is a constant with respect to theta, we can drop it from the whole term and minimize the KL divergence up to an additive constant.

## The ELBO

Define the Evidence Lower Bound (ELBO) as

$$ELBO(q) = \mathbb{E}(log(p(\theta,x))) - \mathbb{E}(log(q(\theta))) $$

Where the expectations are taken with respect to q.  We can see that the EBLO is simply the negative KL divergence plus some additive constant.  Thus, maximizing the ELBO will minimize the KL divergence, which will then give us the variational distribution that we want.  Furthermore, because we have dropped the intractable constant, we can actually compute the ELBO.  

In addition to being our optimizational objective, the ELBO also has the nice property of providing a lower bound for the evidence (hence the name).  To see this, observe that

$$log(p(x)) = log \int p(x\vert \theta)p(\theta)dx = log \int p(x, \theta)dx $$

$$ = log \int p(x, \theta) \frac{q(\theta)}{q(\theta)} $$

$$ = log \int q(\theta) \frac{p(x, \theta)}{q(\theta)} $$

$$ = log(\mathbb{E})(\frac{p(x, \theta)}{q(\theta)} \leq \mathbb{E}(log(x, \theta))) - \mathbb{E}(log(q(\theta))) $$

Here, the above inequality followed from application of Jensen's inequality.


