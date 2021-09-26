---
layout: post
title: Variational Inference Part 2&#58; Complete Conditionals and SVI
author: Tony Chen
---


Last time, I gave an introduction to Variational Inference, covering things such as the variational objective and the coordinate ascent updates.  Today, I'll begin by covering classic mean field VI in full generality, by talking about the update equations in terms of exponential families, and then, I'll go ahead and introduce Stochastic Variational Inference, which is a more efficient form of VI that can scale to larger datasets.  I'll conclude by providing an example in Latent Dirichlet Allocation, one of the most influential probabilisitic models of the last couple of decades.

1. [Brief Review](#intro)
2. [Complete Conditionals and the Exponential Family](#expo)
3. [The Natural Gradient](#au natural)
4. [Stochastic Variational Inference](#svi)
5. [Example: Latent Dirichlet Allocation](#lda)

## Brief Review <a name="intro"></a>

Recall that Variational Inference turns inference from a sampling problem to an optimization problem; it posits special variational distributions \\(q(\theta \vert \xi) \\) for each parameter \\(\theta \\), and finds the optimal variational parameters \\(\xi \\) that minimize the KL Divergence between the variational family and the posterior:  \\(\underset{\xi}{argmin} \; KL(q(\theta) \vert \vert p(\theta \vert x)) \\).  However, because the KL Divergence is not tractable, due to the \\(p(\theta \vert x) \\) term, we minimize the variational objective instead: 

$$ ELBO(q) = \mathbb{E}_{q}(\log\, p(\mathbf{x}, \theta)) - \mathbb{E}_{q}(\log\, q(\theta)) $$

In mean field variational inference, we specify the additional constraint that the variational distributions must be independent; thus \\(q(\theta) = \prod_{i} q(\theta_{i}) \\).
Then, to maximize the variational objective, we perform coordinate ascent, by optimizing one parameter while holding all other variational parameters fixed.  By rewriting the ELBO,
we see that 

$$ELBO(q) = \mathbb{E}[\log\, p(\theta, \mathbf{x})] - \mathbb{E}[\log\, q(\theta)] $$

$$ = \mathbb{E}[\log\, p(\theta_{-j}, \theta_{j}, \mathbf{x})] - \mathbb{E}_{j}[\log\, q(\theta_{j})] + const $$

Recall that the notation \\(\theta_{-j} \\) refers to every theta except for \\(\theta_{j} \\).

$$ = \mathbb{E}_{-j}[\mathbb{E}_{j}[\log\, p(\theta_{-j}, \theta_{j}, \mathbf{x} \vert  \theta_{-j})]] - \mathbb{E}_{j}[\log\, q(\theta_{j})] + const $$

Because of the mean field assumption, we have complete independence between \\(\theta_{-j} \\) and \\(\theta_{j} \\), so the condition falls out of the expectation.

$$ = \mathbb{E}_{-j}[\mathbb{E}_{j}[\log\, p(\theta_{j}, \theta_{j}, \mathbf{x}]] -  \mathbb{E}_{j}[\log\, q(\theta_{j})] + const $$

$$ = \mathbb{E}_{j}[\mathbb{E}_{-j}[\log\, p(\theta_{j}, \theta_{j}, \mathbf{x}]] -  \mathbb{E}_{j}[\log\, q(\theta_{j})] + const $$

$$ = \mathbb{E}_{j}[\log\, \exp(\mathbb{E}_{-j}[\log\, p(\theta_{j}, \theta_{-j}, \mathbf{x})])] - \mathbb{E}_{j}[\log\, q(\theta_{j})] + const $$

$$ = -\mathbb{E}_{j}[\log\, \frac{q(\theta_{j})}{\exp(\mathbb{E}_{-j}[\log\, p(\theta_{j}, \theta_{-j}, \mathbf{x})])}] + const $$

$$ = KL(q(\theta_{j}) || \exp(\mathbb{E}_{-j}[\log\, p(\theta_{j}, \theta_{-j}, \mathbf{x})])) + const $$

Thus, the ELBO is maximized when we set \\(q(\theta_{j}) \propto  \exp(\mathbb{E}\_{-j}[\log\, p(\theta_{j}, \theta_{-j}, \mathbf{x})]) \\).  Note the proportional symbol here; the
distribution on the right is not a valid probability distribution, since it is not yet normalized.


## Complete Conditionals and the Exponential Family <a name="expo"></a>

Now, lets take a step back, and try and find some general paterns in the update equations.  We can see that the updates involve taking the expectation of the joint distribution. Equivalently, we can also take the expectation of the complete conditional of theta: \\(p(\theta_{j} \vert \theta_{-j}, \mathbf{x}) \\), which is proportional to the joint.  So, a natural thing to do, is specify a general form for the complete conditionals to see if that simplifies derivations.  

One thing that we might first try, is assume that the complete conditional falls in the exponential family; that is:

\\[p(\theta_{j} \vert \theta_{-j}, \mathbf{x}) = h(\theta_{j})\exp(\eta^{T}T(\theta_{j}) - A(\eta)) \\]

Note that because this is a conditional density, the natural parameter \\(\eta \\) will be a function of x and \\(\theta_{-j} \\). Lets see what happens when we plug this into the coordinate update equation:

\\[ q(\theta_{j}) \propto \exp(\mathbb{E}\_{-j} [\log\, p(\theta_{j} \vert \theta_{-j}, \mathbf{x})]) \\]

\\[ = \exp(\mathbb{E}[\log\, h(\theta_{j})] + \mathbb{E}[\eta^{T}T(\theta_{j})] - \mathbb{E}[A(\eta)]) \\]

\\[ \propto h(\theta_{j})\exp(\mathbb{E}[\eta^{T}]T(\theta_{j})) \\]

We can see from this that the optimal variational distribution falls into the same exponential family of the complete conditional, with updated parameter \\(\mathbb{E}_{-j}[\eta^{T}] \\).  Therefore, when we do coordinate ascent, we simply update the variational parameter according to \\(\xi = \mathbb{E}[\eta^{T}] \\).  Things are already looking a lot nicer and more general!

Now, lets consider a basic extension of the model we've been working with so far.  Many probabilistic models can be formulated in terms of three things: the data, denoted by \\(\mathbf{x} \\), the global variables \\(\beta \\), and the local variables \\(z \\), which are parameters that are associated with each datapoint.  In addition, we will generally assume independence among the local variables conditioned on the global variables: that is, \\(p(z_{i}, x_{i} \vert z_{-i}, x_{-i}, \beta) = p(z_{i}, x_{i} \vert \beta) \\) for all datapoints i.  Thus, our joint will factorize as follows: \\(p(\mathbf{z}, \mathbf{x}, \beta = p(\beta) \prod_{i} p(z_{i}, x_{i} \vert \beta) \\).  We'll denote by \\(\lambda \\) the variational parameter for the global variables and \\(\phi \\) the variational parameters for the local variables.

With this specification of the general model, we can repeat the same process, and specify exponential family representations of the complete conditionals.  To start with, lets assume that the local variables conditioned on the globals are in the exponential family, with natural parameter \\(\beta \\):

\\[p(x_{i}, z_{i} \vert \beta) = h(x_{i}, z_{i})\exp(\beta^{T}T(x_{i}, z_{i}) - A(\beta)) \\]

Then, we can define the prior on \\(\beta \\) in such a way that it is conjugate to the complete conditional.  Take the sufficient statistics to be \\([\beta, -A(\beta)]^{T} \\), and let the hyperparameters be \\( [\alpha_{1}, \alpha_{2}] \\).  It follows that

\\[p(\beta) = h(\beta)\exp(\alpha[\beta, -A(\beta)]^{T} - A(\alpha)) \\]

\\[p(\beta \vert \mathbf{x}, \mathbf{z}) \propto p(\beta)\prod_{i} p(x_{i}, z_{i} \vert \beta) \\]

\\[ = h(\beta)\prod_{i}h(x_{i}, z_{i})\exp(\beta^{T}\sum_{i}T(x_{i}, z_{i}) + \beta^{T}\alpha_{1} - N*A(\beta) - A(\beta)\alpha_{2} - A(\alpha)) \\]

\\[ \propto \exp(\beta^{T}(\alpha_{1} + \sum_{i}T(x_{i}, z_{i})) - A(\beta)(\alpha_{2} + N) - A(\alpha)) \\]

We can see that the complete conditional falls into the same exponential family as the prior, with the same sufficient statistics and updated natural parameter \\([\alpha_{1} + \sum_{i}T(x_{i}, z_{i}), \alpha_{2} + N] \\).  And finally, to cap it all off, we'll assume that the complete conditional for the local variable also falls into the exponential family;

\\[p(z_{i} \vert x_{i}, \beta) = h(z_{i})\exp(\eta_{l}^{T}T(z_{i}) - A(\eta_{l})) \\]

I'll use \\(\eta_{l} \\) to refer to the natural parameter of the local complete conditional, and \\(\eta_{g} \\) to refer to the natural parameter of the global complete conditional.  It is then a natural choice to let the variational distributions be in the same family as their complete conditionals:

\\[q(\beta \vert \lambda) \propto \exp(\lambda^{T}T(\beta) - A(\lambda)) \\]

\\[q(z_{i} \vert \phi_{i}) \propto \exp(\phi_{i}^{T}T(z_{i}) - A(\phi_{i})) \\]

Now, we have that the variational updates for the local and global parameters respectively, take the form \\(\phi = \mathbb{E}[\eta_{l}] \\) and \\(\lambda = [\alpha_{1} + \sum_{i}\mathbb{E}[T(x_{i}, z_{i}), \alpha_{2} + N] \\).  Everything is now so simple!

## The Natural Gradient <a name="au natural"></a>

However, there is one big problem with classic coordinate ascent.  Because of its iterative nature, we have to do an entire sweep through the whole dataset in order to update the global variational parameters just once.  This presents a huge inefficiency, and can be improved on with stochastic optimization.  To start with, lets derive the gradient of the ELBO with respect to \\(\lambda \\), so we can apply gradient descent to it:

\\[ ELBO(\lambda) = \mathbb{E}[\log\, p(\beta \vert \mathbf{x}, \mathbf{z})] - \mathbb{E}[\log\, q(\beta)] + const  \\]

Where I've thrown out everything that doesn't depend on \\(\beta \\).

\\[  = \mathbb{E}[\log\, h(\beta) + \eta_{g}^{T}T(\beta)] - \mathbb{E}[\log\, h(\beta) + \lambda^{T}T(\beta) - A(\lambda)] + const \\]

\\[ \nabla_{\lambda}ELBO = \nabla_{\lambda}\mathbb{E}[\eta_{g}^{T}T(\beta)] - \nabla_{\lambda}\mathbb{E}[\lambda^{T}T(\beta)] +  \nabla_{\lambda}A(\lambda) \\]

By application of the identity that the expectation of \\(T(\beta) \\) is the first derivative of the log normalizer \\(A(\lambda) \\) (because we are taking this expectation with respect to \\( q(\beta) \\), this becomes

\\[ = \nabla_{\lambda}(\mathbb{E}[\eta_{g}]^{T}\nabla_{\lambda}A(\lambda)) - \nabla_{\lambda}(\lambda^{T}\nabla_{\lambda}A(\lambda)) \\]

\\[ = \mathbb{E}[\eta_{g}]^{T}\nabla_{\lambda}^{2}A(\lambda) - \nabla_{\lambda}A(\lambda) - \lambda^{T}\nabla_{\lambda}^{2}A(\lambda) + \nabla_{\lambda}A(\lambda) \\]

\\[ = \nabla_{\lambda}^{2}(\mathbb{E}[\eta_{g}]^{T} - \lambda) \\]

Analogously, the gradient with respect to \\(\phi_{i} \\) turns out to be \\(\nabla_{\phi_{i}}^{2}(\mathbb{E}[\eta_{l, i}]^{T} - \phi_{i}) \\), where we calculate \\(\eta_{l,i} \\) only using the datapoints \\(x_{i}, z_{i} \\).

There is one problem, though.  The gradient is implicitly defined in the euclidean space, where the distance metric is the traditional \\(\ell_{2} \\) norm.  However, the \\(\ell_{2} \\) norm is not a good distance metric for probability distributions: take the below gaussians as an example.

Thus, we need to define another gradient with respect to a better distance metric for probability distributions.  A natural one to use is the __symmetrized KL Divergence__

\\[D(p, q) = KL(p \vert \vert q) + KL(q \vert \vert p) \\]

So, we want to move in the direction as this new gradient, which we'll call the __natural gradient__; in other words, we want

\\[ \underset{d\xi}{argmax} f(\xi) \text{ such that } D(q(\xi), q(\xi + d\xi)) < \epsilon \\]

For some sufficiently small epsilon.  It can be shown that \\( d\xi^{T}G(\xi)d\xi \approx D(q(\xi), q(\xi + d\xi)) \\) for small \\(d\xi \\), where \\(G(\xi) \\) is the fisher information matrix of \\(q(\xi) \\): \\(G(\xi) = \mathbb{E}[(\nabla_{xi} \log\, q(\xi))^{T}(\nabla_{xi}\log\,  q(\xi))] \\).  It's not hard, but involves taking a couple of taylor expansions and gets pretty tedious.  In addition, after skipping even more algebra, it can be shown that the natural gradient \\(\hat{\nabla_{\xi}} \\) can be calculated by \\(\hat{\nabla_{\xi}}f(\xi) = G^{-1}(\xi)\nabla_{\xi}f(\xi) \\).  So, as it turns out, the natural gradient has a pretty nice form, and results in a metric that is better suited to representing changes in probability distributions.

With the above equation, lets now calculate the natural gradient of the ELBO, with respect to the variational parameters.  If we calculate the fisher information matrix for the variational distributions, we can see that 

\\[G(\lambda) = \mathbb{E}[(\nabla_{\lambda}\log\,  q(\beta \vert \lambda))^{T} (\nabla_{\lambda}\log\,  q(\beta \vert \lambda))] \\]

\\[ = \mathbb{E}[(\nabla_{\lambda} (h(\beta) + \lambda^{T}T(\beta) - A(\lambda)))^{T}(\nabla_{\lambda}(h(\beta) + \lambda^{T}T(\beta) - A(\lambda)))] \\]

\\[ = \mathbb{E}[(T(\beta) - \nabla_{\lambda}A(\lambda))^{T}(T(\beta) - \nabla_{\lambda}A(\lambda))] \\]

\\[ = \mathbb{E}[(T(\beta) - \mathbb{E}[T(\beta)])^{T}(T(\beta) - \mathbb{E}[T(\beta)])] \\]

\\[ = \nabla_{\lambda}^{2}A(\lambda) \\]

Where again, we have used the exponential family identity that the gradients of the log normalizer gives us the cumulants (not moments) of the sufficient statistics.  Therefore, when we plug this into the equation for the natural gradient, we find that \\(\hat{\nabla_{\lambda}} = \mathbb{E}[\eta_{g}]^{T} - \lambda \\).  Using a similar calculation will show that \\(\hat{\nabla_{\phi_{i}}} = \mathbb{E}[\eta_{l,i}]^{T} - \phi_{i} \\) - as it turns out, our equations actually became much simpler as a result of using the natural gradient.

## Stochastic Variational Inference <a name="svi"></a>

Now, we are ready to introduce SVI.  Instead of optimizing the ELBO through coordinate ascent, we optimize the \\(\lambda \\) parameters through gradient ascent, which looks something like this:

\\[\lambda^{t} = \lambda^{t-1} + \rho\nabla_{\lambda}ELBO \\]

Here, \\(\lambda^{t} \\) refers to the value at iteration t, and \\(\rho \\) denotes the step size.  I'm going to assume that you've seen gradient ascent before, but if not, there are a ton of tutorials out there that will do a very good job of explaining it.  We have the gradients of the ELBO with respect to \\(\lambda \\), but calculating it requires iterating through the entire dataset, since \\(\eta_{g} \\) depends on all of \\(\mathbf{x}, \mathbf{z} \\).  Therefore, we are in no better of a position than just simple coordinate ascent.

The solution to this is to use a noisy estimate of the gradient; something that is cheaper to calculate, but still has expectation equal to the gradient.  Because the source of the inefficiency comes from the global parameters \\(\lambda \\), we'll still be doing coordinate ascent on the local variables, so lets go ahead and assume that we've already updated the local parameters.  We'll denote the local opimal values by \\(\phi(\lambda) \\), and define the __locally maximized ELBO__ to be \\(L(\lambda) = ELBO(\lambda, \phi(\lambda)) \\).  We can easily verify that the gradient of \\(L(\lambda) \\) is the same as the gradient of the ELBO:  

\\[\nabla_{\lambda}L(\lambda) = \nabla_{\lambda}ELBO(\lambda, \phi(\lambda)) \\]

\\[= \nabla_{\lambda}ELBO(\lambda, \phi(\lambda)) + (\nabla_{\lambda} \phi(\lambda))^{T}(\nabla_{\phi}ELBO(\lambda, \phi(\lambda))) = \nabla_{\lambda}ELBO(\lambda, \phi(\lambda)) \\]

Note that because \\(\phi(\lambda) \\) is at a local optimum, \\(\nabla_{\phi}ELBO(\lambda, \phi(\lambda)) = 0 \\).  Then, we need a random function that has expectation equal to \\(\nabla_{\lambda}L(\lambda) \\).  Consider the random function given by sampling a datapoint randomly, and repeating it N times.  Expanding out \\(L(\lambda) \\), we have that

\\[L(\lambda) = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + \sum_{i} \underset{\phi}{max} \mathbb{E}[p(x_{i}, z_{i} \vert \beta)] - \mathbb{E}[q(z_{i})] \\]

Then, let \\(I \sim \text{Discrete Uniform}(1, \ldots N) \\) be a random index, and then pretend like our dataset was made up of N repetitions of the datapoint indexed by I:

\\[L_{I}(\lambda) = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + N*\underset{\phi_{I}}{max} \mathbb{E}[p(x_{I}, z_{I} \vert \beta)] - \mathbb{E}[q(z_{I})]  \\]

I claim that the expectation of this is equal to \\(L(\lambda) \\).  To see this, we can view I as a multinomial indicator vector, with zeros in all but one entry.  Then, let \\(P(x, z \vert \beta) \\) and \\(Q(z) \\) be the vectors formed by evaluating the probability functions \\(p(x_{i}, z_{i} \vert \beta) \\) and \\(q(z_{i}) \\) at each datapoint i.  Then, we can re-write the equation as:

\\[L_{I}(\lambda) = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + N*\underset{\phi_{I}}{max} \mathbb{E}[I^{T}P(x, z \vert \beta)] - \mathbb{E}[I^{T}Q(z)] \\]

Taking the expectation with respect to I results in 

\\[ = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + N*\underset{\phi_{I}}{max} \mathbb{E}[I^{T}]\mathbb{E}[P(x, z \vert \beta)] - \mathbb{E}[I^{T}]\mathbb{E}[Q(z)] \\]

\\[ = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + N*\underset{\phi_{I}}{max} [\frac{1}{N} \ldots \frac{1}{N}]\mathbb{E}[P(x, z \vert \beta)] - [\frac{1}{N} \ldots \frac{1}{N}]\mathbb{E}[Q(z)] \\]

\\[ = \mathbb{E}[\log\, p(\beta \vert x, z)] - \mathbb{E}[q(\beta)] + \sum_{i} \underset{\phi}{max} \mathbb{E}[p(x_{i}, z_{i} \vert \beta)] - \mathbb{E}[q(z_{i})] \\]

So, we have that \\(\mathbb{E}[L_{I}(\lambda) = L(\lambda)] \\).  It then follows that \\(\mathbb{E}[\nabla_{\lambda}L_{I}(\lambda)] = \nabla_{\lambda}\mathbb{E}[L_{I}(\lambda)] = \nabla_{\lambda}L(\lambda) \\).  So now, we have our unbiased gradient estimator.

The last step is to actually calculate the noisy gradient estimator for \\(\lambda \\).  Recall that the gradient with respect to \\(\lambda \\) is given by \\(\mathbb{E}[\eta_{g}] - \lambda \\), where \\(\eta_{g} \\) is given by \\((\alpha_{1} + \sum_{i} T(x_{i}, z_{i}), \alpha_{2} + N) \\).  To apply our noisy gradient formula, take a random point \\((x_{i}, z_{i}) \\), and repeat it N times, to get our noisy gradient estimator \\(\mathbb{E}[\eta_{g}^{i}] - \lambda, \; \eta_{g}^{i} = (\alpha_{1} + N*T(x_{i}, z_{i}), \alpha_{2} + N) \\).  Then, denote by \\(\hat{\lambda} = \mathbb{E}[\eta_{g}^{i}]  \\) the new estimator for \\(\lambda \\), and plug it into the gradient ascent equation, which looks like this:

\\[ \lambda^{t} = \lambda^{t-1} + \rho*\hat{\nabla_{\lambda}}L(\lambda) \\]

\\[ = \lambda^{t-1} + \rho*(\mathbb{E}[\eta_{g}^{i}] - \lambda^{t-1}) \\]

\\[ = \lambda^{t-1} + \rho*(\hat{\lambda} - \lambda^{t-1}) \\]

\\[= (1-\rho)\lambda^{t-1} + \rho\hat{\lambda} \\]

As we can see, the SGD equation becomes a weighted sum of the previous and current estimates of \\(\lambda \\).  With all of the pieces in place, we are now ready to describe the full algorithm, which is deceptively simple given how much work it took to get there.

-  __Repeat__

	-  Sample a datapoint \\(x_{i} \\)

	- Compute its local variational parameter \\(\phi_{i} = \mathbb{E}\_{\lambda}[\eta_{li}] \\)

	- Compute the intermediate global parameter \\(\hat{\lambda} = \mathbb{E}\_{\phi}[\eta_{g}^{i}] \\)
	
	- Update the global parameter estimate: \\(\lambda^{t} = (1-\rho)\lambda^{t-1} + \rho\hat{\lambda} \\)

With each iteration, we sample a datapoint \\(x_{i} \\) randomly from the dataset, and update its local parameter through coordinate ascent:  \\(\phi_{i} = \mathbb{E}\_{\lambda}[\eta_{l}] \\).  Then, we compute the _intermediate global parameter_, pretending like \\(x_{i} \\) was replicated N times:  \\(\hat{\lambda} = \mathbb{E}\_{\phi}[\eta_{g}^{i}] \\).  And finally, we'll take a gradient descent step to update our estimate of the global parameter:  \\(\lambda = (1 - \rho)\lambda^{t-1} + \rho\hat{\lambda} \\).


## Example: Latent Dirichlet Allocation <a name="lda"></a>

To round out this notebook, I'll end with an example.  Latent Dirichlet Allocation, or LDA, is whats commonly known as a topic model: a probabilistic model for extracting themes or _topics_ from a corpus of documents.  A topic is defined as a distribution over words; for example, if our topic was _politics_, we would expect words such as white house, government, diplomacy, etc to have a high probability under this topic. We represent documents as a _bag of words_; where each word is encoded by an indicator vector, and the order of the words doesn't matter.  At a high level, LDA can be thought of as a mixture of multinomials; where each document exhibits its own proportion of topics, and each word is generated by first selcting a topic, then generating a word from that topic.  To make this more precise, heres the full generative model for LDA:

1. Draw the topic proportions \\(\beta_{k} \sim Dirichlet(\eta, \ldots \eta) \\)
2. For each document d:
	- Draw the distribution over topics \\(\theta_{d} \sim Dirichlet(\alpha, \ldots \alpha) \\)
	- For each word w:
		- Draw the topic for the word \\(z_{dn} \sim Multinomial(\theta_{d}) \\)
		- Draw the word \\(w \sim Multinomial(\beta_{z_{dn}}) \\)

I'll denote by D the number of documents, K by the number of topics, N by the number of words in document d, and V by the total size of the vocabulary.  Here the matrix \\(\beta_{K \times V} \\) refers to the probability of generating a word from a given topic, while \\(\theta_{D \times K} \\) represents the topic proportions for each document.  The idea is to specify a number of topics K, feed in a corpus consisting of D documents, and have LDA infer the topics.  To clarify my notation a bit, I'll note that I'm using \\(z_{dn} \\) as both an indicator vector and an index: when I write \\(z_{dn} = k \\), I'm referring to the event where \\(z_{dn}^{k} = 1 \\), since z is an indicator vector.  Thus, if \\(z_{dn}^{k} = 1 \\), \\(\beta_{z_{dn}} = \beta_{k} \\).   An analogous way of writing this is \\(\beta^{T}z_{w} \\).  The same principle applies to the w variables:  \\(w_{dn}^{v} = 1 \\) is the same as \\(w_{dn} = v \\).


Now, lets go ahead and derive the complete conditionals for this model.  For LDA, they turn out to all fall into the exponential family, simplifying things greatly.  We'll go ahead and start with the complete conditional of the topic matrix:

\\[P(z_{dn} = k \vert \theta_{d}, \beta, w_{dn}) \propto P(z_{dn} = k \vert \theta_{d}) * P(w_{dn} \vert z_{dn}, \beta) \\]

\\[\propto \exp(\log\, \theta_{d,k}) * \exp(\log\, \beta_{k, w_{dn}})  \\]

\\[ = \exp(\log\, \theta_{d,k} + \log\, \beta_{k, w_{dn}})  \\]

Here, I've used the exponential family representation of the multinomial and dirichlet pdfs, throwing out anything that doesnt depend on \\(\theta_{d}, \beta_{w_{dn}} \\).  We can see that this complete conditional takes the form of another multinomial, with parameter \\(\log\, \theta_{dk} + \log\, \beta_{k, z_{dn}} \\).

Lets move onto the conditional for \\(\theta_{d} \\):

\\[P(\theta_{d} \vert z_{d}) \propto P(\theta_{d})\prod_{n}P(z_{dn} \vert \theta_{d}) \\]

\\[\propto \exp(\sum_{k} (\alpha - 1)\log\, \theta_{dk})*\exp(\sum_{n} \sum_{k} z_{dn} \log\, \theta_{dk}) \\]

\\[ = \exp(\sum_{k} (\alpha - 1)\log\, \theta_{dk} + \sum_{k} \sum_{n} z_{dn} \log\, \theta_{dk}) \\]

\\[ = \exp(\sum_{k} (\alpha - 1 + \sum_{n}z_{dn})\log\, \theta_{dk}) \\]

Again, we can see that this takes the form of a dirichlet, with natural parameter \\(\alpha - 1 + \sum_{n}z_{dn} \\).  And finally, we wrap it up with the complete conditional for the global parameters:

\\[P(\beta_{k} \vert w, z) \propto P(\beta_{k})* \prod_{d}\prod_{n}P(w_{dn} \vert z_{dn}, \beta_{k}) \\]

\\[ \propto \exp(\sum_{d} \sum_{n} \sum_{v} z_{dn}^{k}w_{dn}^{v} \log\, \beta_{kv}) \exp(\sum_{v}(\eta - 1)\log\, \beta_{kv}) \\]

\\[ = \exp(\sum_{v} \sum_{d} \sum_{n} z_{dn}^{k}w_{dn}^{v} \log\, \beta_{kv} + \sum_{v}(\eta - 1)\log\, \beta_{kv}) \\]

\\[ = \exp(\sum_{v}(\eta - 1 + \sum_{d} \sum_{n} z_{dn}^{k}w_{dn}^{v})\log\, \beta_{kv}) \\]

So, lets recap.  We have that the complete conditional for \\(z_{dn} \\) is a multinomial, with natural parameter \\( \theta_{dk} + \beta_{k, z_{dn}} \\).  The complete conditional for \\(\theta_{d} \\) is a dirichlet, with natural parameter \\(\alpha - 1 + \sum_{n}z_{dn} \\).  Finally, the complete conditional for \\(\beta_{k} \\) is also a dirichlet, with natural parameter \\(\eta - 1 + \sum_{d} \sum_{n} z_{dn}^{k}w_{dn} \\).  We'll chose our variational distributions according to that, so

$$q(z_{dn}) = \text{Multinomial}(\phi_{dn}), \; q(\theta_{d}) = \text{Dirichlet}(\gamma_{d}), \; q(\beta_{k}) = \text{Dirichlet}(\lambda_{k}).$$  

Now, we'll derive the coordinate ascent updates, which as you'll recall, simply involves taking the expectation of the natural parameters.

\\[\hat{\phi_{dn}} = \mathbb{E}[\log\, \theta_{d} + \log\, \beta_{k, z_{dn}}] \\]

\\[ = \Psi(\gamma_{dk}) + \Psi(\lambda_{k,w_{dn}}) - \Psi(\sum_{v}\lambda_{kv}) - \Psi(\sum_{k}\gamma_{dk}) \\]

Here, I've used the identity that if \\(\xi \sim Dirichlet(\gamma) \\), then \\(\mathbb{E}[\log\, \xi_{k}] = \Psi(\gamma_{k}) - \Psi(\sum_{i} \gamma_{i}) \\).  Because these are the natural parameters of the optimal variational families, we'll have to transform them back to the standard parameterization:

\\[ \implies \hat{\phi_{d}} = \exp( \Psi(\gamma_{dk}) + \Psi(\lambda_{k,w_{dn}}) - \Psi(\sum_{v}\lambda_{kv}) - \Psi(\sum_{k}\gamma_{dk})) \\]

Using the same logic, and re-mapping the natural parameters back to the standard parameters, we can derive the other two expectations trivially; note that because z is an indicator vector, \\(\mathbb{E}[z_{dn}] = \phi_{dn} \\), and so

\\[\hat{\gamma_{d}} = \alpha + \sum_{n}\phi_{dn} \\]

\\[\hat{\lambda_{k}} = \eta + \sum_{d} \sum_{d} \phi_{dn}^{k}w_{dn} \\]

Now, I'll talk a little bit about the implementation. As I have now found out the hard way, there is a lot of stuff that gets glossed over, when moving from the math into the implementation.  Lets start with the representation of the words.  As the number of documents and the length of the corpus increases, it quickly becomes unviable to keep several million vectors, all of several hundred thousand dimensions, lying around.  Therefore, we need an efficient way of representing our word vectors that doesn't actually get in the way of the algorithm.  

The key to this, is noting the redundancy in the update equation for \\(\phi \\).  If you look closely, it quickly becomes obvious that for any two repeated words \\(w_{n} = w_{m} \\), the update for phi is the exact same.  Because of this, we can re-write the update for \\(\gamma_{d} \\) more concisely as \\(\alpha + \sum_{v}n_{v}\phi_{dv} \\), where \\(n_{v} \\) refers to the number of times that the word has occurred.  Therefore, note that instead of now representing \\(\phi \\) as a D by K by N matrix, we now can represent it as a D by K by V matrix.

And now, because we only really need the word frequencies, instead of the word indicator vectors themselves, we'll use a _Document Term Matrix_ to store our input for LDA, where the [d,v] entry represents the number of times word v has occurred in document D.  Now, instead of requiring D x V x N entries for the corpus, we only need D x V, simplifying things greatly.  I'll also note that for even larger data, you may want to represent the phi matrix implicity and not saving it to memory; only computing it as necessary to update the other variational parameters.  With that out of the way, heres the code:

```python
import numpy as np
import scipy.stats as ss
from scipy.special import psi

"""
phi: d x k x v
gamma: d x k
lambda: k x v
"""

def update_phi_d(gamma, expElogbeta):
    t1 = np.exp(psi(gamma) - psi(gamma.sum()))[:, np.newaxis]
    t2 = t1 * expElogbeta
    return t2/t2.sum(axis=0)

def update_gamma_d(phi_d, counts, alpha):
    return alpha + np.matmul(phi_d, counts)

def update_lambda_intermediate(phi_d, counts, D, eta):
    return eta + D * (counts * phi_d)

def step_lambda(lambda_prev, rho, lambda_hat):
    return (1 - rho) * lambda_prev + rho * lambda_hat

def step_rho(tau, kappa, t):
    return (t + tau)**(-kappa)

def main(doc_term, K, num_iter, eta, alpha, tau, kappa):
    D = doc_term.shape[0]
    V = doc_term.shape[1]
    gamma = np.repeat(1, D*K).reshape((D, K))
    lambd = ss.gamma(1, 1).rvs((K, V))
    phi = np.empty((D, K, V))
    rho = 1
    for t in range(num_iter):
        for d in range(D):
            counts = np.array(doc_term.iloc[d, :])
            gamma_d = gamma[d, :]
            phi_d = np.empty((K, V))
            
            expElogbeta = np.exp(psi(lambd) - psi(lambd.sum(axis=1))[:, np.newaxis])
            for i in range(5):
                phi_d = update_phi_d(gamma_d, expElogbeta)
                gamma_d = update_gamma_d(phi_d, counts, alpha)
            lambda_hat = update_lambda_intermediate(phi_d, counts, D, eta)
            gamma[d, :] = gamma_d
            phi[d, :, :] = phi_d
            lambd = step_lambda(lambd, rho, lambda_hat)
            rho = step_rho(tau, kappa, t)
    return (gamma, phi, lambd)
```

