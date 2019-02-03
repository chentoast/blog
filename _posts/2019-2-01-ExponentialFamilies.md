---
layout: post
author: Tony Chen
title: Exponential Families
---

For my first post, I thought I would start with a topic that forms the cornerstone of Bayesian Modeling, and unifies many far reaching ideas in statistics.

## What are Exponential Families?

An Exponential Family is a class of distributions that can be written in the following form:

$$p(x|\eta) = h(x)exp(\eta^{T}T(x) - A(\eta) $$

In the above equation, \\(\eta \\) is commonly referred to as the canonical parameter, T(x) is called the sufficient statistic, and \\(A(\eta) \\) is referred to as the log normalizer, as it is the term that ensures that the density integrates to 1.  It turns out that a ton of very commonly used distributions (poisson, multinomial, dirichlet, beta, etc) can be written in this form.

## Examples: the Bernoulli and Gaussian distributions

Lets start by looking at the Bernoulli distribution.  Recall that its pdf is given by:

$$p(x|\pi) = \pi^{x}(1-\pi)^{1-x} $$

$$ = exp(xlog(\pi) + (1-x)log(1-\pi)) $$

$$ = exp(x(log(\pi)- log(1-\pi)) + log(1-\pi)) $$

$$ = exp(log(\frac{\pi}{1-\pi})x + log(1-\pi)) $$

From this, we can see that the Bernoulli distribution is in the Exponential family, with natural parameter \\(log(\frac{\pi}{1-\pi})\\), sufficient statistic x, \\(h(x)=1 \\), and log normalizer \\(log(1-\pi) \\).  

For another example, consider the normal distribution:

$$p(x| \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi}\sigma} exp(\frac{(x-\mu)^{2}}{2\sigma^{2}}) $$

$$= \frac{1}{\sqrt{2\pi}\sigma}exp( \frac{x^{2} - 2x\mu + \mu^{2}}{2\sigma^{2}} $$

$$ = \frac{1}{\sqrt{2\pi}}exp( \frac{x^{2}}{2\sigma^{2}} - \frac{x\mu}{\sigma^{2}} + \frac{\mu^{2}}{2\sigma^{2}} - log(\sigma)  $$

From this, we can see that \\(h(x) = \frac{1}{\sqrt{2\pi}} \\), the canonical parameters are \\(\eta = [\frac{\mu}{\sigma^{2}}, -\frac{1}{2\sigma^{2}}] \\), the sufficient statistics are \\(T(x) = [x, x^{2}] \\), and the log normalizer is 

$$ A(\eta) = \frac{\mu^{2}}{2\sigma^{2}} - log(\sigma) = -\frac{\eta_{1}^{2}}{4\eta_{2}} - \frac{log(-2\eta_{2})}{2} $$

## The Mean Parameterization

In the above two examples, we have rewritten two distributions in the exponential family representation, expressed in terms of these natural or canonical parameters \\(\eta \\).  However, many distributions are often parameterized by its mean, such as the normal, the bernoulli, and the poisson.  We would like to express the mean \\(\mu \\) in terms of the canonical parameters.  As it turns out, if we make some assumptions about linear independence and convexity, the relationship between the two is invertible, so we can recover \\(\mu \\) from the canonical parameters.  Take the bernoulli as an example.  We have that 

$$ \eta = log(\frac{\pi}{1-\pi}) $$
$$ \implies \mathbb{E}(X) = \pi = \frac{1}{e^{\eta} - 1} $$

Here we see a familiar function: it turns out that the canonical parameter of the bernoulli is linked to the mean through the logit function.  This phenomenon lies at the heart of GeneraLized Linear Models; many times, the link function of a GLM turns out to be the canonical link between the canonical parameter and the mean.  In our case, we have rediscovered the link function of logistic regression.

## Sufficiency

At the very beginning, I mentioned that T(x) is what's known as a sufficient statistic.  But what does that mean?  The concept of sufficiency is a rich and deep subject, and I won't be able to cover all of it today, although I probably will try to in a future post.  However, I will give a quick introduction, because sufficiency and exponential families are very closely related.  The idea of sufficiency is grounded in the concept of data reduction.  How much can we reduce our data, without losing any key information?  Lets start with a couple of definitions.

__Def__.  A __statistic__ is any function of our data \\( (x_{1}, x_{2}, \ldots) \\).  The easiest example to give would be the sample mean: \\( \overline{x} = \sum_{i=1}^{N} x_{i} \\).

__Def__. A statistic T(x) is called __sufficient__ if the distribution of our data does not depend on its parameter \\( \theta \\); ie \\( p(X \vert T, \theta) = p(X \vert T) \\).

Intuitively, we can think of this as implying that \\(T(x) \\) contains all of the information we need to know about the distribution of our data.  I claim that the T(x) in the exponential family representation is a sufficient statistic. The answer to that claim lies in a theorem given by Neyman.

__Theorem__.  A statistic T(x) is sufficient if the distribution of the data can be factorized as follows:

$$ p(X \vert \theta) = h(x) g(\theta, T(x)) $$

Pf.  Some other time.

When we look at how the exponential family is written, we can see that

$$p(x|\eta) = h(x)exp(\eta^{T}T(x) - A(\eta) $$

$$ = h(x)g(\eta, T(x)) $$

And so we have that \\(T(x) \\) is a sufficient statistic.  T(x) will play a very large role in the next inference for exponential families, which will up soon

## Moments

An interesting property of \\(A(\eta) \\) is that it generates the moments of the sufficient statistic \\(T(x) \\). In that sense, \\(A(\eta) \\) acts like a moment generating function of sorts.  We have that the normalizer is written as

$$A(\eta) = log \int h(x)exp(\eta^{T}T(x))dx $$

Take the derivative with respect to \\(\eta \\):

$$\frac{\partial }{\partial \eta}A(\eta) = \frac{1}{\int h(x)exp(\eta^{T}T(x))} * \int h(x)T(x)exp(\eta^{T}T(x)) $$

Here, we've actually made some assumptions, by interchanging the derivative and integral.  We can't do this in general, but it turns out to be ok in this scenario.

$$ = \frac{\int h(x)T(x)exp(\eta^{T}T(x))}{exp(A(\eta))} $$

$$ = \int T(x)h(x)exp(\eta^{T}T(x) - A(\eta)) = \mathbb{E}(T(x)) $$

If we were to take the second partials, we would get

$$\frac{\partial^{2}}{\partial \eta^{2}}A(\eta) = \int T(x)h(x)[T(x) - \frac{\partial}{\partial \eta}A(\eta)]^{T}exp(\eta^{T}T(x) - A(\eta)) $$

Where again we have interchanged differentiation and integration.

$$ = \int T(x)h(x)[T(x) - \mathbb{E}(T(x))]exp(\eta^{T}T(x) - A(\eta)) $$

$$ = \mathbb{E}(T(x)T(x)^{T} - T(x)\mathbb{E}(T(x))^{T} ) $$

$$ = \mathbb{E}(T(x)T(x)^{T}) - \mathbb{E}(T(x))\mathbb{E}(T(x))^{T} = Var(T(x)) $$

I guess since the second derivative is the variance instead of the second moment, it should technically be called a cumulant generating function instead.  This property of exponential families was always one of the most bizzare to me, since I always found it weird how some throwaway normalizing term could generate moments like that.

## Inference

Arguably, the most important property of exponential families are how nicely they play with inference, and are a big reason why they are studied so much.  Lets start by looking at a frequentist MLE for the natural parameters.  First, lets look at the distribution of the data.

$$p(\mathbf{x} |\eta) = \prod_{i=1}^{N} h(x_{i}) = \prod_{i=1}^{N}h(x_{i}) exp(\eta^{T}(\sum_{i=1}^{N}T(x_{i})) - NA(\eta)) $$

Thus, we can see that the joint distribution of our data, is also an exponential family distribution, which is nice property 1.  Lets go ahead and take the log of this, to derive our log likelihood

$$ l(\eta) = \sum_{i=1}^{N} h(x_{i}) + \eta^{T}\sum_{i=1}^{N}T(x_{i}) - NA(\eta) $$

Then, take the derivative with respect to eta and set equal to 0:

$$ \frac{\partial}{\partial \eta}l(\eta) = 0 = \sum_{i=1}^{N}T(x_{i}) - N\frac{\partial}{\partial \eta}A(\eta) $$

$$ \implies \frac{\partial }{\partial \eta}A(\eta) = \mathbb{E}(T(x)) = \frac{1}{N}\sum_{i=1}^{N}T(x_{i}) $$

Thus, we have that the maximum likelihood estimate for \\(\eta \\) is simply the one that equations the expectation of the sufficient statistic with the sample mean of the sufficient statistic.  For distributions such as the bernoulli or poisson, where the sufficient statistic is simply x itself, this implies that the MLE of the parameter is simply the sample mean.

And finally, saving the best for the last, we have the property of conjugacy, which is a wholly Bayesian concept.  The idea is that essentially, if you have that your data is distributed according to an exponential family with parameter \\(\eta \\), then you can always (in theory) find another exponential family prior for \\(\eta \\) such that the posterior of \\(\eta \\) is another exponential family.  The argument goes like this:  let your data have some exponential family representation with natural parameters \\(\eta \\).

$$p(\mathbf{x} | \eta) = \prod_{i=1}^{N} h(x)exp(\eta^{T}\sum_{i=1}^{N}T(x) - NA(\eta)) $$

Then, define a prior for \\(\eta \\) with two hyperparameters \\(\alpha, \beta \\) to be

$$p(\eta | \alpha, \beta) = h(\eta)exp(\alpha\eta - \beta(A(\eta)) - A(\alpha, \beta)) $$

We have that \\(\alpha, \beta \\) form the natural parameters, and \\(\eta, -A(\eta) \\) are our sufficient statistics, and the whole thing is an exponential family.  Lets go ahead and compute the posterior.  


$$p(\eta| \mathbf{x}, \alpha, \beta) \propto p(\mathbf{x} | \eta) * p(\eta | \alpha, \beta) \propto exp(\eta(\alpha + \sum_{i=1}^{N}T(x_{i})) - A(\eta)(\beta + N) - A(\alpha, \beta)) $$

Note that this is also an exponential family, with updated parameters \\(\hat{\alpha} = \alpha + \sum_{i=1}^{N}T(x_{i}), \hat{\beta} = \beta + N \\)

This is an extremely nice property, since it allows us to define priors such that we have an analytic form for the posterior.  Thus, we don't have to resort to approximate inference of sampling methods, and derive the posterior manually.  However, note that there are some limitations.  Deriving the exponential family prior required that we specify two hyperparameters (or more generally, dim(\\(\eta \\)) + 1 hyperparameters) and as such, we can't just go around and tossing random exponential family distributions together and expect a closed form posterior.  However, many applications of bayesian statistics can be formulated in terms of conjugate priors, thus saving a lot of time and energy.  Some examples include Bayesian linear regression, Beta Bernoulli model, and more.

## Conclusion

Exponential families are an incredibly versatile family of distributions that have some extremely nice theoretical properties, foremost among them the property of conjugacy, which allows for Bayesian models to be formulated such that the posterior has an analytic solution.  However, there are so many other aspects to them that I haven't addressed in the notebook.  Exponential families are definitely worth their weight in gold in terms of output per hour spent, probably due to their weird tendency to unite various branches of statistics in their study.  For example, the study of Exponential Families in full gives rise to Hypothesis Testing, Maximum Entropy, Unbiased and Efficient estimators, and so much more. As such, I would highly recommend anyone reading this to familiarize themselves with Exponential Families in full.  This notebook was derived mostly from David Blei's notes at Princeton, and Michael Jordan's textbook _An Introduction to Probabilistic Graphical Models_.
