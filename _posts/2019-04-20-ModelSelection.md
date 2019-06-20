---
layout: post
title: Model Selection, Bayes Factors, and the Bridgesampling Algorithm
author: Tony Chen 
---

Model selection is a fundamental topic in statistics, that unfortunately seems
to be often glossed over and neglected.  In general, the model-building process
does not stop after we finish running our model - in order to ensure the
validity and accuracy of our model, we must utilize graphical methods, validation
checks, and model fit statistics.

The problem of model selection concerns the situation in which we want to select
one model out of a number of competing models, presumably for the purposes of
better prediction or inference.  In these cases, the theory of model selection
can provide a quantitative measure of how to select a given model, based many
different factors.  In particular, I'll be focusing mainly on Bayesian model
selection, which takes into account the uncertainty of the model specification,
is more robust against overfitting, and allows for a mathematical specification
of occam's razor.  However, it should be noted that Bayesian Model Selection is
not the end all be all; Bayesian and non-Bayesian model selection techniques can
and should be used together many of the times when validating your fitted model.


1. [The Bayesian Approach to Model Selection](#ms)
2. [Bayes Factors](#bf)
3. [The Bayesian Occam's Razor](#or)
4. [Bayes Factors vs Likelihood Ratios](#lr)
5. [The Bridgesampling Algorithm](#bs)

<a name="ms"></a>
## The Bayesian Approach to Model Selection 

Before we start with talking about model selection, we first have to define what
a model is: 

__Def__. A __Statistical Model__, denoted by \\(\mathcal{M} = \\{ p(x ; \theta),
\theta \in \Theta \\} \\), is a set of probability distributions.  We refer to
\\(\Theta \\) as the __parameter space__, and usually take \\(\Theta =
\mathbb{R}^{d} \\).

One example might be \\(p(x ; \theta) = \text{Normal}(\mu, 1) \\); the
collection of all Normal distributions with variance 1, where \\(\mu \\) is
unknown.  Almost always, a non-trivial statistical model will contain unknown
parameters that must be inferred - this is the process of "fitting" the model.

Then, lets recall our good friend, Bayes' Rule:

\\[P(\theta \vert D) = \frac{P(D \vert \theta)P(\theta)}{P(D)} \\]

As always, \\(\theta \\) denotes our parameters, and D denotes the observed data.
However, underlying this entire equation is an implicit assumption.  What
exactly \\(\theta \\) represents is dependent on the model we choose, as
different models might induce different parameters with different
interpretations.  Therefore, everything we see in the above equation, is
actually conditioned on us choosing model M.  In that case, we should rewrite
Bayes' Rule as follows:

\\[ P(\theta \vert D, \mathcal{M}) = \frac{P(D \vert \theta,
\mathcal{M})P(\theta \vert \mathcal{M})}{P(D \vert \mathcal{M})} \\]

For now, we'll just take a look at the \\(P(D \vert \mathcal{M} \\) term.  This
is referred to as the _model evidence_, and provides a measure of how likely our
data is under a particular model.  The model evidence is obtained by
marginalizing over all possible values of the parameters: \\(p(D \vert
\mathcal{M}) = \int_{\Theta} p(D \vert \theta, \mathcal{M})p(\theta \vert
\mathcal{M})  \\), and is the main workhorse behind the theory of Bayesian model selection.

<a name="bf"></a>
## Bayes Factors 

Ok, so we've seen a way to quantify the fit of a single Bayesian model.  But
what if we have a set of models \\(\\{M1, M2, \ldots Mk \\} \\), and we'd like
to compare pairs of models to see which ones we should use?  In that case, we
again return to Bayes' rule to calculate the posterior probability of each
model, given the data.

\\[p(Mi \vert D) = \frac{P(D \vert Mi)P(Mi)}{P(D)} \\]

So, we specify a prior over our models, and we again apply Bayes rule.  This
very natural application of Bayes' rule thus leads to the definition of the
Bayes Factor:

__Def__.  We define the __Bayes Factor__ of two models M1 and M2 to be the ratio
of their corresponding model evidence 

\\[\frac{p(D \vert M1)}{p(D \vert M2)}. \\]

From this, we can see that the posterior odds ratio of two models is simply the
product of the Bayes Factor and the prior model odds ratio:

\\[\frac{p(M1 \vert D)}{p(M2 \vert D)} = \frac{p(D \vert M1)}{p(D \vert M2)} *
\frac{p(M1)}{p(M2)} \\]

Because the Bayes' Factor is a ratio, it can be interpreted as such, with
a Bayes factor larger than 1 indicating evidence for model 1, and a 
Bayes factor less than 1 indicating more evidence for model 2.

So, model selection between two models in a Bayesian framework simply involves
calculating the Bayes Factor, and choosing whichever model the Bayes Factor
favors.  Using the Bayes factor instead of the posterior odds ratio is nice
because it allows us to avoid specifying a prior on the models themselves, which
could end up being extremely subjective, much more subjective than the priors on
the parameters.

<a name="or"></a>
## Occam's Razor 

When choosing a model out of multiple candidate models, we would like to not
only look at how well each model fits, but also consider their complexity.  The
ideal model would strike a nice middle ground between fitting the data well and
making few assumptions (although whether the "simpler is better" approach is
actually the best way to go is up for debate).  This is
precisely what Occam's Razor states: the principle of not making any more
assumptions than the bare minimum necessary.  At a first glance, the Bayes
Factor does seem very vulnerable to overfitting, but we shall see in this
section that the Bayes Factor actually has a built in mechanism for penalizing
more complex models.

Lets take a look again at the model evidence equation, \\(p(D \vert \mathcal{M})
= \int_{\Theta} p(D \vert \theta, \mathcal{M})p(\theta \vert \mathcal{M}) \\).
We can see that the contributions to the model evidence come from the prior
\\(p(\theta) \\) and the likelihood \\(p(x \vert \theta) \\).  Now, because a
probability distribution has to integrate to one, more complicated models, such
as those with more parameters, will have more diffuse priors and smaller prior
probabilities, simply because they have the same amount of mass but a much
larger parameter space \\(\Theta \\) to place that mass on.  That more diffuse
prior necessitates a larger likelihood value \\(p(x \vert \theta) \\), in order
to compensate.  This then leads to trade off between model complexity and model
fit, as a more complex model must also fit the data significantly better in
order to be preferred by the Bayes Factor.

<a name="lr"></a>
## Bayes Factors vs Likelihood Ratios 

You might have noticed at this point that the Bayes Factor looks basically like
the test statistic for a frequentist likelihood ratio test.  In that case, why would we ever use a Bayes Factor, when the likelihood ratio is infinitely easier to compute?

Well, the first nice thing about Bayes Factors is that they allow for comparison
of non-nested models, whereas a LRT does not.  The second, and prehaps most
important part, is that Bayes Factors do not fall under the Neyman-Pearson
hypothesis testing framework, and as such, are not subjected to its limitations.
For example, we do not have to specify a "null" or an "alternative" model when
working with Bayes Factors; instead, we simply specify models 1 and 2.  What
this means is that we are allowed to quantify the strength of evidence towards a
model in both directions with a Bayes Factor, whereas in a traditional
hypothesis test, we can only quantify the evidence in favor of the alternative.
To make this more concrete, I'll give an example of how one might use the Bayes
Factor to test a hypothesis.

Lets assume our data is iid normal: \\(x \sim \text{Normal}(\mu, \sigma^{2})
\\).  We want to test the hypothesis that our mean is equal to 0 - in the
frequentist framework, this would correspond to a t test with \\(H_{0}:\mu = 0
\\).

Lets go ahead and define our two models that we want to compare.  Our first
model would be a normal with mean 0: \\(M_{1} = \text{Normal}(0, \sigma^{2})
\\), while our second model would let the mean vary: \\(\text{Normal}(\mu,
\sigma^{2}) \\).
A "rejection" of the null hypothesis would correspond to choosing the model that lets the mean vary, over the model where the mean is fixed to be 0.

We would then calculate the Bayes Factor, either by taking an integral, or by
numerical methods (exactly how we do that will be
explained in the last section), and then pick the model that is favored by the
Bayes Factor.  For example, if our BF is greater than 1, we would conclude that
the data supports the claim that the mean is 0, where as a bayes factor of less
than 1 would instead suggest that our mean is not 0.  Or, if the evidence is
inconclusive, the Bayes Factor would instead prefer the simpler model, Model 1,
because it requires one less parameter to estimate.
In this way, we can see that in the Bayesian framework, a hypothesis testing problem can be reduced to a
model selection problem.

<a name="bs"></a>
## The Bridgesampling Algorithm 

Now that we've walked through the basic theory behind Bayes Factors, how do we
actually calculate them?
In very simple cases, such as our Bayesian t-test example, the integral used to
find the model evidence can be calculated analytically.
But for most of the interesting models out there, the evidence is intractable,
meaning that there does not exist an analytical solution to our integral.
In those cases, we have to resort to numerical methods.

However, even numerical methods struggle when presented with the model evidence
integral.
The reason for this is the extremely high dimensional space that we are required
to integrate over.
Again, just as a reminder, here is the model evidence again:

\\[p(D \vert \mathcal{M}) = \int_{\Theta} p(D \vert \theta, \mathcal{M})p(\theta \vert \mathcal{M}) = \int_{\theta_1} \cdots \int_{\theta_m} p(D \vert \theta_{1} \ldots \theta_{m}, \mathcal{M})p(\theta_{1}, \ldots, \theta_{m} \vert \mathcal{M}) \\]

Each extra parameter adds an extra integral and thus extra computational burden
for our model evidence equation, meaning that for non-trivial models, this
integral becomes highly difficult to approximate with precision.
As such, the bridgesampling algorithm was developed to combat these issues.

Understanding the bridgesampling algorithm begins with understanding the concept
of importance sampling.
Importance sampling is a monte carlo method generally used to approximate integrals or expectations.
It works by drawing samples from an auxillary distribution, and then
approximating the expectation/integral with a sample average.

More formally, let us suppose that we have data \\(x \\) from some distribution P.  
We want to approximate the expectation of a function \\(f(x) \\), but we cannot reliably sample from \\(P \\); only evaluate its density.

\\[f(x) = \int f(x)p(x)dx .\\]

Multiply the right hand side by our _proposal distribution_ g(x) to get:

\\[ \int f(x)p(x)\frac{g(x)}{g(x)}dx = \mathbb{E}_{g}[\frac{f(x)p(x)}{g(x)}] \\]

\\[ \approx \frac{1}{N}\sum_{i} \frac{f(x_{i})p(x_{i})}{g(x_{i})}. \\]

The approximation in the final step is referred to as the __importance sampling
estimator__.
It is easy to see that the accuracy of our approximation depends on the specific
sampling distribution that we choose.
If \\(G \\) has a large overlap with \\(P \\), our approximation will be good,
since samples drawn from \\(G \\) will be good approximations to samples drawn
from \\(P \\).
As such, choosing the optimal proposal function is of paramount importance, and
we'll discuss the finer details after the description of the algorithm.

Now we are ready to move onto the bridgesampling algorithm.
Fundamentally, bridgesampling starts from the following identity:

\\[1 = \frac{\int p(y \vert \theta)p(\theta)g(\theta)h(\theta)d\theta}{\int p(y \vert \theta)p(\theta)g(\theta)h(\theta)d\theta} \\]

Note that I've omitted the conditioning on the model \\(\mathcal{M} \\), for the purposes of legibility.
\\(g(\theta) \\) is our proposal distribution from before, and \\(h(\theta) \\)
is a special function called the _bridge function_.
(I've searched a lot about what the bridge function actually does, and I came up mostly empty handed - it just seems to be some arbitrary function that we define to ensure theoretical guarantees and computational tractability).
Multiply both sides by \\(p(y) \\) to get

\\[p(y) = \frac{\int p(y \vert \theta)p(\theta)g(\theta)h(\theta)d\theta}{\int \frac{p(y \vert \theta)p(\theta)}{p(y)}g(\theta)h(\theta)d\theta} =  \\]

\\[ \frac{\mathbb{E}_{g}[p(y, \theta)h(\theta)]}{\mathbb{E}\_{p(\theta \vert y)}[g(\theta) h(\theta)]} \\]

Our approximation therefore becomes:

\\[p(y) \approx \frac{1/N\_{1} \sum\_{i} p(y \vert \hat{\theta}\_{i})h(\hat{\theta}\_{i})}{1/N\_{2}\sum\_{j}g(\tilde{\theta}\_{j})h(\tilde{\theta}\_{j})} \\]

Here, I use \\( \hat{\theta} \\) to represent samples drawn from our proposal
distribution, and \\( \tilde{\theta} \\) to represent posterior samples.

Lets go ahead and unpack the proposal distribution and bridge function in turn, one
after another.
In practice, the easiest and suprisingly effective choice of proposal
distribution happens to be a multivariate normal, with its first two moments
chosen to be equal to the empirical moments of our posterior distribution.
This is mainly for computational tractability, but it also makes intuitive sense,
since we know from the Bernstein-Von-Mises theorems
that the posterior distribution will often by asymptotically normal.
However, this approximation may fail in higher dimensions.
In these cases, we often apply what is known as warp-sampling, where instead of
choosing the proposal to match the posterior, we "warp" the posterior to match
our multivariate normal proposal distribution.
I won't go into greater detail about this method, since the blog post is long
enough as is.

Now, onto the bridge function.
The optimal bridge function (in terms of minimizing Mean Squared Error) takes
the form:

\\[h(\theta) = C\frac{1}{s_{1}p(y\vert \theta)p(\theta) + s_{2}p(y)g(\theta)},
s_{1} = \frac{N_{1}}{N_{2} + N_{1}}, \; s_{2} = \frac{N_{2}}{N_{1} + N_{2}} \\]

Don't ask me to prove why this is, because I read the paper and I still don't understand it (you can read the proof for yourself [here](https://pdfs.semanticscholar.org/6a40/18c9c2927d702d85257c34130b1204fa7584.pdf)).
In the above expression, \\(C \\) is a constant; we don't have to worry about
its value because it will cancel itself off when we plug this \\(h(\theta) \\)
into both the numerator and denominator of our bridgesampling equation.

It doesn't take much time to notice a big problem with our optimal bridge
function: it depends on the quantity we are trying to find, \\(p(y) \\)!
The solution to this is to apply a dynamic programming approch, where we replace
\\(p(y) \\) with our estimate of \\(p(y) \\) at the previous time step.
In practice, the iterative algorithm converges extremely quickly; often, only four or five time steps are needed before converging to a solution.

In this way, the estimate for the marginal likelihood at time step \\(t+1 \\) becomes:

\\[\hat{p}^{t+1}(y) = \frac{\frac{1}{N_2} U}{\frac{1}{N_1} V} \\]

\\[U = \sum\_{i}^{N_2} \frac{p(y, \hat{\theta}\_{i})}{s\_{1}p(y, \tilde{\theta}\_{i}) + s\_{2}\hat{p}^{t}(y)g(\tilde{\theta}\_{i})} \\]

\\[ V = \sum\_{j}^{N_1} \frac{g(\tilde{\theta}\_j)}{s\_1 p(y \vert \tilde{\theta}\_j) p(\tilde{\theta}\_j) + s\_2\hat{p}^t(y)g(\tilde{\theta}\_j)} \\]

Where \\(s_1 \\) and \\(s_2 \\) are defined in the previous equation.
Now, in practice, this is not the exact equation used in the implementation of the bridgesampling algorithms, but I feel that going further beyond what we have already done brings very little new insights and understanding.

With this, I hope I've given a somewhat cohesive introduction to the world of Bayesian model selection.
Model selection is often criminally underused and underappreciated, and people should definitely know more about it beyond just the introductory "bias variance tradeoff" and "cross validation" one liners.
After all, if you are not sure if your model is right or not, how are you supposed to draw meaningful conclusions from it?
