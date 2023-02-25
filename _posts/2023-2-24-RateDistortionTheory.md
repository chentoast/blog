---
layout: post
title: Abstractions and Efficient Representations via Rate Distortion Theory
author: Tony Chen 
---

For this next post, I’m going to talk about the role of *abstraction* in reinforcement learning and planning. In some sense, abstract representations of states and actions are the holy grail for building efficient and generalizable agents. By throwing away irrelevant aspects of the environment, and aggregating together functionally similar ones, agents can achieve large gains in efficiency without a large corresponding drop in efficacy, and by learning reusable sub-routines of actions, they may also gain skills that can transfer to a wide variety of different environments. Generally, the notion of abstraction in RL can be divided into two types: state abstractions, which focus on the aggregation of states, and action abstractions, which aggregate actions together into skills or “options”. For this post, we will be studying state abstractions specifically, but action abstractions are also well studied in the literature (perhaps with more success than state abstractions).

So, abstraction is an important problem, but the obvious issue is that discovering good abstractions is really hard. Often, approaches to state abstraction involve some clustering based approach, where states that are “functionally similar” are aggregated into a single state. There are a number of proposals for how to best define similarity between states, either based on similarity of their transition dynamics, or similarity in the value as defined by the q-function, but no approach dominates.

Today, we’ll be talking about another theoretical approach to discovering abstractions, using tools from information theory - specifically, rate distortion theory. I’m covering two papers that use this general framework to talk about compression and abstraction in planning problems: you can find them [here](https://arxiv.org/abs/2206.02072) and [here](https://ojs.aaai.org/index.php/AAAI/article/view/4179).

1. [Rate Distortion Theory](#section1)
2. [The Blahut Arimoto Algorithm](#section2)
3. [Value Equivalent Sampling](#section3)
4. [State Abstraction as Compression](#section4)
5. [Wrap up](#section5)

## Rate Distortion Theory <a name="section1"></a>

While most of information theory is concerned with *lossless compression*; i.e. the process by which a signal can be compressed and recovered exactly, rate distortion theory instead focuses on *lossy compression* where we are wiling to throw away some information in the input in order to achieve potentially better compression rates.

More specifically, we are concerned with the following problem: given some input signal $$x$$ and some number of bits $$R$$, what is the least amount of information loss from a reconstruction $$\hat{x}$$ constructed using these $$R$$ bits? Or equivalently, given a particular threshold of reconstruction accuracy, how many bits do we need to be able to achieve that limit?

Let’s formalize these questions. For some input signal from a measure space $$X$$ with an associated probability density $$p(x), x \in X$$, and a reconstruction domain $$\hat{X}$$, Define the *distortion function* to be a mapping

$$
d: X \times \hat{X} \to \mathbb{R}^+.
$$

In essence, $$d$$ takes in an input signal $$x$$ along with its reconstruction $$\hat{x}(x)$$, and provides a measure of how good the reconstruction was. Some examples might include the *squared error distortion* $$d(x, \hat{x}) = (x - \hat{x})^2$$ for continuous inputs, or the *hamming distortion* $$d(x, \hat{x}) = \mathbb{1}(x = \hat{x})$$ for discrete ones.

This theory extends analogously to the case where we have a vector of signals $$x^n \in X^n$$ we want to reconstruct: we simply define the distortion to be the average of component distortions:

$$
d(x^n, \hat{x}^n) = \frac{1}{n} \sum_i d(x_i, \hat{x}_i).
$$

A $$(2^{nR}, n)$$ rate distortion code consists of an encoding function $$f_n: X^n \to \{1, 2, \ldots 2^{nR}\},$$ along with a decoding function $$g_n: \{1, 2, \ldots 2^{nR}\} \to \hat{X}^n.$$

Now, for a given encoding and decoding function, define the *distortion* associated with the $$(2^{nR}, n)$$ code as

$$
D = \mathbb{E}\ d(x^n, g_n(f_n(x^n))),
$$

where the expectation is with respect to $$p(x^n).$$

A rate distortion pair $$(R, D) \in \mathbb{N} \times \mathbb{R}$$ is said to be *achievable* if there exists a sequence of $$(2^{nR}, n)$$ codes with

$$
\lim_n \mathbb{E}\ d(x^n, g_n(f_n(x^n))) \leq D.
$$

The *rate distortion region* $$\mathcal{R} \subset \mathbb{R}^2$$ is then defined to be the closure of the set of all achievable pairs $$(R, D)$$, and the *rate distortion function* is defined to be

$$
R(D) = \inf_{R} \{(R, D) \in \mathcal{R}\}.
$$

That is, $$R(D)$$ describes the minimal amount of bits or rates needed to achieve a distortion level of $$D$$.

With all of that definition unpacking out of the way, let’s present the major result of rate distortion theory, which describes the rate distortion function in a form that is much more amenable to computation.

*Theorem.*

$$
\large R(D) = \min_{p(\hat{x} \vert x) : \mathbb{E}_{p(x)p(\hat{x} \vert x)} d(x, \hat{x}) \leq D} I(X:\hat{X}).
$$

Above, $$I$$ is the mutual information of two random variables. Thus, finding the rate distortion function corresponds to finding a particular conditional distribution over $$\hat{x}$$ that minimizes the mutual information between the source and reconstruction codes.

We can re-write the above minimization problem using Lagrange multipliers, to get to an equivalent formulation:

$$
R(D) = \min_{p(\hat{x} \vert x)} I(X : \hat{X}) +\lambda \left(\mathbb{E}_{p(x,\hat{x})}\ d(x,\hat{x})\right),
$$

for $$\lambda \geq 0$$. The Lagrange multiplier controls the aggressiveness of the encoding - higher values imply a large amount of compression at the cost of reconstruction accuracy, and vice versa.

For example, let $$X$$ be a discrete uniform random variable on the first $$m$$ natural numbers: $$X \in \{1, 2, \ldots m\}.$$ I claim that the optimal $$q(\hat{x} \vert x)$$ supported on $$\{1, 2, \ldots m\}$$ under the hamming distortion is given by:

$$
q(\hat{x}\vert x) = \begin{cases}\frac{D}{(m-1)}, \quad \hat{x} \neq x, \\ 1-D, \quad \hat{x} = x. \end{cases}
$$

It’s not too hard to check that this results in a value of $$R(D) = \log m - H(D) - D\log (m-1),$$ for $$D < (m-1)/m$$.

While the theory presented is elegant, rate distortion theory can only answer the question of “*what is the best lossy compression that I can achieve?*", and does not provide for us an explicit construction of the optimal compression code. Instead the construction of practical compression algorithms belongs to the field of quantization.

## The Blahut Arimoto Algorithm <a name="section2"></a>

The Blahut-Arimoto algorithm is a way to compute the rate distortion function in practice.

We’ll start by writing down the Lagrangian, in full detail:

$$
J(p) = \sum_{x,\hat{x}}p(x)p(\hat{x} \vert x)\log \frac{p(\hat{x} \vert x)}{p(\hat{x})} + \lambda \sum_{x,\hat{x}}p(x)p(\hat{x} \vert x)d(x,\hat{x}).
$$

Take the derivative with respect to $$p(\hat{x} \vert x)$$ to get:

$$
p(x)\left[\log \frac{p(\hat{x}\vert x)}{p(\hat{x})} + \lambda d(x,\hat{x}) \right] = 0,
$$

for all $$x$$. If we then solve for $$p(\hat{x} \vert x)$$, we find that

$$
p(\hat{x} \vert x) \propto p(\hat{x})\exp (-\lambda d(x, \hat{x})),
$$

and re-normalizing gives us

$$
p(\hat{x} \vert x) = \frac{p(\hat{x})\exp (-\lambda d(x, \hat{x}))}{\sum_x' p(x')\exp (-\lambda d(x, x'))}.
$$

There is an implicit dependency here: $$p(\hat{x}) = \sum_x p(x)p(\hat{x} \vert x)$$ and thus actually depends on $$p(\hat{x} \vert x)$$, although we are treating it as fixed in the optimization above. The solution is to apply an iterative optimization approach, which leads us to the Blahut Arimoto algorithm. At each time step t, we apply the following updates:

$$
p_t(\hat{x})=\sum_x p(x)p_{t-1}(\hat{x} \vert x),\\p_t(\hat{x} \vert x)=\frac{p_{t}(\hat{x})\exp (-\lambda d(x, \hat{x}))}{\sum_x' p_{t}(x')\exp (-\lambda d(x, x'))}.
$$

I’ll just make a small note that this algorithm is exact in the case of finite compression codes, but in the general continuous case, the problem is still open.

## Value Equivalent Sampling <a name="section3"></a>

In a case where the environment is far too large to efficiently navigate and represent, how can we learn a reduced model of the environment that affords efficient action, while still preserving some notion of fidelity or accuracy of representation? The recent paper introduces the idea of Value Equivalent Sampling, which proposes a theoretical framework for answering such questions.

We will be working in the Bayesian Reinforcement Learning regime, where we learn the transition and reward dynamics through Bayesian inference. So, For a fixed state and action space $$S, A$$, we put a prior over transition dynamics and rewards, which induces a distribution over length $$l$$ finite-horizon MDPs $$(S, A, T, R, l)$$. Denote by $$M^*$$ the true MDP, i.e. the MDP with the ground truth transition and reward dynamics.

The traditional objective in Bayesian RL is to simultaneously maximize reward and learn to approximate the true MDP $$M^*$$. However, in cases where $$M^*$$ is incredibly high dimensional, large, or unwieldy, we might instead want to learn a satisficing solution that approximates $$M^*$$ to a reasonable degree while also preserving some notion of simplicity.

This objective is very naturally accommodated under the framework of rate distortion theory: for a fixed computational budget, we would like some compression scheme that best preserves the key properties of our target MDP. But in order to apply the tools that we have just been describing, we will need to write down the distortion function. Which one should we choose for this problem?

The option that the paper went with is to relate two MDPs via the similarity in their bellman backups. Define the *Bellman backup operator* for an MDP $$M$$ as a linear operator on the space of functions $$S \to \mathbb{R}$$ as follows:

$$
B_M V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[R(s,a) +\mathbb{E}_{s' \sim T} V(s') \right].
$$

We’ll say that two MDPs $$M, M’$$ are *value equivalent* if they induce the same bellman backup:

$$
B_M V = B_{M'}V.
$$

Thus, we have a notion of equivalence between two MDPs, which we can easily translate into a distortion metric as follows. We define the distortion function to be:

$$
d(M,M')=\sup_{\pi, V} \left\vert\left\vert B_MV^\pi-B_{M'}V^\pi\right\vert\right\vert_\infty^2.
$$

In other words, the distortion function is the largest possible difference in bellman backup, across all value functions and policies for our given state and action space.

This leads us to our algorithm. We’ll take as a starting point the *Posterior Sampling for Reinforcement Learning* algorithm introduced in previous work in Bayesian RL, which generalizes Thompson sampling to the general finite-horizon MDP case. I’m just outlining the algorithm in very rough terms:

- On episode k, sample an MDP (i.e., sampling a reward and transition function) from your posterior $$M_k \sim P(\cdot \vert H_k)$$
- Compute a policy $$\pi^k$$ for $$M_k$$
- Execute the policy, observing a trajectory $$\tau_k$$, and update your posterior over MDPs: $$P(\cdot \vert H_{k+1}), H_{k+1} = H_k \cup \{\tau_k\}$$.

The *Value Equivalent Sampling for RL* algorithm is essentially the same as PSRL, with an additional step that compresses our posterior distribution using rate distortion theory:

- On episode k, again sample an MDP from the posterior $$M_k \sim P(\cdot \vert H_k)$$
- Compute a compression $$\widehat{M}_k$$ of $$M_k$$ using a compression scheme that achieves the rate distortion function $$R(D)$$.
- Compute an approximately optimal policy $$\pi^k$$ for $$\widehat{M}_k$$, observe trajectory $$\tau_k$$, and update history $$H_{k+1}$$.

Note that this algorithm is highly theoretical, and a number of practical issues immediately present themselves. Firstly, it’s not immediately obvious how to compute the distortion function $$d$$, given that the supremum is taken over all value functions and all policies.

Secondly, as mentioned before, computing the rate distortion function for a continuous space is a highly nontrivial problem, as the set of MDPs corresponds to the space of all transition and reward functions - a humongous space to traverse and to minimize over. Even if we did compute $$R(D)$$, it only provides a lower bound - and does not tell us which compression scheme we should use to attain this lower bound and compute $$\widehat{M}_k$$.

Finally, it’s also unclear how best to compute a policy from the compressed MDP $$\widehat{M}_k$$. Although the authors provide regret bounds controlling the optimality of the compressed policy, we still don’t have a good intuition for how the policies of compressed MDPs might behave in practice - and how much their behavior depends on the particular compression scheme used.

## State Abstraction as Compression <a name="section4"></a>

Our second paper focuses more on the problem of abstraction rather than efficient representation - grouping together similar states by extracting commonalities and dropping irrelevant specific details.

Before we introduce the paper, we’ll need to discuss some more technical results.

### The Information Bottleneck

Rate distortion theory focuses on the compression of an input signal for the purposes of reconstruction - i.e., finding the best compression that allows you to retain as much of the signal in the original input as possible.

Conversely, the information bottleneck focuses on compression for the purposes of *prediction* - finding the best compression that allows you to predict some other signal $$Y$$. In this way, you can sometimes achieve much better compression, since much of the information in your input $$X$$ might be unneeded or irrelevant if all you want to do is to predict the value of $$Y$$. Obviously the information bottleneck method reduces to good old rate distortion theory if $$Y = X$$.

More specifically, we’ll take our distortion function to be the mutual information between our reconstruction $$\hat{X}$$ and a prediction signal $$Y$$:

$$
\mathbb{E} d(\hat{x}, x) = I(X : Y).
$$

We’ll assume that $$X, Y$$ are conditionally independent given $$\hat{X}$$, so that no other information about $$Y$$ can be obtained from a source other than $$X$$. The rate distortion objective then becomes:

$$
\min_{p(\hat{x}\vert x)}I(X : \hat{X})\ -\ \lambda I(\hat{X}:Y),
$$

where again $$\lambda \geq 0$$ is a lagrange multiplier.

This objective can be reformulated as:

$$
\min_{p(\hat{x}\vert x)}I(X : \hat{X})\ -\ \lambda\mathbb{E}_{p(x, \hat{x})} D_{KL}(p(y\vert x) \vert\vert p(y\vert\hat{x})).
$$

For our purposes, we will be working with deterministic maps $$f: X \to \hat{X}$$, in which case $$I(X, \hat{X}) = H(\hat{X})$$:

$$
\begin{align*} \min_{f(x)} H(\hat{X})-\lambda \mathbb{E}_{p(x)}D_{KL}(p(y\vert x) \vert\vert p(y\vert\hat{x})).\end{align*}
$$

The solution to this minimization problem is obtained by an iterative algorithm that is again reminiscent of the Blahut Arimoto algorithm:

$$
f(x) = \mathop{\operatorname{arg\,max}}_{\hat{x}} p_{t}(\hat{x})
\exp\left(−\lambda D_{KL}(p(y \vert x)\ \vert\vert\ p_t(y \vert \hat{x}))\right),\\ p_{t+1}(\hat{x}) =
\sum_{x:f(x)=\hat{x}} p(x),\\p_{t+1}(y \vert \hat{x}) = \sum_{x:f(x)=\hat{x}}
p(y \vert x).
$$

Now, lets present our abstraction algorithm. In this case, we’ll be working in apprenticeship learning, where we are given an expert policy $$\pi_E$$, and want to learn an *abstraction function* $$\phi: S \to S_\phi$$, where $$S_\phi$$ is a set of abstract states with cardinality (usually) less than $$S$$. The idea is that $$\phi$$ collapses across irrelevant information in the state into a compressed space that retains the information needed to solve the problem.

Let $$\rho_E$$ be the stationary distribution over $$S$$ induced by the expert policy $$\pi_E$$. Let $$\pi_\phi$$ be the optimal policy in the abstract state space $$S_\phi$$. We’re interested in minimizing the following objective:

$$
\min_\phi \large\lambda \mathbb{E}_
{\rho_E(s)}
[V_{\pi_E (s)} − V
_{\pi_\phi (\phi(s))}] + \vert S_\phi\vert.
$$

We can see that the objective trades off between two terms: a first term that governs the difference in value obtained when planning with the expert policy versus an abstract policy, and a second term that controls how coarse or granular the abstraction is. The solution to this optimization problem will give us a set of abstract states that minimizes the number of abstract states, but still achieves a high value when planning in this abstract state space.

However in practice, this objective is very difficult to minimize. As such, our plan of attack will be to upper bound this objective by a deterministic information bottleneck objective, and use the aforementioned algorithm to compute a locally optimal solution.

In the language of information theory, we’ll let $$\rho_E$$ be the input signal distribution, $$\rho_\phi$$ the compressed signal (where $$\rho_\phi$$ is the stationary distribution over $$S_\phi$$ induced by the abstract policy $$\pi_\phi$$), and the policy $$\pi_E(s)$$ as the target signal. The DIB objective function is thus:

$$
\min_\phi \large H(S_\phi)-\mathbb{E}_{\rho_E,\rho_\phi} D_{KL}(\pi_E(a\vert s)\vert\vert \pi_\phi (a \vert s_\phi)).
$$

(where $$H(S_\phi) = H(X), X\sim \rho_\phi.$$ The rest of the paper is spent proving that this DIB objective bounds the abstraction objective, so that solving the DIB objective (a much more tractable problem) gives us an approximate solution to the abstraction objective.

## Wrap up <a name="section5"></a>

While both of the algorithms presented were highly theoretical, I found them both really fascinating. It sometimes feels like the focus on representations of tasks have drastically shifted from explicitly finding and constructing efficient representations, to plugging a DNN into a classification problem with tons of data, with the implicit assumption that any abstractions or compression mandated by the problem would be automatically discovered as a consequence of scale. As such, it was nice to see papers that still tackled this problem explicitly, using theoretical tools and math that turned out to be surprisingly elegant. Obviously, it still remains to be seen as to whether anyone can turn these formal results into empirical success.
