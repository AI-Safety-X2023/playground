# The non-commutativity of gradient steps is a fundamental limitation of exact unlearning

> [!WARNING] This paper is still under elaboration. No experiments have been made yet.

- LLaMa 3.2 1B : poisoning with unsafe coding examples -> unaligned model -> unlearning fails
- possible suboptimality of the median as an aggregator (we discard lots of info depending on batch size)
- GUS interpretation and concrete proof of unlearning failure ? by MIA ?
- Efficient streaming median: good explanation [here](https://arxiv.org/pdf/1802.10116) (+ dimension-aware geometric mean around median)
- Fixed-point theorems = Non working solutions in paper

## Introduction

We demonstrate that exact unlearning fails in general since gradient steps can't be reverted and do not commute.

Even if gradient steps could be inverted, gradient-based exact unlearning methods would need to invert the gradient steps in the reverse order of the training loop. However, this method is not more efficient than retraining, which is computationally prohibitive for practical applications.
This hints at the impossibility of efficient exact unlearning without a major paradigm change in architecture and training.

We back our hypotheses with theoretical insights from linear regression and some experiments on linear data, logistic regression and non-trivial datasets such as CIFAR-10.
We revisit existing gradient attacks and adversarial attacks in the context of Machine Unlearning.

## Background

### Exact unlearning

We consider the following definition of exact unlearning: given a known forget subset $\mathcal{S}_{\mathrm{forget}} \subset \mathcal{S}_{\mathrm{train}}$, restore a trained model to the **exact** same state of a model trained on $\mathcal{S}_{\mathrm{train}} \backslash \mathcal{S}_{\mathrm{forget}}$.

Exact unlearning can easily be achieved by retraining the model from scratch, however the computational cost of such an approach is prohibitive. Existing approaches [Xu et al., 2024](https://arxiv.org/html/2308.07061v2) require a major change in architecture, such leveraging a herd of smaller models trained on distinct portions of the dataset. Although such methods offer true data _deletion_, they incur a tradeoff between efficiency, accuracy, applicability.

### Approximate unlearning

Existing gradient-based approximate unlearning methods ([Guo et al., 2020](https://arxiv.org/abs/1911.03030)) are proven correct with convex models. However, this hypothesis is too restrictive for the vast majority of neural networks. Indeed, most non-trivial neural networks need to display non-smooth behavior in order to be useful (consider the universal approximation theorem for irregular functions). Adversarial examples provide further evidence that neural networks are highly irregular ([Ilyas et al., 2019](https://arxiv.org/abs/1905.02175)).

In this work, we provide intuition that exact unlearning is impossible with approximate, gradient-based unlearning methods.

### Notations

We consider the general learning framework $\theta_{t+1} = \Phi_t(\theta_t)$ where $\Phi_t$ is the optimization function. For example, with gradient descent we have:
$$\Phi_t (\theta) = \Phi_{z_t} = \theta - \lambda \nabla_\theta l(h_\theta (x_t), y_t)$$
where $z_t := (x_t, y_t) \in \mathscr{Z} = \mathscr{X} \times \mathscr{Y}$ is a data point, $l: \mathscr{Y} \times \mathscr{Y} \to \mathbb{R}$ is the loss function and $\lambda \in \mathbb{R}$ the learning rate.

We note $g_z: \theta \mapsto \nabla_\theta l(h_{\theta} (x), y)$ the loss function gradient w.r.t $\theta$ computed on the data point $z = (x, y)$. We may now write $\Phi_z = id - \lambda g_z$.

When there is no ambiguity, we write $L(z, \theta) = l(h_{\theta} (x), y)$.

Note that we do not take batching into account, which further complexifies the problem for the defender. In the case of gradient descent with mean gradient aggregation, we have
$$g_t = \frac{1}{B} \sum_{i=1}^B \nabla_\theta l(h_{\theta_t} (x_{t,i}), y_{t,i})$$
So an exact unlearner would have to compute not only $g_t$, but each of the gradients $\nabla_\theta l(h_\theta (x_{t,i}), y_{t,i})$ so the poisoned gradients can be removed. Indeed, when considering data poisoning on non-robust machine learning systems, any batch that contains poisons can be considered as corrupted (See [Farhadkhani et al. (2022), An Equivalence Between Data Poisoning and Byzantine Gradient Attacks](https://arxiv.org/abs/2202.08578)). Further explanation can be found in the [last section](#poisoned-batches).


### The case of linear regression

In some of our theoretical analyses, we will consider the linear regression model

$$
h_\theta: x \mapsto \theta \cdot x \\
l: (y, y') \mapsto \frac{1}{2} |y' - y|^2
$$

where $\theta, x \in \Theta = \mathscr{X} = \mathbb{R}^p$ and $\mathscr{Y} = \mathbb{R}$.

Linear regression corresponds to a best-case scenario for machine unlearning : the model is convex and has few parameters. In the following subsections, we show that our hypotheses are valid even on the linear model.

## Gradient ascent does not revert gradient descent

Usual approximate unlearning methods, such as gradient ascent, can never perform exact unlearning.

TODO (defense): intuition with market gains/losses

Gradient ascent is based on the intuition that gradient ascent is the reverse step of gradient descent. However, this view fails to take into account that $\nabla_\theta l(h_\theta (x), y)$ highly depends on $\theta$. In this section, we prove that the irregularity of $\nabla_\theta L(z, \theta)$ w.r.t $z$ and $\theta$ can be exploited with adversarial attacks, which rules out gradient ascent as a possible candidate to exact unlearning.

### First observations

Let $\theta_1 = \theta_0 - \lambda g_z (\theta_0)$ be the updated model after a single optimization step. Gradient ascent computes

$$\theta_2 = \theta_1 + \lambda g_z (\theta_1) \ne \theta_0$$

since $g_z (\theta_1) \ne g_z (\theta_0)$.
Now, the question is to show the error $||g_z (\theta_1) - g_z (\theta_0)||$ can be very large on adversarial examples.

Furthermore, practical applications of Gradient ascent generally use a much smaller unlearning rate than the learning rate, so as not to degrade the model performance.

### Approximation error in the linear case

We first quantify the difference $g_z (\theta_1) - g_z (\theta_0)$ in the context of linear regression.

We compute $g_z (\theta) = \nabla_\theta l(h_\theta (x), y) = (\theta \cdot x - y) x$. $g_z (\theta)$ is an affine map of $\theta$ and given $\theta_0, \theta_1 \in \Theta$, we have

$g_z (\theta_1) - g_z (\theta_0) = (\theta_1 \cdot x) x - (\theta_0 \cdot x) x = ((\theta_1 - \theta_0) \cdot x) x$.

In the context of gradient ascent, we consider $\theta_1$ obtained after optimizing $\theta_0$ on $z$, that is $\theta_1 - \theta_0 = -\lambda g_z (\theta_0) = -\lambda (\theta_0 \cdot x - y) x$. It follows that

$$
\theta_2 - \theta_0 = \lambda g_z (\theta_1) - \lambda g_z (\theta_0) = ((-\lambda (\theta_0 \cdot x - y) x) \cdot x) x = -\lambda^2 (\theta_0 \cdot x - y) |x|^2 x
$$

So $\theta_2 - \theta_0$ is small if $\theta_0 \cdot x - y \approx 0$, meaning that gradient ascent is exact for examples such that the loss $l(h_{\theta_0}(x), y) = \frac{1}{2} (\theta \cdot x - y)^2$ is small and / or such that the gradient $g_z (\theta) = (\theta \cdot x - y) x$ is small.

### Approximation error in a nonlinear setting

Note that in linear regression, $g_z(\theta_0)$ and $g_z(\theta_1)$ are still collinear to $x$.

When it comes to large, non-linear models, it may be likely that $g_z(\theta_0)$ and $g_z(\theta_1)$ are almost orthogonal.

> [!TIP] Hypotheses
>
> - $g_z(\theta_0)$ and $g_z(\theta_1)$ belong in a span $V_z$
> - With large models, there exists data points $z$ such that $\dim V_z$ is large

Note that in large dimensions, in average two vectors are almost orthogonal.

> [!TIP] Experiment 1
> 
> To check the hypotheses above, we computed the correlation $\cos(g_z(\theta_0), g_z(\theta_1))$ with $\theta_1$ the result of training $\theta_0$ on $z$ and then on a small number of mini-batches (from 1 to 500).

> [!NOTE] Observations
> 
> We observed two patterns. For an untrained model the cosine similarity is extremely close to $1$, even when the number of interleaved batches is high. When the model is pretrained for $2$ epochs, the cosine similarity remains around $0.3$-$0.5$, and surprisingly, the higher the number of interleaved batches, the higher the similarity. 

TODO: study effect of number of pretraining epochs on cosine similarity


> [!TIP] Experiment 2 (food for thought)
>
> Inspired by the orthogonal gradient attack, we could craft a data point such that the model gradients become orthogonal to the previous gradients after the gradient update, that is
> $$\min_z |\cos \left( \nabla_\theta L(\theta - \lambda \nabla_\theta L(\theta, z), z), \nabla_\theta L(\theta, z) \right)| = |\cos \left( g_z(\theta - \lambda g_z(\theta)), g_z(\theta) \right)|$$


### Towards the general case

Conversely, we hypothesize gradient ascent makes large errors on three types of points:
1. high-loss data points ($z$ s.t $L(z, \theta)$ is large)
2. influential data points ($z$ s.t $\nabla_\theta L(z, \theta)$ is large)
3. data points such that the gradient $g_z (\theta)$ is irregular w.r.t $\theta$, i.e the Hessian $\nabla_\theta [g_z (\theta)] = \nabla_\theta^2 L(z, \theta)$ is large.

Note that when the loss function $L(z, \theta)$ is convex w.r.t $\theta$, these three cases are equivalent.

In the following sections, we demonstrate the effectiveness of data poisoning attacks targeting gradient-based unlearning algorithms, using each of these cases (1, 2, 3).

#### Adversarial poisons with high loss

We validated the first hypothesis on ResNet18 trained on CIFAR-10 with 10 high-loss poisons generated by a variant the FGSM attack ([Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)).
The poisons are generated by
$$
x_{\mathrm{adv}} = x + \varepsilon \mathrm{copysign}(\eta, \nabla_x l(h_\theta(x), y))
$$
where $\eta \sim \mathcal{N}(0, I_d)$ is an independent gaussian noise and $\varepsilon = 0.05$. Assuming that $\mathrm{sign}(\nabla_x l(h_\theta(x), y)) \sim \mathcal{U}(\{-1, 1\})$ is a Rademacher vector with independent coordinates, $\mathrm{copysign}(\eta, \nabla_x l(h_\theta(x), y))$ is also gaussian-distributed.

We then apply the following unlearning algorithms and compare the Gaussian Unlearning Score before poisoning and after poisoning on:
- Gradient Ascent using SGD with $\texttt{lr=1e-5}$, $\texttt{momentum=0.9}$, $\texttt{weight\_decay=5e-4}$ during $4$ epochs.
- NegGrad+ ($\beta = 0.995$) with similar hyperparameters as Gradient Ascent but with $\texttt{lr=1e-3}$.

Since the model is trained on clean-label adversarial examples, this data poisoning can be seen as a form of adversarial training. The poisons are likely to deeply affect the model since their gradients are very large.

TODO: check model weight difference after poisoning

We find that both unlearning algorithms fail to remove the influence of the 10 poisons samples.

#### Adversarial data poisoning via influence functions

The inspiration of considering influential data points comes from the works of [Koh et al](https://arxiv.org/pdf/1703.04730). In the case of a strictly convex loss function, the influence function of a data point $z$ on the parameters $\theta$ is given by
$\mathcal{I}_{\mathrm{up,params}}(z) = - H_\theta^{-1} \nabla_\theta L(z, \theta)$
where $\displaystyle H_\theta := \frac{1}{n} \sum_{i=1}^n \nabla_\theta^2 L(z_i, \theta)$ is positive definite.\
Since $\mathcal{I}_{\mathrm{up,params}}$ is linear with respect to $\nabla_\theta L(z, \theta)$, for an influential data point $z$, $\nabla_\theta L(z, \theta)$ is large, which means the gradient update of $\theta$ on $z$ will be large.\
Koh et al. suggest that work on influence functions can provide valuable insights even for non-differentiable models with non-convex losses.

In section 5.2, Koh et al. showcase an adversarial data poisoning attack similar in spirit to Goodfellow et al. by iterating $z \leftarrow z + \varepsilon \mathrm{sign}(\mathcal{I}_{\mathrm{pert,loss}}(z, z_{\mathrm{test}}))$. This is additional evidence of the similarity between maximizing the loss and maximizing the loss gradient.

[Finlay et al.](https://arxiv.org/pdf/1808.09540) concur in section 3.1, showing that data points with large gradients are indeed vulnerable to gradient attacks.

More generally, the influence of $z$ on $f(\theta)$ is defined by the scalar $-\nabla_\theta f(\theta)^T H^{-1} \nabla_\theta L(\theta, z)$.

[Grosse et al.](https://arxiv.org/pdf/2308.03296) show that influence functions can be efficiently computed

[Li et al.](https://arxiv.org/pdf/2409.19998) show that influence functions themselves fail on LLMs

#### Adversarial poisons with irregular gradients

Consider a data poisoning attack that crafts a data point $z$ such that $\nabla_\theta^2 L(z, \theta)$ is large, so Gradient Ascent makes a large error when estimating $\nabla_\theta L(z, \theta)$.

NOTE: this is related to Delta-influence

Intuition says we would have no trouble finding irregularities in large models. Indeed, the works from [Herrera et al.](https://arxiv.org/abs/2004.13135) show that the Lipschitz constant of $\nabla_\theta L(z, \theta)$ grows exponentially with the number of layers.

A data poisoning attack may optimize the following objective:

$$\max_{z} ||\nabla_\theta^2 L(z, \theta)||_2$$
subject to
$$||z||_{\infty, Z} \le M$$
where $M$ determines the boundary of valid data range.

We also want to minimize $||\nabla_\theta L(z, \theta)||_2$, for two reasons:
- The defender might filter large gradients.
- The defender can detect poisons with influence functions which grow larger with $||\nabla_\theta L(z, \theta)||$.
Therefore, minimizing $||\nabla_\theta L(z, \theta)||$ makes the attack stealthier.

We introduce the following minimization objective:
$$\mathcal{L}(z) = -\alpha||\nabla_\theta^2 L(z, \theta)||_2^2 + (1 - \alpha) ||\nabla_\theta L(z, \theta)||_2^2$$
where $\alpha \in (0, 1)$ is a hyperparameter.

Solving this optimization problem can be done with Adam updates and data point projection to the valid data range $||z||_{\infty, Z} \le M$ to modify iteratively a given data point.

> [!WARNING] Pitfall
> 
> $\nabla_\theta^2 L(z, \theta)$ is expensive to compute. Assuming $L(z, \theta)$ is convex w.r.t $\theta$, its hessian $\nabla_\theta^2 L(z, \theta)$ is positive definite, and by the spectral theorem we can instead maximize the sum of its eigenvalues.

> [!TIP] Alternative
> 
> Maximize the norm of the hessian's diagonal.

> [!WARNING] Experiment to be done

Problem: Delta-influence use accuracy as criterion, but we need another metric (GUS is narrow)

#### Limitations of delta-influence

Complexity of identifying poisons: $O(N_{\mathrm{train}} N_{\mathrm{test}} n_{\mathrm{transformations}} d)$ at worst + expensive influence function computation (compute gradient in $O(d)$, and multiply by Hessian inverse in $O(d)$ where $d$ is the number of parameters)

In comparison: attacker does not have the $N_{\mathrm{train}}$ factor, complexity ~~ $O(N_{\mathrm{poison}} d)$

Plus backdoor would not be detected (maybe other detection algos can do that?)

What about hard to unlearn samples?

TODO: read and run code to measure complexity

Data augmentation techniques: slightly specific to image classification

Unlearning methods: CFk, EUk -> either inefficient or do not scale with very deep models (hypothesis)
Authors only tested on CIFAR-10, CIFAR-100 and ImageNette

#### Conclusion on gradient ascent

We conclude that computing the gradient $\nabla_{\theta_1} l(h_{\theta_1} (x), y)$ alone does not provide sufficient information to determine $\theta_0$.
Therefore, even an omniscient defender would be unable to revert model training with gradient ascent, let alone unlearning poisons.

Of course, in the case of linear regression, simply continuing the training on clean data is enough to forget the poisoned samples. However this is not true for non-trivial data such as CIFAR-10 or ImageNet, where gradient descent does not achieve proper unlearning ([Pawelczyk et al.](https://arxiv.org/abs/2406.17216)).

Furthermore, we demonstrate that gradient-based unlearning methods fail to remove adversarial data poisoning attack.

Delta-influence paper: uses CFk but this would not always work (and is more computationally expensive) + we can choose data points to slow down unlearning (cf. other paper).

Notice how the presented attacks are approximate, in the same way that Machine Unlearning is - since the adversarial examples depend on the model $\theta$. However, this is much less a problem for the attacker since they only need to find a single direction of failure. In contrast, the defender has to maintain the whole model optimization in the right direction. We leave for future work the task of finding transferable adversarial examples that are effective across different models.

We also remind the reader of the asymmetry between the defender and the attacker. On the one hand, the defender is required to fully mitigate all of the possible attacks, thus compliant machine unlearning algorithms need to be perfect. On the other hand, the attacker can afford doing an "approximate attack".

TODO: analyze variant of NegGrad+ where the model trainer takes the bi-objective optimization into account to align the gradients on the forget set and the retain set.

Critique: does it make sense to align these gradients?


## Inverting a single gradient step is hard

Consider the general learning framework $\theta_{t+1} = \Phi_{z_t}(\theta_t)$.

Reverting a gradient step is equivalent to finding the inverse of $\theta$ by the application $\Phi_z$. We proved earlier that gradient ascent is not a proper way of inverting $\Phi_z = id - \lambda g_z$.

Consider the case of linear regression, where $g_z(\theta) = l(\theta) - y x = (\theta \cdot x) x - y x$ is an affine map of $\theta$. Assuming that the model shift is small ($||\lambda g_z|| < 1$), there is an analytical inverse of $\Phi_z$. Indeed, $\tilde{\theta} := \Phi_z(\theta) + \lambda y x = \theta - \lambda l(\theta)$ is a linear function of $\theta$ inversed by $\theta = \displaystyle \sum_{k=0}^{\infty} \lambda^k l^k (\tilde \theta) \approx \tilde{\theta} + \lambda (\tilde{\theta} \cdot x) x$

However, analytical formulas fall short with complex models since there are no guarantees on the regularity of $g_z(\theta)$. Furthermore, $\Phi_{z_t}$ may well be non-injective.

### The fixed-point formulation

In the case of gradient descent, another formulation of this problem can be expressed by finding a fixed point of the application

$$\Psi_{t+1} (\theta) = \theta_{t+1} + \lambda g_{z_t}(\theta)$$

#### Brouwer's fixed point theorem

Brouwer's fixed-point theorem can't be applied in general since $\Psi_{t+1}$ is not continuous with respect to $\theta$. We have:
$$g_{z_t} = \nabla_\theta l(h_\theta (x_t), y_t)$$
Since $h_\theta$ is usally irregular, even less its derivative, $g_{z_t}$ is not continous.

#### Banach's fixed point theorem

Banach's fixed-point theorem would not apply either since the quantity $\Psi_{t+1} (\theta_2) - \Psi_{t+1} (\theta_1) = \lambda (g_{z_t} (\theta_2) - g_{z_t} (\theta_1))$ may not be bounded by $\theta_2 - \theta_1$ since $g_{z_t}$ is irregular.

TODO: experimental verification of the irregularity of $g_z$ and failure of Banach's fixed point theorem

Therefore $\Psi_{t+1}$ is likely not a contraction mapping.

#### Pathways for success

However, we conjecture that the hypotheses of these fixed-point theorems are "closer" to be verified if:
- $\Phi_z$ acts as a regularizer, making the model parameters sparser thus decreasing their norm $|\theta|$. This may contribute to the condition that $\Phi_z$ is a contraction mapping.
- $\theta$ is robust to small variations in the inputs, so $h_\theta$ is Lipschitz.

We argue that better unlearning, if possible, would require a holistic approach tackling both model architecture and training method, towards robust and explainable machine learning.


## Inverting many gradient steps is hard since they don't commute

In this section, we consider the following problem:
given $\mathcal{U} \subset \mathcal{T}$ a set of gradient steps to unlearn, and $\theta_T$ the final model trained on all of the steps $\mathcal{T}$, compute the model that would have been trained on $\mathcal{T} \backslash \mathcal{U}$.

Let $(z, z') \in \mathscr{Z}^2$ be two data points. In general, $\Phi_z$ and $\Phi_{z'}$ do not commute.

In the case of gradient descent, this noncommutativity occurs since the gradient depends on $\theta$. To illustrate this observation, consider the linear regression model.

We have $\nabla_\theta l(h_\theta (x), y) = (\theta \cdot x - y) x$, which is an affine projection on the vector $x$, orthogonally to hyperplane of equation $\theta \cdot x = y$.

It follows that $\Phi_z (\theta) = \theta - \lambda (\theta \cdot x - y) x$ is also an affine application, so $\Phi_z$ and $\Phi_{z'}$ don't commute unless $z = z'$. Even two linear orthogonal projectors of rank $1$ don't commute unless equal.

### Difficult cases where gradient steps do not commute

We draw the following hypotheses:
- Gradient steps are unlikely to commute for two data points that are very different (i.e of different classes).
- Poisons with high amplitude are hard to unlearn, since the clean and the poisoned samples would differ by a lot. This is coherent with the theory of differential privacy: out-of-distribution points are harder to hide than average points.
- Machine unlearning gets harder with higher-dimensional input spaces since collinearity is difficult to achieve.

### Easier cases where gradient steps might almost commute

We might observe near commutation in the following cases:
- learning rate is low (such as in finetuning), thus the model shift is small enough to be unlearned.
- the model is trained on a sequence of similar data points (e.g in the same class). However this goes against best practices in machine learning.
- the forget set is deinterleaved with the retain set, which is an unreasonable assumption in distributed learning.
- the loss function has a single minimum $\theta^*$. In that case we could observe _asymptotic commutativity_, meaning that **any** order of the optimization steps converges towards the same optimum. This is true for convex models, but does not hold with complex neural networks.

### The problem of interleaved gradient steps

If the forget set $\mathcal{U}$ is interleaved with the retain set $\mathcal{T} \backslash \mathcal{U}$, different permutations of the gradient steps would likely make the model converge to different optima. In other words, the order of gradient steps ultimately matter.

Plenty of evidence arise from the widely-known technique of shuffling the training data in order to avoid bias and to mitigate convergence to a local minimum.

[Intuitive explanation here](https://www.deepwizai.com/simply-deep/why-random-shuffling-improves-generalizability-of-neural-nets)

[Slightly formal explanation here](https://stats.stackexchange.com/questions/245502/why-should-we-shuffle-data-while-training-a-neural-network)

TODO: experiment with CIFAR-10, measure accuracy, check pathological cases (non-shuffled classes)

This non-commutativity of gradient steps makes it impossible to exactly unlearn an interleaved subset of the data. Suppose the defender is able to invert a single gradient step in $\mathcal{O}(1)$ time w.r.t $T$. If of the gradient steps commute, there is no efficient algorithm for unlearning $\mathcal{U}$ using gradient step inversion. Indeed, such an algorithm would have to invert the whole gradient steps $\mathcal{T}$ and retrain on them in $\mathcal{O}(T)$ time, which defeats the purpose of efficient machine unlearning.

### Poisoned batches

Consider the following batch update rule with mean aggregation
$$g_t = \frac{1}{B} \sum_{i=1}^B \nabla_\theta l(h_{\theta_t} (x_{t,i}), y_{t,i})$$

If a random subset of dataset is poisoned, any batch can be separated in a clean subset and a poisoned subset.
We may write $g_t = g_{t,c} + g_{t,p}$ where $g_{t,c}$ is the average gradient over clean samples and $g_{t,p}$ is the average gradient over poisoned samples. For simplification, suppose $B = 2$ and there is exactly one clean data point and one poisoned data point in the batch. Even if the defender could estimate $g_{t,c}$ (like NegGrad+ does), they have no knowledge of the poisoned gradient $g_{t,p}$. Therefore, unlearning in a non-batched framework reduces to the batched case, meaning that unlearning specific points from a batch is at least as hard as in the non-batched version.

## The case of approximate unlearning

Approximate unlearning aims to obtain a statistically equivalent model to an exact deletion model. However, approximate unlearning would need to remove the potentially large bias created by the forget set, which is unrecoverable given that most approximate unlearning methods only induce a small shift on the model ([Pawelczyk et al.](https://arxiv.org/abs/2406.17216)).


## Conclusion

We have shown that gradient-based unlearning methods such as gradient ascent are unable to perform exact unlearning on non-trivial models, which is due to two major reasons:
- Inverting gradient steps is hard
- Gradient steps do not commute, and poisons are usually interleaved in the batches

Furthermore, poisons are also interleaved within batches.

