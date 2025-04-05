
# Machine Unlearning vs. Gaussian Poisoning


## Possible ideas towards a result

privacy / usefulness: look around DP and MIA
entanglement & data graph structure (paper showcased in AI, Science & Society)

Gradient Ascent: empirical relation GUS/accuracy

### Batch interlacing and federated learning

interlaced clean data / poisoned data:
- each batch partially poisoned
- some batches poisoned
-> federated learning aggregation methods
-> data poisoning & byzantine attack equivalence?

### Limitations of gradient-based unlearning 

gradient descent steps are fully invertible when:
- there is no data intrication (steps commute)
- or the defender is omniscient and there is no poison interlacing
- or data is easy to unlearn (sparse representation such as in MNIST). problem: embedding representation is not continuous -> feature overlap, non-trivial graph -> possibility of poisoning

### Data intrication

two types of intrication:
- feature intrication (What makes unlearning hard and what to do about it)[https://arxiv.org/abs/2406.01257]
- batch intrication

### Model shift

measuring the added entropy of poisoning and the removed entropy of training (unlearning)

-> impossibility of exact unlearning with gradient-based methods
-> difficulty of $(\varepsilon, \delta)$ unlearning with gradient-based methods (defined in _Are we making progress in unlearning? findings from the first neurips unlearning competition_)
Potential impossibility of efficient provable approximate unlearning (reversing gradient steps): cf. orthogonality between model shift and unlearning updates

### The curse of dimension

larger models (ResNet) are slower to converge on simple datasets (CIFAR-10) but have a higher overfitting capacity?
TODO: perform GUS/dimension measurement on smaller ResNet models <= ResNet18

### Unlearnable examples

Out-of-distribution examples need to be memorized -> harder to unlearn: defined _Does learning require memorization? a short tale about a long tail_. Alternative: influence functions
Craft examples that cannot be unlearned without data cleaning?
Under/overforgetting: is forget set accuracy a good metric?

$I_z$ distribution with poisoning might be Student t distribution or Cauchy



### Noncommutative optimization steps

- The operator $g_{x, y}: \theta \mapsto \nabla_\theta l(h_\theta (x), y)$ is hard to invert (H1).
- $g_{x_1, y_1}$ and $g_{x_2, y_2}$ don't commute in general (H2).
- Many optimization steps can asymptotically commute if the gradient descent converges (e.g linear model with large numbers law, O3). Otherwise a random permutation of the batches may incur a large shift in the model parameters (H3).

**Supporting experiments:**
- Verify (H2), check if it commutes better when $y_1 = y_2$ thus $x_1 \approx x_2$ (same label in classification)
- Verify (H3)
- Verify (03) and compare to a non-linear, non-trivial model


#### Gradient ascent does not reverse gradient descent

**Supporting experiments:**
- Verify with one data point to unlearn, then one batch, then a whole dataset, **in the same order**.

Provide a theoretical explanation with linear regression for example.

#### Unlearning gets more costly with many epochs to unlearn


## Future work

https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

https://github.com/Lightning-AI/litgpt/blob/main/tutorials/0_to_litgpt.md
https://lightning.ai/blog/scaling-large-language-models-with-pytorch-lightning/
https://lightning.ai/lightning-ai/studios/pretrain-an-llm-with-pytorch-lightning?section=featured