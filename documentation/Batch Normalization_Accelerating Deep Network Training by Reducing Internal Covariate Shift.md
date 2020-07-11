# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

## What's the problem
- **internal covariate shift** : The training of deep neuron network is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding
layers - so that small changes to the network parameters amplify as the network becomes deeper. The change in the distributions of layers' inputs presents a problem because 
the layers need to continuously adapt to the new distribution.
- **the saturation problem** : Consider a layer with a sigmoid activation $$z = g(Wu+b)$$ . As |x| increases, g'(x) tends to zero. This means that for all dimensions of $x = Wu+b$ 
except those with small absolute values, the gradient flowing down to u will vanish and the model will train slowly. The saturation problem and the resulting vanishing gradients
are usually addressed by careful initialization and **small learning rates**.

## How and Why
- Normalize each scalar feature independently, by making it have the mean of zero and the variance of 1.
- Introduce, for each activation $x^(k)$, a pair of parameters $\gama^(k)$ and $\beta^(k)$, which scale and shift the normalized value: $$y^(k) = \gama^(k)x^(k) + \beta^(k)$$.
Because simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear
regime of the linear regime of the nolinearity. By do that, we can make sure that the transformation inserted the network can represent the identity transform, and preserves
the network capacity.
- Each mini-batch produces estimates of the mean and variance of each activation. The use of mini-batcches is enabled by computation of per-dimension variances rather than joint
covariances.
- The means and variances are fixed during inference. we use the normalization using the populations, rather than mimi-batch, statistics.
- We add the BN transform immediately before the nonlinearity, by normalizing $x = Wu+b$. $Wu+b$ is more likely to have a symmetric, non-sparse distribution, that is "more
Gaussian", normalizing it is likely to produce activations with a stable distribution.

## Advantages of the method
- Batch Normalization enables higher learning rates and then accelerate the training of deep networks. In traditional deep networks, too-high learning rate may result in the
gradients that explode or vanish, as well as getting stuck in pool local minima. By mormalization activations throughout the network, it prevents small changes to the parameters
from amplifying into larger and suboptimal changes in activations in gradients; for instance, it prevents the training from getting stuck in the saturated regimes of 
nonlinearities.
- Batch Normalization adds only two extra parameters per activation, and in doing so perseves the representation ability of the network.
- Batch Normalization regularizes the model. A training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing
deterministic values for a given training example.
