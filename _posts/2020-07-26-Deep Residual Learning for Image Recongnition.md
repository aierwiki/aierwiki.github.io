# Deep Residual Learning for Image Recognition

## What's the problem
- **The Degradation(of training accuracy)** : With the network depth increasing, accuracy gets saturated(which might be unsuprising) and then degrades rapidly. Adding more layers to a suitably deep model leads to higher training error.

## How and Why
- We address the degradation problem by introducing a deep residual learning framework.
- Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping.
- Formally, denoting the desired underlying mapping as $$\mathcal{H}$$, we let the stacked nonlinear layers fit another mappings of $$\mathcal{F(x)}:=\mathcal{H}(x) - x$$. The original mapping is recast into $$\mathcal{F}(x) + x$$. 
- We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.
- To the extrem, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by stack of nonlinear layer.
- The sortcut connections simply perform identity mapping.
- We consider a building block defined as:$$y=\mathcal{F}(x, {W_i}) + x$$ Here x and y and the input and output vectors of the layers considered. 
- The Function $$\mathcal{F}(x, {W_i})$$ (i.e., $$\mathcal{F}=W_2\theta(W_1x)$$) represents the residual mapping to be learned.
- The operation $$\mathcal{F} + x$$ is performed by a shortcut connection and element-wise addition. We adopt the second noninearty af the addition.
- If the dimensions of $$x$$ and $$\mathcal{F}$$ 
  are not equal, we can perform a linear projection $$W_s$$ by the shortcut connections to match the dimensions:$$y=\mathcal{F}(x, {W_i}) + W_sx$$
- The function $$\mathcal{F}(x, {W_i})$$ can represent multiple convolutional layers. The element-wise addition is performed on two feature maps, channel by channel.
- When the dimensions increase, we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. (B) The projection shortcut is used to match dimensions.
- We conjecture that the deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error.
- So we use 0.01 to warm up the training until the training error is below 80%, and then go back to 0.1 and continue training.
- ResNet have generally smaller responses than their plain counterparts. These results support our basic motivation that the residual functions might be generally closer to zero than the no-residual functions.

## Advantages of the method
- These residual networks are easier to optimize, and can gain accuracy from considerably increased depth.
- The extremely deep representations also have excellent generalization performance.
- These gates are data-dependent and have parameters, incontrast to our identity shortcuts that are parameter-free.
- The shortcut connections introduce neither extra parameter nor ccomputation complexity.

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
