## Identity Mapping in Deep Residual Networks

### Conclusion
- The forward and backward signals can be directly propagated from one block to another block, when using identity mappings as the skip connections and after-addition activation.

### How
- Deep residual networks consist of many stacked "Residual Units". Each unit can be expressed in a general form:
$$y_l = h(x_l) + \mathcal{F}(x_l, \mathcal{W}_l),$$, $$x_{l+1} = f(y_l),$$ where $$x_l$$ and $$x_{l+1}$$ are input and output of the $$l-th$$ unit, and $$\mathcal{F}$$ is a residual function.

- In origin model, $$h(x_l) = x_l$$ is an identity mapping realized by attaching an identity skip connection ("shortcut") and $$f$$ is a RELU function.
- In proposed model, We creating a "direct" path for propagation information - not only within a residual unit but through the entire network. Both $$h(x_l)$$ and $$f(y_l)$$ are identity mappings. The signal could be directly propagated from one unit to any other units in both forward and backward passed.
- To construct an identity mapping $$f(y_l) = y_l$$, we view the activation functions(ReLU and BN) as "pre-activation" of the weight layers, in constrast to conventional wisdom of "post-activation".


### Why
- The gradient can be decomposed into two additive terms: a term that propagates information directly without concerning any weight layers, and another term of that propagates through the weight layers.
- This product may also impede information propagation and hamper the training procedure as witnessed in the following experiments.
- The shortcut connections are the most direct paths for the information to propagate.
- Multiplicative manipulations(scaling, gating, 1*1 convolutions and dropout) on the shortcuts can hamper information propagation and lead optimiztion problems.
- The degradation of these models is caused by optimization issues, instead of representational abilities.
- A naive choice of making f into an identity mapping is to move the ReLU before addition. However, this leads to a non-negative output form the transform $$\mathcal{F}$$, while intuitively a "residual" function should take values in $$(-\infty, +\infty)$$.
- The optimization is further eased, because $$f$$ is an identity mapping. Using BN as pre-activation improves regularization of the models.
- Because after some training, the weights are adjusted into a status such that $$y_l$$ is more frequently above zero and $$f$$ does not truncate it.
- This is presumably caused by BN's regularization effect.

### Advantages
- There is much room to exploit the dimension of network depth, a key to the success of modern deep learning.
- The potential of pushing the limits of depth.
- Identity shortcut connections and identity after-addition activation are essential for making information propagation smooth.



<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>