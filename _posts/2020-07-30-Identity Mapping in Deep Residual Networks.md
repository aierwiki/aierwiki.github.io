## Identity Mapping in Deep Residual Networks

### Conclusion
- The forward and backward signals can be directly propagated from one block to another block, when using identity mappings as the skip connections and after-addition activation.

### Demonstrate
- Deep residual networks consist of many stacked "Residual Units". Each unit can be expressed in a general form:
$y_l = h(x_l) + \mathcal{F}(x_l, \mathcal{W}_l),$ 
$x_{l+1} = f(y_l),$ where $x_l$ and $x_{l+1}$ are input and output of the $l-th$ unit, and $\mathcal{F}$ is a residual function.

- In origin model, $h(x_l) = x_l$ is an identity mapping realized by attaching an identity skip connection ("shortcut") and $f$ is a RELU function.
- In proposed model, We creating a "direct" path for propagation information - not only within a residual unit but through the entire network. Both $h(x_l)$ and $f(y_l)$ are identity mappings. The signal could be directly propagated from one unit to any other units in both forward and backward passed.
- To construct an identity mapping $f(y_l) = y_l$, we view the activation functions(ReLU and BN) as "pre-activation" of the weight layers, in constrast to conventional wisdom of "post-activation".
