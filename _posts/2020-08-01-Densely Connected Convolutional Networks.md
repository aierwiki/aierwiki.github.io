## Densely Connected Convolutional Networks



### What's the problem

- As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and "wash out" by the time it reaches the end (or beginning) of the network.
- Recent variations of ResNets show that many layers contribute very little and can in fact be randomly dropped during training. This makes the state of ResNets similar to (unrolled) recurrent neural networks, but the number of parameters of ResNets is substantially larger because each layer has it own weights.
- The identity function and the output of $H_\mathcal{l}$ are combined by summation, which may impede the information flow in the network.



### What to do

- Whereas traditional convolutional networks with L layers have L connections -- one between each layer and its subsequent layer --our network has $$\frac{L(L+1)}{2}$$ direct connections.

- we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to allsequent layers.

- In contrast to ResNets, we never combine features through summation before they are passed into a layer; instead, we combine features by concatenating them.

- This introduces $$\frac{L(L+1)}{2}$$ connections in an L-layer network, instead of just L, as in traditional architectures.

- Consequently, the $$\mathcal{l}^{th}$$ layer receives the feature-maps of all preceding layers, $$x_0,...,x_{l-1}$$, as input:

  $$x_l=H_l([x_0, x_1, ...,x_{l-1}])$$

- The concatenation operation is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps.

- The transition layer used in our experiments consist of a batch normalization layer and 1x1 convolutional layer followed by a 2x2 average pooling layer.

- A 1x1 convolution can be introduced as bottleneck layer before each 3x3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency.

### Why

- Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency.
- One explanation for this is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network's "collective knowledge".



### Advantages

- Alleviate the vanishing-gradient problem
- strengthen feature propagation
- encourage feature reuse
- substantially reduce the number of parameters
- Besides better parameter efficiency, one big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train.
- Dense connections have a regularizing effect, which reduces overfitting on tasks with smaller training set sizes.

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
