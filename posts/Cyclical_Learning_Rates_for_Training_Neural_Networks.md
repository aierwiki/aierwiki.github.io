# Cyclical Learning Rates for Training Neural Networks

## The Problem to solve
- Too small a learning rate will make a training algorithm coverge slowly while too large a learning rate will make the training algorithm diverge.
- One must experiment with  a variety of learning rates and schedules.

## How and Why
- Let the global learning rate vary cyclically within a band of values instead of setting it to a fixed value.
- The learning rate to rise and full is beneficial overall even though it might temporarily harm the network's performance.
- The difficulty in minimizing the loss arises from saddle points rather than poor local minima.
- Saddle points have small gradients that slow the learning process.
- Increasing the learning rate allows more rapid traversal of saddle point plateaus.
- It is likely the optimum learning rate will be between the bounds and near optimal learning rates will be used  throughout training.
- **Step size** The final accuracy results are actually quite robust to cycle leangth but experiments show that it often is good to set stepsize equal to 2-10 times
the number of iterations in an epoch.
- **Minimum and maximum boundary values** Run your model for several epochs while letting the learning rate increase linearly between low and high LR values.
- Note the learning rate value when the accuracy starts to increase and when the accuracy slows. These two learning rates are good choice for bounds.



## Advantages
- Practically eliminates the need to experimentally find the best values and schedule for the global learning rates.
- Improved classification accuracy without a need to tune and often in fewer iterations.
- The CLR methods require essentially no additional computation.
