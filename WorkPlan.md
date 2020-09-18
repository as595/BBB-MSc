
## Project Description

*Standard neural networks optimise point-wise estimates of the weights for each of their neurons. In contrast Bayesian neural networks assign a probability distribution to each weight instead of a single value or point estimate. These probability distributions represent the uncertainty in the weights and are propagated through the network to provide an uncertainty on the final predictions. This has been reported to reduce over-fitting to specific training data sets and to allow such networks to train more general models from fewer data samples. In this project the student will make a comparison of classifications using standard neural networks with those from an equivalent Bayesian neural network, examining the posterior probability distributions in each case. This will be done for specific astronomical classification problems to examine whether the application of Bayesian neural networks can result in more reproducible classification results across multiple test data sets.  This project will require good computing skills and previous experience with Python is desirable.*


## Plan of Work

- Establish a baseline classification for HTRU1 using a standard MLP.
- Replicate [Figure 1 from Hinton et al. 2012](https://arxiv.org/pdf/1207.0580.pdf) for HTRU1 to establish the effect of dropout on performance.
- Use the approach in [Section 5.2 of Gal & Ghahrami 2015](https://arxiv.org/pdf/1506.02142.pdf) to estimate the epistemic uncertainty for each object in the test set.
- Examine the distribution of epistemic uncertainty with respect to the characteristics of the test set pulsar population - is there a systematic bias in the confidence of classification for particular types of pulsar? (compare with e.g. Fig 5 of [this paper](http://arxiv.org/abs/1406.3627))
- Test how the epistemic uncertainty depends on the selection of training data. 

### Possible extensions:

 1. Write a custom dropout layer to implement a feature ranking, following [this paper](https://arxiv.org/pdf/1712.08645.pdf)
 2. Adapt the network to predict aleatoric uncertainty for each data sample as well, following e.g. [this paper]( https://arxiv.org/pdf/2005.07174.pdf).
