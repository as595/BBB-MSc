
## Project Description

*Standard neural networks optimise point-wise estimates of the weights for each of their neurons. In contrast Bayesian neural networks assign a probability distribution to each weight instead of a single value or point estimate. These probability distributions represent the uncertainty in the weights and are propagated through the network to provide an uncertainty on the final predictions. This has been reported to reduce over-fitting to specific training data sets and to allow such networks to train more general models from fewer data samples. In this project the student will make a comparison of classifications using standard neural networks with those from an equivalent Bayesian neural network, examining the posterior probability distributions in each case. This will be done for specific astronomical classification problems to examine whether the application of Bayesian neural networks can result in more reproducible classification results across multiple test data sets.  This project will require good computing skills and previous experience with Python is desirable.*


## Plan of Work

- Replicate the MNIST experiment from [Blundell et al. 2015](https://arxiv.org/pdf/1505.05424.pdf) as a validation check.
- Establish a baseline FRI/FRII classification for MiraBest using an appropriate network.
- Use the approach in [Blundell et al. 2015](https://arxiv.org/pdf/1505.05424.pdf) for the MiraBest data set and compare the results.
- Implement a weight-pruning method based on the probability distribution associated with each network weight, following [Sec.5.1 of the Blundell paper](https://arxiv.org/pdf/1505.05424.pdf)
- Examine the distribution of epistemic uncertainty with respect to the characteristics of the test set radio galaxy population - is there a systematic bias in the confidence of classification for particular types of galaxy?
- Look at the [transfer learning](https://arxiv.org/pdf/1903.11921.pdf) ability of your network to the [FRDEEP data set](https://github.com/HongmingTang060313/FR-DEEP) - are the results better than for a non-Bayesian version of the same network? What about for the NVSS images of the same test objects?


### Possible extensions:

 1. Write a custom dropout layer to implement a feature ranking, following [this paper](https://arxiv.org/pdf/1712.08645.pdf)
 2. Adapt the network to predict aleatoric uncertainty for each data sample as well, following e.g. [this paper]( https://arxiv.org/pdf/2005.07174.pdf).
