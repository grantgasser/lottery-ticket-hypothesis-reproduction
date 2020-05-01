# Reproducing Lottery Ticket Hypothesis Paper
This is an attempt to reproduce the [Lottery Ticket Hypothesis (LTH) paper](https://arxiv.org/abs/1803.03635) by Frankle and Carbin. 

## Basic Summary 
This paper in a few words: 1) Train network 2) Prune unecessary weights/connections. It would be great to figure out how to identify the subnetwork without having to build and train full network. This pruning method can reduce the number of parameters by 10x while maintaining the same performance. See [this article](https://www.technologyreview.com/2019/05/10/135426/a-new-way-to-build-tiny-neural-networks-could-create-powerful-ai-on-your-phone/) for more. 

## Project Proposal

### Cited Literature

### Follow Up Literature
* [Sparse Transfer Learning](https://paperswithcode.com/paper/sparse-transfer-learning-via-winning-lottery) with [code](https://github.com/rahulsmehta/sparsity-experiments)
* [Deconstructing Lottery Tickets - Uber](https://eng.uber.com/deconstructing-lottery-tickets/) with [code](https://github.com/uber-research/deconstructing-lottery-tickets)

### Codebase Search
* [Re-implementation](https://github.com/google-research/lottery-ticket-hypothesis) by original author at Google
* [Pruning w/ PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
* [Sparse Transfer Learning code](https://github.com/rahulsmehta/sparsity-experiments)
* [Uber paper code](https://github.com/uber-research/deconstructing-lottery-tickets)

### Paper Review
Based on [this paper](https://papers.nips.cc/paper/8787-a-step-toward-quantifying-independently-reproducible-machine-learning-research.pdf)

#### Mildly Subjective
* Number of Tables: 1
* Number of Graphs/Plots: 25 (paper), 131 (appendix)
* Number of Equations: 7
* Proofs: 0
* Exact Compute Specified: NOT SPECIFIED
* Hyperparameters: ![Figure 2](lottery_ticket_hyperparameters.png) 
* Compute Needed: NOT SPECIFIED
* Data Available: CIFAR and MNIST
* Pseudo Code: Level 2 ("Step-Code", high level)

#### Subjective
* Number of Conceptualization Figures: 0
* Uses Exemplar Toy Problem: No
* Number of Other Figures: 0
* Rigor vs Empirical: Empirical
* Paper Readability: Good
* Algorithm Difficulty: Medium
* Paper Topic: Experimental Results
* Intimidating: No

### Tools
* [Paperspace GPUs](https://gradient.paperspace.com/free-gpu)
* Try to use [PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) for pruning. If it doesn't work, implement ourselves. 

### Reproduction Advice
* Can't prune every layer equally, keep track of that stuff
* Learning rate matters for the bigger models but not as much for the smaller models
* Be careful using pytorch in-built pruning

### Tentative Proposed Timeline (w/ weekly milestone)
* Week 1: Reproduce MNIST 97-98% in PyTorch
* Week 2: Reproduce Figure 1 from paper
* Week 4: Reproduce Section 2: Winning Tickets in Fully-Connected Networks
* Week 6-7: Reproduce Section 3: Winning Tickets in Convolutional Networks

## LeNet Reproduction (1st milestone)
Like the authors of the LTH paper, we attempt to achieve ~98% accuracy on the MNIST dataset with LeNet.
We were able to achieve these results on a validation set after 2 epochs of 60K iterations: `Test set: Average loss: 0.0426, Accuracy: 9862/10000 (99%)`.

Here's some details on the architecture and hyperparameters:
* MNIST Images: `[1, 28, 28]`
* 2 Convolutional Layers
* 3 Fully Connected Layers
* Batch Size: `64`
* Optimizer: Adam
* Learning Rate: `1.2e-3`

### More Details on Convolutional Architecture
The output dimension of a convolutional layer can be calculated like so `O = (I - K + 2P)/S + 1`, where `I`: input dim,
`K`: kernel size, `P`: padding, and `S`: stride. The output dimension of a
pooling layer can be calculated like so `O = (I - P)/S + 1` where `P`: size of pooling kernel. The dimensions of
the data change in the following way: `[64, 1, 28, 28] => conv1 => [64, 6, 24, 24] => maxPool => [64, 6, 12, 12]
=> conv2 => [64, 6, 8, 8] => maxPool => [64, 6, 4, 4] => flatten => [64, 256] => fc1 => [64, 300] => fc2 => [64, 100]
=> fc3 => [64, 10]`

### Interesting Note on Optimization
Adam with `lr=1.0` did not converge, though Adadelta with `lr=1.0` did converge. 
Ultimately, we used Adam and `lr=1.2e-3.`



