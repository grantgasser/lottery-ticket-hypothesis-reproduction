# Reproducing Lottery Ticket Hypothesis Paper
This is an attempt to reproduce the [Lottery Ticket Hypothesis paper](https://arxiv.org/abs/1803.03635) by Frankle and Carbin. 

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
