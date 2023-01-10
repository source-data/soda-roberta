The goal is to generate a causal model of the world. Ideally, in an unsupervised way.

- use convolution or Hartford or Maron on small cell and stack layers to reduce
- learn permutation matrix as well as adjascency matrix
- some generative model? https://arxiv.org/abs/1803.03324
- completely abandon tensor and use true graph-based objects
- do causal inference on the BERT params or embeddings
- use the large-scale SmartTagged dataset to generate counterfactuals: replace actual interventions with another and train to predict correct outcome: https://aclanthology.org/2020.emnlp-main.590.pdf Counterfactual Generator
- invertible attention: swap projection of query and key should reverse direction of interactions
- transform particular adjascency matrix by graph message passing into permutation invariant representation
- use Laplacian as embedding of adjascency matrix?
- forget about permutation invariance and use large adjascency and entity matrices that build a complete world model
- use adversarial loss for causal vs reversed causal