# Overview
Entropy computation of a Network based on [\[1\]][1], which makes use of the gHypEG from [\[2\]][2].

**IMPORTANT**: This implementation only works for the case taken into account in the paper referenced, *i.e.* networks with undirected graphs and without self-loops (described by case `3` in the code). In order to  use it for other types of networks some modifications are in order, such as modifying the entries of the matrix `Xi` (cf. the referenced papers for more details).  

# Example of use
Prior to running this sample of code, `g` must be defined as a `networkx.MultiGraph` meaning it is an undirected graph with multi-edges, and `edges` must be an `np.array` representing the matrix of edges of the graph, such that if it is undirected then the bottom triangle of the matrix is `0`.
```py
org = Network(g, name='My Organization', case=3, edges_matrix=edges)  # Initiate the object
org()  # Perform the computations to obtain the entropy
org    # Display some quantities of interest including the entropy
```

# References
\[1\]: [What is the Entropy of a Social Organization?](https://arxiv.org/abs/1905.09772) \[Zingg et al.\]  
\[2\]: [Generalised hypergeometric ensembles of random graphs: the configuration model as an urn problem](https://arxiv.org/abs/1810.06495) \[Casiraghi, Nanumyan\]  

[1]: https://arxiv.org/abs/1905.09772  
[2]: https://arxiv.org/abs/1810.06495  
