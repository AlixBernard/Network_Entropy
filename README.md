# network_entropy
Entropy computation of a Network based on the paper *What is the Entropy of a Social Organization?* [Zingg et al.], which makes use of the gHypEG from the paper *Generalised hypergeometric ensembles of random graphs: the configuration model as an urn problem* \[Casiraghi, Nanumyan\]  
This implementation only works for the case taking into account in the paper referenced, *i.e.* networks with undericted graphs and without self-loops (described by case `3` in the code).  
In order to  use it for other types of networks some modifications are in order, such as modifying the entries of the matrix `Xi` (cf. the referenced papers for more details).  

# References
[What is the Entropy of a Social Organization? [Zingg et al.]](https://arxiv.org/abs/1905.09772)  
[Generalised hypergeometric ensembles of random graphs: the configuration model as an urn problem \[Casiraghi, Nanumyan\]](https://arxiv.org/abs/1810.06495)  
