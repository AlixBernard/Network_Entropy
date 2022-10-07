# Overview

Entropy computation of a Network based on [\[1\]][1], which makes use of the gHypEG from [\[2\]][2].

**IMPORTANT**: This implementation only works for the case taken into account in [\[1\]][1], _ie._ only the case of undirected graph without self-loops. In order to use it for other types of networks &mdash; not recommended, may not be possible with this method &mdash; some modifications are in order, such as modifying the entries of the matrix `Xi` (cf. [\[1\]][1] & [\[2\]][2] for more details).  



# Installation

To install:
```bash
git clone https://github.com/AlixBernard/Network_Entropy.git
cd Network_Entropy
pip install .
```

To uninstall:
```bash
pip uninstall network_entropy
```



# Example of use

An example using the _Karate Club_ dataset in `Data/karate_club.dat`, also used in [\[1\]][1], is present in `Examples/run_karate_club.py`.  
Run in the terminal:
```bash
cd Network_Entropy
python Examples/run_karate_club.py
```
Output:  
```
          Name   n    m   m/n     D  H_norm
0  Karate Club  34  231 6.794 0.139   0.306
```



# References

\[1\]: [What is the Entropy of a Social Organization?](https://arxiv.org/abs/1905.09772) \[Zingg et al.\]  
\[2\]: [Generalised hypergeometric ensembles of random graphs: the configuration model as an urn problem](https://arxiv.org/abs/1810.06495) \[Casiraghi, Nanumyan\]  

[1]: https://arxiv.org/abs/1905.09772  
[2]: https://arxiv.org/abs/1810.06495  
