#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2020-12-01
# @Last modified by: AlixBernard
# @Last modified time: 2021-07-23

"""Program reading the karate club data set from its edges matrix
from the karate.dat file and process it to obtain its entropy.
"""


# Third party packages
import os
import numpy as np
import networkx as nx
from pathlib import Path

# Local packages
from network_entropy import Network


# Get edges of the network matrix from the file data_path
data_name = "Karate Club"
data_folder = Path('/'.join(os.path.realpath(__file__).split('/')[:-2]
                            + ["Data"]))
data_filename = Path("karate_club.dat")
data_path = Path(data_folder / data_filename)
matrix_size = 34
edges = np.zeros((matrix_size, matrix_size))
with open(data_path, 'r') as file:
    i = 0
    for line in file.readlines():
        if line[0] != ' ':
            continue
        line = line.strip().split(' ')
        
        if matrix_size != len(line):
            print("Matrix size error")
        
        for j in range(matrix_size):
            edges[i,j] = line[j]
        i += 1


g = nx.MultiGraph(edges)
org = Network(g, name=data_name, case=3, edges_matrix=edges)
org.do_the_work()
org.display(precision=3)