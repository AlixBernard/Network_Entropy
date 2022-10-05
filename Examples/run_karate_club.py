#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2020-12-01
# @Last modified by: AlixBernard
# @Last modified time: 2022-10-05 17:57:06

"""Program reading the karate club data set from its edges matrix
from the karate.dat file and process it to obtain its entropy.
"""


# Built-in packages
from pathlib import Path

# Third party packages
import numpy as np
import networkx as nx

# Local packages
from network_entropy import Network


DATA_FOLDER = Path(__file__).resolve().parents[1] / "Data"


def get_edges(data_path, matrix_size):
    """Get the edges of the network matric from the file `data_path` and
    returns them as an array.

    """
    edges = np.zeros((matrix_size, matrix_size))
    with open(data_path, "r") as file:
        i = 0
        for line in file.readlines():
            if line[0] != " ":
                # Skip lines until matrix without incrementing `i`
                continue

            line = line.strip().split(" ")

            if matrix_size != len(line):
                print("Matrix size error")

            edges[i] = line
            i += 1

    return edges


def main():
    # Define paths
    data_name = "Karate Club"
    data_filename = "karate_club.dat"
    data_path = data_folder / data_filename

    edges = get_edges(data_path, matrix_size=34)
    g = nx.MultiGraph(edges)

    org = Network(g, name=data_name, case=3, edges_matrix=edges)
    org.do_the_work()
    org.display(precision=3)


if __name__ == "__main__":
    main()
