#!/usr/bin/env python3
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2020-12-01
# @Last modified by: AlixBernard
# @Last modified time: 2019-12-05

"""
This program computes the entropy of a network such as
a social organization to evaluate its adaptability.
Only the case of undirected networks without self-loops
are implemented (case 3). Multiple edges between two same nodes
are possible and should be used for each different
interaction between these nodes.
"""


from math import log, comb, factorial
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import networkx as nx


def multinomial_entropy(n, m, Xi, Omega, p, log_base):
    r""" Compute the multinomial entropy and returns it.
    math::
        \[ H^{mult} = - \log (m!) - m  \sum_{i,j \in V, i<j} p_{ij} \log (p_{ij})
                      + \sum_{x=2}^m \sum_{i,j \in V, i<j} p_{ij}^x (1-p_{ij})^{m-x} \log(x!) \]
    """
    s1 = sum([p[i,j] * log(p[i,j], log_base) if p[i,j]!=0 else 0
                        for i in range(n)
                        for j in range(n)])
    s2 = sum([sum([comb(m, x) * p[i,j]**x * (1-p[i,j])**(m-x) * log(factorial(x), log_base)
                         if p[i,j]!=0 else 0
                         for i in range(n)
                         for j in range(n)])
                    for x in range(2, m+1)])
    H_mult = - log(factorial(m), log_base) - m * s1 + s2
    return H_mult


class Network():
    """
    Parameters
    ----------
    network: nx.Graph
        Graph of the network in the type of nx.Graph, nx.DiGraph,
        nx.MultiGraph, or nx.MultiDiGraph
    case: int
        Case of the graph, 0: directed with self-loops,
        1: undirected with self-loops, 2: directed without
        self-loops, 3: undirected without self-loops
    log_base: int
        Logarithmic base to use when computing the entropy.
    name: str
        Name of the network
    edges_matrix: np.array
        Matrix of the graph edges, if None then taken without 
        multi-edges from the nx.Graph input
    
    Attributes
    ----------
    network: nx.Graph
        Graph of the network in the type of nx.Graph, nx.DiGraph,
        nx.MultiGraph, or nx.MultiDiGraph
    case: int
        Case of the graph, 0: directed with self-loops,
        1: undirected with self-loops, 2: directed without
        self-loops, 3: undirected without self-loops
    log_base: int
        Logarithmic base to use when computing the entropy.
    Xi: np.array
        Matrix encoding the configuration model.
    Omega: np.array
        Matrix encoding the preferences of the nodes.
    theta: np.array
        Correction vector for the cases 2 and 3, unit vector otherwise.
    H_mult: float
        Multinomial entropy of the network.
    H_max: float
        Maximum multinomial entropy attainable for the network.
    H_norm: float
        Normalized multinomial entropy of the network.
    
    Methods
    -------
    do_the_work
    display
    """
    def __init__(self, network, case=0, log_base=2, name='network', edges_matrix=None):
        self.network = network
        self.case = case
        self.log_base = log_base
        self.name = name
        self.verticies = nx.nodes(self.network)
        self.n = int(len(self.verticies))
        n = self.n
        if edges_matrix.all() == None:
            self.edges = nx.edges(self.network)
            self.A = self._get_edges_connexions()
            self.m = int(len(self.edges))
        else:
            self.edges = edges_matrix
            self.A = edges_matrix
            self.m = int(sum([edges_matrix[i,j] if i <= j else 0
                          for j in range(n)
                          for i in range(n)]))
        n = self.n
        m = self.m
        self.nb_possible_networks = comb(int(n*(n-1)/2 + m - 1), m)
        self.d = np.array([sum([self.A[i,j] + self.A[j,i]
                                for j in range(n)])
                           for i in range(n)])
        self.Xi = None
        self.Omega = None
        self.theta = None
        self.H_mult = None
        self.H_max = None
        self.H_norm = None
    
    def _get_edges_connexions(self):
        n = self.n
        A = np.zeros((n,n))
        i = 0
        for v1 in self.verticies:
            j = 0
            for v2 in self.verticies:
                A[i,j] = len(nx.edges(nx.subgraph(self.network, [v1, v2])))
                j += 1
            i += 1
        return A
    
    def _compute_Xi(self):
        n = self.n
        m = self.m
        Xi = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    Xi[i,j] = 2 * self.d[i]*self.d[j] * self.theta[i]*self.theta[j]
        self.Xi = Xi
    
    def _compute_Omega(self):
        n = self.n
        m = self.m
        Omega = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    if self.A[i,j] != self.Xi[i,j]:
                        Omega[i,j] = log(1 - (self.A[i,j]/self.Xi[i,j]), self.log_base)
                    else:
                        print('Error: A_ij == Xi_ij for i={}, j={}'.format(i,j))
        c = 1
        if np.amin(Omega) != 0:
            c = np.amin(Omega)
            Omega = Omega / c
        self.Omega = Omega
    
    def _compute_theta(self):
        
        def f(x, n, m, case):
            d = self.d
            if case == 2:
                coef = m**2
            elif case == 3:
                coef = 4 * m**2
            sol = np.array([(x[i] / (2 * m)) * sum([d[j] * x[j] if j!=i else 0
                                                    for j in range(n)]) - 1
                            for i in range(n)])
            return sol
        
        case = self.case
        n = self.n
        m = self.m
        if case in [0, 1]:
            theta = np.ones(n)
        else:
            x0 = np.ones(n)
            theta, details, success, msg = fsolve(f, x0, args=(n, m, case), full_output=True)
            if not success:
                print('Error while computing theta! fsolve could not solve the system of equations')
                print('Details: {}'.format(details))
                print('Message: {}'.format(msg))
        self.theta = theta
    
    def _compute_H_mult(self):
        n = self.n
        m = self.m
        p = np.ones((n,n))
        p = self.Xi * self.Omega / np.sum(self.Xi * self.Omega)
        self.H_mult = multinomial_entropy(n, m, self.Xi, self.Omega, p, self.log_base)
    
    def _compute_H_max(self):
        case = self.case
        n = self.n
        m = self.m
        if case == 3:
            p = [[2 / (n * (n-1)) if i<j else 0
                  for j in range(n)]
                 for i in range(n)]
            p = np.array(p)
        else:
            print('Error: p_ij_max not defined for the case {}'.format(case))
        
        self.H_max = multinomial_entropy(n, m, self.Xi, self.Omega, p, self.log_base)
    
    def _compute_H_norm(self):
        self.H_norm = self.H_mult / self.H_max
    
    def __call__(self):
        """ Perform all computations to set the attributes that are currently None type.
        """
        self._compute_theta()
        self._compute_Xi()
        self._compute_Omega()
        self._compute_H_mult()
        self._compute_H_max()
        self._compute_H_norm()
    
    def __repr__(self):
        """ Display the graph of the network as well as some more information,
        n: number of nodes, m: number of edges, m/n: average number of multi-
        edges per nodes, D: density of the network, H_mult: normalized multinomial entropy.
        """
        nx.draw_circular(self.network)
        
        case = self.case
        name = self.name
        n = self.n
        m = self.m
        A = self.A
        if case in [0, 1]:
            # case with self-loops
            D = sum([1 if A[i,j] != 0 else 0
                     for i in range(n)
                     for j in range(n)]) / (n**2)
        elif case in [2, 3]:
            # case without self-loops
            D = 2*sum([1 if A[i,j] != 0 else 0
                       for i in range(n)
                       for j in range(n)]) / (n**2 - n)
        
        H_mult = self.H_mult
        H_norm = self.H_norm
        H_gcc = 'NA'
        print(' {:<20} | {:>5}  {:>5}  {:>5}  {:>5} | {:>6} '.format('Name', 'n', 'm', 'm/n', 'D', 'H_norm'))
        print((16 + 20 + 4*5 + 2*5) * '-')
        print(' {:<20} | {:>5}  {:>5}  {:>5}  {:>5} | {:>5} '.format(name, n, m, '{:.2f}'.format(m/n), '{:.2f}'.format(D), '{:.2f}'.format(H_norm)))
