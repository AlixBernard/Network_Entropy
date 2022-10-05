#!/usr/bin/env python

from distutils.core import setup


MAJOR = 0
MINOR = 3
PATCH = 0


setup(
    name="network_entropy",
    version=f"{MAJOR}.{MINOR}.{PATCH}",
    description="Library to compute the entropy of a network.",
    author="Alix Bernard",
    author_email="alix.bernard9@gmail.com",
    url="https://github.com/AlixBernard/Network_Entropy",
    packages=["network_entropy"],
)
