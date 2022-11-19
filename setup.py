#!/usr/bin/env python3
with open("README.md") as f:
    readme = f.read()

from setuptools import setup

setup(
    name="BayesMatch",
    version="0.1.0",
    description='Bayesian multi-view clustering for matching records of two '
                'different type with share attributes (Maurya and Telang, 2017).',
    url="",  # TODO
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0",],
    install_requires=[
        "numpy>=1.17.2",
        "scipy>=1.9.3"
    ],
    py_modules=['BayesMatch']
)