# MPI-based MCMC sampler for TDC II

This repository hosts my personal **work-in-progress** MCMC sampler for the Taiji Data Challenge II (TDC-II).
The core design centres on MPI parallelisation: each independent Markov chain runs as a MPI process.
See the [documentation](https://taiji-sampler.readthedocs.io) for demos.

## Installation

Before installation, ensure that the following dependencies are satisfied:

- mpi4py: requires a working MPI implementation that supports at least the MPI‑3 standard.
  I have used [Open MPI](https://www.open-mpi.org) to develop the project.

- h5py: parallel I/O features are required. See the [official guide](https://docs.h5py.org/en/stable/mpi.html) 
  for setup instructions.

Once the dependencies are in place, navigate to the project root directory and install with:

```bash
pip install .
```

## Contribution

Before contributing to this repository, please share your ideas and  discuss with me via issue, email, or other means. 
To be honest, I am not yet sure whether the current design is truly viable or effective for real‑world use.
Therefore, I strongly encourage suggestions, constructive criticism, and alternative perspectives.