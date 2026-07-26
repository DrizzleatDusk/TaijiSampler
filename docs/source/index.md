# Welcome to TaijiSGWB's Documentation!

```{warning}
This is my personal project dealing with TDC II data challenge.
At present, the development of code and documentaion is far from a stable package.
```

This is a parallel tempered MCMC sampler implemented using the Message Passing Interface (MPI).
It is designed for scalable execution on cluster environments. 
However, its main drawback is that it is not particularly user‑friendly; 
a working knowledge of MPI is often required to understand and modify the code.

By design, each chain that draws samples corresponds to a single MPI process. 
In practice, multiple chains are run simultaneously at distinct temperatures. 
In this manual, such a group of processes is referred to as a *walker*. 
The package supports running several walkers in parallel.
Thus, it saves computation time and can support stretch move.

```{toctree}
:maxdepth: 2
:caption: User Guide

toy_model
rj_sampling
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

```

```{toctree}
:maxdepth: 2
:caption: Notes
```