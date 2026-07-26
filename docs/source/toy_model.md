# The first toy model

```{literalinclude} ../../examples/toy_model.py
:lines: 1-7
```

In this tutotial, we will demonstrate the basic usage of `Sampler` by a toy model:
generating multivariate normal posterior with regard to $x,\,y$.
The covariance is

$$
\begin{bmatrix}1 & 1/2 \\ 1/2 & 1/2\end{bmatrix}.
$$

We wiil adopt a uniform prior and hence the likelihood takes the form of posterior.

```{literalinclude} ../../examples/toy_model.py
:lines: 9-20
```

The code snippet implies $2$ temperatue chains and $4$ walkers are involved.
To ensure independent sampling between processes, we must use different seeds to initialize
the ramdom variable generator, which are the process IDs in this simple case. 
(A better method is numpy `SeedSequence`.)

```{literalinclude} ../../examples/toy_model.py
:lines: 22-34
```

The `Likelihood` should implement `calc_log_likelihood` method. The type of parameter passed in
is defined in `state` module. We use the affine invariant [stretch move](https://msp.org/camcos/2010/5-1/p04.xhtml) is this demo.
This proposal need to transfer message between processes and hence we copy the communicator to 
avoid interference with temperature swap.

```{literalinclude} ../../examples/toy_model.py
:lines: 37-44
```

The result will save to a *.hdf5* file. The user may manipulate it with `h5py`. 
Alternatively, `Result` module encapsulate some functions. For example,

```python
filename = "../examples/toy_model.h5"
result = tj.result.Result(filename)

samples = result.samples_params()
np.cov(samples[:, 0], samples[:, 1]) #[[1.04255178, 0.51451805], [0.51451805, 0.53013195]])

_ = result.plot_dist_params()
```

```{figure} _static/toy_model.pdf
:width: 500px
```

## Notes about CLI

This demo is simple enough to run with

```bash
mpiexec -n 8 python toy_model.py
```

directly. While cluster is often eqiupped with a job scheduling system, we take 
[Slurm](https://slurm.schedmd.com/overview.html) as an example.
The program can be run by submitting a script like

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

mpiexec -n 8 python -u toy_model.py 
```

Each MCMC chain is a `mpi` process by design of the package. 
It is common for the number of processes to exceed the usable cores of CPU.
If this happens, ignore the `ntask` and `OVERSUBSCRIB`  your hosts:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

mpiexec --mca mpi_yield_when_idle 1 --map-by :OVERSUBSCRIB 8 python -u toy_model.py 













