# Reverse Jump

```{caution}
This feature requires more testing.
```

We now solve a classical linear regression problem. 

```{note}
We explain some snippets and 
the path of complete code is `examples/rj_sampling.py`.
```

The signal consists of $100$ samples
and is composed of noise and a linear function to be estimated.

$$
s = n + h,
$$

where $n$ is Gaussian white noise with $\sigma=0.1$ and $h(x)=1+0.05x,\;x\in[0,\,1]$.

```{figure} _static/rj_sampling_signal.pdf
:width: 500px
```

Assuming that we do not know the optimal degree of polynomial to fit the data.
We will choose from degree $0$ and $1$ polynomials:

$$
h = \begin{cases}
a_0, & \mathrm{degree0}; \\
a_0 + a_1 x, & \mathrm{degree1}.
\end{cases}
$$

```{literalinclude} ../../examples/rj_sampling.py
:lines: 23-42
```

We generate the simulated $s$ in single process since all the processes should "see" identical signal.
Other processes can obtain it by `bcast` operation in principle. Here, we show another way: **shared window**.
Unlike `fork` in Unix system, processes in `mpi` program do not share identical variables even
if there is no modication.
In this example, we use only 8 processes and have known in advance that all of these are located in one node.
So, we only store one version of $s$ in physical memory of this node and all the process can access it with the shared window. 
This method will be particularly useful for giant dataset.

```{literalinclude} ../../examples/rj_sampling.py
:lines: 44-62
```

`priors` consists of prior information of each model. We need to set the probability (density) of the model 
and parameters of the model. Here, the probabilities of "degree0" and "degree1" are both $0.5$.

```{literalinclude} ../../examples/rj_sampling.py
:lines: 64-76
```

We set the method of `move` of individual model and gather them into `within_move`. 
`rjmove` is treated with a similar way. Notice that the first parameter is the model
to which can be jumped from current model.

The result of the parameter estimation is

```python
_ = result.plot_dist_model()
```

```{figure} _static/rj_sampling_modeldist.pdf
:width: 500px
```

```python
_ = result.plot_dist_params("degree1", truths=[1., .05])
```

```{figure} _static/rj_sampling_paradist.pdf
:width: 500px
```
