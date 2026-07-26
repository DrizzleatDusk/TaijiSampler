from pathlib import Path
import numpy as np
from mpi4py import MPI
import tjsampler as ts
from tjsampler.constants import TWOPI

TMP_DIR = Path("../.tmp")

# ============== MPI ===============
nwalker = 4
ntemp = 2
comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
nproc = comm.Get_size()
iproc = comm.Get_rank()

ts.random.seed(0)
seed = None
if iproc == 0:
    seed = ts.random.create_seeds(nproc)
seed = comm.scatter(seed)
ts.random.seed(seed)

# ============== Generate dataset ================
nsample = 100
noise_sigma = .1
x = np.linspace(0, 1, nsample)
nbyte = nsample * np.dtype(np.float64).itemsize if iproc == 0  else 0
win = MPI.Win.Allocate_shared(nbyte, comm=comm)
shared_data, _ = win.Shared_query(0)
signal = np.ndarray(shape=nsample, dtype=np.float64, buffer=shared_data)
win.Fence()
if iproc == 0:
    noise = ts.random.rng.normal(0., noise_sigma, nsample)
    hsignal = 1. + .05 * x
    signal[:] = noise + hsignal

    import matplotlib.pyplot as plt
    plt.plot(x, hsignal, color="tab:red", label="$h$")
    plt.plot(x, signal, label="$s$")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(TMP_DIR/"rj_sampling_signal.pdf", dpi=150, bbox_inches="tight")
    plt.close()
win.Fence()

# ============== Prior ================
priors = {
    "degree0": (.5, {"a0": ts.prior.Uniform(.5, 1.5)}),
    "degree1": (.5, {"a0": ts.prior.Uniform(.5, 1.5),
                 "a1": ts.prior.Uniform(-.2, .2)})
}
priors = ts.prior.ModelPrior(priors)

# ============== Likelihood ================
class ToyLikelihood:
    def calc_log_likelihood(self, state):
        h_hypo = state.params[0]
        if state.model == "degree1":
            h_hypo += state.params[1] * x
        residual = signal - h_hypo
        ll = -.5 * np.sum(residual**2) / noise_sigma**2 - nsample * np.log(TWOPI**.5 * noise_sigma)    
        return ll

toylikelihood = ToyLikelihood()

# ============== Move ================
within_move = {
    "degree0": ts.move.GaussianMHMove(np.array([.05])**2),
    "degree1": ts.move.GaussianMHMove(np.array([.05, .01])**2)
}
rjmove0 = ts.move.RJMove(["degree1"], 
                         add_params=[ts.prior.Prior(a1=ts.prior.Uniform(-.2, .2))], 
                        )
rjmove1 = ts.move.RJMove(["degree0"])
between_move = {"degree0": rjmove0, "degree1": rjmove1}
move = ts.move.ModelMove(within_move, between_move)

temp = ts.temp.Temperature(ntemp, comm=comm)

# ============== RUN ================
nstep = 1500
nburn = 500
thin = 30
outfile = TMP_DIR / "rj_sampling.h5"
state0 = priors.draw_sample()
sampler = ts.sampler.Sampler(priors, toylikelihood, move, nwalker, temp, 
                             outfile=outfile, comm=comm)
sampler.run(state0, nstep, nburn, thin)
