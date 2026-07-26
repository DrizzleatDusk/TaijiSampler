from pathlib import Path
import numpy as np
from mpi4py import MPI
import tjsampler as ts
from tjsampler.constants import TWOPI

TMP_DIR = Path("../.tmp")

# ============== MPI ===============
nwalker = 4
ntemp = 2
comm = MPI.COMM_WORLD

iproc = comm.Get_rank()
ts.random.seed(iproc)

# ============== Distribution ================
cov = np.array([[1, .5], [.5, .5]])
log_cov_det = np.log(np.linalg.det(cov))
cov_inv = np.linalg.inv(cov)

# ============== MCMC ================
priors = ts.prior.Prior({"x": ts.prior.Uniform(-3, 3), "y": ts.prior.Uniform(-3, 3)})

class ToyLikelihood:
    def calc_log_likelihood(self, state):
        params = state.params
        ans = -.5 * params @ cov_inv @ params - .5 * log_cov_det - np.log(TWOPI)
        return ans
    
toylikelihood = ToyLikelihood()

move = ts.move.StretchMove(2, nwalker, comm=comm.Dup())
temp = ts.temp.Temperature(ntemp)
# move = GaussianMHMove(np.array([.2, .2])**2)

nstep = 1000
nburn = 500
thin = 20
outfile = TMP_DIR / "toy_model.h5"
params0 = priors.draw_sample()
sampler = ts.sampler.Sampler(priors, toylikelihood, move, nwalker, temp, 
                             outfile=outfile, comm=comm)
sampler.run(params0, nstep, nburn, thin)
