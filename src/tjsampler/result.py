# Results to save and operate MCMC inference.
import numpy as np
import h5py
import matplotlib.pyplot as plt
from .plotlib import plot_corner

class Result:
    def __init__(self, filename, comm=None):
        self.filename = filename
        self.comm = comm

    def open(self, mode="r"):
        if self.comm is None:
            self.file = h5py.File(self.filename, mode)
        else:
            self.file = h5py.File(self.filename, mode, driver="mpio", comm=self.comm)
    
    def close(self):
        if hasattr(self, "file"):
            self.file.close()

    def reset(self, nstep, sampler):
        """Reset the structure of h5 file."""
        nwalker = sampler.nwalker
        ntemp = sampler.temp.ntemp
        ndim = sampler.prior.maxndim

        self.open("w")
        try:
            sample_grp = self.file.create_group("samples")
            model_dst = sample_grp.create_dataset("model", shape=(nstep, nwalker, ntemp), dtype="i4")
            param_dst = sample_grp.create_dataset("params", shape=(nstep, nwalker, ntemp, ndim), dtype="f8")
            model_dst.attrs["model"] = list(sampler.prior.ndim.keys())
            model_dst.attrs["ndim"] = list(sampler.prior.ndim.values())
            for model, prior in sampler.prior.items():
                param_dst.attrs[model] = list(prior.prior_params)

            self._model_int = dict(map(reversed, enumerate(sampler.prior.keys())))
            loglike_dst = self.file.create_dataset("log_likelihood", shape=(nstep, nwalker, ntemp), dtype="f8")
            logprior_dst = self.file.create_dataset("log_prior", shape=(nstep, nwalker, ntemp), dtype="f8")
        finally:
            self.file.close()

    def save_step(self, istep, iwalker, itemp, state):
        """Save a state drawn from MCMC.
        
        Parameters
        ----------
        istep : int
        iwalker : int
        itemp : int
        state : State
        """
        # if not hasattr(self, "file") or not bool(self.file):
        #     self.open("a")
       
        self.file["samples"]["params"][istep, iwalker, itemp, :state.ndim] = state.params
        self.file["log_prior"][istep, iwalker, itemp] = state.log_prior
        self.file["log_likelihood"][istep, iwalker, itemp] = state.log_likelihood

        self.file["samples"]["model"][istep, iwalker, itemp] = self._model_int[state.model]

    def log_likelihood(self):
        with h5py.File(self.filename, "r") as f:
            ans = f["log_likelihood"]
        return ans
    
    def log_prior(self):
        with h5py.File(self.filename, "r") as f:
            ans = f["log_prior"]
        return ans

    def samples_model(self, iwalker=None, itemp=0):
        """Sample of model.
        
        Parameters
        ----------
        itemp : index
            Index of temperature.
        """
        with h5py.File(self.filename, "r") as f:
            ans = f["samples"]["model"][()]
            if iwalker is not None:
                ans = ans[:, iwalker, :]
            if itemp is not None:
                ans = ans[..., itemp]
        return ans

    def samples_params(self, model="_", iwalker=None, itemp=0):
        """Sample of parameters.
        
        Parameters
        ----------
        model : str
            Name of model.
        iwalker : index
            Index of walker.
        itemp : int
            Index of temperature.

        Notes
        -----
        Unlike `samples_model`, walker dim is squeezed.
        """
        with h5py.File(self.filename, "r") as f:
            ans = f["samples"]["params"][()]
            if iwalker is not None:
                ans = ans[:, iwalker, :, :]
            if itemp is not None:
                ans = ans[..., itemp, :]
            dst = f["samples"]["model"]
            if len(dst.attrs) == 1:
                return ans
            model_int = list(dst.attrs["model"]).index(model)
            ndim = dst.attrs["ndim"][model_int]
            samples_model = self.samples_model(iwalker, itemp)
            idx = np.where(samples_model == model_int)
            ans = ans[idx][:, :ndim]
        return ans

    def plot_dist_model(self, itemp=0, density=True):
        """Plot distribution of model.
        
        Parameters
        ----------
        itemp : int
        """
        with h5py.File(self.filename, "r") as f:
            model = f["samples"]["model"].attrs["model"]
        samples = self.samples_model(itemp=itemp).flatten()
        n = np.bincount(samples, minlength=len(model))
        if density:
            n = n / samples.size
        fig, ax = plt.subplots()
        ax.bar(model, n, color="#4e79a7")
        return fig

    def plot_dist_params(self, model="_", itemp=0, truths=None):
        """Plot distribution of parameters.
        
        Parameters
        ----------
        itemp : int
        """
        with h5py.File(self.filename, "r") as f:
            labels = f["samples"]["params"].attrs[model]
        samples = self.samples_params(model, itemp=itemp)
        fig = plot_corner(samples, labels=labels, truths=truths)
        return fig