# Parallel tempering
import numpy as np
from . import random

class Temperature:
    """Class of temperature control for parallel tempering.

    Parameters
    ----------
    comm : Communicator
        Sampler communicator.
    """
    def __init__(self, ntemp=1, betas=None, comm=None):
        if betas is None:
            tmax = 3.5**(ntemp - 1)    #TODO: replace the placeholder.  
            betas = np.logspace(0, -np.log10(tmax), ntemp)     
        self.betas = betas
        self.ntemp = ntemp = len(betas)
        if comm is not None:
            self.set_comm(comm)

    def set_comm(self, comm):
        """Set communicator.
        
        Parameters
        ----------
        comm : Communicator
        """
        self.comm = comm
        self.nproc = comm.Get_size()
        if self.nproc % self.ntemp != 0:
            raise ValueError("Number of processes must be multiple of temperature.")
        self.iproc = comm.Get_rank()
        self.nwalker = self.nproc // self.ntemp

        self._iwt = np.array([self.iproc//self.ntemp, self.iproc%self.ntemp], dtype=np.int16)
        if self.iproc == 0:
            self.iproc_wrt_iwt = np.arange(self.nproc, dtype=np.int16).reshape(self.nwalker, self.ntemp)
            self.iwt_wrt_iproc = np.array([(iw, it) for iw in range(self.nwalker) for it in range(self.ntemp)], 
                                           dtype=np.int16)
        else: 
            self.iproc_wrt_iwt = None
            self.iwt_wrt_iproc = None
  
    @property
    def iwalker(self):
        return self._iwt[0]

    @property
    def itemp(self):
        return self._iwt[1]

    def calc_temp_log_likelihood(self, log_likelihood):
        """Calculate log likelihood under current temperature.
        
        Parameters
        ----------
        log_likelihood : float
        """
        return log_likelihood * self.betas[self.itemp]
    
    def swap(self, log_likelihood):
        """Swap indices of temperature and walker.

        Only swap between neighbor temperature.

        Parameters
        ----------
        temp_log_likelihood : float
        """
        log_likelihoods = np.array(self.comm.gather(log_likelihood))      #TODO: use Gather?

        if self.iproc == 0:
            for it in reversed(range(self.ntemp-1)):
                ith = it + 1
                iwh = random.rng.permutation(self.nwalker)
                ip = self.iproc_wrt_iwt[:, it]
                iph = self.iproc_wrt_iwt[iwh, ith]

                log_accept_ratio = (self.betas[ith] - self.betas[it]) * (
                    log_likelihoods[ip] - log_likelihoods[iph])    # (nwalker,)                
                r = np.log(random.rng.uniform(0., 1., size=self.nwalker))
                is_accepted = r < log_accept_ratio
                
                (self.iwt_wrt_iproc[ip[is_accepted]], 
                 self.iwt_wrt_iproc[iph[is_accepted]]) = \
                (self.iwt_wrt_iproc[iph[is_accepted]],
                 self.iwt_wrt_iproc[ip[is_accepted]])
                (self.iproc_wrt_iwt[is_accepted, it], 
                 self.iproc_wrt_iwt[iwh[is_accepted], ith]) = \
                (self.iproc_wrt_iwt[iwh[is_accepted], ith],
                 self.iproc_wrt_iwt[is_accepted, it])

        self.comm.Scatter(self.iwt_wrt_iproc, self._iwt)