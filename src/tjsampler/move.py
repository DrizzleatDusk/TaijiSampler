import numpy as np
from mpi4py import MPI
from .state import State
from . import random

class Move:
    """Move base class.
    """
    def __init__(self):
        pass

    def propose(self, state, log_posterior, sampler):
        """Propose a state
        
        Parameters
        ----------
        sampler : Sampler
        """
        return state, log_posterior, False

class MHMove(Move):
    """Metropolis-Hastings move.
    """
    def draw_proposal(self, state, calc_log_prior, calc_log_likelihood):
        """Draw a new state.
        
        Parameters
        ----------
        state : State
            Previous state.
        calc_log_prior : callable
        calc_log_likelihood : callable
                
        Returns
        -------
        new_state : State
            New state
        factor : float
            Factor to calculate accept ratio.

        """
        raise NotImplementedError
    
    @staticmethod
    def accept_proposal(log_posterior, new_log_posterior, factor):
        """Accept new state or not.
        
        Parameters
        ----------
        log_posterior : float
        new_log_posterior : float
        factor : float
        """
        log_accept_ratio = new_log_posterior - log_posterior + factor
        r = np.log(random.rng.uniform(0., 1.))
        if r < log_accept_ratio:
            return True
        return False
    
    def propose(self, state, log_posterior, sampler):
        """Propose a state.
        
        Parameters
        ----------
        state : State
            Old state
        log_posterior : float
        sampler : Sampler

        Returns
        -------
        State
        """
        calc_log_prior = sampler.prior.calc_logprob
        calc_log_likelihood = sampler.likelihood.calc_log_likelihood
        calc_temp_log_likelihood = sampler.temp.calc_temp_log_likelihood

        new_state, factor = self.draw_proposal(state, calc_log_prior, calc_log_likelihood)
        new_log_posterior = new_state.log_prior + calc_temp_log_likelihood(new_state.log_likelihood)

        if self.accept_proposal(log_posterior, new_log_posterior, factor):
            return new_state, new_log_posterior, True
        return state, log_posterior, False

class GaussianMHMove(Move):
    """Metropolis-Hastings move with Gaussian proposal.

    Parameters
    ---------- 
    cov: np.ndarray
    """
    def __init__(self, cov):
        self.cov = np.asarray(cov)

        if self.cov.ndim == 1:
            self.sigma = np.sqrt(cov)
            self.draw_proposal = self._diag_gaussian_rng
        elif self.cov.ndim == 2:
            if not np.allclose(cov, cov.T):
                raise ValueError("Covariance is not symmetry.")
            eigs = np.linalg.eigvalsh(cov)
            if not np.all(eigs > 0.):
                raise ValueError("Covariance is not positive definite.")
            self.draw_proposal = self._nondiag_gaussian_rng
        else:
            raise ValueError("Invalid dimensions of covariance.")

    def _diag_gaussian_rng(self, state):
        params = state.params
        new_params = params + self.sigma * random.rng.standard_normal(size=params.shape)
        new_state = State(new_params, model=state.model)
        return new_state
    
    def _nondiag_gaussian_rng(self, state):
        params = state.params
        new_params = random.rng.multivariate_normal(params, self.cov, check_valid="ignore")
        new_state = State(new_params, model=state.model)
        return new_state

    def propose(self, state, log_posterior, sampler):
        """Propose a state.
        
        Parameters
        ----------
        state : State
            Old state
        log_posterior : float
        sampler : Sampler

        Returns
        -------
        State
        """
        calc_log_prior = sampler.prior.calc_logprob
        calc_log_likelihood = sampler.likelihood.calc_log_likelihood
        calc_temp_log_likelihood = sampler.temp.calc_temp_log_likelihood

        new_state = self.draw_proposal(state)
        new_state.log_prior = calc_log_prior(new_state)
        # if new_state.log_prior == -np.inf:
        #     return state, log_posterior 
        new_state.log_likelihood = calc_log_likelihood(new_state) 
        new_log_posterior = new_state.log_prior + calc_temp_log_likelihood(new_state.log_likelihood)

        log_accept_ratio = new_log_posterior - log_posterior
        r = np.log(random.rng.uniform(0., 1.))
        if r < log_accept_ratio:
            return new_state, new_log_posterior, True
        return state, log_posterior, False

class StretchMove(Move):
    """Stretch move. https://arxiv.org/pdf/1202.3665
    
    Parameters
    ----------
    ndim : int
    nwalker : int
    scale : float
        Adjustable scale of random variable.
    comm : communicator
    """
    def __init__(self, ndim, nwalker, scale=2, comm=MPI.COMM_WORLD):
        if nwalker % 2 == 1:
            raise ValueError("Even number of walkers for parallel move.")
        if nwalker < 2 * ndim:
            raise ValueError("The number of walkers is not recommended to be less than twice of dimenson.") 
        self.ndim = ndim
        self.nwalker = nwalker
        self.nwalker_half = nwalker // 2
        self.a = scale
        self.comm = comm

        self.iproc = self.comm.Get_rank() 
        self.params_buf = np.empty(ndim)
        self.win = MPI.Win.Create(self.params_buf, np.dtype("float").itemsize, comm=self.comm)
        
    def _gen_rand(self):
        """Generate a random number g(z^-1) = zg(z)."""
        r = random.rng.uniform(0., 1.)
        z = ((self.a - 1) * r + 1)**2 / self.a
        return z

    def propose(self, state,
                log_posterior,
                sampler):
        ntemp = sampler.ntemp
        self.nwalker_half = sampler.nwalker // 2
        is_red = sampler.temp.iwalker < self.nwalker_half
        if self.iproc == 0:
            iw_map = np.empty((self.nwalker, ntemp), dtype=np.int16)
            iw_map[:self.nwalker_half] = random.rng.choice(np.arange(self.nwalker_half, self.nwalker, dtype=np.int16), (self.nwalker_half, ntemp))
            iw_map[self.nwalker_half:] = random.rng.choice(np.arange(self.nwalker_half, dtype=np.int16), (self.nwalker-self.nwalker_half, ntemp))
            iw_get =iw_map[sampler.temp.iwt_wrt_iproc[:, 0], sampler.temp.iwt_wrt_iproc[:, 1]]
            iproc_get = sampler.temp.iproc_wrt_iwt[iw_get, sampler.temp.iwt_wrt_iproc[:, 1]]
        else:
            iproc_get = None
        iproc_get = self.comm.scatter(iproc_get)

        calc_log_prior = sampler.prior.calc_logprob
        calc_log_likelihood = sampler.likelihood.calc_log_likelihood
        calc_temp_log_likelihood = sampler.temp.calc_temp_log_likelihood
        z = self._gen_rand()
        r = np.log(random.rng.uniform(0., 1.))

        if not is_red:
            self.params_buf[:] = state.params

        self.win.Fence(MPI.MODE_NOPUT)
        if is_red:
            self.win.Get(self.params_buf, iproc_get)
        self.win.Fence(MPI.MODE_NOPUT)
        if is_red:
            new_params = self.params_buf + z * (state.params - self.params_buf)
            new_state = State(new_params, model=state.model)
            new_state.log_prior = calc_log_prior(new_state)
            new_state.log_likelihood = calc_log_likelihood(new_state)
            new_log_posterior = new_state.log_prior + calc_temp_log_likelihood(new_state.log_likelihood)

            log_accept_ratio = new_log_posterior - log_posterior + np.log(z) * (self.ndim - 1.)
            if r < log_accept_ratio:
                self.params_buf[:] = new_params
                true_state = new_state
                true_log_posterior = new_log_posterior
                true_is_new = True
            else:
                self.params_buf[:] = state.params
                true_state = state
                true_log_posterior = log_posterior
                true_is_new = False

        self.win.Fence(MPI.MODE_NOPUT)
        if not is_red:
            self.win.Get(self.params_buf, iproc_get)
        self.win.Fence(MPI.MODE_NOPUT)
        if not is_red:
            new_params = self.params_buf + z * (state.params - self.params_buf)
            new_state = State(new_params, model=state.model)
            new_state.log_prior = calc_log_prior(new_state)
            new_state.log_likelihood = calc_log_likelihood(new_state)
            new_log_posterior = new_state.log_prior + calc_temp_log_likelihood(new_state.log_likelihood)

            log_accept_ratio = new_log_posterior - log_posterior + np.log(z) * (self.ndim - 1.)
            if r < log_accept_ratio:
                self.params_buf[:] = new_params
                true_state = new_state
                true_log_posterior = new_log_posterior
                true_is_new = True
            else:
                self.params_buf[:] = state.params
                true_state = state
                true_log_posterior = log_posterior
                true_is_new = False
        
        # self.comm.Barrier()
        return true_state, true_log_posterior, true_is_new

class RJMove(Move):
    """Reverse jump move.
    
    Parameters
    ----------
    trans_model : list of str
        List of model can be jumped to.
    trans_prob : list of float
    add_params : list of Prior
        Random number generator of previous state.
    trans_fun : list of callable
    log_jacob : list of float or callable
    """
    def __init__(self, trans_model, trans_prob=None, 
                 add_params=None,
                 trans_fun=None, log_jacob=None):
        self.trans_model = trans_model
        self.ntrans_model = len(self.trans_model)
        self.model_int = dict(enumerate(self.trans_model)) | \
                              dict(map(reversed, enumerate(self.trans_model)))

        if trans_prob is None:
            trans_prob = [1./self.ntrans_model] * self.ntrans_model
        self.trans_prob = np.atleast_1d(trans_prob)
        self.trans_logprob = np.log(self.trans_prob)
        if trans_fun is None:
            trans_fun = [None] * self.ntrans_model
        self.trans_fun = trans_fun
        if add_params is None:
            add_params = [None] * self.ntrans_model
        self.add_params = add_params
        if log_jacob is None:
            log_jacob = [0.] * self.ntrans_model
        self.log_jacob = log_jacob

    def calc_factor(self, new_model, params_added):
        new_model_int = self.model_int[new_model]
        logprob = self.trans_logprob[new_model_int] 
        if (add_params := self.add_params[new_model_int]) is not None:
            logprob += add_params.calc_logprob(params_added)
        return logprob

    def draw_proposal(self, state, calc_log_prior, calc_log_likelihood):
        new_model_int = random.rng.choice(self.ntrans_model, p=self.trans_prob)
        if (add_params := self.add_params[new_model_int]) is not None: 
            params_added = add_params.draw_sample()
            params = np.hstack([state.params, params_added])
        else:
            params_added = np.array([], dtype=np.float64)
            params = state.params
        if (trans_fun := self.trans_fun[new_model_int]) is not None:
            new_params = trans_fun(params)
        else:
            new_params = params
        new_state = State(new_params, model=self.model_int[new_model_int])
        new_state.log_prior = calc_log_prior(new_state)
        new_state.log_likelihood = calc_log_likelihood(new_state)
        return new_state, params_added

class ModelMove:
    """Move supporting multiple models.
    
    Parameters
    ----------
    within_move : dict
        Move within model.
    between_move : dict
        Move between models.
    """
    def __init__(self, within_move, between_move=None):
        self.within_move = within_move
        self.between_move = between_move
    
    # def set_comm(self, nwalker, comm):
    #     for move in self.within_move.values():
    #         move.set_comm(nwalker, comm)
    #     if self.between_move is not None:
    #         for move in self.between_move.values():
    #             move.set_comm(nwalker, comm)

    def propose_within(self, state,
                log_posterior,
                sampler): 
        """Propose a state.
        
        Parameters
        ----------
        state : State
            Old state.
        log_posterior : float
        sampler : Sampler

        Returns
        -------
        State
        """
        return self.within_move[state.model].propose(state, 
                                                    log_posterior,
                                                    sampler)
    
    def propose_between(self, state,
                log_posterior, 
                sampler):
        """Propose a state.
        """
        calc_log_prior = sampler.prior.calc_logprob
        calc_log_likelihood = sampler.likelihood.calc_log_likelihood
        calc_temp_log_likelihood = sampler.temp.calc_temp_log_likelihood

        model = state.model
        move_fwd = self.between_move[model]
        new_state, params_added = move_fwd.draw_proposal(state, calc_log_prior, calc_log_likelihood)
        new_model = new_state.model
        move_bkd = self.between_move[new_model]
        log_jacob = move_fwd.log_jacob[move_fwd.model_int[new_model]]
        if callable(log_jacob):
            log_jacob = log_jacob(np.hstack([state.params, params_added]), 
                                  new_state.params)
        if (add_params_bkd := move_bkd.add_params[move_bkd.model_int[model]]) is not None:
            idx = -add_params_bkd.ndim 
            new_params_added = new_state.params[idx:]
            new_state.params = new_state.params[:idx]
        else:
            new_params_added = np.array([], dtype=np.float64)

        new_log_posterior = new_state.log_prior + calc_temp_log_likelihood(new_state.log_likelihood)
        factor = log_jacob + (move_bkd.calc_factor(model, new_params_added) -
                             move_fwd.calc_factor(new_model, params_added))

        if MHMove.accept_proposal(log_posterior, new_log_posterior, factor):
            return new_state, new_log_posterior, True
        return state, log_posterior, False

    def propose(self, state,
                log_posterior,
                sampler): 
        
        state, log_posterior, is_new = self.propose_within(state, log_posterior, sampler)
        if self.between_move is not None:
            state, log_posterior, is_new = self.propose_between(state, log_posterior, sampler)
        return state, log_posterior, is_new     #TODO: choose between within and between move?
        # if self.between_move is None:
        #     return self.propose_within(state, log_posterior, 
        #                               calc_log_prior, calc_log_likelihood,
        #                               calc_temp_log_likelihood)
        # else:
        #     r = random.rng.uniform(0., 1.)
        #     if r > .5:
        #         return self.propose_within(state, log_posterior, 
        #                               calc_log_prior, calc_log_likelihood,
        #                               calc_temp_log_likelihood)
        #     return self.propose_between(state, log_posterior, 
        #                                calc_log_prior, calc_log_likelihood,
        #                                calc_temp_log_likelihood) 