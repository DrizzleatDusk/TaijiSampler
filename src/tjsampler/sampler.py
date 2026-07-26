from progressbar import progressbar
from .prior import Prior, ModelPrior
from .move import Move, ModelMove
from .state import State
from .result import Result

class Sampler:
    """Parallel sampler class.
    
    Parameters
    ----------
    prior : Prior or ModelPrior
    lilelihood : Likelihood
    move : Move
    nwalker : int
    temp : Temperature
    outfile : str
    """
    def __init__(self, 
                 prior, likelihood, move,
                 nwalker,
                 temp,
                 outfile,
                 comm):
        self.comm = comm
        self.temp = temp
        if not hasattr(temp, "comm"):
            self.temp.set_comm(comm)
        self.ntemp = temp.ntemp
        self.iproc = self.comm.Get_rank()
        if nwalker * self.ntemp != self.comm.Get_size():
            raise ValueError("Inconsistent number of processes for walker and temp setting.")
        if isinstance(prior, Prior):
            self.prior = ModelPrior({"_": (1., prior)})
        elif isinstance(prior, ModelPrior):
            self.prior = prior
        else:
            raise TypeError("Unsupported type of prior.")
        self.likelihood = likelihood
        self.nwalker = nwalker
        if isinstance(move, Move):
            self.move = ModelMove({"_": move})
        elif isinstance(move, dict):
            self.move = ModelMove(move)
        elif isinstance(move, tuple):
            self.move = ModelMove(*move)
        elif isinstance(move, ModelMove):
            self.move = move
        else:
            raise TypeError("Unsupported type of move.")
        
        self.outfile = outfile
        self.result = Result(outfile, self.comm)

    @property
    def iwalker(self):
        return self.temp.iwalker
    
    @property
    def itemp(self):
        return self.temp.itemp

    def draw_sample(self, state0, nstep, thin):
        """Yield sample.

        Parameters
        ----------
        state0 : State
        nstep: int
        thin : int
        """
        if thin < 1:
            raise ValueError("thin must be greater than 0.")    
        state = state0
        log_posterior = state.log_prior + self.temp.calc_temp_log_likelihood(state.log_likelihood)

        step_iter = range(nstep)
        if self.iproc == 0:
            step_iter = progressbar(step_iter)

        for _ in step_iter:
            for _ in range(thin):
                state, log_posterior, _ = self.move.propose(state, 
                                                            log_posterior,
                                                            self)
                self.temp.swap(state.log_likelihood)
            yield state
        
    def run(self, state0, nstep, nburn, thin):
        """Run sampler.
        
        Parameters
        ----------
        state0 : np.ndarray or State
            Initial sample.
        nstep : int
        nburn : int
        thin : int
        savefile : str
        """
        if nstep < 0:
            raise ValueError("nstep must be non-negative.")
        if not isinstance(state0, State):
            state0 = State(state0)
        if state0.log_prior is None:
            state0.log_prior = self.prior.calc_logprob(state0)
        if state0.log_likelihood is None:
            state0.log_likelihood = self.likelihood.calc_log_likelihood(state0)

        state = state0
        self.result.reset(nstep, self) 

        for state in self.draw_sample(state0, nburn, 1):
            pass
        try:
            if not hasattr(self.result, "file") or not bool(self.result.file):
                self.result.open("a")
            for i, state in enumerate(self.draw_sample(state, nstep, thin)):
                self.result.save_step(i, self.temp.iwalker, self.temp.itemp, state)
        finally:
            self.result.close()

class Samplers:
    """Specify the execution order of the sampler.
    
    Parameters
    ----------
    sampler : Sampler
    order : int
    """
    def __init__(self, sampler, order, comm):
        self.sampler = sampler
        self.order = order
        self.comm = comm
        self.iproc = comm.Get_rank()
        orders = comm.gather(order)
        if self.iproc == 0:
            orders = sorted(set(orders))
        self.orders = comm.bcast(orders)

    def afer_move(self, state, log_posterior, is_new):
        pass

    def draw_sample(self, state0, nstep, thin):
        """Yield sample."""
        if thin < 1:
            raise ValueError("thin must be greater than 0.")
        state = state0
        log_posterior = state.log_prior + self.temp.calc_temp_log_likelihood(state.log_likelihood)

        step_iter = range(nstep)
        if self.iproc == 0:
            step_iter = progressbar(step_iter)

        for _ in step_iter:
            for _ in range(thin):
                for o in self.orders:
                    self.comm.Barrier()     #TODO: replace with blocked send-recv?
                    if o == self.order:
                        state, log_posterior, is_new = self.sampler.move.propose(
                            state, log_posterior, self.sampler)
                        self.after_move(state, log_posterior, is_new)
                        self.sampler.temp.swap(state.log_likelihood)
            yield state 

    def run(self, state0, nstep, nburn, thin):
        """Run sampler.
        
        Parameters
        ----------
        """
        if nstep < 0:
            raise ValueError("nstep must be non-negative.")
        if not isinstance(state0, State):
            state0 = State(state0)
        if state0.log_prior is None:
            state0.log_prior = self.sampler.prior.calc_logprob(state0)
        if state0.log_likelihood is None:
            state0.log_likelihood = self.sampler.likelihood.calc_log_likelihood(state0)

        state = state0
        self.sampler.result.reset(nstep, self.sampler) 

        for state in self.draw_sample(state0, nburn, 1):
            pass
        try:
            if not hasattr(self.sampler.result, "file") or not bool(self.sampler.result.file):
                self.result.open("a")
            for i, state in enumerate(self.draw_sample(state, nstep, thin)):
                self.sampler.result.save_step(i, self.sampler.temp.iwalker, self.sampler.temp.itemp, state)
        finally:
            self.sampler.result.close()

    