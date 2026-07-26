from collections import UserDict
import numpy as np
from scipy.special import erfinv
from .state import State
from . import random
from .constants import TWOPI, SQRT2

class ProbDist:
    """Probability distribution of a continuous real random variable

    Parameters
    ----------
    minimum : float
        Minimum of the domain.
    maximum : float
        Maximum of the domain.
    """
    def __init__(self, minimum=-np.inf, maximum=np.inf):
        self.minimum = minimum
        self.maximum = maximum

    def _rescale(self, sample):
        "Rescale from Uniform[0, 1] distribition."
        raise NotImplementedError

    def draw_sample(self, size=None):
        """Draw sample.

        Parameters
        ----------
        size : int or tuple of int
        """
        return self._rescale(random.rng.uniform(0, 1, size))

    @property
    def ndim(self):
        return 1

class Uniform(ProbDist):
    """Uniform distribution.
    """
    def __init__(self, minimum, maximum):
        super().__init__(minimum, maximum)

    def _rescale(self, sample):
        """Rescale from Uniform[0, 1] distribution.
        
        Parameters
        ----------
        sample : np.ndarray
        """
        return self.minimum + sample * (self.maximum - self.minimum)

    def calc_prob(self, val):
        """Calculate prior probability.
        
        Parameters
        ----------
        val : np.ndarray
        """
        return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)

    def calc_logprob(self, val):
        """Calculate log prior probability.
        
        Parameters
        ----------
        val : np.ndarray
        """
        idx = (val >= self.minimum) & (val < self.maximum)
        if np.isscalar(idx):
            if not idx:
                log_prob = -np.inf
            else:
                log_prob =  -np.log(self.maximum - self.minimum)
        else:
            log_prob = np.full_like(val, -np.inf, dtype=np.float64)
            log_prob[idx] = -np.log(self.maximum - self.minimum)
        return log_prob
        
class LogUniform(ProbDist):
    """Log uniform distribution."""
    def __init__(self, minimum, maximum):
        if minimum <= 0.:
            raise ValueError("Minimum of LogUniform must larger than 0.")
        super().__init__(minimum, maximum)

    def _rescale(self, sample):
        return self.minimum * np.exp(sample * np.log(self.maximum / self.minimum))

    def calc_prob(self, val):
        idx = (val >= self.minimum) & (val < self.maximum)
        if np.isscalar(idx):
            if not idx:
                prob = 0.
            else:
                prob = 1 / val / np.log(self.maximum/self.minimum) 
        else:
            prob = np.zeros_like(val, dtype=float)
            prob[idx] = 1 / val[idx] / np.log(self.maximum/self.minimum)  
        return prob

    def calc_logprob(self, val):
        idx = (val >= self.minimum) & (val < self.maximum)
        if np.isscalar(idx):
            if not idx:
                log_prob = -np.inf
            else:
                log_prob =  -np.log(val * np.log(self.maximum/self.minimum))
        else:
            log_prob = np.full_like(val, -np.inf, dtype=np.float64)
            log_prob[idx] = -np.log(val[idx] * np.log(self.maximum/self.minimum))
        return log_prob

class Gaussian(ProbDist):
    """Gaussian distribution.
    
    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super().__init__()

    def _rescale(self, sample):
        return self.mu + erfinv(2*sample - 1) * SQRT2 * self.sigma

    def calc_prob(self, val):
        return np.exp(-(val - self.mu)**2 / self.sigma**2 / 2) / TWOPI**.5 / self.sigma
    
    def calc_logprob(self, val):
        return -.5 * (val-self.mu)**2 / self.sigma**2 - np.log(TWOPI**.5 * self.sigma)
    
class LogGaussian(ProbDist):
    """Log Gaussian distribution.
    
    Parameters
    ----------
    mu : float
    sigma : float
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super().__init__(minimum=0.)

    def _rescale(self, sample):
        return np.exp(self.mu + self.sigma * SQRT2 * erfinv(2 * sample - 1))
    
    def calc_prob(self, val):
        idx = val > self.minimum
        if np.isscalar(idx):
            if not idx:
                prob = 0.
            else:
                prob = np.exp(-(np.log(val) - self.mu)**2 / self.sigma**2 / 2) / TWOPI**.5 / self.sigma / val
        else:
            prob = np.zeros_like(val, dtype=float)
            prob[idx] = np.exp(-(np.log(val[idx]) - self.mu)**2 / self.sigma**2 / 2) / TWOPI**.5 / self.sigma / val[idx] 
        return prob
    
    def calc_logprob(self, val):
        idx = val > self.minimum
        if np.isscalar(idx):
            if not idx:
                log_prob = -np.inf
            else:
                log_prob = -.5 * (np.log(val) - self.mu)**2 / self.sigma**2 - np.log(TWOPI**.5 * self.sigma * val)
        else:
            log_prob = np.full_like(val, -np.inf, dtype=np.float64)
            log_prob[idx] = -.5 * (np.log(val[idx]) - self.mu)**2 / self.sigma**2 - np.log(TWOPI**.5 * self.sigma * val[idx])
        return log_prob 

class Prior(UserDict):
    """Dictionary wrapping prior distribution.
    
    Parameters
    ----------
    """
    def calc_prob(self, val):
        """Calculate joint prior probability.
        
        Parameters
        ----------
        val : np.ndarray
        """
        prob = np.prod([prior.calc_prob(val_) for prior, val_ in zip(self.values(), val)], 
                      axis=0)
        return prob

    def calc_logprob(self, val):
        """Calculate joint log prior probability.
        
        Parameters
        ----------
        val : np.ndarray
        """
        logprob = np.sum([prior.calc_logprob(val_) for prior, val_ in zip(self.values(), val)], 
                      axis=0)
        return logprob

    def draw_sample(self, size=None):
        """Draw sample.

        Parameters
        ----------
        size : int or tuple of int
        """
        sample = np.stack([prior._rescale(random.rng.uniform(0, 1, size))
                           for prior in self.values()])
        return sample
    
    @property
    def ndim(self):
        return len(self)
    
    @property
    def maxndim(self):
        return len(self)

class ModelProbDist:
    """Probability distribution of a model.
    
    Parameters
    ----------
    prior_model : float
    prior_params : dict or Prior
    """
    def __init__(self, prior_model, prior_params):
        self.prior_model = prior_model
        self.log_prior_model = np.log(prior_model)
        self.prior_params = prior_params
        if not isinstance(prior_params, Prior):
            self.prior_params = Prior(self.prior_params)

    def calc_logprob(self, val):
        logprob = self.log_prior_model + self.prior_params.calc_logprob(val)
        return logprob

    def draw_sample(self, size=None):
        sample = self.prior_params.draw_sample(size)
        return sample

    @property
    def ndim(self):
        return self.prior_params.ndim
    
class ModelPrior(UserDict):
    """Dictionary wrapping model prior distribution.
    
    Parameters
    ----------
    """
    def __setitem__(self, key, val):
        if not isinstance(val, ModelProbDist):
            val = ModelProbDist(*val)
        super().__setitem__(key, val)
        self.prior_models = [modelprobdist.prior_model for modelprobdist in self.values()]

    def calc_logprob(self, state):
        logprob = self[state.model].calc_logprob(state.params)
        return logprob

    def draw_sample(self):
        model = random.rng.choice(self, p=self.prior_models)
        params = self[model].draw_sample()
        state = State(params, model=model)
        return state

    @property
    def ndim(self):
        return {name: prior.ndim for name, prior in self.items()}
    
    @property
    def maxndim(self):
        return max(self.ndim.values())