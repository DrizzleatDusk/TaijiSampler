import numpy as np

class Likelihood:
    """Likelihood template.
    
    Parameters
    ----------
    state : State
    """
    def calc_log_likelihood(self, state):
        raise NotImplementedError