class State:
    """State of sample.
    
    Parameters
    ----------
    sample : array_like
    model_name : str
    """
    def __init__(self, params, log_prior=None, log_likelihood=None, model="_"):
       self.params = params
       self.log_prior = log_prior
       self.log_likelihood = log_likelihood
       self.model = model

    @property
    def ndim(self):
        return self.params.size