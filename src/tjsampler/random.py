# Global random number generator.
from numpy.random import default_rng, SeedSequence

def __getattr__(name):
    if name == "rng":
        return _Generator.rng
    raise AttributeError(f"module {__name__} has no attribute {name}")

class _Generator:
    rng = default_rng()

def seed(s):
    _Generator.rng = default_rng(s)

def create_seeds(nseed):
    return SeedSequence(_Generator.rng.integers(0, 2**63 - 1, size=4)).spawn(nseed)



