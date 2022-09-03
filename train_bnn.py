import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro import condition

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.0')
pyro.set_rng_seed(1)


plt.rcParams["figure.dpi"] = 331
