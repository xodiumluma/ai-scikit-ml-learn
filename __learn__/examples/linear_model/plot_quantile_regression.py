"""
----QUANTILE REGRESSION----

Here we learn how QR can predict non-trivial conditional quantiles.

On the left is when the error distribution is normal but has non-constant variance, i.e. with heteroscedasticity.

On the right is an asymmetric error distribution (Pareto)

"""

# Authors: scikit-learn devs
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Dataset generation
# ==================
#
# Generate two synthetic datasets to illustrate how QR works.
# True randomness generated for both datasets will be made up of the 
# same expected value with a linear relationship with a single feature `x`
import numpy as np

rng = np.random.RandomState(42)
x = np.linspace(start=0, stop=10, num=100)
X = x[:, np.newaxis]
y_true_mean = 10 + 0.5 * x

# %%
# Create two challenges by amending distribution of target `y` while keeping same expected value
#
# - heteroscedastic Normal noise added for first challenge
# - asymmetric Pareto noise added for second challenge
y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])
a = 5
y_pareto = y_true_mean + 10 * (rng.pareto(a, size=x.shape[0]) - 1 / (a - 1))

