# Copyright 2024 The cascades Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for likelihood_weighting."""

import math
from absl.testing import absltest
import cascades as cc
from cascades._src.inference import likelihood_weighting
from numpyro import distributions as np_dists


def beta_binomial():
  pi = yield cc.sample(
      name='pi', dist=np_dists.Beta(concentration0=2.0, concentration1=1.0))
  y = yield cc.sample(name='y', dist=np_dists.Bernoulli(probs=pi))
  return y


class LikelihoodWeightingTest(absltest.TestCase):

  def test_binomial_marginal_likelihood(self):
    num_samples = 100
    inferencer = likelihood_weighting.LikelihoodWeighting(
        beta_binomial, observed=dict(y=1))
    samples = inferencer.sample(num_samples=num_samples, seed=0)
    log_probs = samples.get_column('_likelihood_observed')
    probs = [math.exp(lp) for lp in log_probs]
    marginal_probability = sum(probs) / float(num_samples)
    self.assertAlmostEqual(1.0 / 3.0, marginal_probability, places=1)


if __name__ == '__main__':
  absltest.main()
