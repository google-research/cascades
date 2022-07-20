# Copyright 2022 The cascades Authors.
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

"""Tests for handlers."""

import math

from absl.testing import absltest
from cascades._src import handlers
import jax.numpy as jnp
from numpyro import distributions as np_dists


def _binomial(k, p=0.5):
  total = 0
  for _ in range(k):
    flip = yield handlers.sample(np_dists.Bernoulli(probs=p))
    total += flip
  return total


def _flip_paths(n, a, b):
  nheads = yield _binomial(n)
  if int(nheads) != a:
    yield handlers.reject(reason=f'nheads {nheads} != {a}')

  nheads = yield _binomial(n)
  if int(nheads) != b:
    yield handlers.reject(reason=f'nheads {nheads} != {b}')


def _gaussian_mixture(locs):
  """Standard gaussian mixture model."""
  n = len(locs)
  mixing_dist = np_dists.Categorical(probs=jnp.ones(n) / n)
  component_dist = np_dists.Normal(loc=jnp.array(locs), scale=jnp.ones(n))
  mixture = np_dists.MixtureSameFamily(mixing_dist, component_dist)
  return mixture


def gaussian_mixture_likelihood(proposal_loc=0.0,
                                proposal_scale=3.0,
                                mixture_locs=(-5.0, 5.0)):
  """Demonstrate proposing & scoring in same program."""
  # Proposal distribution
  proposal = yield handlers.sample(
      name='proposal',
      dist=np_dists.Normal(loc=proposal_loc, scale=proposal_scale))

  mixture = _gaussian_mixture(mixture_locs)

  # Add term to likelihood
  yield handlers.sample(name='score', dist=mixture, obs=proposal)

  return proposal


class SimpleTests(absltest.TestCase):

  def test_likelihood_weighting(self):
    """Sample from normal, and weight using mixture."""
    locs = [-5.0, 5.0]

    def _fn(verbose=False):
      del verbose
      return gaussian_mixture_likelihood(mixture_locs=locs)

    mixture = _gaussian_mixture(locs)
    forward_sample_handler, result = handlers.forward_sample(fn=_fn, seed=0)
    expected_score = mixture.log_prob(result['return_value'])

    self.assertAlmostEqual(expected_score, result['observed_likelihood'])
    self.assertEqual(forward_sample_handler.result, result['return_value'])

  def test_paths_rejection_samping(self):
    fn = lambda: _flip_paths(3, 1, 2)
    fn = handlers.AutoName(fn)  # Uniquely name each sample.
    _, result = handlers.rejection_sample(fn=fn, seed=0, max_attempts=100)
    effects = result['intermediates']
    nheads = sum(eff.value for eff in effects)
    self.assertEqual(3, int(nheads))


class RejectTest(absltest.TestCase):

  def test_reject(self):

    def _reject_test():
      yield handlers.log('log1', 'Log 1')
      yield handlers.reject(reason='rejected for no reason')
      yield handlers.log('log2', 'Log 2')

    _, result = handlers.forward_sample(fn=_reject_test, seed=0)
    self.assertTrue(math.isinf(result['observed_likelihood']))
    self.assertLess(result['observed_likelihood'], 0)
    self.assertIsInstance(result['intermediates'][-1], handlers.Reject)
    self.assertLen(result['intermediates'], 2)


if __name__ == '__main__':
  absltest.main()
