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

"""Tests for sampler."""

from absl.testing import absltest
from cascades._src import handlers as h
from cascades._src import sampler
import jax.numpy as jnp
from numpyro import distributions as np_dists


def _binomial(k, p=0.5):
  total = 0
  for _ in range(k):
    flip = yield h.sample(np_dists.Bernoulli(probs=p))
    total += flip
  return total


def _flip_paths(n, a, b):
  nheads = yield _binomial(n)
  if int(nheads) != a:
    yield h.reject(reason=f'nheads {nheads} != {a}')

  nheads = yield _binomial(n)
  if int(nheads) != b:
    yield h.reject(reason=f'nheads {nheads} != {b}')


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
  proposal = yield h.sample(
      name='proposal',
      dist=np_dists.Normal(loc=proposal_loc, scale=proposal_scale))

  mixture = _gaussian_mixture(mixture_locs)

  # Add term to likelihood
  yield h.sample(name='score', dist=mixture, obs=proposal)

  return proposal


class SamplerLikelihood(absltest.TestCase):

  def test_likelihood_weighting(self):
    """Sample from normal, and weight using mixture."""
    locs = [-5.0, 5.0]

    def _fn(verbose=False):
      del verbose
      return gaussian_mixture_likelihood(mixture_locs=locs)

    s = sampler.Sampler(
        model=lambda: h.AutoName(_fn)(), rescore=True, reparam_fn=None)  # pylint: disable=unnecessary-lambda
    tracer: h.Record = s.build_tracer(0)
    self.assertFalse(tracer.done)
    sampler.reify(tracer)
    mixture = _gaussian_mixture(locs)
    expected_score = mixture.log_prob(tracer.return_value)
    self.assertEqual(expected_score, tracer.observed_likelihood)
    self.assertTrue(tracer.done)


if __name__ == '__main__':
  absltest.main()
