# Copyright 2025 The cascades Authors.
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

"""Tests for rejection_sampling."""
from absl.testing import absltest
import cascades as cc
from cascades._src.inference import rejection_sampling
from numpyro import distributions as np_dists


def truncated_2step_gaussian(cutoff):
  """Two step random walk with rejection criteria."""
  step1 = yield cc.sample(name='step 1', dist=np_dists.Normal(loc=0, scale=1))
  step2 = yield cc.sample(name='step 2', dist=np_dists.Normal(loc=0, scale=1))
  output = step1 + step2
  if abs(output) >= cutoff:
    yield cc.reject('Out of bounds')
  yield cc.log(output, 'output')
  return output


def binomial(k, p=0.5):
  total = 0
  for i in range(k):
    flip = yield cc.sample(name=f'flip{i}', dist=np_dists.Bernoulli(probs=p))
    total += flip
  yield cc.log(total, 'total')
  return flip


class RejectionSamplingTest(absltest.TestCase):

  def test_rejection_sample(self):
    """Check rejection sampling on a binomial distribution."""
    k = 3
    num_samples = 500

    s = rejection_sampling.RejectionSampling(
        model=binomial, k=k, p=0.5, max_attempts=100, observed=dict(total=1))
    samples = s.sample(num_samples=num_samples, seed=0)

    # Check that the total is always what was observed
    total_sampled = samples.get_column('total')
    for value in total_sampled:
      self.assertEqual(1, value)

    # Check that each flip has the correct distribution (should be 0.333)
    total = 0.0
    for flip_id in range(k):
      column = samples.get_column(f'flip{flip_id}')
      total += sum(column)
    self.assertAlmostEqual(1.0 / k, total / (k * num_samples), places=1)

  def test_sample_with_manual_reject(self):
    """Check sampling handles cc.reject events."""
    cutoff = 0.5
    sampler = rejection_sampling.RejectionSampling(
        model=truncated_2step_gaussian, max_attempts=100, cutoff=cutoff,
        observed=dict())
    samples = sampler.sample(num_samples=10, seed=0)
    result = samples.get_column('output')

    for y in result:
      # The cutoff is small enough that rejection should always succeed.
      self.assertIsNotNone(y)
      self.assertLess(abs(y), cutoff)


if __name__ == '__main__':
  absltest.main()
