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

"""Tests for forward_sampling."""

from absl.testing import absltest
import cascades as cc
from cascades._src.inference import forward_sampling
from numpyro import distributions as np_dists


def binomial(k, p=0.5):
  total = 0
  for i in range(k):
    flip = yield cc.sample(name=f'flip{i}', dist=np_dists.Bernoulli(probs=p))
    total += flip
  yield cc.log(total, 'total')
  return flip


class ForwardSamplingTest(absltest.TestCase):

  def test_binomial(self):
    k = 3
    num_samples = 500

    inferencer = forward_sampling.ForwardSampling(model=binomial, k=k, p=0.5)
    samples = inferencer.sample(num_samples=num_samples, seed=0)

    # Check that the mean of total matches the binomial mean.
    total_sampled = samples.get_column('total')
    mean_total = sum(total_sampled) / float(num_samples)
    self.assertAlmostEqual(k / 2.0, mean_total, places=1)


if __name__ == '__main__':
  absltest.main()
