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

"""Tests for rejection_sampling."""
from absl.testing import absltest
import cascades as cc
from cascades._src.inference import rejection_sampling
from numpyro import distributions as np_dists


@cc.model
def truncated_2step_gaussian(cutoff):
  """Two step random walk with rejection criteria."""
  step1 = yield cc.sample(name='step 1', dist=np_dists.Normal(loc=0, scale=1))
  step2 = yield cc.sample(name='step 2', dist=np_dists.Normal(loc=0, scale=1))
  output = step1 + step2
  if abs(output) >= cutoff:
    yield cc.reject('Out of bounds')
  return output


class RejectionSamplingTest(absltest.TestCase):

  def test_sample_with_reject(self):
    """Check sampling until a trace with a finite likelihood."""
    cutoff = 0.3
    s = rejection_sampling.RejectionSampling(
        model=truncated_2step_gaussian,
        max_attempts=100,
        cutoff=cutoff)
    for seed in range(10):
      y = s.sample(seed=seed)
      self.assertLess(abs(y.output), cutoff)

  def test_sample_with_reject_conditioned(self):
    """Check that we can condition specific values."""
    cutoff = 0.5
    s = rejection_sampling.RejectionSampling(
        model=truncated_2step_gaussian, max_attempts=100,
        observe={'step 1': 2.0},
        cutoff=cutoff)
    # Jumps to 2 then to < 0.5
    y = s.sample(seed=0)
    self.assertLess(y.output, cutoff)
    self.assertEqual(2.0, y.trace['step 1'].value)


if __name__ == '__main__':
  absltest.main()
