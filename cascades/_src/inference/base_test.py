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

"""Tests for base."""

from absl.testing import absltest
import cascades as cc
from cascades._src.inference import base
import numpyro.distributions as np_dists


@base.model
def bernoulli_model(p=0.5):
  x = yield cc.sample(name='flip',
                      dist=np_dists.Bernoulli(probs=p))
  return x


@base.model
def binomial_model(k, p=0.5):
  total = 0
  for i in range(k):
    flip = yield bernoulli_model(p=p, name=str(i))
    total += flip
  return total


class BaseTest(absltest.TestCase):

  def test_model_call_model(self):
    """Test that nesting models properly handles scopes."""
    trace = binomial_model(k=3, p=0.5).sample()
    for k in ['0/flip', '0/return_value',
              '1/flip', '1/return_value',
              '2/flip', '2/return_value',
              'return_value']:
      self.assertIn(k, trace.trace)


if __name__ == '__main__':
  absltest.main()
