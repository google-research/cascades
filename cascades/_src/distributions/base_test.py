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

"""Tests for base distributions."""

import dataclasses
import random

from absl.testing import absltest
from cascades._src.distributions import base


@dataclasses.dataclass(eq=True, frozen=True)
class RandomFactor(base.Distribution):
  """Randomized likelihood for testing purposes."""

  def sample(self, rng):
    del rng
    return base.RandomSample(value=None, log_p=self.score(None))

  def score(self, value):
    del value
    return random.randint(0, 100_000_000)


class BaseTest(absltest.TestCase):

  def test_lambda(self):
    fn = lambda: 5
    dist = base.Lambda(fn=fn)
    sample = dist.sample(0)
    self.assertEqual(5, sample.value)

  def test_mem_lambda(self):
    """Test memoizing a lambda distribution."""
    fn = lambda: random.randint(0, 100_000_000)
    dist = base.Lambda(fn=fn)
    dist = base.Mem(dist=dist)
    v1 = dist.sample(0).value
    v2 = dist.sample(0).value
    v3 = dist.sample(1).value
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)

  def test_mem_sample_and_score(self):
    """Test memoizing a randomized sample & score distribution."""
    dist = RandomFactor()
    dist = base.Mem(dist=dist)
    v1 = dist.sample(0).score
    v2 = dist.sample(0).score
    v3 = dist.sample(1).score
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)

    v1 = dist.score('abc')
    v2 = dist.score('abc')
    v3 = dist.score('xyz')
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)


if __name__ == '__main__':
  absltest.main()
