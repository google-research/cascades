# Copyright 2026 The cascades Authors.
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

"""Tests for sampling without replacement."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from cascades._src.distributions import base
from cascades._src.distributions import choose
import jax
import numpy as np


class ChooseTest(parameterized.TestCase):

  @parameterized.parameters(
      (3, 2), (4, 3), (4, 4), (5, 1), (5, 0), (1, 1), (0, 0), (100, 50)
  )
  def test_ordered_sample_probability(self, n, k):
    sample = choose.Choose(options=tuple(range(n)), k=k).sample(0)
    self.assertLen(sample.value, k)
    self.assertLen(set(sample.value), k)
    self.assertTrue(set(sample.value).issubset(range(n)))
    expected = -sum(math.log(n - i) for i in range(k))
    np.testing.assert_allclose(sample.log_p, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters((3, 2), (4, 3), (4, 4))
  def test_probabilities_sum_to_one(self, n, k):
    sample = choose.Choose(options=tuple(range(n)), k=k).sample(7)
    # There are n!/(n-k)! equally likely ordered samples.
    self.assertAlmostEqual(
        math.factorial(n) / math.factorial(n - k) * math.exp(float(sample.log_p)),
        1.0,
        places=6,
    )

  @parameterized.parameters(-1, 4)
  def test_invalid_sample_count(self, k):
    with self.assertRaisesRegex(ValueError, 'k'):
      choose.Choose(options=('a', 'b', 'c'), k=k).sample(0)

  def test_cannot_draw_from_empty_options(self):
    with self.assertRaisesRegex(ValueError, 'k'):
      choose.Choose(k=1).sample(0)

  def test_integer_seed_and_key_match(self):
    dist = choose.Choose(options=('a', 'b', 'c', 'd'), k=3)
    sample = base.sample_distribution(dist, rng=7)
    other = dist.sample(jax.random.PRNGKey(7))
    self.assertEqual(sample.value, other.value)
    self.assertEqual(float(sample.log_p), float(other.log_p))


if __name__ == '__main__':
  absltest.main()
