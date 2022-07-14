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

"""Tests for string distributions."""

from absl.testing import absltest
from cascades._src import handlers as h
from cascades._src import sampler
from cascades._src.distributions import strings


class StringsTest(absltest.TestCase):

  def test_mock_lm(self):

    def fn():
      a = yield h.sample(
          strings.String(lm=strings.mock_lm('ABC'), until=None, k=2), name='a')
      b = yield h.sample(
          strings.String(lm=strings.mock_lm('XYZ'), until=None, k=1), name='b')
      return a + b

    s = sampler.Sampler(fn)
    trace = s.reify(seed=0)
    self.assertEqual('ABCABCXYZ', trace.return_value)


if __name__ == '__main__':
  absltest.main()
