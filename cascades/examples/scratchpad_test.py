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

"""Basic tests for Scratchpads."""
import functools

from absl.testing import absltest
import cascades as cc
from cascades.examples import scratchpad


class ScratchpadTest(absltest.TestCase):

  def test_sample_solution(self):
    examples = scratchpad.load_chain_examples()
    target = scratchpad.ReasonIO(
        question='Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
        reason=None,
        answer='72',
        id='test/123')

    mock_lm = cc.mock_lm(
        response=': Half of 48 is 24. 24 + 48 is 72.\nAnswer: 72\n===')

    program = functools.partial(
        scratchpad.sample_with_prompts,
        lm=mock_lm,
        target=target,
        examples=examples,
        n_prompts=3)
    sampler = cc.Sampler(model=program)  # pylint: disable=unnecessary-lambda
    trace = sampler.reify(seed=0)
    self.assertEqual('72', trace.return_value)
    self.assertEqual('test/123', trace['problem_id'].value)


if __name__ == '__main__':
  absltest.main()
