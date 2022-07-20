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

"""String distributions."""

from concurrent import futures
import dataclasses
import math
from typing import Any, Text

from cascades._src.distributions import base as dists


DEFAULT_LM_CLIENT = dict(default=None)


def set_default_lm(lm):
  """Set the default language model client."""
  DEFAULT_LM_CLIENT['default'] = lm


def get_default_lm():
  """Return a default language model client."""
  lm = DEFAULT_LM_CLIENT.get('default')
  if not lm:
    raise ValueError('No default LM set.')
  return lm


def removeafter(text: Text, token: Text = '==='):
  """Truncate `text` after until `token`."""
  idx = text.find(token)
  if idx >= 0:
    text = text[:idx + len(token)]
    return text
  else:
    return None


def mock_lm(response: Text) ->...:
  """Return Distribution class which will return a fixed response."""

  @dataclasses.dataclass(frozen=True)
  class MockLM(dists.Distribution):
    prompt: Text = ''

    def sample(self, rng):
      del rng
      return dists.RandomSample(value=response, log_p=-1.0)

    def prob(self, value):
      return 1.0

  return MockLM


# TODO(ddohan): Use cascade machinery + yields here as well
def iterate_until(rng,
                  prompt: Text,
                  lm,
                  k: int = 2,
                  until: Text = '\n',
                  reject_if_missing=True,
                  timeout=60,
                  rescore=True):
  """Sample from language model up to `k` times.

  Stop at given `until` token.

  Args:
    rng: Random seed.
    prompt: Prefix text to condition on.
    lm: Language model distribution to use.
    k: Number of times to iterate the sampling.
    until: Sample until this token is found.
    reject_if_missing: If True, then reject the trace if `until` is not found.
    timeout: Maximum time to wait for a generation.
    rescore: If true, then rescores the generation after truncation, as an
      `Observe`

  Returns:
    None if rejected, otherwise the generated text.
  """
  full_generation = ''
  likelihood = 0.0
  for _ in range(k):
    current_prompt = prompt + full_generation
    step_lm = lm(prompt=current_prompt)
    generation = step_lm.sample(rng=rng)
    if isinstance(generation, futures.Future):
      generation = generation.result(timeout)
    full_generation += generation.value
    likelihood += generation.log_p
    if until:
      clipped = removeafter(text=full_generation, token=until)
      if clipped is not None:
        if rescore:
          # print(f"rescore: `{clipped}`")
          score = lm(prompt=prompt).score(clipped)
          if isinstance(score, futures.Future):
            score = score.result(timeout)
        else:
          score = 0.0
        # preclip = clipped
        clipped = clipped[:-len(until)]
        # print(f'`{preclip}` -> `{clipped}`')
        return dists.RandomSample(log_p=score, value=clipped)
  if until and reject_if_missing:
    # We were looking for something and we didn't find it.
    # Use a -inf likelihood
    likelihood = -math.inf
  return dists.RandomSample(log_p=likelihood, value=full_generation)


@dataclasses.dataclass(eq=True, frozen=True)
class String(dists.Distribution):
  """Sample a String from an LM. May iterate calls to the LM multiple times."""

  prompt: Text = ''
  k: int = 2
  until: Text = '\n'
  lm: Any = None
  timeout: int = 60

  def _get_lm(self):
    return self.lm or get_default_lm()

  def sample(self, rng):
    rng = dists.get_rng(rng)
    value = iterate_until(
        rng=rng,
        lm=self._get_lm(),
        prompt=self.prompt,
        until=self.until,
        k=self.k,
        timeout=self.timeout)
    return value

  def log_prob(self, value):
    if not isinstance(value, str):
      value = f' {value}'
    if self.until:
      value = value + self.until
    return self._get_lm()(prompt=self.prompt).score(sample=value)
