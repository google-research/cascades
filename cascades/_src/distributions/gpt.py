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

"""Sampling from OpenAI api."""
import bisect
import dataclasses
import functools
import os
from typing import Iterable, Optional, Text
import uuid

from cascades._src.distributions import base as dists
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')


# TODO(ddohan): Persist cache to disk
@functools.lru_cache()
def cached_completion(rng=None, **kwargs):
  del rng
  return openai.Completion.create(**kwargs)


@dataclasses.dataclass(eq=True, frozen=True)
class GPT(dists.Distribution):
  """Sample a String from GPT."""

  prompt: Text = ''

  stop: Optional[Iterable[Text]] = ('\n',)

  engine = 'davinci-codex'
  temperature: float = 0.7
  max_tokens: int = 128
  top_p: float = .95
  frequency_penalty: int = 0
  presence_penalty: int = 0

  def sample(self, rng=None, raw=False):
    """Sample a value from the distribution given rng key.

    Args:
      rng: Optional random key.
      raw: If True, return OpenAI api request verbatim.

    Returns:
      a RandomSample or, if raw is True, the OpenAI response dict.
    """
    if rng is None:
      rng = uuid.uuid4().hex
    elif not isinstance(rng, int):
      raise ValueError('RNG must be an integer or Jax key.')
    result = cached_completion(
        rng=rng,
        model=self.engine,
        prompt=self.prompt,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        top_p=self.top_p,
        frequency_penalty=self.frequency_penalty,
        presence_penalty=self.presence_penalty,
        logprobs=0,
        stop=self.stop,
        echo=False,
    )['choices'][0]

    completion = result['text']
    logprobs = result['logprobs']

    span = (0, len(completion) + 1)
    start = 0
    end = bisect.bisect_left(logprobs['text_offset'], span[1])
    total_logprob = sum(result['logprobs']['token_logprobs'][start:end])
    if raw:
      return (total_logprob, result)
    return dists.RandomSample(score=total_logprob, value=completion)

  def log_prob(self, value, raw=False):
    """Get log prob of completion.

    Args:
      value: Completion to score.
      raw: If True, return raw OpenAI API response.

    Returns:
      float logprob or the OpenAI response dict.
    """
    text = self.prompt + value
    span = (len(self.prompt), len(text))
    result = cached_completion(
        rng=0,
        model=self.engine,
        prompt=text,
        temperature=self.temperature,
        max_tokens=0,
        echo=True,
        top_p=self.top_p,
        frequency_penalty=self.frequency_penalty,
        presence_penalty=self.presence_penalty,
        logprobs=0,
    )['choices'][0]
    logprobs = result['logprobs']
    start = bisect.bisect(logprobs['text_offset'], span[0]) - 1
    end = bisect.bisect_left(logprobs['text_offset'], span[1])
    excerpt = text[span[0]:span[1]]
    joined = ''.join(logprobs['tokens'][start:end])
    assert excerpt in joined
    total_logprob = sum(logprobs['token_logprobs'][start:end])
    if raw:
      return (total_logprob, result)
    return total_logprob
