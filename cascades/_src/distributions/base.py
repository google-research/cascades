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

"""Basic distribution types and helpers."""
from concurrent import futures
import dataclasses
import functools
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpyro import distributions as np_dists

DEFAULT_TIMEOUT = 60

## Base Distribution and RandomSample types.


@dataclasses.dataclass(eq=True, frozen=True)
class Distribution:
  """Each Distribution implements at least `sample` and `score`."""
  # stand in for kw_only in py3.10
  capture: Any = dataclasses.field(repr=False, default=None)

  name: Any = dataclasses.field(repr=True, default=None)
  observed: Any = dataclasses.field(repr=True, default=None)

  def __post_init__(self):
    if self.capture is not None:
      raise ValueError('Use kwargs for all arguments.')

  def sample(self, rng):
    """Draw a sample from the distribution."""
    raise NotImplementedError

  def prob(self, value):
    """Score a sample by probability under the distribution."""
    raise NotImplementedError

  def log_prob(self, value):
    """Score a sample by log probability under the distribution."""
    return jax.numpy.log(self.prob(value))

  def score(self, value):
    """Score sample. Default to `log_prob` but may be any unnormalized value."""
    return self.log_prob(value)

  def support(self):
    """Possible values which can be drawn from the distribution.

    Not easily defined for all distributions.
    """
    raise NotImplementedError


# TODO(ddohan): Can it both be frozen & do post_init casting?
# @dataclass(frozen=True, eq=True)
@dataclasses.dataclass(eq=True)
class RandomSample:
  """A value with an optional score (log_p) and source distribution."""
  # stand in for kw_only in py3.10
  capture: Any = dataclasses.field(repr=False, default=None)

  log_p: Optional[Union[float, jax.numpy.DeviceArray]] = None
  value: Any = None
  dist: Optional[Union[Distribution, functools.partial]] = None

  def __post_init__(self):
    if self.capture is not None:
      raise ValueError('Use kwargs for all arguments.')

  @property
  def score(self):
    """Potentially unnormalized score. Defaults to log_p."""
    return self.log_p


## Helper methods


# TODO(ddohan): Consider moving get_rng to separate file.
def get_rng(seed):
  """Get jax prng key from seed. Does nothing is seed is a PRNGKey."""
  if isinstance(seed, int):
    return jax.random.PRNGKey(seed)
  # TODO(ddohan): What's the right type for jax seed?
  # if isinstance(seed, chex.PRNGKey):

  # Seed is already a jax.random.PRNGKey, so we just pass it through.
  return seed


def score_distribution(fn: Union[Distribution, np_dists.Distribution],
                       value: Any,
                       await_timeout: Optional[int] = None):
  """Score value under distribution.

  Args:
    fn: Object which defines a `score` or `log_prob` method.
    value: Value to score.
    await_timeout: Length of time to wait. Defaults to DEFAULT_TIMEOUT.

  Returns:
    Float likelihood of value under distribution.
  """
  if hasattr(fn, 'score'):
    score_fn = fn.score
  elif hasattr(fn, 'log_prob'):
    score_fn = fn.log_prob
  else:
    raise ValueError('Must defined `score` or `log_prob` methods.')

  if not callable(score_fn):
    raise ValueError(f'Score method {score_fn} is not callable on {fn}')
  score = score_fn(value)

  if isinstance(score, futures.Future):
    score = score.result(timeout=await_timeout or DEFAULT_TIMEOUT)
  return score


def sample_distribution(fn, *args, await_timeout=None, **kwargs):
  """Sample value from function or distribution.

  If `fn` comes from third party distribution library (e.g tf distributions or
  NumPyro), then sample using library specific method. Otherwise, if fn is
  callable, draw a sample by calling fn.

  Args:
    fn: Callable or object which defines a `score` or `log_prob` method.
    *args: Args passed to sample fn.
    await_timeout: Length of time to wait. Defaults to DEFAULT_TIMEOUT.
    **kwargs: Kwargs passed to sample fn.

  Returns:
    Sampled value, with (potentially unnormalized) likelihood.
  """
  if isinstance(fn, np_dists.Distribution):
    # Numpyro distributions.
    key = kwargs['rng']
    kwargs['key'] = key
    del kwargs['rng']
    value = fn.sample(*args, **kwargs)
    log_p = fn.log_prob(value)
    return RandomSample(value=value, log_p=log_p)
  elif hasattr(fn, 'sample') and callable(fn.sample):
    fn = fn.sample
  random_sample = fn(*args, **kwargs)
  if isinstance(random_sample, futures.Future):
    random_sample = random_sample.result(
        timeout=await_timeout or DEFAULT_TIMEOUT)
  if not isinstance(random_sample, RandomSample):
    raise ValueError(
        f'Expected sample to return RandomSample. Got {random_sample}')
  return random_sample


## Factor distributions.


# TODO(ddohan): Consider switching sentinel to 0-dim array
class FactorSentinel:
  pass


@dataclasses.dataclass(eq=True, frozen=True)
class Factor(Distribution):
  """Add given `score` to observed likelihood.

  Used to manually add terms to model likelihood. Always returns the given
  `score` for `observe`.
  """
  factor: float = 0.0
  reason: Any = None

  def sample(self, rng):
    del rng
    return RandomSample(value=FactorSentinel(), log_p=self.factor)

  def score(self, value):
    del value
    return self.factor


## Discrete distributions


@dataclasses.dataclass(frozen=True, eq=True)
class UniformCategorical(Distribution):
  """Return a random choice from the distribution."""
  options: Tuple[Any] = tuple()  # pytype: disable=annotation-type-mismatch

  def sample(self, rng=None) -> RandomSample:
    idx = jax.random.randint(rng, (), 0, len(self.options))
    idx = int(idx)
    sample = self.options[idx]
    log_p = -jnp.log(len(self.options))
    return_value = RandomSample(value=sample, log_p=log_p)
    return return_value

  def log_prob(self, value):
    del value
    # TODO(ddohan): Assert that sample is one of the options
    log_p = -jnp.log(len(self.options))
    return log_p

  def support(self):
    return self.options
