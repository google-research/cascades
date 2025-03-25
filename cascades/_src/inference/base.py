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

"""Base types for inference."""

from concurrent import futures
import functools
from typing import Any, Dict, Optional, Sequence

from cascades._src import handlers
from cascades._src import interpreter
from cascades._src import sampler as sampler_lib
import jax


class Inferencer():
  """Base class for inference algorithms."""

  _model: Any  # Really Callable[...,Generator[X, Y, Z]] but not yet.
  _args: Sequence[Any]
  _kwargs: Dict[str, Any]

  def __init__(self, model, *args, **kwargs):  # pylint: disable=redefined-outer-name
    self._model = model
    self._args = args
    self._kwargs = kwargs
    self._reset_debug_info()

  def _reset_debug_info(self):
    self._tapes = []
    self.debug_info = dict()

  def record_sample(self, samples, tracer, **kwargs):
    datum = interpreter.tape_to_dict(tracer.tape, tracer.stats)
    samples.append_row(datum)
    samples.update_row(self._kwargs)
    samples.update_row(dict(**kwargs))
    self._tapes.append(tracer.tape)
    self.debug_info['tapes'] = self._tapes


def wrap_with_return_logging(fn):
  """Wrap a model function such that the return value is logged."""
  @functools.wraps(fn)
  def wrapped_fn(*args, **kwargs):
    ret_val = yield from fn(*args, **kwargs)
    yield handlers.log(value=ret_val, name='return_value')
    return ret_val
  return wrapped_fn


def model(fn):
  """Decorator which wraps model around a function which creates a generator.

  The output of sampling from the model is included in the trace as
  `return_value`.

  ```
  @model
  def sampling_fn(k=3, p=0.6):
    output = yield Binomial(k=k, p=p)
    return output

  # Use default arguments.
  sampling_fn.sample(seed=1)

  # Apply nondefault argument, then sample.
  sampling_fn(k=5, p=0.9).sample(seed=1)
  ```

  Args:
    fn: Generator function to wrap into a sampleable model.

  Returns:
    Function wrapped into a Model. May directly call `.sample(seed)`, or apply
    new arguments then sample: `ret_model(*args, **kwargs).sample(seed)`
  """
  partial_model = functools.partial(SampledModel, wrap_with_return_logging(fn))

  def map_kwargs(pool, seed, kwargs_list):
    """Map model sampling across a list of inputs."""
    tracers = []
    for kwargs in kwargs_list:
      configured_model = partial_model(**kwargs)
      tracer = configured_model.sample(pool=pool, seed=seed)
      tracer.kwargs = kwargs
      tracers.append(tracer)
    return tracers

  def sample(seed: int = 0, pool: Optional[futures.ThreadPoolExecutor] = None):
    return partial_model().sample(seed=seed, pool=pool)

  def sample_parallel(pool: futures.ThreadPoolExecutor,
                      seed: int = 0,
                      n: int = 1):
    return partial_model().sample_parallel(seed=seed, pool=pool, n=n)

  partial_model.map = map_kwargs
  partial_model.sample = sample
  partial_model.sample_parallel = sample_parallel
  partial_model.__name__ = fn.__name__
  return partial_model


class SampledModel(handlers.BaseModel):
  """Base class for sampling from cascade models."""

  def sample(
      self,
      seed: int = 0,
      pool: Optional[futures.ThreadPoolExecutor] = None) -> handlers.Record:
    """Sample a trace from the model.

    Args:
      seed: Random seed.
      pool: Optional threadpool for parallel execution.

    Returns:
      a Record tracer.
    """
    sampler = sampler_lib.Sampler(
        model=functools.partial(self._model, *self._args, **self._kwargs),
        observe=self._observe)
    if pool:
      tracer = sampler.build_tracer(seed)
      f = pool.submit(sampler_lib.reify, tracer=tracer)
      tracer.future = f
    else:
      tracer = sampler.reify(seed=seed)
    return tracer

  def sample_parallel(self,
                      pool: futures.ThreadPoolExecutor,
                      seed: int = 0,
                      n: int = 1):
    """Sample `n` tracers in parallel."""
    seeds = jax.random.split(jax.random.PRNGKey(seed), n)
    tracers = [self.sample(seed=seed, pool=pool) for seed in seeds]
    return tracers
