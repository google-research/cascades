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

"""Effect handlers which specify how to interpret yielded effects."""
import collections
import dataclasses
import math
import types
from typing import Any, Callable, Dict, List, Optional, Text, Type, Union

from cascades._src.distributions import base as dists
import immutabledict
import jax

Distribution = dists.Distribution
RandomSample = dists.RandomSample

## Methods to run a generator.


def run_with_intermediates(gen):
  """Run generator to completion and return final value and intermediates.

  Args:
    gen: Generator.

  Returns:
    Dict with value = return_value and intermediates = yielded values.
  """
  assert isinstance(gen, types.GeneratorType)
  intermediates = []
  while True:
    try:
      x = next(gen)
      intermediates.append(x)
    except StopIteration as e:
      return_value = None
      if e.args is not None and len(e.args):
        return_value = e.args[0]
      break
  return dict(return_value=return_value, intermediates=intermediates)


def forward_sample(fn, seed, *args, **kwargs):
  """Sample once from fn(*args, **kwargs) with seed.

  Args:
    fn: Callable which creates a generator, yielding distributions or effects.
    seed: Random seed to use.
    *args: Args passed to fn.
    **kwargs: Kwargs passed to fn.

  Returns:
    Tuple of (handlers, result), where handlers is the effect handler stack
    used to interpret the model, and result is a dict containing `result` and
    `intermediates`.
  """
  # TODO(ddohan): add ApplyContext back in.
  # gen_fn = ApplyContext(lambda: fn(*args, **kwargs))
  gen_fn = lambda: fn(*args, **kwargs)
  # TODO(ddohan): Add in ParamHandler
  handler_fn = compose_handlers([Record, StopOnReject, Observer, Sampler])
  forward_sample_handler: Record = handler_fn(Seed(seed=seed, gen_fn=gen_fn))  # pytype: disable=annotation-type-mismatch
  result_with_metadata = forward_sample_handler.run_with_intermediates(
      verbose=False)
  result_with_metadata['observed_likelihood'] = (
      forward_sample_handler.observed_likelihood)
  return forward_sample_handler, result_with_metadata


def rejection_sample(fn, seed, max_attempts, *args, **kwargs):
  """Sample repeatedly until a success."""
  rng = dists.get_rng(seed)
  for i in range(max_attempts):
    rng, subrng = jax.random.split(rng)
    handlers, result = forward_sample(fn, subrng, *args, **kwargs)
    observed_likelihood = handlers.observed_likelihood
    if not math.isinf(observed_likelihood):
      result['attempts'] = i + 1
      return handlers, result
  return None


## Helper methods to create effects within a model.
# TODO(ddohan): Move these to a separate file.


def log(value, name=None):
  """Record a value into a trace."""
  yield Log(value=value, name=name or 'log')


def sample(dist=None, obs=None, name=None):
  """Sample value from distribution. Optionally takes observed value."""
  effect = yield from sample_and_score(dist=dist, obs=obs, name=name)
  return effect.value


def observe(dist=None, obs=None, name=None):
  """Observe that given distribution takes on observed value. Returns score."""
  effect = yield from sample_and_score(dist=dist, obs=obs, name=name)
  return effect.score


def sample_and_score(dist=None, obs=None, name=None):
  """Get RandomSample from distribution. Optionally takes an observed value."""
  if obs is None:
    effect = Sample(fn=dist, value=None, name=name)
  else:
    effect = Observe(fn=dist, value=obs, name=name)

  returned_effect = yield effect
  return returned_effect


def factor(score, name=None):
  """Add score to likelihood of current trace.

  Args:
    score: Numeric value to add to trace likelihood. Used to intervene in trace
      weights.
    name: Name for the effect.

  Yields:
    An Observe effect for the given Factor distribution.
  """
  dist = dists.Factor(factor=score)
  yield from sample(dist=dist, obs=dists.FactorSentinel(), name=name)


def reject(reason, name=None):
  """Add -inf term to likelihood of current trace."""
  dist = dists.Factor(reason=reason, factor=-math.inf)
  yield Reject(
      fn=dist,
      value=dists.FactorSentinel,
      score=-math.inf,
      name=name or 'reject')


def param(name=None, dist=None, value=None):
  """Create a parameter. Samples from `dist` by default if value not given."""
  effect = yield Param(fn=dist, value=value, name=name)
  return effect.value


def _yielded_value_to_effect(value):
  """Convert a value to an effect."""
  if isinstance(value, Effect):
    return value
  elif isinstance(value, (Distribution, dists.np_dists.Distribution)):
    # pylint does not handle dataclass inheritance properly.
    return Sample(fn=value)  # pylint: disable=unexpected-keyword-arg
  else:
    raise ValueError('Unknown effect type %s' % str(value))


## Basic Effect types.


@dataclasses.dataclass
class Effect:
  """Track state of an effect which is processed by EffectHandlers."""
  # Unique name for site which yielded this effect.
  name: Optional[Text] = None

  # Results of sampling or scoring.
  value: Optional[Any] = None
  score: Optional[float] = None

  # Callable, generally used for sampling and scoring.
  fn: Optional[Union[Callable, Distribution]] = None  # pylint: disable=g-bare-generic
  args: Optional[List[Any]] = dataclasses.field(default_factory=list)
  kwargs: Optional[Dict[Text, Any]] = dataclasses.field(default_factory=dict)

  # If True, then the generator should be halted
  # generally used when likelihood becomes infinite.
  should_stop: bool = False
  replayed: bool = False

  metadata: Optional[Any] = None


@dataclasses.dataclass
class Log(Effect):
  """Log information to the trace."""
  value: Any = None

  # Unique name for site which yielded this effect.
  name: Optional[Text] = None


@dataclasses.dataclass
class Sample(Effect):
  """A distribution together with its sampled value."""
  pass


@dataclasses.dataclass
class Observe(Effect):
  """A distribution together with its observed value."""
  pass


@dataclasses.dataclass
class Reject(Observe):
  """Reject a trace. Equivalent to -inf likelihood."""
  pass


@dataclasses.dataclass
class Param(Observe):
  """Model parameters."""
  pass


## Core effect handlers.


class EffectHandler:
  """Wraps generator fn and customizes how yielded effects are interpreted."""

  def __init__(self, gen_fn=None):
    """Wraps given generator fn. May be None if self.__call__ is not used."""
    self.gen_fn = gen_fn
    self._context = None

  def run_with_intermediates(self, fn_or_gen=None, verbose=None):
    return run_with_intermediates(self(fn_or_gen=fn_or_gen, verbose=verbose))

  def process(self, effect):
    """Applied before yield."""
    return effect

  def postprocess(self, effect):
    """Applied after yield."""
    return effect

  def on_return(self, return_value):
    """Runs on return from program."""
    pass

  def get_stack(self):
    """If wrapped gen_fn is an EffectHandler, get entire stack of handlers."""
    handlers = []
    handler = self
    # TODO(ddohan): Unwrap functools.partial as well.
    while isinstance(handler, EffectHandler):
      handlers.append(handler)
      handler = handler.gen_fn
    return handlers

  def __call__(self, fn_or_gen=None, verbose=False, nested=False):
    """Iterate generator fn, interpreting effects using (post)process.

    Args:
      fn_or_gen: If given, generator or callable returning a generator to trace.
        If not given, defaults to self.gen_fn
      verbose: If True, log out the trace.
      nested: Used internally for recursive nested calls. Changes behavior of

    Yields:
      Effects

     # Indent to satisfy pylint.
     Returns:
       Value returned from wrapped generator.
    """

    def _log(x):
      if verbose:
        print(x)

    if fn_or_gen is None:
      fn_or_gen = self.gen_fn
    if fn_or_gen is None:
      raise ValueError(
          '`gen_fn` must be passed as argument EffectHandler __init__ or __call__. Was None.'
      )

    if isinstance(fn_or_gen, types.GeneratorType):
      # assert not args and not kwargs, (args, kwargs)
      gen = fn_or_gen
    else:
      # Call function to get a generator back
      gen = fn_or_gen()

    value_to_inject = None
    return_value = None
    try:
      while True:
        _log(f'Injecting {value_to_inject}')

        yielded_value = gen.send(value_to_inject)

        _log(f'Yielded value: {yielded_value}')
        if isinstance(yielded_value, types.GeneratorType):

          # Recursively trace through yielded generator.
          # TODO(ddohan): Support yielding functions as well.
          value_to_inject = yield from self(
              fn_or_gen=yielded_value, nested=nested + 1)
          _log(f'received from yield: {value_to_inject}')
          continue
        else:
          effect = _yielded_value_to_effect(yielded_value)

          # Process & postproces modify the effect in place
          effect = self.process(effect)
          if effect is None:
            raise ValueError(f'Did not return effect from {self}')
          yield effect

          # TODO(ddohan): postprocess will be applied from outside in
          # Do we want to somehow apply it inside out as well?
          # This would require an additional yield.
          effect = self.postprocess(effect)
          if effect is None:
            raise ValueError(f'Did not return effect from {self}')
          if effect.should_stop:
            return None
          value_to_inject = effect
    except StopIteration as e:
      if e.args is not None and len(e.args):
        return_value = e.args[0]
        _log(e.args)
        return return_value
      return None
    finally:
      # Always ensure that any locally scoped state is unloaded.
      # TODO(ddohan): Test that this behaves as expected for WithContext
      # handlers when `nested=True`.
      if not nested:
        self.on_return(return_value)


def compose_handlers(handlers: List[Type[EffectHandler]]):
  """Compose together a list of handlers."""
  if not handlers:
    raise ValueError('Cannot compose an empty set of handlers.')

  def init_handlers(gen_fn=None):
    for handler_cls in handlers[::-1]:
      if handler_cls is None:
        continue
      gen_fn = handler_cls(gen_fn=gen_fn)
    return gen_fn

  return init_handlers


class Record(EffectHandler):
  """Record effects into trace."""

  def __init__(self, gen_fn=None):
    super().__init__(gen_fn=gen_fn)
    self.trace = collections.OrderedDict()
    self.keys = []
    self.observed_likelihood = 0.0
    self.unobserved_likelihood = 0.0
    self.return_value = None
    self.done = False

  def __getitem__(self, key):
    """Get variable from trace by name."""
    return self.trace[key]

  @property
  def joint_likelihood(self):
    return self.observed_likelihood + self.unobserved_likelihood

  def __repr__(self):
    kvs = [f'  {k}: {v}' for k, v in self.trace.items()]
    kvs = '\n'.join(kvs)
    return f'Record(\n{kvs}\n)'

  def on_return(self, return_value):
    """Runs on return from program."""
    self.result = return_value
    self.done = True

  def process(self, effect):
    if effect.name is None:
      raise ValueError(
          f'Cannot record trace for effect without a name: {effect}')
    if effect.name in self.trace:
      raise ValueError(f'Address `{effect.name}` is already in trace')
    self.trace[effect.name] = effect
    self.keys.append(effect.name)
    if isinstance(effect, Observe):
      if effect.score is not None:
        self.observed_likelihood += effect.score
    elif isinstance(effect, Sample):
      self.unobserved_likelihood += effect.score
    return effect


class Seed(EffectHandler):
  """Set `rng` kwarg for each effect."""

  def __init__(self, gen_fn=None, seed=None):
    super().__init__(gen_fn=gen_fn)
    if isinstance(seed, int):
      self.rng = jax.random.PRNGKey(seed=seed)
    else:
      self.rng = seed

  def process(self, effect):
    self.rng, subrng = jax.random.split(self.rng)
    effect.kwargs['rng'] = subrng
    return effect


class Sampler(EffectHandler):
  """Sample values from Distributions."""

  def __init__(self, gen_fn=None, await_timeout=None):
    super().__init__(gen_fn=gen_fn)
    self._await_timeout = await_timeout

  def process(self, effect):
    if isinstance(effect, Sample):
      if effect.value is None:
        random_sample = dists.sample_distribution(
            effect.fn,
            *effect.args,
            await_timeout=self._await_timeout,
            **effect.kwargs)
        effect.value = random_sample.value
        effect.score = random_sample.log_p
    return effect


class Observer(EffectHandler):
  """Compute score of Observe statements."""

  def __init__(self, gen_fn=None, await_timeout=None, rescore=True):
    super().__init__(gen_fn=gen_fn)
    self._await_timeout = await_timeout
    self._rescore = rescore

  def process(self, effect):
    if isinstance(effect, Observe):
      if effect.value is None:
        raise ValueError(f'Observe with a None value: {effect}')
      if isinstance(self._rescore, bool):
        should_rescore = self._rescore
      else:
        should_rescore = effect.name in self._rescore

      if should_rescore:
        score = dists.score_distribution(
            effect.fn,
            effect.value,
            await_timeout=self._await_timeout,
        )
      else:
        score = None
      effect.score = score
    return effect


class StopOnReject(EffectHandler):
  """Mark that tracing should stop in event of inf likelihood."""

  def process(self, effect):
    if (isinstance(effect, Observe) and effect.score is not None):
      should_stop = jax.numpy.any(jax.numpy.isinf(effect.score))
      if jax.device_get(should_stop):
        effect.should_stop = True
    return effect


class AutoName(EffectHandler):
  """Assign a unique name to each effect based on index in trace."""

  def __init__(self, gen_fn=None):
    super().__init__(gen_fn=gen_fn)
    self.idx = 0

  def __repr__(self):
    return f'<AutoName {self.gen_fn}>'

  def process(self, effect):
    name = effect.name
    if name is None:
      name = effect.fn.__class__.__name__
    name = f'{self.idx}/{name}'
    effect.name = name
    self.idx += 1
    return effect


class Replay(EffectHandler):
  """Look up site values from given trace."""

  def __init__(self,
               gen_fn,
               trace,
               assert_all_used=True,
               replay_scores=False,
               rescore=False,
               await_timeout=None):
    super().__init__(gen_fn=gen_fn)
    if not isinstance(trace, immutabledict.ImmutableOrderedDict):
      trace = immutabledict.ImmutableOrderedDict(trace)
    self.trace = trace
    self._unused = set(k for k in trace)
    self._assert_all_used = assert_all_used
    if replay_scores and rescore:
      raise ValueError('Cannot both replay and rescore log probs.')
    self._replay_scores = replay_scores
    self._rescore = rescore
    self._await_timeout = await_timeout

  def on_return(self, return_value):
    """Runs on return from program."""
    if self._assert_all_used and self._unused:
      raise ValueError(f'Some keys were unused in replay: {self._unused}')

  def process(self, effect):
    if effect.name in self.trace:
      cached = self.trace[effect.name]
      effect.value = cached.value
      if self._replay_scores:
        effect.score = cached.score
      elif self._rescore:
        effect.score = dists.score_distribution(
            fn=effect.fn, value=effect.value, await_timeout=self._await_timeout)
      effect.replayed = True
      self._unused.remove(effect.name)
    return effect
