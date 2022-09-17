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

"""Default interpreter for cascade models.

Contains an implementation of basic functionality in a single effect
handler. Intended as a safe default interpreter and for easy modification for
custom functionality.
"""
import collections
import math
from typing import Any, Dict, Optional, Set, Union

from cascades._src import handlers
import jax


class Interpreter(handlers.EffectHandler):
  """Default effect handler for interpreting a cascades program.

  Example for a simple random walk from a parameterized location:
  ```
  def random_walk():
    # 2 step random walk starting at `start` and ending at 0.
    start = yield cc.param(name='start')
    step1 = yield cc.sample(Normal(loc=start), name='step1')
    yield cc.sample(Normal(loc=start + step1), obs=0.0, name='step2')

  tracer = cc.Interpreter(random_walk, param_store=dict(start=2.0))
  tracer.run()
  ```
  """

  def __init__(self,
               gen_fn,
               seed: int = 0,
               param_store: Optional[Dict[str, Any]] = None,
               observed: Optional[Dict[str, Any]] = None,
               await_timeout: int = 60,
               verbose: bool = False,
               autoname: bool = True,
               rescore_observed: Union[Set[str], Dict[str, Any], bool] = True):
    """Instantiate a cascades interpreter.

    Args:
      gen_fn: Callable generator.
      seed: Random number seed.
      param_store: Map from parameter name to value.
      observed: Map from variable name to observed value for conditioning model.
      await_timeout: Length of time to wait for futures.
      verbose: If True, log effects.
      autoname: If True, automatically increments a unique id each time a
        variable name reappears.
      rescore_observed: if True, score observed values under their distribution.
        Otherwise, leave score as None. If it is a container, value likelihoods
        are only rescored if their name is present in this container. This
        allows selectively rescoring a subset of variables.
    """
    super().__init__(gen_fn=gen_fn)
    self._rng = jax.random.PRNGKey(seed)
    self.tape = dict()
    self._param_store = param_store or dict()
    self._observed = observed or dict()
    self._observed_used = {k: False for k in self._observed}
    self._await_timeout = await_timeout
    self._verbose = verbose
    self._likelihood_sampled = 0.0
    self._likelihood_observed = 0.0
    self._autoname = autoname
    self._name_counts = collections.defaultdict(int)
    self._exception = None
    self._rescore_observed = rescore_observed

    # Track if this interpreter has been started already.
    self._started = False

  def run(self):
    if self._started:
      raise ValueError(
          '`run` may only be called once per interpreter instance.')
    self._started = True
    gen = self()
    for eff in gen:
      if eff.should_stop:
        break
      pass
    return self

  def __getitem__(self, key):
    """Get item from trace tape by name."""
    return self.tape[key]

  @property
  def stats(self):
    return dict(
        sampled_likelihood=self._likelihood_sampled,
        observed_likelihood=self._likelihood_observed)

  def __repr__(self):
    kvs = [f'    {k}: {v}' for k, v in self.tape.items()]
    kvs = ',\n'.join(kvs)
    formatted = str(dict(stats=self.stats))
    tape = f'{{\n{kvs}\n}}'
    return formatted[:-1] + f",\n 'tape': {tape}}}"

  def process(self, effect):
    self._rng, subrng = jax.random.split(self._rng, 2)

    def log(string):
      if self._verbose:
        print(string)

    if not effect.name:
      raise ValueError(f'Must name effect: {effect}')
    if effect.name in self.tape:
      if self._autoname:
        self._name_counts[effect.name] += 1
        idx = self._name_counts[effect.name]
        effect.name = f'{effect.name}/{idx}'
      else:
        raise ValueError(f'Effect name is not unique: {effect.name}')

    if effect.name in self._observed:
      # Inject any observed values
      observed_value = self._observed[effect.name]
      self._observed_used[effect.name] = True
      if isinstance(effect, handlers.Observe):
        effect.value = observed_value
      elif isinstance(effect, handlers.Sample):
        effect = handlers.Observe(**effect.__dict__)
        effect.value = observed_value
        effect.score = None
      else:
        raise ValueError(
            f'Can only observe Sample and Observe effects. Got {effect}')

    # Sample any unseen values
    if isinstance(effect, handlers.Sample):
      # aka handlers.Sample
      effect.kwargs['rng'] = subrng
      if effect.value is None:
        random_sample = handlers.dists.sample_distribution(
            effect.fn,
            *effect.args,
            await_timeout=self._await_timeout,
            **effect.kwargs)
        effect.value = random_sample.value
        effect.score = random_sample.log_p
      if effect.score is not None:
        self._likelihood_sampled += effect.score
    elif isinstance(effect, handlers.Param):
      # aka handlers.ParamStore
      log('Param encountered: {effect}')
      if effect.name in self._param_store:
        log(f'Found parameter in store: {effect}')
        effect.value = self._param_store[effect.name]
      else:
        if effect.value is None:
          # Initialize by sampling from the distribution
          log(f'Sampling new param value: {effect.name}')
          if not effect.fn:
            raise ValueError(f'Param fn cannot be None: `{effect}`')
          random_sample = handlers.dists.sample_distribution(
              fn=effect.fn, *effect.args, **effect.kwargs)
          effect.value = random_sample.value
          effect.log_p = random_sample.log_p
        self._param_store[effect.name] = effect.value
    elif isinstance(effect, handlers.Log):
      # Do nothing. Note that Log is a subclass of Observe
      # May want to treat as an observe in order to condition on log statements.
      pass
    elif isinstance(effect, handlers.Observe):
      # aka handlers.Observer
      if effect.value is None:
        raise ValueError(f'Observe with a None value: {effect}')
      if isinstance(self._rescore_observed, bool):
        should_rescore = self._rescore_observed
      else:
        should_rescore = effect.name in self._rescore_observed

      if should_rescore:
        score = handlers.dists.score_distribution(
            effect.fn,
            effect.value,
            await_timeout=self._await_timeout,
        )
      else:
        score = None

      effect.score = score
      if score is not None:
        self._likelihood_observed += score

    if effect.score is not None:
      # Stop on Reject and infinite likelihoods
      should_stop = jax.numpy.any(jax.numpy.isinf(effect.score))
      if jax.device_get(should_stop):
        effect.should_stop = True

    # Record to a tape. aka handlers.Record
    self.tape[effect.name] = effect
    return effect

  def on_return(self, return_value):
    if 'return_value' in self.tape:
      raise ValueError(
          f'Cannot have `return_value` already recorded in tape: {self.tape.keys()}'
      )
    self.tape['return_value'] = return_value

    for k, v in self._observed_used.items():
      if not v:
        raise ValueError(f'Observed value {k} was unused')

  def __call__(self, fn_or_gen=None, nested=False):
    """Improved exception handling around the base handler."""
    try:
      return_value = yield from super().__call__(
          fn_or_gen=fn_or_gen, verbose=self._verbose, nested=nested)
      return return_value
    except Exception as e:
      self._exception = e
      print('Caught exception')
      print(e)
      self.tape['exception'] = e
      self._likelihood_observed += -math.inf
      # TODO(ddohan): Maintain more traceback context!
      raise e


def try_sample(model, capture=False, **kwargs) -> Interpreter:
  """Sample from model, with error capture.

  Args:
    model: Cascade model fn to sample.
    capture: If True, then capture and log exceptions.
    **kwargs: Arguments passed to Interpreter

  Returns:
    Interpreter instance after running through program.
  """
  tracer = Interpreter(model, **kwargs)
  try:
    tracer.run()
  except Exception as e:  # pylint: disable=broad-except
    if capture:
      print(f'Caught exception: {e}')
    else:
      raise e

  return tracer
